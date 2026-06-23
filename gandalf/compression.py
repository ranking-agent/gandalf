"""Zstandard HTTP body compression + decompression middleware.

A single ASGI middleware that, using the same code path:

* **Decompresses** incoming request bodies whose ``Content-Encoding`` is
  ``zstd`` before they reach route handlers, and
* **Compresses** outgoing response bodies with zstandard when the client
  advertises ``Accept-Encoding: zstd``.

It is implemented as a raw ASGI middleware (rather than Starlette's
``BaseHTTPMiddleware``) because it needs to rewrite both the request body the
application reads and the response body the client receives — which requires
manipulating the ASGI ``scope``, ``receive`` and ``send`` directly.
"""

from typing import Awaitable, Callable, Iterable, List, Tuple

import zstandard

ASGIApp = Callable[..., Awaitable[None]]
Scope = dict
Message = dict
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]

# Statuses that never carry a body — never compress these.
_BODILESS_STATUSES = frozenset({204, 304})


def _get_header(headers: Iterable[Tuple[bytes, bytes]], name: bytes) -> bytes:
    """Return the (lowercased) value of an ASGI header, or ``b""`` if absent."""
    name = name.lower()
    for key, value in headers:
        if key.lower() == name:
            return value
    return b""


async def _read_request_body(receive: Receive) -> bytes:
    """Drain the full request body from the ASGI ``receive`` channel."""
    chunks: List[bytes] = []
    more_body = True
    while more_body:
        message = await receive()
        if message["type"] != "http.request":
            # e.g. http.disconnect — stop reading.
            break
        chunks.append(message.get("body", b""))
        more_body = message.get("more_body", False)
    return b"".join(chunks)


class ZstdCompressionMiddleware:
    """Bidirectional zstandard compression for HTTP request/response bodies."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        minimum_size: int = 500,
        level: int = 4,
        max_request_size_mb: int = 1000,  # 1 GB
        decompress_requests: bool = True,
        compress_responses: bool = True,
    ) -> None:
        self.app = app
        self.minimum_size = minimum_size
        self.level = level
        self.max_decompressed_bytes = max_request_size_mb * 1024 * 1024
        self.decompress_requests = decompress_requests
        self.compress_responses = compress_responses

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers: List[Tuple[bytes, bytes]] = list(scope.get("headers", []))

        # --- Inbound: decompress the request body if zstd-encoded ---------
        if (
            self.decompress_requests
            and _get_header(headers, b"content-encoding").lower() == b"zstd"
        ):
            try:
                receive, headers = await self._decompress_request(receive, headers)
            except _RequestTooLarge:
                await self._send_413(send)
                return
            except zstandard.ZstdError:
                await self._send_400(send, b"Malformed zstd request body")
                return
            scope = {**scope, "headers": headers}

        # --- Outbound: compress the response if the client accepts zstd ---
        client_accepts_zstd = (
            b"zstd" in _get_header(headers, b"accept-encoding").lower()
        )
        if self.compress_responses and client_accepts_zstd:
            await self._compress_response(scope, receive, send)
        else:
            await self.app(scope, receive, send)

    async def _decompress_request(
        self, receive: Receive, headers: List[Tuple[bytes, bytes]]
    ) -> Tuple[Receive, List[Tuple[bytes, bytes]]]:
        """Buffer + decompress the request body, capping the output size.

        Returns a replacement ``receive`` that replays the decompressed bytes
        and a header list with ``content-encoding`` removed and
        ``content-length`` updated.
        """
        compressed = await _read_request_body(receive)

        decompressor = zstandard.ZstdDecompressor()
        # Stream with a hard cap so a small "bomb" body cannot blow up memory.
        # max_output_size + 1 lets us detect overflow without materialising it.
        with decompressor.stream_reader(compressed) as reader:
            body = reader.read(self.max_decompressed_bytes + 1)
        if len(body) > self.max_decompressed_bytes:
            raise _RequestTooLarge()

        new_headers = [
            (key, value)
            for key, value in headers
            if key.lower() not in (b"content-encoding", b"content-length")
        ]
        new_headers.append((b"content-length", str(len(body)).encode("latin-1")))

        sent = False

        async def replay_receive() -> Message:
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            # Body already delivered — block on the original channel for any
            # subsequent events (e.g. http.disconnect).
            return await receive()

        return replay_receive, new_headers

    async def _compress_response(
        self, scope: Scope, receive: Receive, send: Send
    ) -> None:
        """Buffer the response body and zstd-compress it before sending."""
        start_message: Message = {}
        body_chunks: List[bytes] = []

        async def buffering_send(message: Message) -> None:
            nonlocal start_message
            msg_type = message["type"]
            if msg_type == "http.response.start":
                start_message = message
            elif msg_type == "http.response.body":
                body_chunks.append(message.get("body", b""))
                if message.get("more_body", False):
                    return
                await self._flush(start_message, b"".join(body_chunks), send)
            else:
                await send(message)

        await self.app(scope, receive, buffering_send)

    async def _flush(self, start_message: Message, body: bytes, send: Send) -> None:
        """Compress ``body`` (if eligible) and emit the buffered response."""
        status = start_message.get("status", 200)
        headers: List[Tuple[bytes, bytes]] = list(start_message.get("headers", []))

        already_encoded = bool(_get_header(headers, b"content-encoding"))
        eligible = (
            len(body) >= self.minimum_size
            and status not in _BODILESS_STATUSES
            and not already_encoded
        )

        if eligible:
            body = zstandard.ZstdCompressor(level=self.level).compress(body)
            headers = [
                (key, value)
                for key, value in headers
                if key.lower() != b"content-length"
            ]
            headers.append((b"content-encoding", b"zstd"))
            headers.append((b"content-length", str(len(body)).encode("latin-1")))

        # Advertise that the response varies on Accept-Encoding (cache safety),
        # whether or not we compressed this particular response.
        _add_vary_accept_encoding(headers)

        await send({**start_message, "headers": headers})
        await send({"type": "http.response.body", "body": body, "more_body": False})

    @staticmethod
    async def _send_413(send: Send) -> None:
        await _send_plain(send, 413, b"Decompressed request body too large")

    @staticmethod
    async def _send_400(send: Send, detail: bytes) -> None:
        await _send_plain(send, 400, detail)


class _RequestTooLarge(Exception):
    """Raised when a decompressed request body exceeds the configured cap."""


def _add_vary_accept_encoding(headers: List[Tuple[bytes, bytes]]) -> None:
    """Append ``Accept-Encoding`` to the ``Vary`` header without duplicating."""
    for i, (key, value) in enumerate(headers):
        if key.lower() == b"vary":
            existing = {v.strip().lower() for v in value.split(b",")}
            if b"accept-encoding" not in existing:
                headers[i] = (key, value + b", Accept-Encoding")
            return
    headers.append((b"vary", b"Accept-Encoding"))


async def _send_plain(send: Send, status: int, detail: bytes) -> None:
    """Send a minimal plain-text error response."""
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"text/plain; charset=utf-8"),
                (b"content-length", str(len(detail)).encode("latin-1")),
            ],
        }
    )
    await send({"type": "http.response.body", "body": detail, "more_body": False})
