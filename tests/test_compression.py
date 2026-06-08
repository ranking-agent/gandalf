"""Tests for the zstandard compression/decompression middleware.

Most cases exercise ``ZstdCompressionMiddleware`` in isolation on a tiny echo
app so the compression parameters can be controlled precisely, without pulling
in the heavyweight graph fixtures.  One case asserts the middleware is actually
wired onto the real GANDALF ``APP``.
"""

import json

import zstandard
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from gandalf.compression import ZstdCompressionMiddleware


def _make_client(**middleware_kwargs) -> TestClient:
    """Build an echo app wrapped in the compression middleware."""
    app = FastAPI()

    @app.post("/echo")
    async def echo(request: Request):
        return await request.json()

    app.add_middleware(ZstdCompressionMiddleware, **middleware_kwargs)
    return TestClient(app)


def _zstd(data: bytes) -> bytes:
    return zstandard.ZstdCompressor().compress(data)


def _maybe_unzstd(resp) -> dict:
    """Return the JSON body, decompressing manually if httpx didn't."""
    raw = resp.content
    try:
        return json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        return json.loads(zstandard.ZstdDecompressor().decompress(raw))


# ---------------------------------------------------------------------------
# Response compression (outbound)
# ---------------------------------------------------------------------------


def test_response_compressed_when_client_accepts_zstd():
    client = _make_client(minimum_size=0)
    payload = {"hello": "world", "items": list(range(50))}

    resp = client.post("/echo", json=payload, headers={"Accept-Encoding": "zstd"})

    assert resp.status_code == 200
    assert resp.headers["content-encoding"] == "zstd"
    assert "accept-encoding" in resp.headers.get("vary", "").lower()
    assert _maybe_unzstd(resp) == payload


def test_response_not_compressed_without_accept_encoding():
    client = _make_client(minimum_size=0)
    payload = {"hello": "world"}

    resp = client.post("/echo", json=payload, headers={"Accept-Encoding": "identity"})

    assert resp.status_code == 200
    assert "zstd" not in resp.headers.get("content-encoding", "")
    assert resp.json() == payload


def test_response_below_minimum_size_not_compressed():
    client = _make_client(minimum_size=10_000)
    payload = {"hello": "world"}

    resp = client.post("/echo", json=payload, headers={"Accept-Encoding": "zstd"})

    assert resp.status_code == 200
    assert "zstd" not in resp.headers.get("content-encoding", "")
    # Even uncompressed responses advertise Vary for cache correctness.
    assert "accept-encoding" in resp.headers.get("vary", "").lower()
    assert resp.json() == payload


# ---------------------------------------------------------------------------
# Request decompression (inbound)
# ---------------------------------------------------------------------------


def test_request_body_decompressed():
    client = _make_client()
    payload = {"a": 1, "b": [1, 2, 3], "msg": "decompress me"}
    body = _zstd(json.dumps(payload).encode())

    resp = client.post(
        "/echo",
        content=body,
        headers={
            "Content-Type": "application/json",
            "Content-Encoding": "zstd",
            "Accept-Encoding": "identity",
        },
    )

    assert resp.status_code == 200, resp.text
    assert resp.json() == payload


def test_round_trip_compressed_request_and_response():
    client = _make_client(minimum_size=0)
    payload = {"round": "trip", "items": list(range(50))}
    body = _zstd(json.dumps(payload).encode())

    resp = client.post(
        "/echo",
        content=body,
        headers={
            "Content-Type": "application/json",
            "Content-Encoding": "zstd",
            "Accept-Encoding": "zstd",
        },
    )

    assert resp.status_code == 200, resp.text
    assert resp.headers["content-encoding"] == "zstd"
    assert _maybe_unzstd(resp) == payload


def test_decompressed_body_too_large_returns_413():
    client = _make_client(max_request_size_mb=1)
    # ~2 MB of highly compressible data -> tiny compressed, but exceeds the cap.
    payload = {"data": "x" * 2_000_000}
    body = _zstd(json.dumps(payload).encode())

    resp = client.post(
        "/echo",
        content=body,
        headers={
            "Content-Type": "application/json",
            "Content-Encoding": "zstd",
            "Accept-Encoding": "identity",
        },
    )

    assert resp.status_code == 413


def test_malformed_zstd_request_returns_400():
    client = _make_client()

    resp = client.post(
        "/echo",
        content=b"this is not zstd",
        headers={
            "Content-Type": "application/json",
            "Content-Encoding": "zstd",
            "Accept-Encoding": "identity",
        },
    )

    assert resp.status_code == 400


def test_uncompressed_request_passthrough():
    client = _make_client()
    payload = {"plain": "request"}

    resp = client.post("/echo", json=payload, headers={"Accept-Encoding": "identity"})

    assert resp.status_code == 200
    assert resp.json() == payload


# ---------------------------------------------------------------------------
# Wiring on the real app
# ---------------------------------------------------------------------------


def test_middleware_registered_on_app(monkeypatch):
    monkeypatch.setenv("GANDALF_SKIP_PRELOAD", "true")
    monkeypatch.setenv("GANDALF_OTEL_ENABLED", "false")
    from gandalf import server

    assert any(m.cls is ZstdCompressionMiddleware for m in server.APP.user_middleware)
