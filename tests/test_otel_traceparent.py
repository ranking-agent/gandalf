"""End-to-end tests for W3C ``traceparent`` propagation through Gandalf.

``FastAPIInstrumentor`` already handles the inbound side for both ``/query``
and ``/asyncquery``: it extracts the incoming ``traceparent`` and makes the
server span a child of the caller's span, so Jaeger joins the two sides by
trace_id.  The only piece that does NOT happen for free is the async
callback POST — httpx is not auto-instrumented, and the FastAPI background
task runs on a worker thread that does not inherit the request's OTel
contextvars.  Without explicit propagation the callback receiver would
start a brand-new disconnected trace.

These tests verify that ``/asyncquery`` forwards the originating trace
context to the eventual callback POST so the full chain stays linked.

They reload :mod:`gandalf.server` with OpenTelemetry enabled; the shared
``conftest.py`` disables it by default so the rest of the suite does not
require the instrumentation packages.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest

# Skip the whole module if OTel instrumentation isn't available — the CI
# matrix installs only ``[dev,server]`` extras, which omit OpenTelemetry.
pytest.importorskip("opentelemetry.instrumentation.fastapi")

from fastapi.testclient import TestClient  # noqa: E402

import gandalf.otel

from tests.search_fixtures import graph  # noqa: F401, E402

_QUERY_MESSAGE = {
    "query_graph": {
        "nodes": {
            "n0": {"ids": ["CHEBI:6801"]},
            "n1": {"categories": ["biolink:Gene"]},
        },
        "edges": {
            "e0": {
                "subject": "n0",
                "object": "n1",
                "predicates": ["biolink:affects"],
            },
        },
    },
}

# A valid W3C traceparent: version-trace_id-parent_id-flags.
# trace_id chosen from the W3C spec example so failures are easy to spot.
_INCOMING_TRACEPARENT = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
_INCOMING_TRACE_ID = "0af7651916cd43dd8448eb211c80319c"

# A well-formed ``pk`` baggage value: 36 chars of [0-9A-Za-z-], matching the
# regex guard in gandalf.otel (a UUID string qualifies).
_VALID_PK = "0af76519-16cd-43dd-8448-eb211c80319c"


@pytest.fixture
def otel_server(monkeypatch, graph, bmt):  # noqa: F811
    """Reload ``gandalf.server`` with OpenTelemetry enabled and the test graph."""
    monkeypatch.setenv("GANDALF_SKIP_PRELOAD", "true")
    monkeypatch.setenv("GANDALF_OTEL_ENABLED", "true")
    # Console exporter avoids requiring an OTLP collector during tests.
    monkeypatch.setenv("GANDALF_OTEL_USE_CONSOLE_EXPORTER", "true")

    import gandalf.config

    importlib.reload(gandalf.config)
    # Reload otel before server so the freshly-imported settings take effect
    # and the module-level _initialized/_provider state is reset per test.
    import gandalf.otel

    importlib.reload(gandalf.otel)
    import gandalf.server

    importlib.reload(gandalf.server)

    gandalf.server.GRAPH = graph
    gandalf.server.BMT = bmt
    # init_otel() is deferred (post_fork / lifespan startup); the bare
    # TestClient used below does not run lifespan, so initialize explicitly.
    gandalf.otel.init_otel(gandalf.server.APP)

    try:
        yield gandalf.server
    finally:
        # Restore the OTel-disabled state that the shared conftest expects so
        # later tests still see a fresh, unconfigured server module.
        monkeypatch.undo()
        importlib.reload(gandalf.config)
        importlib.reload(gandalf.otel)
        importlib.reload(gandalf.server)


def _trace_id_of(traceparent: str) -> str:
    """Return the trace_id segment of a W3C ``traceparent`` header."""
    parts = traceparent.split("-")
    assert len(parts) == 4, f"malformed traceparent: {traceparent!r}"
    return parts[1]


class _RecordingHandler(BaseHTTPRequestHandler):
    """Capture POSTed callback requests for assertion in tests."""

    received: list[dict[str, Any]] = []

    def do_POST(self):  # noqa: N802 (BaseHTTPRequestHandler API)
        length = int(self.headers.get("content-length", "0"))
        body = self.rfile.read(length) if length else b""
        self.__class__.received.append(
            {
                "path": self.path,
                "headers": {k.lower(): v for k, v in self.headers.items()},
                "body": json.loads(body) if body else None,
            }
        )
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, fmt, *args):  # silence test output
        pass


@pytest.fixture
def callback_server():
    """Run a tiny localhost HTTP server that records incoming POSTs."""
    _RecordingHandler.received = []
    server = HTTPServer(("127.0.0.1", 0), _RecordingHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}/cb", _RecordingHandler.received
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def _wait_for(predicate, timeout=5.0, interval=0.05):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def test_async_callback_forwards_request_traceparent(otel_server, callback_server):
    """The async callback POST must carry the originating ``traceparent``."""
    callback_url, received = callback_server
    client = TestClient(otel_server.APP)
    resp = client.post(
        "/asyncquery",
        headers={"traceparent": _INCOMING_TRACEPARENT},
        json={"callback": callback_url, "message": _QUERY_MESSAGE},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "Accepted"

    assert _wait_for(lambda: len(received) > 0), "callback never received the POST"
    cb = received[0]
    assert "traceparent" in cb["headers"], cb["headers"]
    assert _trace_id_of(cb["headers"]["traceparent"]) == _INCOMING_TRACE_ID


def test_query_records_pk_baggage_on_span(otel_server):
    """``/query`` records a well-formed incoming ``pk`` baggage value on its span.

    The composite propagator extracts the ``baggage`` header into the request
    context, but baggage is never attached to spans automatically -- Gandalf
    records the ``pk`` entry (when it matches the expected shape) as a ``pk``
    span attribute so it is visible in the trace backend.
    """
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    gandalf.otel._provider.add_span_processor(SimpleSpanProcessor(exporter))

    client = TestClient(otel_server.APP)
    resp = client.post(
        "/query",
        headers={"baggage": f"pk={_VALID_PK}"},
        json={"message": _QUERY_MESSAGE},
    )
    assert resp.status_code == 200, resp.text

    # Merge attributes across all spans emitted for the request; the pk
    # attribute lives on the FastAPI server span.
    attrs: dict[str, Any] = {}
    for span in exporter.get_finished_spans():
        attrs.update(span.attributes or {})

    assert attrs.get("pk") == _VALID_PK


@pytest.mark.parametrize(
    "headers",
    [
        {},  # no baggage header at all
        {"baggage": "pk=not-a-valid-pk"},  # present but malformed (not 36 chars)
        {"baggage": "team=ranking-agent"},  # baggage present, but no pk entry
    ],
)
def test_query_does_not_record_pk_when_absent_or_malformed(otel_server, headers):
    """No ``pk`` attribute is set unless a well-formed ``pk`` baggage entry arrives."""
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    gandalf.otel._provider.add_span_processor(SimpleSpanProcessor(exporter))

    client = TestClient(otel_server.APP)
    resp = client.post("/query", headers=headers, json={"message": _QUERY_MESSAGE})
    assert resp.status_code == 200, resp.text

    pk_values = [
        span.attributes["pk"]
        for span in exporter.get_finished_spans()
        if "pk" in (span.attributes or {})
    ]
    assert pk_values == []


def test_asyncquery_records_pk_baggage_on_span(otel_server, callback_server):
    """``/asyncquery`` records the incoming ``pk`` on its span and forwards baggage.

    The server span (the request that returns ``accepted``) gets the ``pk``
    attribute, and because ``inject`` carries baggage out of the request
    context, the eventual callback POST also keeps the ``baggage`` header so
    the chain stays linked.
    """
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    gandalf.otel._provider.add_span_processor(SimpleSpanProcessor(exporter))

    callback_url, received = callback_server
    client = TestClient(otel_server.APP)
    resp = client.post(
        "/asyncquery",
        headers={"baggage": f"pk={_VALID_PK}"},
        json={"callback": callback_url, "message": _QUERY_MESSAGE},
    )
    assert resp.status_code == 200, resp.text

    attrs: dict[str, Any] = {}
    for span in exporter.get_finished_spans():
        attrs.update(span.attributes or {})
    assert attrs.get("pk") == _VALID_PK

    # Baggage also rides along to the callback via the injected headers.
    assert _wait_for(lambda: len(received) > 0), "callback never received the POST"
    cb_baggage = received[0]["headers"].get("baggage", "")
    assert f"pk={_VALID_PK}" in cb_baggage


def test_async_callback_includes_traceparent_without_incoming(
    otel_server, callback_server
):
    """Even without an incoming traceparent, the callback gets one from the server-side span."""
    callback_url, received = callback_server
    client = TestClient(otel_server.APP)
    resp = client.post(
        "/asyncquery",
        json={"callback": callback_url, "message": _QUERY_MESSAGE},
    )
    assert resp.status_code == 200, resp.text

    assert _wait_for(lambda: len(received) > 0), "callback never received the POST"
    cb = received[0]
    assert "traceparent" in cb["headers"], cb["headers"]
    trace_id = _trace_id_of(cb["headers"]["traceparent"])
    assert len(trace_id) == 32
    assert int(trace_id, 16) != 0


@pytest.fixture
def uvicorn_url(otel_server, request):
    """Serve the OTel-enabled app with a real uvicorn server on a free port.

    ``TestClient`` bypasses uvicorn's socket handling, which is where the
    keep-alive context leak lives, so these tests need a real server.  The
    HTTP protocol implementation comes from ``request.param``.
    """
    import uvicorn

    config = uvicorn.Config(
        otel_server.APP,
        host="127.0.0.1",
        port=0,
        http=request.param,
        lifespan="off",
        log_level="warning",
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    assert _wait_for(lambda: server.started), "uvicorn never started"
    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=5.0)


def _traceparent(n: int) -> str:
    """A distinct, valid W3C ``traceparent`` for the *n*-th request."""
    return f"00-{n + 1:032x}-{n + 1:016x}-01"


# Large enough that uvicorn pauses reading the socket while the app consumes
# the body (its flow-control high-water mark is 64 KiB).
_LARGE_BODY_PADDING = 2 * 1024 * 1024


@pytest.mark.parametrize(
    "uvicorn_url",
    [
        "h11",
        pytest.param(
            "httptools",
            marks=pytest.mark.skipif(
                importlib.util.find_spec("httptools") is None,
                reason="httptools not installed",
            ),
        ),
    ],
    indirect=True,
)
def test_keepalive_requests_after_large_body_keep_their_own_traceparent(
    uvicorn_url,
):
    """Each ``/query`` on a reused connection is a SERVER span under its own caller.

    A large request body makes uvicorn pause and resume reading the socket
    from inside that request's context, which used to leave its server span
    active for every later request on the connection: they became INTERNAL
    children of the first request and their ``traceparent`` was ignored.
    """
    import httpx
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )
    from opentelemetry.trace import SpanKind

    exporter = InMemorySpanExporter()
    gandalf.otel._provider.add_span_processor(SimpleSpanProcessor(exporter))

    query = json.dumps({"message": _QUERY_MESSAGE}).encode()
    # Trailing whitespace is valid JSON, so the large body is still a valid query.
    bodies = [query + b" " * _LARGE_BODY_PADDING, query, query]
    with httpx.Client(base_url=uvicorn_url, timeout=30.0) as client:
        for n, body in enumerate(bodies):
            resp = client.post(
                "/query",
                content=body,
                headers={
                    "content-type": "application/json",
                    "traceparent": _traceparent(n),
                },
            )
            assert resp.status_code == 200, resp.text

    request_spans = [
        span for span in exporter.get_finished_spans() if span.name == "POST /query"
    ]
    assert [
        (span.kind, f"{span.context.trace_id:032x}", f"{span.parent.span_id:016x}")
        for span in request_spans
    ] == [
        (SpanKind.SERVER, _traceparent(n).split("-")[1], _traceparent(n).split("-")[2])
        for n in range(len(bodies))
    ]
