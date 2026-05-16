"""End-to-end tests for W3C ``traceparent`` propagation through Gandalf.

The server is Plater-compatible, which means it advertises OpenTelemetry
support.  Two behaviours must hold:

1.  Synchronous ``POST /query`` echoes the active server-side ``traceparent``
    in the response headers so callers can correlate their request with the
    span Gandalf produced (and verify the same trace_id when they supplied
    an incoming ``traceparent``).
2.  ``POST /asyncquery`` forwards the originating trace context to the
    eventual callback POST, so the asynchronous result the caller receives
    later is still linked to their trace.

These tests reload :mod:`gandalf.server` with OpenTelemetry enabled.  The
shared ``conftest.py`` disables it by default so other tests do not require
the instrumentation packages.
"""

from __future__ import annotations

import importlib
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

# A valid W3C traceparent: version-tracestate-parent_id-flags.
# trace_id chosen from the W3C spec example so failures are easy to spot.
_INCOMING_TRACEPARENT = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
_INCOMING_TRACE_ID = "0af7651916cd43dd8448eb211c80319c"


@pytest.fixture
def otel_server(monkeypatch, graph, bmt):  # noqa: F811
    """Reload ``gandalf.server`` with OpenTelemetry enabled and the test graph."""
    monkeypatch.setenv("GANDALF_SKIP_PRELOAD", "true")
    monkeypatch.setenv("GANDALF_OTEL_ENABLED", "true")
    # Console exporter avoids requiring an OTLP collector during tests.
    monkeypatch.setenv("GANDALF_OTEL_USE_CONSOLE_EXPORTER", "true")

    import gandalf.config

    importlib.reload(gandalf.config)
    import gandalf.server

    importlib.reload(gandalf.server)

    gandalf.server.GRAPH = graph
    gandalf.server.BMT = bmt

    try:
        yield gandalf.server
    finally:
        # Restore the OTel-disabled state that the shared conftest expects so
        # later tests still see a fresh, unconfigured server module.
        monkeypatch.undo()
        importlib.reload(gandalf.config)
        importlib.reload(gandalf.server)


def _trace_id_of(traceparent: str) -> str:
    """Return the trace_id segment of a W3C ``traceparent`` header."""
    parts = traceparent.split("-")
    assert len(parts) == 4, f"malformed traceparent: {traceparent!r}"
    return parts[1]


# ---------------------------------------------------------------------------
# Sync /query
# ---------------------------------------------------------------------------


def test_sync_response_echoes_incoming_traceparent_trace_id(otel_server):
    """When the client sends a ``traceparent``, the response carries the same trace_id."""
    client = TestClient(otel_server.APP)
    resp = client.post(
        "/query",
        headers={"traceparent": _INCOMING_TRACEPARENT},
        json={"message": _QUERY_MESSAGE},
    )
    assert resp.status_code == 200, resp.text
    assert "traceparent" in resp.headers, resp.headers
    assert _trace_id_of(resp.headers["traceparent"]) == _INCOMING_TRACE_ID


def test_sync_response_carries_traceparent_without_incoming(otel_server):
    """Without an incoming ``traceparent`` the server still emits its own."""
    client = TestClient(otel_server.APP)
    resp = client.post("/query", json={"message": _QUERY_MESSAGE})
    assert resp.status_code == 200, resp.text
    assert "traceparent" in resp.headers
    trace_id = _trace_id_of(resp.headers["traceparent"])
    assert len(trace_id) == 32
    # All-zero trace_id is reserved for "invalid" — the server-side span must be valid.
    assert int(trace_id, 16) != 0


# ---------------------------------------------------------------------------
# Async /asyncquery
# ---------------------------------------------------------------------------


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
    assert resp.json()["status"] == "accepted"
    # The /asyncquery response itself should also echo the trace context.
    assert "traceparent" in resp.headers
    assert _trace_id_of(resp.headers["traceparent"]) == _INCOMING_TRACE_ID

    assert _wait_for(lambda: len(received) > 0), "callback never received the POST"
    cb = received[0]
    assert "traceparent" in cb["headers"], cb["headers"]
    assert _trace_id_of(cb["headers"]["traceparent"]) == _INCOMING_TRACE_ID


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
