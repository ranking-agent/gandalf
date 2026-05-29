"""Tests for the nested ``parameters`` request object and ``rehydration``.

These exercise the refactor that moved request configuration out of URL query
params / top-level body fields into a single ``parameters`` object, kept
``profile`` as a URL query param, renamed ``gandalf_annotators`` ->
``annotator_config``, wired ``filter_config`` into ``lookup()``, and added a
presence-based ``rehydration`` field that skips lookup and only enriches the
supplied knowledge graph.
"""

import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import orjson
import pytest
from fastapi.testclient import TestClient

from gandalf.models import AsyncTRAPIQuery, QueryParameters, TRAPIQuery
from tests.search_fixtures import graph  # noqa: F401

_ONE_HOP = {
    "message": {
        "query_graph": {
            "nodes": {
                "n0": {"ids": ["CHEBI:6801"]},
                "n1": {"ids": ["MONDO:0005148"]},
            },
            "edges": {
                "e0": {
                    "subject": "n0",
                    "object": "n1",
                    "predicates": ["biolink:treats"],
                },
            },
        },
    },
}


# ---------------------------------------------------------------------------
# Model-level tests
# ---------------------------------------------------------------------------


class TestQueryParametersModel:
    def test_parameters_parsed_into_nested_model(self):
        q = TRAPIQuery(
            **_ONE_HOP,
            parameters={
                "subclass": False,
                "subclass_depth": 3,
                "dehydrated": True,
                "filter_config": {"max_node_degree": 5},
                "annotator_config": {"some_plugin": {}},
                "rehydration": {},
            },
        )
        assert isinstance(q.parameters, QueryParameters)
        assert q.parameters.subclass is False
        assert q.parameters.subclass_depth == 3
        assert q.parameters.dehydrated is True
        assert q.parameters.filter_config == {"max_node_degree": 5}
        assert q.parameters.annotator_config == {"some_plugin": {}}
        assert q.parameters.rehydration == {}

    def test_parameters_absent_dumps_clean(self):
        q = TRAPIQuery(**_ONE_HOP)
        raw = q.model_dump(exclude_none=True)
        assert "parameters" not in raw

    def test_parameters_present_round_trips(self):
        q = TRAPIQuery(**_ONE_HOP, parameters={"subclass": True})
        raw = q.model_dump(exclude_none=True)
        assert raw["parameters"] == {"subclass": True}

    def test_async_accepts_parameters(self):
        q = AsyncTRAPIQuery(
            callback="https://example.com/cb",
            **_ONE_HOP,
            parameters={"dehydrated": True},
        )
        assert q.parameters.dehydrated is True


# ---------------------------------------------------------------------------
# Endpoint tests
# ---------------------------------------------------------------------------


@pytest.fixture
def server(graph, bmt, monkeypatch):  # noqa: F811
    """Return the patched server module + a TestClient."""
    monkeypatch.setenv("GANDALF_SKIP_PRELOAD", "true")
    monkeypatch.setenv("GANDALF_OTEL_ENABLED", "false")
    from gandalf import server as gandalf_server

    monkeypatch.setattr(gandalf_server, "GRAPH", graph)
    monkeypatch.setattr(gandalf_server, "BMT", bmt)
    return gandalf_server, TestClient(gandalf_server.APP)


class TestParametersWiring:
    def test_filter_config_passed_to_lookup(self, server, monkeypatch):
        gandalf_server, client = server
        captured = {}

        def fake_lookup(graph, query, **kwargs):
            captured.update(kwargs)
            return {"message": query["message"]}

        monkeypatch.setattr(gandalf_server, "lookup", fake_lookup)

        body = dict(_ONE_HOP)
        body["parameters"] = {"filter_config": {"max_node_degree": 5}}
        resp = client.post("/query", json=body)
        assert resp.status_code == 200, resp.text
        assert captured["filter_config"] == {"max_node_degree": 5}

    def test_subclass_read_from_parameters(self, server, monkeypatch):
        gandalf_server, client = server
        captured = {}

        def fake_lookup(graph, query, **kwargs):
            captured.update(kwargs)
            return {"message": query["message"]}

        monkeypatch.setattr(gandalf_server, "lookup", fake_lookup)

        body = dict(_ONE_HOP)
        body["parameters"] = {"subclass": False, "subclass_depth": 4}
        resp = client.post("/query", json=body)
        assert resp.status_code == 200, resp.text
        assert captured["subclass"] is False
        assert captured["subclass_depth"] == 4

    def test_legacy_query_param_and_body_field_ignored(self, server, monkeypatch):
        """Hard cutover: ?subclass=false and a top-level body subclass are ignored."""
        gandalf_server, client = server
        captured = {}

        def fake_lookup(graph, query, **kwargs):
            captured.update(kwargs)
            return {"message": query["message"]}

        monkeypatch.setattr(gandalf_server, "lookup", fake_lookup)

        body = dict(_ONE_HOP)
        body["subclass"] = False  # top-level field is no longer read
        resp = client.post("/query?subclass=false", json=body)
        assert resp.status_code == 200, resp.text
        # Neither source is honoured; default True is used.
        assert captured["subclass"] is True

    def test_annotator_config_passed_to_annotate_response(self, server, monkeypatch):
        gandalf_server, client = server
        captured = {}

        monkeypatch.setattr(
            gandalf_server,
            "lookup",
            lambda graph, query, **kwargs: {"message": query["message"]},
        )

        def fake_annotate(response, graph, annotator_config):
            captured["annotator_config"] = annotator_config

        monkeypatch.setattr(gandalf_server, "annotate_response", fake_annotate)

        body = dict(_ONE_HOP)
        body["parameters"] = {"annotator_config": {"my_plugin": {"opt": 1}}}
        resp = client.post("/query", json=body)
        assert resp.status_code == 200, resp.text
        assert captured["annotator_config"] == {"my_plugin": {"opt": 1}}


class TestRehydration:
    def _dehydrated_body(self):
        """A request whose message already carries a (dehydrated) KG."""
        body = dict(_ONE_HOP)
        body["message"] = {
            "query_graph": _ONE_HOP["message"]["query_graph"],
            "knowledge_graph": {
                "nodes": {"CHEBI:6801": {}, "MONDO:0005148": {}},
                "edges": {},
            },
            "results": [],
        }
        body["parameters"] = {"rehydration": {}}
        return body

    def test_sync_rehydration_skips_lookup_and_enriches(self, server, monkeypatch):
        gandalf_server, client = server

        def boom(*args, **kwargs):
            raise AssertionError("lookup must not be called during rehydration")

        monkeypatch.setattr(gandalf_server, "lookup", boom)

        resp = client.post("/query", json=self._dehydrated_body())
        assert resp.status_code == 200, resp.text
        nodes = resp.json()["message"]["knowledge_graph"]["nodes"]
        assert nodes["CHEBI:6801"]["name"] == "Metformin"
        assert nodes["MONDO:0005148"]["name"] == "Type 2 Diabetes"

    def test_async_rehydration_skips_lookup_and_posts_enriched(
        self, server, monkeypatch
    ):
        gandalf_server, client = server

        def boom(*args, **kwargs):
            raise AssertionError("lookup must not be called during rehydration")

        monkeypatch.setattr(gandalf_server, "lookup", boom)

        callback_url, received = _start_callback_server()
        try:
            body = self._dehydrated_body()
            body["callback"] = callback_url
            resp = client.post("/asyncquery", json=body)
            assert resp.status_code == 200, resp.text
            assert resp.json()["status"] == "accepted"

            assert _wait_for(lambda: len(received) > 0), "callback never received POST"
            posted = received[0]
            nodes = posted["message"]["knowledge_graph"]["nodes"]
            assert nodes["CHEBI:6801"]["name"] == "Metformin"
        finally:
            _stop_callback_server()


# ---------------------------------------------------------------------------
# Tiny recording callback server (mirrors tests/test_otel_traceparent.py)
# ---------------------------------------------------------------------------

_received: list = []
_server = None
_thread = None


class _RecordingHandler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        _received.append(orjson.loads(raw))
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, *args):  # silence
        pass


def _start_callback_server():
    global _server, _thread
    _received.clear()
    _server = HTTPServer(("127.0.0.1", 0), _RecordingHandler)
    port = _server.server_address[1]
    _thread = threading.Thread(target=_server.serve_forever, daemon=True)
    _thread.start()
    return f"http://127.0.0.1:{port}/cb", _received


def _stop_callback_server():
    if _server is not None:
        _server.shutdown()
        _server.server_close()
    if _thread is not None:
        _thread.join(timeout=2.0)


def _wait_for(predicate, timeout=5.0, interval=0.05):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()
