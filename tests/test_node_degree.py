"""Tests for the ``GET /node_degree/{curie}`` endpoint."""

import pytest
from fastapi.testclient import TestClient

from tests.search_fixtures import graph  # noqa: F401


@pytest.fixture
def client(graph, monkeypatch):  # noqa: F811
    monkeypatch.setenv("GANDALF_SKIP_PRELOAD", "true")
    monkeypatch.setenv("GANDALF_OTEL_ENABLED", "false")
    from gandalf import server as gandalf_server

    monkeypatch.setattr(gandalf_server, "GRAPH", graph)
    return TestClient(gandalf_server.APP)


def test_node_degree_returns_total_degree(client):
    # CHEBI:6801 has 11 outgoing + 0 incoming edges in the fixture graph.
    resp = client.get("/node_degree/CHEBI:6801")
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"id": "CHEBI:6801", "degree": 11}


def test_node_degree_counts_incoming_and_outgoing(client):
    # MONDO:0005148 has 2 outgoing + 7 incoming edges.
    resp = client.get("/node_degree/MONDO:0005148")
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"id": "MONDO:0005148", "degree": 9}


def test_node_degree_unknown_curie_returns_404(client):
    resp = client.get("/node_degree/CHEBI:DOESNOTEXIST")
    assert resp.status_code == 404
    assert "CHEBI:DOESNOTEXIST" in resp.json()["detail"]


def test_node_degree_503_when_graph_not_loaded(monkeypatch):
    monkeypatch.setenv("GANDALF_SKIP_PRELOAD", "true")
    monkeypatch.setenv("GANDALF_OTEL_ENABLED", "false")
    from gandalf import server as gandalf_server

    monkeypatch.setattr(gandalf_server, "GRAPH", None)
    client = TestClient(gandalf_server.APP, raise_server_exceptions=False)
    resp = client.get("/node_degree/CHEBI:6801")
    assert resp.status_code == 503
