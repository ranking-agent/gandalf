"""Liveness, readiness and Prometheus metrics endpoints, and the job envelope.

Everything here runs without Redis.  The queue-backed paths are exercised
against a real ``redis-server`` in ``tests/test_queue_redis.py``.
"""

import time

import pytest
from fastapi.testclient import TestClient

import gandalf.server as gandalf_server
from gandalf.config import settings
from gandalf.execute import execute_to_bytes, run_query, serialize_response
from gandalf.jobs import Job
from gandalf.search.gc_utils import gc_disabled
from gandalf.trapi import Deadline

from tests.search_fixtures import graph  # noqa: F401

ONE_HOP = {
    "message": {
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
                }
            },
        }
    }
}


@pytest.fixture
def client(graph, bmt, monkeypatch):  # noqa: F811
    monkeypatch.setattr(gandalf_server, "GRAPH", graph)
    monkeypatch.setattr(gandalf_server, "BMT", bmt)
    return TestClient(gandalf_server.APP)


# ---------------------------------------------------------------------------
# /health and /ready
# ---------------------------------------------------------------------------


def test_health_is_always_ok(monkeypatch):
    monkeypatch.setattr(gandalf_server, "GRAPH", None)
    resp = TestClient(gandalf_server.APP).get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_ready_needs_the_graph(monkeypatch):
    monkeypatch.setattr(gandalf_server, "GRAPH", None)
    resp = TestClient(gandalf_server.APP).get("/ready")
    assert resp.status_code == 503
    assert resp.json() == {"status": "graph not loaded"}


def test_ready_with_graph_and_no_queue(client):
    resp = client.get("/ready")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ready"}


def test_ready_needs_the_queue_when_configured(client, monkeypatch):
    """With a queue configured but not opened, the pod must not take traffic."""
    monkeypatch.setattr(settings, "queue_url", "redis://nowhere.invalid:6379/0")
    monkeypatch.setattr(gandalf_server, "QUEUE", None)
    resp = client.get("/ready")
    assert resp.status_code == 503
    assert resp.json() == {"status": "queue unreachable"}


# ---------------------------------------------------------------------------
# /metrics
# ---------------------------------------------------------------------------


def test_metrics_count_requests_by_route_template(client):
    client.get("/health")
    client.get("/node_degree/CHEBI:6801")
    client.get("/node_degree/NCBIGene:5468")
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    text = resp.text
    assert (
        'gandalf_http_requests_total{method="GET",route="/health",status="200"}' in text
    )
    # One label value for the route, not one per CURIE.
    assert 'route="/node_degree/{curie}"' in text
    assert "CHEBI:6801" not in text
    assert "gandalf_http_request_duration_seconds_bucket" in text


def test_metrics_label_unmatched_paths(client):
    client.get("/no/such/route")
    text = client.get("/metrics").text
    assert 'route="unmatched",status="404"' in text


# ---------------------------------------------------------------------------
# The execution core
# ---------------------------------------------------------------------------


def test_execute_to_bytes_is_the_serialized_run_query(graph, bmt):  # noqa: F811
    """The worker's bytes are the server's bytes, apart from log timestamps."""
    body = execute_to_bytes(graph, bmt, dict(ONE_HOP))
    with gc_disabled():
        expected = serialize_response(run_query(graph, bmt, dict(ONE_HOP)))
    import orjson

    a, b = orjson.loads(body), orjson.loads(expected)
    a.pop("logs", None), b.pop("logs", None)
    assert a == b
    assert a["message"]["results"]


def test_execute_to_bytes_honours_an_expired_deadline(graph, bmt):  # noqa: F811
    import orjson

    body = execute_to_bytes(
        graph, bmt, dict(ONE_HOP), deadline=Deadline.started_ago(1.0, 5.0)
    )
    response = orjson.loads(body)
    assert response["status"] == "Timeout"
    assert response["message"]["results"] == []


# ---------------------------------------------------------------------------
# Job envelope
# ---------------------------------------------------------------------------


def test_job_round_trips_through_bytes():
    job = Job.new(
        dict(ONE_HOP),
        profile=True,
        budget=12.5,
        callback="http://cb.invalid/x",
        trace_headers={"traceparent": "00-abc-def-01"},
        request_id="r1",
    )
    assert Job.from_bytes(job.to_bytes()) == job
    assert len(job.job_id) == 32


def test_job_deadline_counts_queue_time():
    job = Job(job_id="j", query={}, budget=10.0, accepted_at=time.time() - 4.0)
    deadline = job.deadline()
    assert deadline.budget == 10.0
    assert 3.9 < deadline.elapsed < 4.5
    assert not deadline.expired
    assert (
        Job(job_id="j", query={}, budget=1.0, accepted_at=time.time() - 2)
        .deadline()
        .expired
    )


@pytest.mark.parametrize(
    "budget, expected",
    [
        (None, settings.queue_sync_max_wait_seconds),
        (30.0, 30.0 + settings.queue_sync_grace_seconds),
    ],
)
def test_job_wait_seconds(budget, expected):
    assert Job.new({}, budget=budget).wait_seconds() == expected


def test_metrics_survive_a_queue_outage(client, monkeypatch):
    """A scrape still answers when Redis is down; only the queue gauges are stale."""
    from gandalf.jobs import JobQueue
    import redis

    dead = JobQueue(
        redis.Redis(host="127.0.0.1", port=1, socket_connect_timeout=0.1),
        stream="x",
        group="x",
        dead_stream="x",
        max_deliveries=1,
        claim_idle_seconds=1,
    )
    monkeypatch.setattr(gandalf_server, "QUEUE", dead)
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "gandalf_http_requests_total" in resp.text
