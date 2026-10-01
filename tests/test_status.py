"""The status page and snapshot without a queue.

Queue-mode sections (workers, jobs, dead letters) are covered against a
real Redis in ``tests/test_queue_redis.py``.
"""

import pytest
from fastapi.testclient import TestClient

import gandalf.server as gandalf_server
from gandalf.status import aggregate, per_minute, percentile, snapshot

from tests.search_fixtures import graph  # noqa: F401


@pytest.fixture
def client(graph, bmt, monkeypatch):  # noqa: F811
    monkeypatch.setattr(gandalf_server, "GRAPH", graph)
    monkeypatch.setattr(gandalf_server, "BMT", bmt)
    monkeypatch.setattr(gandalf_server, "QUEUE", None)
    return TestClient(gandalf_server.APP)


def test_status_page_is_self_contained(client):
    resp = client.get("/status")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    html = resp.text
    assert "Gandalf status" in html
    assert 'fetch("status.json"' in html  # relative, so a path prefix works
    assert "<script src=" not in html and "<link" not in html  # nothing external


def test_status_json_in_process_mode(client):
    client.get("/health")
    data = client.get("/status.json").json()
    assert data["mode"] == "in-process"
    assert "queue" not in data and "workers" not in data and "jobs" not in data
    server = data["server"]
    assert server["graph_loaded"] is True
    assert server["graph"]["nodes"] == 11
    assert server["graph"]["edges"] == 20
    assert server["rss_anon_kb"] > 0
    health = [r for r in data["http"] if r["route"] == "/health"]
    assert health and health[0]["status"] == "200" and health[0]["count"] >= 1


def test_status_json_without_a_graph(monkeypatch):
    monkeypatch.setattr(gandalf_server, "GRAPH", None)
    monkeypatch.setattr(gandalf_server, "QUEUE", None)
    data = TestClient(gandalf_server.APP).get("/status.json").json()
    assert data["server"]["graph_loaded"] is False
    assert "graph" not in data["server"]


def test_snapshot_without_queue_has_no_queue_sections(graph):  # noqa: F811
    data = snapshot(graph, None, None, None)
    assert set(data) == {"generated_at", "mode", "server", "http"}


@pytest.mark.parametrize(
    "values, fraction, expected",
    [([1, 2, 3, 4], 0.5, 2), ([1, 2, 3, 4], 0.95, 4), ([7], 0.5, 7), ([], 0.5, None)],
)
def test_percentile(values, fraction, expected):
    assert percentile(values, fraction) == expected


def test_aggregate_counts_every_outcome_and_only_the_window():
    now = 1000.0
    records = [
        {
            "finished_at": now - 1,
            "outcome": o,
            "duration_s": i + 1.0,
            "wait_s": 0.1 * i,
            "bytes": 5,
        }
        for i, o in enumerate(["ok", "ok", "timeout", "expired", "error", "poisoned"])
    ] + [
        {
            "finished_at": now - 500,
            "outcome": "ok",
            "duration_s": 99.0,
            "wait_s": 99.0,
            "bytes": 1,
        }
    ]
    agg = aggregate(records, 300, now)
    assert agg["jobs"] == 6
    assert agg["outcomes"] == {
        "ok": 2,
        "timeout": 1,
        "expired": 1,
        "error": 1,
        "poisoned": 1,
    }
    assert agg["failed"] == 2
    assert agg["duration_max_s"] == 6.0
    assert agg["duration_p95_s"] == 6.0
    assert agg["bytes"] == 30
    assert agg["per_minute"] == pytest.approx(6 / 5)


def test_per_minute_buckets_cover_the_window_oldest_first():
    now = 7200.0
    buckets = per_minute(
        [
            {"finished_at": now - 30, "outcome": "ok"},
            {"finished_at": now - 30, "outcome": "poisoned"},
            {"finished_at": now - 90, "outcome": "timeout"},
            {"finished_at": now - 5000, "outcome": "ok"},  # outside
        ],
        3,
        now,
    )
    assert [b["minute"] for b in buckets] == [now - 120, now - 60, now]
    # now-30 falls in the minute starting at now-60; now-90 in the one before.
    assert buckets[1] == {"minute": now - 60, "ok": 1, "timed_out": 0, "failed": 1}
    assert buckets[0]["timed_out"] == 1
    assert buckets[2] == {"minute": now, "ok": 0, "timed_out": 0, "failed": 0}
