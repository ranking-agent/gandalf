"""The status page and snapshot without a queue.

Queue-mode sections (workers, jobs, dead letters) are covered against a
real Redis in ``tests/test_queue_redis.py``.
"""

import pytest
from fastapi.testclient import TestClient

import gandalf.server as gandalf_server
from gandalf.jobs import latency_bucket
from gandalf.status import (
    aggregate_counters,
    histogram_percentile,
    minute_bucket,
    snapshot,
)

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


def _bucket(start: float, jobs: list[dict], size: int = 60) -> dict:
    """A counter bucket holding *jobs*, as JobHistory.minutes reads one back."""
    bucket: dict = {
        "start": start,
        "size": size,
        "jobs": len(jobs),
        "outcomes": {},
        "bytes": 0,
        "duration_hist": {},
        "wait_hist": {},
        "duration_max_s": None,
        "wait_max_s": None,
    }
    for job in jobs:
        outcome = job.get("outcome", "error")
        bucket["outcomes"][outcome] = bucket["outcomes"].get(outcome, 0) + 1
        bucket["bytes"] += job.get("bytes", 0)
        for field, hist, top in (
            ("duration_s", "duration_hist", "duration_max_s"),
            ("wait_s", "wait_hist", "wait_max_s"),
        ):
            if field in job:
                index = latency_bucket(job[field])
                bucket[hist][index] = bucket[hist].get(index, 0) + 1
                bucket[top] = max(bucket[top] or 0.0, job[field])
    return bucket


@pytest.mark.parametrize(
    "values, fraction",
    [
        (list(range(1, 101)), 0.5),
        (list(range(1, 101)), 0.95),
        ([0.01 * i for i in range(1, 1001)], 0.95),
        ([0.3] * 50 + [45.0] * 50, 0.5),
    ],
)
def test_histogram_percentile_is_within_its_bucket(values, fraction):
    hist: dict[int, int] = {}
    for v in values:
        hist[latency_bucket(v)] = hist.get(latency_bucket(v), 0) + 1
    exact = sorted(values)[max(1, -(-int(fraction * 1000) * len(values) // 1000)) - 1]
    assert histogram_percentile(hist, fraction, max(values)) == pytest.approx(
        exact, rel=0.07
    )


def test_histogram_percentile_of_nothing_is_none():
    assert histogram_percentile({}, 0.5) is None


def test_aggregate_counts_every_outcome_and_only_the_window():
    now = 1000.0
    jobs = [
        {"outcome": o, "duration_s": i + 1.0, "wait_s": 0.1 * (i + 1), "bytes": 5}
        for i, o in enumerate(["ok", "ok", "timeout", "expired", "error", "poisoned"])
    ]
    buckets = [
        _bucket(now - 600, [{"outcome": "ok", "duration_s": 99.0, "bytes": 1}]),
        _bucket(now - 240, jobs[:3]),
        _bucket(now - 60, jobs[3:]),
    ]
    agg = aggregate_counters(buckets, 300, now)
    assert agg["jobs"] == 6
    assert agg["outcomes"] == {
        "ok": 2,
        "timeout": 1,
        "expired": 1,
        "error": 1,
        "poisoned": 1,
        "refused": 0,
    }
    assert agg["failed"] == 2
    assert agg["duration_max_s"] == 6.0
    assert agg["duration_p95_s"] == pytest.approx(6.0, rel=0.07)
    assert agg["duration_p50_s"] == pytest.approx(3.0, rel=0.07)
    assert agg["bytes"] == 30
    # Over the 240 s the counted minutes span, not the 300 s window.
    assert agg["per_minute"] == pytest.approx(6 / 4)


def test_a_minute_partly_before_the_window_is_left_out():
    buckets = [_bucket(0, [{"outcome": "ok"}]), _bucket(60, [{"outcome": "ok"}])]
    assert aggregate_counters(buckets, 120, now=150)["jobs"] == 1
    assert aggregate_counters(buckets, 150, now=150)["jobs"] == 2


def test_a_refused_job_counts_as_failed():
    agg = aggregate_counters([_bucket(960, [{"outcome": "refused"}])], 300, 1000.0)
    assert agg["outcomes"]["refused"] == 1
    assert agg["failed"] == 1


def test_an_empty_window_has_no_latencies():
    agg = aggregate_counters([_bucket(960, [])], 300, 1000.0)
    assert agg["jobs"] == 0 and agg["per_minute"] == 0
    assert agg["duration_p95_s"] is None and agg["duration_max_s"] is None


def test_minute_bucket_stacks_outcomes_into_the_charts_classes():
    bucket = minute_bucket(
        _bucket(
            60,
            [
                {"outcome": "ok"},
                {"outcome": "poisoned"},
                {"outcome": "timeout"},
                {"outcome": "expired"},
            ],
        )
    )
    assert bucket == {
        "minute": 60,
        "ok": 1,
        "timed_out": 2,
        "failed": 1,
        "wait_p50_s": None,
        "wait_p95_s": None,
        "wait_max_s": None,
    }


def test_minute_bucket_carries_the_queue_wait_of_its_jobs():
    waits = [0.5, 1.0, 1.5, 2.0, 40.0]
    bucket = minute_bucket(_bucket(60, [{"outcome": "ok", "wait_s": w} for w in waits]))
    assert bucket["wait_p50_s"] == pytest.approx(1.5, rel=0.07)
    assert bucket["wait_p95_s"] == bucket["wait_max_s"] == 40.0
