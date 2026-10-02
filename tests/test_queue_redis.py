"""Queue mode against a real Redis: the stream, the result store, the worker.

Marked ``integration`` and deselected by default, since it needs a
``redis-server``.  ``GANDALF_TEST_REDIS_URL`` points at one (CI runs a
service container); without it a local ``redis-server`` binary is started
on a free port for the session.

    python -m pytest -m integration tests/test_queue_redis.py
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Callable

import orjson
import pytest
import redis
import zstandard
from fastapi.testclient import TestClient

import gandalf.server as gandalf_server
from gandalf.config import settings
from gandalf.jobs import (
    Delivery,
    Job,
    JobHistory,
    JobQueue,
    ResultStore,
    WorkerRegistry,
)
from gandalf.worker import Worker

from tests.search_fixtures import graph  # noqa: F401

pytestmark = pytest.mark.integration

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


def _wait_for(predicate: Callable[[], bool], timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


# ---------------------------------------------------------------------------
# Redis
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def redis_url():
    """A Redis to test against: the configured one, or a local server."""
    configured = os.environ.get("GANDALF_TEST_REDIS_URL")
    if configured:
        yield configured
        return
    binary = shutil.which("redis-server")
    if binary is None:
        pytest.fail("set GANDALF_TEST_REDIS_URL or install redis-server")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    proc = subprocess.Popen(
        [binary, "--port", str(port), "--save", "", "--appendonly", "no"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"redis://127.0.0.1:{port}/0"
    probe = redis.Redis.from_url(url)

    def _up() -> bool:
        try:
            return bool(probe.ping())
        except redis.ConnectionError:
            return False

    assert _wait_for(_up, 10.0), "redis-server did not start"
    try:
        yield url
    finally:
        proc.terminate()
        proc.wait(timeout=10)


@pytest.fixture
def client(redis_url):
    r = redis.Redis.from_url(redis_url)
    r.flushdb()
    yield r
    r.close()


@pytest.fixture
def queue(client):
    q = JobQueue(
        client,
        stream="t:jobs",
        group="t:workers",
        dead_stream="t:dead",
        max_deliveries=2,
        claim_idle_seconds=0.2,
    )
    q.ensure_group()
    q.ensure_group()  # idempotent
    return q


@pytest.fixture
def store(client):
    # Tiny chunks so a small body exercises the chunking.
    return ResultStore(
        client, prefix="t:result", ttl_seconds=60, chunk_bytes=16, zstd_level=3
    )


# ---------------------------------------------------------------------------
# The stream
# ---------------------------------------------------------------------------


def test_enqueue_receive_ack(queue):
    job = Job.new(dict(ONE_HOP), budget=5.0)
    queue.enqueue(job)
    assert queue.stats().lag == 1

    delivery = queue.receive("a", block_seconds=0.1)
    assert delivery is not None
    assert delivery.job == job
    assert delivery.deliveries == 1
    assert not delivery.poisoned
    assert queue.stats() == queue.stats().__class__(lag=0, pending=1)

    queue.ack(delivery.entry_id)
    assert queue.stats().pending == 0
    assert queue.receive("a", block_seconds=0.05) is None


def test_stale_job_is_taken_over_then_dead_lettered(queue, client):
    job = Job.new(dict(ONE_HOP))
    queue.enqueue(job)

    first = queue.receive("died-1", block_seconds=0.1)
    assert first is not None and first.deliveries == 1
    time.sleep(0.3)  # past claim_idle_seconds, never acknowledged

    second = queue.receive("survivor", block_seconds=0.1)
    assert second is not None
    assert second.job == job
    assert second.deliveries == 2
    assert not second.poisoned
    time.sleep(0.3)

    third = queue.receive("judge", block_seconds=0.1)
    assert third is not None
    assert third.poisoned
    assert third.deliveries == 3
    # Dead-lettered and acknowledged: nothing pending, one entry on the dead stream.
    assert queue.stats().pending == 0
    dead = client.xrange("t:dead")
    assert len(dead) == 1
    _, fields = dead[0]
    assert Job.from_bytes(fields[b"job"]) == job
    assert fields[b"source_id"] == first.entry_id
    assert fields[b"deliveries"] == b"3"


def test_touch_keeps_a_running_job_from_being_claimed(queue):
    job = Job.new(dict(ONE_HOP))
    queue.enqueue(job)
    running = queue.receive("runner", block_seconds=0.1)
    assert running is not None

    end = time.monotonic() + 0.6
    while time.monotonic() < end:
        queue.touch(running.entry_id, "runner")
        time.sleep(0.05)
    assert queue.receive("thief", block_seconds=0.05) is None

    time.sleep(0.3)  # the runner stopped touching: now it is stale
    claimed = queue.receive("thief", block_seconds=0.1)
    assert claimed is not None and claimed.job == job
    # Touching did not count as deliveries.
    assert claimed.deliveries == 2


# ---------------------------------------------------------------------------
# The result store
# ---------------------------------------------------------------------------


def test_result_store_round_trip_in_chunks(store, client):
    body = orjson.dumps({"message": {"results": list(range(200))}})
    compressed_size = store.put("job-1", body, http_status=200)
    assert int(client.hget("t:result:job-1:meta", "chunks")) == -(
        -compressed_size // 16
    )

    result = store.wait("job-1", timeout=1.0)
    assert result is not None
    assert result.http_status == 200
    assert result.size == len(body)
    assert len(result.compressed) == compressed_size
    assert result.decompress() == body

    store.delete("job-1")
    assert store.get("job-1") is None
    assert not client.keys("t:result:job-1:*")


def test_result_store_wait_gives_up(store):
    assert store.wait("never", timeout=0.05) is None


# ---------------------------------------------------------------------------
# The worker
# ---------------------------------------------------------------------------


@pytest.fixture
def worker(graph, bmt, queue, store, tmp_path):  # noqa: F811
    """A worker consuming on a thread for the duration of the test."""
    heartbeat = tmp_path / "heartbeat"
    w = Worker(
        graph,
        bmt,
        queue,
        store,
        consumer="w1",
        block_seconds=0.05,
        keepalive_seconds=0.05,
        heartbeat_file=str(heartbeat),
    )
    thread = threading.Thread(target=w.run, daemon=True)
    thread.start()
    assert _wait_for(heartbeat.exists)
    try:
        yield w
    finally:
        w.stop()
        thread.join(timeout=10)
        assert not thread.is_alive()


@pytest.fixture
def api(graph, bmt, queue, store, monkeypatch):  # noqa: F811
    """The server in queue mode, on the test queue."""
    monkeypatch.setattr(gandalf_server, "GRAPH", graph)
    monkeypatch.setattr(gandalf_server, "BMT", bmt)
    monkeypatch.setattr(gandalf_server, "QUEUE", queue)
    monkeypatch.setattr(gandalf_server, "RESULTS", store)
    monkeypatch.setattr(settings, "queue_url", "redis://configured.invalid/0")
    return TestClient(gandalf_server.APP)


def _in_process_response(graph, bmt, body):  # noqa: F811
    """What the in-process server answers, for comparison."""
    from gandalf.execute import execute_to_bytes

    return orjson.loads(execute_to_bytes(graph, bmt, dict(body)))


def _without_logs(response: dict) -> dict:
    response = dict(response)
    response.pop("logs", None)
    return response


def test_sync_query_through_the_queue_matches_in_process(
    api, worker, graph, bmt
):  # noqa: F811
    resp = api.post("/query", json=ONE_HOP, headers={"Accept-Encoding": "identity"})
    assert resp.status_code == 200, resp.text
    assert resp.headers.get("content-encoding") != "zstd"
    answered = resp.json()
    assert answered["message"]["results"]
    assert _without_logs(answered) == _without_logs(
        _in_process_response(graph, bmt, ONE_HOP)
    )
    assert worker.jobs_done == 1


def test_sync_query_passes_the_stored_zstd_through(
    api, worker, graph, bmt
):  # noqa: F811
    resp = api.post("/query", json=ONE_HOP, headers={"Accept-Encoding": "zstd"})
    assert resp.status_code == 200
    assert resp.headers["content-encoding"] == "zstd"
    raw = resp.content
    if raw[:4] == b"\x28\xb5\x2f\xfd":  # zstd magic: httpx left it compressed
        raw = zstandard.ZstdDecompressor().decompress(raw, max_output_size=10_000_000)
    assert _without_logs(json.loads(raw)) == _without_logs(
        _in_process_response(graph, bmt, ONE_HOP)
    )


def test_sync_query_validation_errors_never_reach_the_queue(api, queue):
    body = {"message": {"query_graph": {"nodes": {}, "edges": {}}}}
    resp = api.post("/query", json=body)
    assert resp.status_code == 400
    assert queue.stats().lag == 0


def test_sync_query_times_out_when_nobody_answers(api, queue, monkeypatch):
    """No worker: the API waits the budget plus grace, then answers 504."""
    monkeypatch.setattr(settings, "queue_sync_grace_seconds", 0.1)
    body = dict(ONE_HOP)
    body["parameters"] = {"timeout": 1.0}
    t0 = time.monotonic()
    resp = api.post("/query", json=body)
    assert resp.status_code == 504, resp.text
    assert 1.0 <= time.monotonic() - t0 < 5.0
    assert queue.stats().lag == 1  # still there for the next worker


def test_ready_reports_the_queue(api, queue):
    assert api.get("/ready").json() == {"status": "ready"}


def test_metrics_report_queue_lag_and_pending(api, queue):
    queue.enqueue(Job.new(dict(ONE_HOP)))
    queue.enqueue(Job.new(dict(ONE_HOP)))
    queue.receive("someone", block_seconds=0.1)
    text = api.get("/metrics").text
    assert "gandalf_queue_lag 1.0" in text
    assert "gandalf_queue_pending 1.0" in text


class _Recorder(BaseHTTPRequestHandler):
    received: list[dict[str, Any]] = []

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("content-length", "0"))
        body = self.rfile.read(length)
        self.__class__.received.append(
            {
                "headers": {k.lower(): v for k, v in self.headers.items()},
                "body": json.loads(body),
            }
        )
        self.send_response(200)
        self.end_headers()

    def log_message(self, fmt, *args):
        pass


@pytest.fixture
def callback_server():
    _Recorder.received = []
    server = HTTPServer(("127.0.0.1", 0), _Recorder)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/callback", _Recorder.received
    finally:
        server.shutdown()


def test_async_query_through_the_queue_posts_the_callback(
    api, worker, callback_server, graph, bmt
):  # noqa: F811
    url, received = callback_server
    body = dict(ONE_HOP)
    body["callback"] = url
    resp = api.post("/asyncquery", json=body)
    assert resp.status_code == 200, resp.text
    accepted = resp.json()
    assert accepted["status"] == "Accepted"
    assert len(accepted["job_id"]) == 32

    assert _wait_for(lambda: len(received) == 1)
    posted = received[0]
    assert posted["headers"]["content-type"] == "application/json"
    assert _without_logs(posted["body"]) == _without_logs(
        _in_process_response(graph, bmt, ONE_HOP)
    )


def test_budget_spent_in_the_queue_is_a_timeout_response(queue, store, worker):
    job = Job(
        job_id="late", query=dict(ONE_HOP), budget=1.0, accepted_at=time.time() - 5
    )
    queue.enqueue(job)
    result = store.wait("late", timeout=10.0)
    assert result is not None
    assert result.http_status == 200
    answered = orjson.loads(result.decompress())
    assert answered["status"] == "Timeout"
    assert answered["message"]["results"] == []
    assert answered["message"]["query_graph"] == ONE_HOP["message"]["query_graph"]


def test_poisoned_job_is_answered_with_an_error(graph, bmt, queue, store):  # noqa: F811
    """What the worker does with a delivery the queue already dead-lettered."""
    job = Job.new(dict(ONE_HOP))
    queue.enqueue(job)
    delivery = queue.receive("w", block_seconds=0.1)
    assert delivery is not None
    w = Worker(graph, bmt, queue, store, consumer="w")
    outcome = w.handle(Delivery(delivery.entry_id, job, deliveries=3, poisoned=True))
    assert outcome == "poisoned"
    assert queue.stats().pending == 0
    result = store.get(job.job_id)
    assert result is not None and result.http_status == 500
    answered = orjson.loads(result.decompress())
    assert answered["status"] == "Error"
    assert answered["logs"][0]["level"] == "ERROR"
    assert "2 attempts" in answered["description"]


def test_a_failing_job_is_answered_and_the_worker_lives_on(
    graph, bmt, queue, store
):  # noqa: F811
    w = Worker(graph, bmt, queue, store, consumer="w", max_jobs=2, block_seconds=0.05)
    broken = Job.new(
        {
            "message": {
                "query_graph": {
                    "nodes": {"n0": {"ids": ["X:1"]}},
                    "edges": {"e0": {"subject": "n0", "object": "n9"}},
                }
            }
        }
    )
    queue.enqueue(broken)
    good = Job.new(dict(ONE_HOP))
    queue.enqueue(good)

    assert w.run() == 2  # stopped by max_jobs after both

    failed = store.get(broken.job_id)
    assert failed is not None and failed.http_status == 500
    assert orjson.loads(failed.decompress())["status"] == "Error"
    ok = store.get(good.job_id)
    assert ok is not None and ok.http_status == 200
    assert orjson.loads(ok.decompress())["message"]["results"]
    assert queue.stats().pending == 0


def test_stop_finishes_the_current_job_first(
    graph, bmt, queue, store, tmp_path
):  # noqa: F811
    w = Worker(
        graph,
        bmt,
        queue,
        store,
        consumer="w",
        block_seconds=0.05,
        heartbeat_file=str(tmp_path / "hb"),
    )
    thread = threading.Thread(target=w.run, daemon=True)
    thread.start()
    queue.enqueue(Job.new(dict(ONE_HOP)))
    assert _wait_for(lambda: w.jobs_done == 1)
    w.stop()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert (tmp_path / "hb").exists()


# ---------------------------------------------------------------------------
# Status: worker registry, job history, /status.json
# ---------------------------------------------------------------------------


@pytest.fixture
def registry(client):
    return WorkerRegistry(client, prefix="t:worker", ttl_seconds=0.3)


@pytest.fixture
def history(client):
    return JobHistory(client, stream="t:history", maxlen=100)


def test_registry_reports_expire_and_are_typed(registry):
    registry.report(
        "w1",
        pid=12,
        state="running",
        jobs_done=3,
        job_query={"nodes": 2},
        rss_anon_kb=512,
        stopping=False,
    )
    registry.report(
        "w2", pid=13, state="idle", jobs_done=0, rss_anon_kb=256, stopping=False
    )
    reports = {r["name"]: r for r in registry.workers()}
    assert set(reports) == {"w1", "w2"}
    assert reports["w1"]["pid"] == 12 and reports["w1"]["jobs_done"] == 3
    assert reports["w1"]["job_query"] == {"nodes": 2}
    assert reports["w1"]["stopping"] == "false"
    assert isinstance(reports["w1"]["last_seen"], float)

    registry.forget("w2")
    assert [r["name"] for r in registry.workers()] == ["w1"]
    time.sleep(0.4)  # past the TTL with no refresh: a dead worker
    assert registry.workers() == []


def test_history_keeps_newest_first_and_types_numbers(history):
    for i in range(3):
        history.record(
            job_id=f"j{i}",
            outcome="ok",
            duration_s=0.5 * i,
            bytes=100 * i,
            query={"nodes": 1},
        )
    records = history.recent(10)
    assert [r["job_id"] for r in records] == ["j2", "j1", "j0"]
    assert records[0]["duration_s"] == 1.0 and records[0]["bytes"] == 200
    assert records[0]["query"] == {"nodes": 1}
    assert history.recent(1)[0]["job_id"] == "j2"


@pytest.fixture
def reporting_worker(
    graph, bmt, queue, store, registry, history, tmp_path
):  # noqa: F811
    w = Worker(
        graph,
        bmt,
        queue,
        store,
        consumer="w1",
        block_seconds=0.05,
        keepalive_seconds=0.05,
        heartbeat_file=str(tmp_path / "hb"),
        registry=registry,
        history=history,
    )
    thread = threading.Thread(target=w.run, daemon=True)
    thread.start()
    assert _wait_for(lambda: bool(registry.workers()))
    try:
        yield w
    finally:
        w.stop()
        thread.join(timeout=10)


def test_status_json_in_queue_mode(
    api, reporting_worker, registry, history, monkeypatch, callback_server
):
    monkeypatch.setattr(gandalf_server, "WORKERS", registry)
    monkeypatch.setattr(gandalf_server, "HISTORY", history)

    before = api.get("/status.json").json()
    assert before["mode"] == "queue"
    assert before["workers"]["alive"] == 1 and before["workers"]["busy"] == 0
    assert before["workers"]["reports"][0]["name"] == "w1"
    assert before["jobs"]["recent"] == []

    assert api.post("/query", json=ONE_HOP).status_code == 200
    url, received = callback_server
    body = dict(ONE_HOP)
    body["callback"] = url
    api.post("/asyncquery", json=body)
    assert _wait_for(lambda: len(received) == 1)
    assert _wait_for(lambda: len(history.recent(10)) == 2)

    data = api.get("/status.json").json()
    assert data["queue"]["lag"] == 0 and data["queue"]["pending"] == 0
    assert data["queue"]["dead_letters"] == 0
    recent = data["jobs"]["recent"]
    assert [j["mode"] for j in recent] == ["callback", "sync"]
    assert all(j["outcome"] == "ok" and j["worker"] == "w1" for j in recent)
    assert recent[0]["query"]["ids"] == ["CHEBI:6801"]
    assert recent[0]["bytes"] > 1000 and recent[0]["duration_s"] >= 0
    hour = data["jobs"]["windows"]["1h"]
    assert hour["jobs"] == 2 and hour["outcomes"]["ok"] == 2 and hour["failed"] == 0
    assert sum(b["ok"] for b in data["jobs"]["per_minute"]) == 2
    assert len(data["jobs"]["per_minute"]) == 60
    worker = data["workers"]["reports"][0]
    assert (
        worker["jobs_done"] == 2
        and worker["last_outcome"] == "ok"
        and worker["state"] == "idle"
    )


def test_status_shows_a_running_job_and_a_dead_letter(
    api, queue, registry, history, graph, bmt, client, monkeypatch
):  # noqa: F811
    monkeypatch.setattr(gandalf_server, "WORKERS", registry)
    monkeypatch.setattr(gandalf_server, "HISTORY", history)
    # A job delivered to a worker that never finishes: in flight on the page.
    stuck = Job.new(dict(ONE_HOP))
    queue.enqueue(stuck)
    delivery = queue.receive("ghost", block_seconds=0.1)
    registry.report(
        "ghost",
        state="running",
        job_id=stuck.job_id,
        job_started_at=time.time() - 5,
        jobs_done=0,
    )
    # And one dead-lettered by a worker that found it poisoned.
    w = Worker(
        graph,
        bmt,
        queue,
        store=ResultStore(client, ttl_seconds=60, chunk_bytes=1024, zstd_level=1),
        consumer="judge",
        registry=registry,
        history=history,
    )
    poisoned = Job.new(dict(ONE_HOP))
    queue.enqueue(poisoned)
    pdelivery = queue.receive("judge", block_seconds=0.1)
    queue._dead_letter(pdelivery.entry_id, {b"job": poisoned.to_bytes()}, 3)
    w.handle(Delivery(pdelivery.entry_id, poisoned, deliveries=3, poisoned=True))

    data = api.get("/status.json").json()
    assert data["queue"]["pending"] == 1
    assert data["queue"]["pending_entries"][0]["consumer"] == "ghost"
    assert data["queue"]["oldest_pending_s"] >= 0
    ghost = next(r for r in data["workers"]["reports"] if r["name"] == "ghost")
    assert ghost["state"] == "running" and ghost["job_elapsed_s"] >= 5
    assert data["queue"]["dead_letters"] == 1
    assert data["queue"]["recent_dead_letters"][0]["job_id"] == poisoned.job_id
    assert data["jobs"]["recent"][0]["outcome"] == "poisoned"
    assert data["jobs"]["windows"]["5m"]["failed"] == 1
    queue.ack(delivery.entry_id)


# ---------------------------------------------------------------------------
# Alerts: the log, throttling, the monitor, the worker's events
# ---------------------------------------------------------------------------

from gandalf.notify import (  # noqa: E402
    BACKLOG_FLAG,
    KNOWN_WORKERS,
    LOG_STREAM,
    Event,
    Monitor,
    Notifier,
    SlackWebhook,
    mark_worker_leaving,
)
from tests.test_notify import webhook_server  # noqa: E402, F401


@pytest.fixture
def notifier(client, webhook_server):  # noqa: F811
    url, hook = webhook_server
    n = Notifier(
        client,
        webhook=SlackWebhook(url),
        kinds=frozenset(
            {
                "worker_started",
                "worker_stopped",
                "worker_restarted",
                "worker_lost",
                "job_retried",
                "job_dead_lettered",
                "job_failed",
                "queue_backlog",
                "queue_backlog_cleared",
                "queue_stuck",
            }
        ),
        throttle_seconds=0.3,
        log_maxlen=50,
    )
    yield n, hook


def test_alert_log_keeps_every_event_newest_first(notifier):
    n, _ = notifier
    n.emit(Event("worker_started", "Worker started", "w1 up", {"worker": "w1"}))
    n.emit(Event("worker_recycled", "Worker recycled"))  # not a Slack kind here
    recent = n.recent()
    assert [a["kind"] for a in recent] == ["worker_recycled", "worker_started"]
    assert recent[1]["fields"] == {"worker": "w1"}
    assert recent[1]["severity"] == "info"
    assert isinstance(recent[1]["at"], float)


def test_noisy_kinds_are_throttled_and_counted(notifier):
    n, hook = notifier
    sent = [n.emit(Event("job_failed", "Job failed", f"job {i}")) for i in range(3)]
    assert sent == [True, False, False]
    time.sleep(0.35)
    assert n.emit(Event("job_failed", "Job failed", "job 3")) is True
    assert n.webhook.flush(5.0)
    assert len(hook.received) == 2
    assert (
        "_2 similar events since the last message_"
        in hook.received[1]["blocks"][0]["text"]["text"]
    )
    # every one of them is in the log regardless
    assert sum(1 for a in n.recent() if a["kind"] == "job_failed") == 4


def test_unthrottled_kinds_always_send(notifier):
    n, hook = notifier
    assert [n.emit(Event("worker_lost", "Worker lost")) for _ in range(3)] == [True] * 3
    assert n.webhook.flush(5.0)
    assert len(hook.received) == 3


@pytest.fixture
def monitor(client, queue, registry, notifier):
    n, _ = notifier
    return Monitor(
        client,
        queue,
        registry,
        n,
        interval_seconds=0.2,
        lag_threshold=2,
        stuck_seconds=0.3,
    )


def test_monitor_reports_a_worker_that_vanished(monitor, registry):
    registry.report(
        "w1",
        state="running",
        job_id="abcdef0123",
        job_query={"nodes": 2, "edges": 1, "ids": ["X:1"]},
        job_started_at=time.time() - 42,
        jobs_done=7,
        rss_anon_kb=2048 * 1024,
    )
    assert monitor.tick() == []  # learned about w1
    time.sleep(0.4)  # the registry TTL (0.3s) passes without a refresh
    events = monitor.tick()
    assert [e.kind for e in events] == ["worker_lost"]
    lost = events[0]
    assert lost.fields["worker"] == "w1" and lost.fields["jobs done"] == 7
    assert lost.fields["was running"].startswith(
        "abcdef01 (2 nodes / 1 edges, ids X:1) for 4"
    )
    assert lost.fields["anon RSS"] == "2048 MB"
    assert monitor.tick() == []  # reported once


def test_monitor_ignores_a_worker_that_left_on_purpose(monitor, registry, client):
    registry.report("w1", state="idle", jobs_done=1)
    monitor.tick()
    mark_worker_leaving(client, "w1")
    registry.forget("w1")
    assert monitor.tick() == []
    assert not client.hexists(KNOWN_WORKERS, "w1")


def test_monitor_alerts_on_backlog_once_and_on_clearing(monitor, queue, registry):
    registry.report("w1", state="idle", jobs_done=0)
    for _ in range(3):
        queue.enqueue(Job.new(dict(ONE_HOP)))
    events = monitor.tick()
    assert [e.kind for e in events] == ["queue_backlog"]
    assert events[0].fields == {"waiting": 3, "running": 0, "workers": 1}
    assert monitor.tick() == []  # still over, already said
    for _ in range(3):
        d = queue.receive("w1", block_seconds=0.1)
        queue.ack(d.entry_id)
    events = monitor.tick()
    assert [e.kind for e in events] == ["queue_backlog_cleared"]
    assert monitor.tick() == []


def test_monitor_alerts_on_a_stuck_job(monitor, queue):
    queue.enqueue(Job.new(dict(ONE_HOP)))
    d = queue.receive("slow", block_seconds=0.1)
    assert monitor.tick() == []
    time.sleep(0.35)
    events = monitor.tick()
    assert [e.kind for e in events] == ["queue_stuck"]
    assert events[0].fields["worker"] == "slow"
    assert monitor.tick() == []  # the same stuck job is not repeated
    queue.ack(d.entry_id)
    assert monitor.tick() == []


def test_only_one_monitor_is_active(client, queue, registry, notifier):
    n, _ = notifier
    first = Monitor(client, queue, registry, n, interval_seconds=0.2, lag_threshold=1)
    second = Monitor(client, queue, registry, n, interval_seconds=0.2, lag_threshold=1)
    queue.enqueue(Job.new(dict(ONE_HOP)))
    assert first.is_leader() is True
    assert second.is_leader() is False
    assert second.tick() == []
    assert [e.kind for e in first.tick()] == ["queue_backlog"]
    client.delete("gandalf:notify:monitor")  # the leader died
    assert second.is_leader() is True


def test_worker_emits_its_lifecycle_and_job_events(
    graph, bmt, queue, store, registry, history, notifier, client
):  # noqa: F811
    n, hook = notifier
    w = Worker(
        graph,
        bmt,
        queue,
        store,
        consumer="w1",
        max_jobs=3,
        block_seconds=0.05,
        registry=registry,
        history=history,
        notifier=n,
    )
    retried = Job.new(dict(ONE_HOP))
    queue.enqueue(retried)
    first = queue.receive("died", block_seconds=0.1)  # a worker took it and died
    assert first.job == retried
    broken = Job.new(
        {
            "message": {
                "query_graph": {
                    "nodes": {"n0": {"ids": ["X:1"]}},
                    "edges": {"e0": {"subject": "n0", "object": "n9"}},
                }
            }
        }
    )
    queue.enqueue(broken)
    queue.enqueue(Job.new(dict(ONE_HOP)))
    time.sleep(0.3)  # past claim_idle_seconds: the abandoned job is claimable

    assert w.run() == 3
    kinds = [a["kind"] for a in reversed(n.recent())]
    assert kinds == [
        "worker_started",
        "job_retried",
        "job_failed",
        "worker_recycled",
    ] or kinds == ["worker_started", "job_failed", "job_retried", "worker_recycled"]
    assert not client.hexists(KNOWN_WORKERS, "w1")
    assert n.webhook.flush(5.0)
    texts = [r["text"] for r in hook.received]
    assert any("Worker started" in t for t in texts)
    assert any("Job retried" in t for t in texts)
    assert any("Job failed" in t for t in texts)
    assert not any("recycled" in t for t in texts)  # not an enabled kind


def test_worker_reports_a_dead_letter(graph, bmt, queue, store, notifier):  # noqa: F811
    n, hook = notifier
    job = Job.new(dict(ONE_HOP))
    queue.enqueue(job)
    delivery = queue.receive("w", block_seconds=0.1)
    w = Worker(graph, bmt, queue, store, consumer="w", notifier=n)
    w.handle(Delivery(delivery.entry_id, job, deliveries=3, poisoned=True))
    assert n.recent()[0]["kind"] == "job_dead_lettered"
    assert n.webhook.flush(5.0)
    assert "Job dead-lettered" in hook.received[0]["text"]


def test_status_json_lists_alerts(api, notifier, monkeypatch):
    n, _ = notifier
    monkeypatch.setattr(gandalf_server, "NOTIFIER", n)
    n.emit(Event("queue_backlog", "Queue backlog", "7 waiting", {"waiting": 7}))
    alerts = api.get("/status.json").json()["alerts"]
    assert alerts["slack"] is True
    assert "worker_lost" in alerts["kinds"]
    assert alerts["recent"][0]["title"] == "Queue backlog"
    assert alerts["recent"][0]["fields"] == {"waiting": 7}


# ---------------------------------------------------------------------------
# Redis timeouts and outages
# ---------------------------------------------------------------------------

from gandalf.jobs import blocking_slice, redis_client  # noqa: E402


def test_redis_client_defaults_and_url_override(redis_url, monkeypatch):
    monkeypatch.setattr(settings, "queue_socket_timeout_seconds", 12.0)
    kwargs = redis_client(redis_url).connection_pool.connection_kwargs
    assert kwargs["socket_timeout"] == 12.0 and kwargs["socket_connect_timeout"] == 5.0
    kwargs = redis_client(
        redis_url + "?socket_timeout=2"
    ).connection_pool.connection_kwargs
    assert kwargs["socket_timeout"] == 2.0  # the URL wins, as redis-py has it


def test_blocking_reads_stay_under_a_short_socket_timeout(redis_url):
    """A ?socket_timeout shorter than the waits raised TimeoutError; now it is sliced."""
    short = redis.Redis.from_url(redis_url + "?socket_timeout=1")
    assert blocking_slice(short, 5.0) == 0.5
    q = JobQueue(
        short,
        stream="t:jobs",
        group="t:workers",
        dead_stream="t:dead",
        max_deliveries=2,
        claim_idle_seconds=1,
    )
    q.ensure_group()
    t0 = time.monotonic()
    assert q.receive("w", block_seconds=5.0) is None  # one slice, no exception
    assert time.monotonic() - t0 < 1.5

    store = ResultStore(
        short, prefix="t:result", ttl_seconds=60, chunk_bytes=1024, zstd_level=1
    )
    t0 = time.monotonic()
    assert store.wait("nobody", timeout=2.2) is None  # several slices, no exception
    assert 2.0 <= time.monotonic() - t0 < 3.5

    def _later():
        time.sleep(1.3)
        store.put("late", b'{"ok":1}')

    threading.Thread(target=_later, daemon=True).start()
    result = store.wait("late", timeout=5.0)
    assert result is not None and result.decompress() == b'{"ok":1}'


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _start_redis(port: int) -> subprocess.Popen:
    binary = shutil.which("redis-server")
    if binary is None:
        pytest.fail("redis-server binary needed for the outage test")
    proc = subprocess.Popen(
        [binary, "--port", str(port), "--save", "", "--appendonly", "no"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    probe = redis.Redis(port=port, socket_connect_timeout=0.2)

    def _up() -> bool:
        try:
            return bool(probe.ping())
        except redis.ConnectionError:
            return False

    assert _wait_for(_up, 10.0), "redis-server did not start"
    return proc


def test_worker_survives_a_redis_outage(graph, bmt):  # noqa: F811
    """Redis goes away and comes back; the worker waits it out and keeps working."""
    port = _free_port()
    proc = _start_redis(port)
    url = f"redis://127.0.0.1:{port}/0?socket_timeout=1"
    client = redis_client(url)
    queue = JobQueue(
        client,
        stream="o:jobs",
        group="o:w",
        dead_stream="o:dead",
        max_deliveries=2,
        claim_idle_seconds=5,
    )
    store = ResultStore(
        client, prefix="o:result", ttl_seconds=60, chunk_bytes=1024, zstd_level=1
    )
    registry = WorkerRegistry(client, prefix="o:worker", ttl_seconds=30)
    w = Worker(
        graph,
        bmt,
        queue,
        store,
        consumer="w1",
        block_seconds=0.2,
        keepalive_seconds=0.1,
        registry=registry,
    )
    thread = threading.Thread(target=w.run, daemon=True)
    thread.start()
    try:
        queue.enqueue(Job.new(dict(ONE_HOP)))
        assert _wait_for(lambda: w.jobs_done == 1)

        proc.terminate()
        proc.wait(timeout=10)
        time.sleep(2.5)  # several failed polls: the loop must still be alive
        assert thread.is_alive()

        proc = _start_redis(port)
        queue.ensure_group()  # the group died with the (unpersisted) server
        queue.enqueue(Job.new(dict(ONE_HOP)))
        # the retry backoff was growing during the outage (1, 2, 4 ... up to 30s)
        assert _wait_for(lambda: w.jobs_done == 2, timeout=40.0)
        assert thread.is_alive()
    finally:
        w.stop()
        thread.join(timeout=10)
        proc.terminate()
        proc.wait(timeout=10)


def test_worker_reports_a_restart_after_a_crash(
    graph, bmt, queue, store, registry, notifier
):  # noqa: F811
    """A live report under its own name tells a starting worker its last instance died."""
    n, hook = notifier
    registry.report(
        "w1",
        pid=1,
        state="running",
        started_at=time.time() - 20,
        jobs_done=7,
        job_id="deadbeef0000",
        job_query={"nodes": 3, "edges": 2, "ids": ["CHEBI:6801"]},
        job_started_at=time.time() - 4,
        rss_anon_kb=9 * 1024 * 1024,
        stopping=False,
    )
    w = Worker(
        graph,
        bmt,
        queue,
        store,
        consumer="w1",
        max_jobs=1,
        block_seconds=0.05,
        registry=registry,
        notifier=n,
    )
    queue.enqueue(Job.new(dict(ONE_HOP)))
    assert w.run() == 1
    kinds = [a["kind"] for a in reversed(n.recent())]
    assert kinds == ["worker_restarted", "worker_recycled"]
    restarted = n.recent()[-1]
    assert restarted["severity"] == "critical"
    assert "killed from outside" in restarted["detail"]
    f = restarted["fields"]
    assert f["jobs done"] == 7 and f["was running"].startswith(
        "deadbeef (3 nodes / 2 edges, ids CHEBI:6801)"
    )
    assert f["anon RSS"] == "9216 MB" and f["previous lifetime"].endswith("s")
    assert "died with" not in f


def test_worker_restart_reports_its_last_words(
    graph, bmt, queue, store, registry, notifier
):  # noqa: F811
    n, _ = notifier
    registry.report(
        "w1",
        pid=1,
        state="idle",
        started_at=time.time() - 5,
        jobs_done=0,
        stopping=False,
    )
    registry.record_last_words(
        "w1", "redis.exceptions.TimeoutError: Timeout reading from socket"
    )
    w = Worker(
        graph,
        bmt,
        queue,
        store,
        consumer="w1",
        max_jobs=0,
        block_seconds=0.05,
        registry=registry,
        notifier=n,
    )
    w.notify(w._start_event())
    event = n.recent()[0]
    assert event["kind"] == "worker_restarted"
    assert "died with the error below" in event["detail"]
    assert event["fields"]["died with"].startswith("redis.exceptions.TimeoutError")
    assert registry.last_words("w1") == ""  # consumed


def test_worker_events_are_throttled_per_worker(notifier):
    n, hook = notifier
    sent = [
        n.emit(Event("worker_started", "Worker started", fields={"worker": "flapper"})),
        n.emit(Event("worker_started", "Worker started", fields={"worker": "flapper"})),
        n.emit(Event("worker_started", "Worker started", fields={"worker": "other"})),
        n.emit(
            Event("worker_restarted", "Worker restarted", fields={"worker": "flapper"})
        ),
    ]
    assert sent == [True, False, True, True]


def test_api_answers_503_when_redis_is_gone(api, graph, bmt, monkeypatch):  # noqa: F811
    dead = redis.Redis(host="127.0.0.1", port=1, socket_connect_timeout=0.1)
    monkeypatch.setattr(
        gandalf_server,
        "QUEUE",
        JobQueue(
            dead,
            stream="x",
            group="x",
            dead_stream="x",
            max_deliveries=1,
            claim_idle_seconds=1,
        ),
    )
    monkeypatch.setattr(
        gandalf_server,
        "RESULTS",
        ResultStore(dead, ttl_seconds=1, chunk_bytes=1, zstd_level=1),
    )
    resp = api.post("/query", json=ONE_HOP)
    assert resp.status_code == 503 and "Queue unavailable" in resp.json()["detail"]
    body = dict(ONE_HOP)
    body["callback"] = "http://cb.invalid/x"
    assert api.post("/asyncquery", json=body).status_code == 503
    assert api.get("/status.json").status_code == 503
