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
from gandalf.queue import Delivery, Job, JobQueue, ResultStore
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
