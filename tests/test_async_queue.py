"""Integration tests for the Redis-backed /asyncquery path and the worker.

Covers three things:
* /asyncquery enqueues to Redis (returns a job_id, runs nothing inline) when
  ``queue_enabled`` is set, and still falls back to an in-process background
  task when it is not;
* the worker drains a job and POSTs the lookup result to the callback URL;
* the worker acks a processed job so it leaves the pending list.

fakeredis stands in for Redis; a threaded HTTPServer stands in for the client's
callback endpoint.
"""

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import fakeredis
import orjson
import pytest
from fastapi.testclient import TestClient

from gandalf.queue import JobQueue
from tests.search_fixtures import graph  # noqa: F401

_ONE_HOP = {
    "callback": "http://localhost:0/cb",
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


@pytest.fixture
def redis_client():
    return fakeredis.FakeStrictRedis()


@pytest.fixture
def server(graph, bmt, monkeypatch, redis_client):  # noqa: F811
    """Server module + TestClient with the queue enabled and fakeredis wired in."""
    monkeypatch.setenv("GANDALF_SKIP_PRELOAD", "true")
    monkeypatch.setenv("GANDALF_OTEL_ENABLED", "false")
    from gandalf import server as gandalf_server

    monkeypatch.setattr(gandalf_server, "GRAPH", graph)
    monkeypatch.setattr(gandalf_server, "BMT", bmt)
    monkeypatch.setattr(gandalf_server.settings, "queue_enabled", True)
    # Inject a fakeredis-backed queue so _get_queue() returns it directly.
    q = JobQueue(stream="s", group="g", consumer="api", client=redis_client)
    q.ensure_group()
    monkeypatch.setattr(gandalf_server, "_QUEUE", q)
    return gandalf_server, TestClient(gandalf_server.APP), q


class TestAsyncEnqueue:
    def test_asyncquery_enqueues_instead_of_running_inline(self, server, monkeypatch):
        gandalf_server, client, q = server

        # If the API ran the lookup inline, this would fire.
        def _boom(*a, **k):
            raise AssertionError("lookup must not run in the API process")

        monkeypatch.setattr(gandalf_server, "lookup", _boom)

        resp = client.post("/asyncquery", json=_ONE_HOP)
        assert resp.status_code == 200, resp.text
        payload = resp.json()
        assert payload["status"] == "accepted"
        assert "job_id" in payload  # only the queue path returns this

        stats = q.stats()
        assert stats["length"] == 1  # job is durably on the stream

    def test_enqueued_job_carries_callback_and_query(self, server):
        _, client, q = server
        resp = client.post("/asyncquery", json=_ONE_HOP)
        assert resp.status_code == 200, resp.text

        ((entry_id, job),) = q.read(count=10, block_ms=50)
        assert job["callback"] == _ONE_HOP["callback"]
        assert job["query"]["message"]["query_graph"]["nodes"]["n0"]["ids"] == [
            "CHEBI:6801"
        ]

    def test_queue_status_endpoint_reports_backlog(self, server):
        _, client, q = server
        client.post("/asyncquery", json=_ONE_HOP)
        status = client.get("/queue_status").json()
        assert status["queue_enabled"] is True
        assert status["length"] == 1


class TestAsyncFallback:
    def test_falls_back_to_background_task_when_queue_disabled(
        self, graph, bmt, monkeypatch  # noqa: F811
    ):
        monkeypatch.setenv("GANDALF_SKIP_PRELOAD", "true")
        monkeypatch.setenv("GANDALF_OTEL_ENABLED", "false")
        from gandalf import server as gandalf_server

        monkeypatch.setattr(gandalf_server, "GRAPH", graph)
        monkeypatch.setattr(gandalf_server, "BMT", bmt)
        monkeypatch.setattr(gandalf_server.settings, "queue_enabled", False)

        added = {}

        def fake_add_task(func, *args, **kwargs):
            added["func"] = func

        # TestClient runs background tasks for real; intercept to keep the test
        # hermetic and just assert the in-process path was chosen.
        monkeypatch.setattr(
            "starlette.background.BackgroundTasks.add_task",
            lambda self, func, *a, **k: added.setdefault("func", func),
        )

        client = TestClient(gandalf_server.APP)
        resp = client.post("/asyncquery", json=_ONE_HOP)
        assert resp.status_code == 200, resp.text
        assert "job_id" not in resp.json()  # background path, no queue id
        assert added.get("func") is gandalf_server._async_lookup


# ---------------------------------------------------------------------------
# Worker end-to-end
# ---------------------------------------------------------------------------


class _CaptureHandler(BaseHTTPRequestHandler):
    received: list = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        _CaptureHandler.received.append(self.rfile.read(length))
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args):  # silence
        pass


@pytest.fixture
def callback_server():
    _CaptureHandler.received = []
    httpd = HTTPServer(("127.0.0.1", 0), _CaptureHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    port = httpd.server_address[1]
    yield f"http://127.0.0.1:{port}/cb"
    httpd.shutdown()


def test_worker_processes_job_and_posts_callback(
    graph, bmt, monkeypatch, redis_client, callback_server  # noqa: F811
):
    from gandalf import jobs
    from gandalf.worker import Worker

    # Stub the heavy lookup so the test targets queue/worker plumbing, not
    # path-finding; the result must still reach the callback verbatim.
    sentinel = {"message": {"results": [{"ok": True}]}}
    monkeypatch.setattr(jobs, "lookup", lambda g, q, **k: sentinel)

    q = JobQueue(stream="s", group="g", consumer="w1", client=redis_client)
    q.ensure_group()
    entry_id = q.enqueue(
        {"callback": callback_server, "query": {"message": {}}, "profile": False}
    )

    worker = Worker(q, graph, bmt)
    ((eid, job),) = q.read(count=1, block_ms=50)
    worker._process(eid, job)

    # Callback received exactly the lookup response.
    assert len(_CaptureHandler.received) == 1
    assert orjson.loads(_CaptureHandler.received[0]) == sentinel
    # And the job was acked (no longer pending, dropped from the stream).
    assert q.stats() == {"stream": "s", "group": "g", "length": 0, "pending": 0}


def test_worker_dead_letters_poison_job_over_max_deliveries(
    graph, bmt, monkeypatch, redis_client  # noqa: F811
):
    from gandalf import worker as worker_mod
    from gandalf.worker import Worker

    monkeypatch.setattr(worker_mod, "_MAX_DELIVERIES", 2)
    # Reclaim anything idle (default threshold is 10 minutes).
    monkeypatch.setattr(worker_mod.settings, "queue_reclaim_idle_ms", 0)

    q = JobQueue(stream="s", group="g", consumer="w1", client=redis_client)
    q.ensure_group()
    q.enqueue({"callback": "http://x/cb", "query": {"message": {}}})
    q.read(count=1, block_ms=50)  # first delivery, left un-acked

    processed = []
    worker = Worker(q, graph, bmt)
    monkeypatch.setattr(worker, "_process", lambda eid, job: processed.append(eid))

    # Reclaim repeatedly; once delivery count exceeds the cap the job is
    # dead-lettered (acked) instead of reprocessed.
    for _ in range(5):
        worker._drain_reclaimed()

    assert q.stats()["pending"] == 0  # eventually acked, not stuck
