"""Unit tests for the Redis-Stream job queue (gandalf.queue).

Backed by fakeredis so no real Redis server is needed. These exercise the
delivery semantics that make queued queries durable: read moves a job to the
pending list, ack removes it, and a job left un-acked by a dead worker is
reclaimable by a live one with an incrementing delivery count.
"""

import fakeredis
import orjson
import pytest

from gandalf.queue import JobQueue


@pytest.fixture
def redis_client():
    return fakeredis.FakeStrictRedis()


@pytest.fixture
def queue(redis_client):
    q = JobQueue(stream="s", group="g", consumer="c1", client=redis_client)
    q.ensure_group()
    return q


def test_ensure_group_is_idempotent(queue):
    # Second call must not raise (BUSYGROUP is swallowed).
    queue.ensure_group()
    queue.ensure_group()


def test_enqueue_then_read_roundtrips_payload(queue):
    job = {"callback": "http://x/cb", "query": {"a": 1}, "profile": True}
    entry_id = queue.enqueue(job)
    assert isinstance(entry_id, str)

    batch = queue.read(count=10, block_ms=50)
    assert len(batch) == 1
    read_id, read_job = batch[0]
    assert read_id == entry_id
    assert read_job == job


def test_read_moves_job_to_pending_until_acked(queue):
    queue.enqueue({"callback": "http://x/cb", "query": {}})
    assert queue.stats()["pending"] == 0

    ((entry_id, _),) = queue.read(count=10, block_ms=50)
    assert queue.stats()["pending"] == 1  # delivered, not yet acked

    queue.ack(entry_id)
    stats = queue.stats()
    assert stats["pending"] == 0
    assert stats["length"] == 0  # XDEL also drops it from the stream


def test_read_returns_empty_when_no_jobs(queue):
    assert queue.read(count=10, block_ms=10) == []


def test_reclaim_recovers_unacked_job_from_dead_worker(redis_client):
    producer = JobQueue(stream="s", group="g", consumer="dead", client=redis_client)
    producer.ensure_group()
    producer.enqueue({"callback": "http://x/cb", "query": {"n": 7}})

    # "dead" worker reads the job but never acks (simulating a crash).
    ((entry_id, _),) = producer.read(count=1, block_ms=50)

    # A live worker reclaims anything idle >= 0ms.
    live = JobQueue(stream="s", group="g", consumer="live", client=redis_client)
    reclaimed = live.reclaim(min_idle_ms=0, count=10)
    assert len(reclaimed) == 1
    r_id, r_job, delivered = reclaimed[0]
    assert r_id == entry_id
    assert r_job["query"]["n"] == 7
    assert delivered >= 2  # original delivery + this reclaim

    live.ack(r_id)
    assert live.stats()["pending"] == 0


def test_reclaim_delivery_count_climbs_across_repeated_reclaims(redis_client):
    q = JobQueue(stream="s", group="g", consumer="c", client=redis_client)
    q.ensure_group()
    q.enqueue({"callback": "http://x/cb", "query": {}})
    q.read(count=1, block_ms=50)

    counts = []
    for _ in range(3):
        rec = q.reclaim(min_idle_ms=0, count=10)
        counts.append(rec[0][2])
    # Monotonically increasing — this is what the worker's dead-letter guard
    # keys off of to stop retrying a poison job forever.
    assert counts == sorted(counts)
    assert counts[-1] > counts[0]


def test_unparseable_payload_is_dropped_not_returned(queue, redis_client):
    # Write a malformed entry directly (bypassing enqueue's orjson encoding).
    redis_client.xadd("s", {b"data": b"not-json{{"})
    batch = queue.read(count=10, block_ms=50)
    assert batch == []  # corrupt record acked+skipped, never surfaced
    assert queue.stats()["pending"] == 0


def test_stats_reports_length_and_pending(queue):
    queue.enqueue({"callback": "http://x/cb", "query": {}})
    queue.enqueue({"callback": "http://x/cb", "query": {}})
    stats = queue.stats()
    assert stats["length"] == 2
    assert stats["group"] == "g"
    assert stats["stream"] == "s"
