"""Jobs: the Redis Streams queue, result store, worker registry and job history.

With ``GANDALF_QUEUE_URL`` set, the API process turns each ``/query`` and
``/asyncquery`` into a :class:`Job` on a Redis Stream and worker processes
(:mod:`gandalf.worker`) execute them.  The stream is read through a consumer
group, which gives the properties the design needs:

* **Ack on completion.**  A worker acknowledges a job only after its result
  has been delivered, so a job whose worker died stays *pending* and is
  taken over by another worker once it has been idle for
  ``queue_claim_idle_seconds`` (``XAUTOCLAIM``).  A running worker refreshes
  its job's idle time every ``queue_keepalive_seconds`` (:meth:`JobQueue.touch`),
  so only a dead worker's job goes idle.
* **Poison protection.**  Redis counts deliveries per pending entry.  A job
  delivered more than ``queue_max_deliveries`` times has killed a worker
  before (an OOM kill, typically), so it is moved to the dead-letter stream
  and answered with an error instead of being run a third time and killing
  every worker in turn.
* **One scaling signal.**  The group's *lag* -- entries not yet delivered to
  any consumer -- is exactly "jobs nobody has started", which is what KEDA's
  ``redis-streams`` scaler reads (``lagCount``).

Results of ``/query`` jobs go through the :class:`ResultStore` rather than
the stream: zstd-compressed, split into chunks below Redis's 512 MB value
cap, with a TTL, and a per-job list the waiting API process blocks on.
``/asyncquery`` results never touch Redis: the worker POSTs them to the
client's callback itself.

Two more pieces of shared state feed the status page (``/status``): each
worker reports what it is doing to the :class:`WorkerRegistry`, and every
finished job is recorded in the :class:`JobHistory`.  Both live in Redis so
any API pod can show the whole deployment.
"""

from __future__ import annotations

import logging
import os
import socket
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Callable, Iterator, Optional

import orjson
import redis
import zstandard
from translator_tom.model_dicts import QueryDict

from gandalf.config import settings
from gandalf.trapi import Deadline

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Job envelope
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Job:
    """One queued query, everything a worker needs to run and answer it.

    Attributes:
        job_id: Unique id, also the key the result is stored under.
        query: The request body, already validated and normalized by the
            API (``server._prepare_query``), so the worker runs it as is.
        profile: Emit per-stage timings into ``message.logs``.
        budget: The query's ``parameters.timeout`` in seconds, or None.
        accepted_at: Wall-clock time (epoch seconds) the API accepted the
            request; the budget runs from here, queue wait included.
        callback: For ``/asyncquery``, where the worker POSTs the result.
            None means a ``/query`` is waiting on the result store.
        trace_headers: W3C trace context captured at the API, forwarded on
            the callback so the trace stays linked.
        request_id: The API's request id, for log correlation.

    Examples:
        >>> job = Job.new({"message": {}}, budget=30.0)
        >>> Job.from_bytes(job.to_bytes()) == job
        True
        >>> job.mode, Job.new({"message": {}}, callback="http://x/cb").mode
        ('sync', 'callback')
    """

    job_id: str
    query: QueryDict
    profile: bool = False
    budget: Optional[float] = None
    accepted_at: float = field(default_factory=time.time)
    callback: Optional[str] = None
    trace_headers: dict[str, str] = field(default_factory=dict)
    request_id: str = ""

    @classmethod
    def new(
        cls,
        query: QueryDict,
        *,
        profile: bool = False,
        budget: Optional[float] = None,
        callback: Optional[str] = None,
        trace_headers: Optional[dict[str, str]] = None,
        request_id: str = "",
    ) -> "Job":
        """A job with a fresh id, accepted now."""
        return cls(
            job_id=uuid.uuid4().hex,
            query=query,
            profile=profile,
            budget=budget,
            callback=callback,
            trace_headers=dict(trace_headers or {}),
            request_id=request_id,
        )

    @property
    def mode(self) -> str:
        """``"callback"`` for an async job, ``"sync"`` for a waiting ``/query``."""
        return "callback" if self.callback else "sync"

    def to_bytes(self) -> bytes:
        """Serialize for the stream entry."""
        data: bytes = orjson.dumps(asdict(self))
        return data

    @classmethod
    def from_bytes(cls, data: bytes) -> "Job":
        """The inverse of :meth:`to_bytes`."""
        return cls(**orjson.loads(data))

    def elapsed(self) -> float:
        """Seconds since the API accepted the request."""
        return time.time() - self.accepted_at

    def deadline(self) -> Deadline:
        """The query's deadline, with the time already spent in the queue."""
        return Deadline.started_ago(self.budget, self.elapsed())

    def wait_seconds(self) -> float:
        """How long the API should wait for this job's result.

        The budget plus a grace period for the worker's own Timeout response
        to arrive, or the configured maximum for a query with no budget.
        """
        if self.budget is None:
            return settings.queue_sync_max_wait_seconds
        return self.budget + settings.queue_sync_grace_seconds


@dataclass(frozen=True)
class Delivery:
    """A job handed to a worker, with what the queue knows about it.

    Attributes:
        entry_id: The stream entry id to acknowledge when done.
        job: The job.
        deliveries: How many times the entry has been delivered, this one
            included.  More than one means a previous worker never finished.
        poisoned: The entry exceeded the delivery limit and has already been
            dead-lettered and acknowledged; the worker must answer it with
            an error, not run it.
    """

    entry_id: bytes
    job: Job
    deliveries: int
    poisoned: bool = False


@dataclass(frozen=True)
class QueueStats:
    """What the consumer group reports about the stream."""

    lag: int  # entries not yet delivered to any consumer
    pending: int  # delivered, not yet acknowledged


def default_consumer_name() -> str:
    """``<hostname>-<pid>``: the pod name under Kubernetes, plus the process."""
    return f"{socket.gethostname()}-{os.getpid()}"


def redis_client(url: Optional[str] = None) -> redis.Redis:
    """A Redis client for the queue, with timeouts so nothing hangs forever.

    Query parameters on the URL (``?socket_timeout=5``) take precedence over
    the configured defaults, as redis-py has it.
    """
    return redis.Redis.from_url(
        url if url is not None else settings.queue_url,
        socket_timeout=settings.queue_socket_timeout_seconds,
        socket_connect_timeout=settings.queue_connect_timeout_seconds,
        health_check_interval=30,
    )


def blocking_slice(client: redis.Redis, wanted: float) -> float:
    """How long one blocking read may last on *client*.

    redis-py applies the socket timeout to blocking commands too, so a
    ``BLPOP`` or ``XREADGROUP BLOCK`` longer than it raises
    ``TimeoutError`` instead of waiting.  Blocking waits here are done in
    slices of at most half the socket timeout and looped.

    Examples:
        >>> blocking_slice(redis.Redis(socket_timeout=None), 5.0)
        5.0
        >>> blocking_slice(redis.Redis(socket_timeout=4), 5.0)
        2.0
    """
    socket_timeout = client.connection_pool.connection_kwargs.get("socket_timeout")
    if socket_timeout is None:
        return wanted
    return min(wanted, max(float(socket_timeout) / 2, 0.05))


def summarize_query(query: QueryDict) -> dict:
    """A few words about a query, for the status page and the job history.

    Node and edge counts, the first pinned CURIEs and predicates, and the
    requested timeout: enough to recognise a query in a list, small enough to
    store with every job.

    Examples:
        >>> summarize_query({"message": {"query_graph": {
        ...     "nodes": {"n0": {"ids": ["CHEBI:6801"]}, "n1": {"categories": ["biolink:Gene"]}},
        ...     "edges": {"e0": {"subject": "n0", "object": "n1", "predicates": ["biolink:affects"]}}}},
        ...     "parameters": {"timeout": 30}})
        {'nodes': 2, 'edges': 1, 'ids': ['CHEBI:6801'], 'predicates': ['biolink:affects'], 'timeout': 30}
        >>> summarize_query({"message": {}})
        {'nodes': 0, 'edges': 0, 'ids': [], 'predicates': []}
    """
    message = query.get("message") or {}
    query_graph = message.get("query_graph") or {}
    nodes = query_graph.get("nodes") or {}
    edges = query_graph.get("edges") or {}
    ids = [i for node in nodes.values() for i in (node.get("ids") or [])]
    predicates = [p for edge in edges.values() for p in (edge.get("predicates") or [])]
    summary: dict = {
        "nodes": len(nodes),
        "edges": len(edges),
        "ids": ids[:3],
        "predicates": predicates[:3],
    }
    parameters = query.get("parameters") or {}
    if "timeout" in parameters:
        summary["timeout"] = parameters["timeout"]
    return summary


def _decode(fields: dict) -> dict:
    """A Redis hash or stream entry as ``str`` keys and values."""
    return {k.decode(): v.decode() for k, v in fields.items()}


# ---------------------------------------------------------------------------
# Queue
# ---------------------------------------------------------------------------


class JobQueue:
    """The Redis Stream of jobs and its consumer group.

    Args:
        client: A ``redis.Redis`` returning bytes (``decode_responses`` off).
        stream: Stream key the jobs go on.
        group: Consumer group the workers read through.
        dead_stream: Where jobs over the delivery limit are moved.
        max_deliveries: Deliveries a job may have before it is dead-lettered.
        claim_idle_seconds: Idle time after which a pending job is taken over.
    """

    def __init__(
        self,
        client: redis.Redis,
        *,
        stream: str,
        group: str,
        dead_stream: str,
        max_deliveries: int,
        claim_idle_seconds: float,
    ):
        self._r = client
        self.stream = stream
        self.group = group
        self.dead_stream = dead_stream
        self.max_deliveries = max_deliveries
        self.claim_idle_ms = int(claim_idle_seconds * 1000)

    @classmethod
    def from_settings(cls, client: Optional[redis.Redis] = None) -> "JobQueue":
        """A queue on the configured stream, group and limits."""
        return cls(
            client if client is not None else redis_client(),
            stream=settings.queue_stream,
            group=settings.queue_group,
            dead_stream=settings.queue_dead_stream,
            max_deliveries=settings.queue_max_deliveries,
            claim_idle_seconds=settings.queue_claim_idle_seconds,
        )

    @property
    def client(self) -> redis.Redis:
        """The underlying Redis client."""
        return self._r

    def ping(self) -> bool:
        """Whether Redis answers; the API's readiness check."""
        return bool(self._r.ping())

    def ensure_group(self) -> None:
        """Create the stream and consumer group if they do not exist yet.

        Idempotent: Redis answers ``BUSYGROUP`` for a group that exists, and
        that is the only error swallowed here.
        """
        try:
            self._r.xgroup_create(self.stream, self.group, id="0", mkstream=True)
        except redis.ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    def enqueue(self, job: Job) -> str:
        """Put *job* on the stream and return its entry id."""
        entry_id: bytes = self._r.xadd(self.stream, {b"job": job.to_bytes()})
        return entry_id.decode()

    def receive(self, consumer: str, block_seconds: float) -> Optional[Delivery]:
        """Take the next job for *consumer*, or None if none arrived in time.

        A pending job another worker left idle for longer than
        ``claim_idle_seconds`` is taken first; otherwise the call blocks up to
        *block_seconds* for a new entry.
        """
        stale = self._claim_stale(consumer)
        if stale is not None:
            return stale
        response = self._r.xreadgroup(
            self.group,
            consumer,
            {self.stream: ">"},
            count=1,
            block=int(blocking_slice(self._r, block_seconds) * 1000),
        )
        if not response:
            return None
        entry_id, fields = response[0][1][0]
        return Delivery(entry_id, Job.from_bytes(fields[b"job"]), deliveries=1)

    def _claim_stale(self, consumer: str) -> Optional[Delivery]:
        """Take over one pending job whose worker went quiet, if there is one."""
        claimed = self._r.xautoclaim(
            self.stream,
            self.group,
            consumer,
            min_idle_time=self.claim_idle_ms,
            start_id="0-0",
            count=1,
        )
        entries = claimed[1]
        if not entries:
            return None
        entry_id, fields = entries[0]
        if fields is None:
            # Redis 6.2 reports an entry deleted from the stream this way;
            # 7.0 drops it from the group itself.  Nothing to run.
            self._r.xack(self.stream, self.group, entry_id)
            return None
        deliveries = self._deliveries(entry_id)
        job = Job.from_bytes(fields[b"job"])
        if deliveries > self.max_deliveries:
            logger.error(
                "job %s delivered %d times (limit %d): dead-lettering it",
                job.job_id,
                deliveries,
                self.max_deliveries,
            )
            self._dead_letter(entry_id, fields, deliveries)
            return Delivery(entry_id, job, deliveries, poisoned=True)
        logger.warning(
            "job %s taken over after %d earlier deliver%s",
            job.job_id,
            deliveries - 1,
            "y" if deliveries == 2 else "ies",
        )
        return Delivery(entry_id, job, deliveries)

    def _deliveries(self, entry_id: bytes) -> int:
        """How many times a pending entry has been delivered."""
        pending = self._r.xpending_range(
            self.stream, self.group, min=entry_id, max=entry_id, count=1
        )
        return int(pending[0]["times_delivered"]) if pending else 1

    def _dead_letter(self, entry_id: bytes, fields: dict, deliveries: int) -> None:
        pipe = self._r.pipeline(transaction=True)
        pipe.xadd(
            self.dead_stream,
            {
                **fields,
                b"source_id": entry_id,
                b"deliveries": str(deliveries).encode(),
                b"dead_at": repr(time.time()).encode(),
            },
        )
        pipe.xack(self.stream, self.group, entry_id)
        pipe.execute()

    def touch(self, entry_id: bytes, consumer: str) -> None:
        """Reset a running job's idle time so no other worker claims it.

        ``XCLAIM ... JUSTID`` re-assigns the entry to the same consumer and
        resets its idle clock without counting as a delivery.
        """
        self._r.xclaim(
            self.stream,
            self.group,
            consumer,
            min_idle_time=0,
            message_ids=[entry_id],
            justid=True,
        )

    def ack(self, entry_id: bytes) -> None:
        """Mark a job done: its result has been delivered."""
        self._r.xack(self.stream, self.group, entry_id)

    def length(self) -> int:
        """Entries on the stream, delivered or not (acknowledged ones are kept)."""
        return int(self._r.xlen(self.stream))

    def pending_entries(self, count: int = 100) -> list[dict]:
        """The jobs delivered and not yet acknowledged, oldest first.

        Each is ``{"entry_id", "consumer", "idle_s", "deliveries"}``.
        """
        return [
            {
                "entry_id": entry["message_id"].decode(),
                "consumer": entry["consumer"].decode(),
                "idle_s": entry["time_since_delivered"] / 1000.0,
                "deliveries": int(entry["times_delivered"]),
            }
            for entry in self._r.xpending_range(
                self.stream, self.group, min="-", max="+", count=count
            )
        ]

    def dead_letters(self, count: int = 20) -> list[dict]:
        """The most recent dead-lettered jobs, newest first.

        Each is ``{"entry_id", "job_id", "query", "deliveries", "dead_at"}``.
        """
        letters = []
        for entry_id, fields in self._r.xrevrange(self.dead_stream, count=count):
            job = Job.from_bytes(fields[b"job"])
            letters.append(
                {
                    "entry_id": entry_id.decode(),
                    "job_id": job.job_id,
                    "mode": job.mode,
                    "query": summarize_query(job.query),
                    "deliveries": int(fields[b"deliveries"]),
                    "dead_at": float(fields[b"dead_at"]),
                }
            )
        return letters

    def dead_letter_count(self) -> int:
        """How many jobs have been dead-lettered (the dead stream's length)."""
        return int(self._r.xlen(self.dead_stream))

    def stats(self) -> QueueStats:
        """Lag and pending counts of the consumer group (zero before it exists)."""
        for group in self._r.xinfo_groups(self.stream):
            if group["name"] == self.group.encode():
                return QueueStats(
                    lag=int(group.get("lag") or 0), pending=int(group["pending"])
                )
        return QueueStats(lag=0, pending=0)


# ---------------------------------------------------------------------------
# Result store
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StoredResult:
    """A job's answer as the store holds it.

    Attributes:
        compressed: The serialized response as one zstd frame.
        size: Its uncompressed length in bytes.
        http_status: The status the API should answer with.
    """

    compressed: bytes
    size: int
    http_status: int

    def decompress(self) -> bytes:
        """The serialized response."""
        body: bytes = zstandard.ZstdDecompressor().decompress(
            self.compressed, max_output_size=self.size
        )
        return body


class ResultStore:
    """Results of ``/query`` jobs, kept in Redis until the API collects them.

    A result is one zstd frame stored in ``chunk_bytes`` pieces under
    ``<prefix>:<job_id>:chunk:<n>``, described by a ``<prefix>:<job_id>:meta``
    hash, and announced by a push onto ``<prefix>:<job_id>:done`` that the
    waiting API process blocks on.  Every key carries the TTL, so a result
    nobody collects (the client gave up) expires on its own.

    Args:
        client: A ``redis.Redis`` returning bytes.
        prefix: Key prefix.
        ttl_seconds: How long an uncollected result lives.
        chunk_bytes: Largest value written to one key.
        zstd_level: Compression level for the stored frame.
    """

    def __init__(
        self,
        client: redis.Redis,
        *,
        prefix: str = "gandalf:result",
        ttl_seconds: int,
        chunk_bytes: int,
        zstd_level: int,
    ):
        self._r = client
        self.prefix = prefix
        self.ttl = ttl_seconds
        self.chunk_bytes = chunk_bytes
        self.zstd_level = zstd_level

    @classmethod
    def from_settings(cls, client: Optional[redis.Redis] = None) -> "ResultStore":
        """A store with the configured TTL, chunk size and compression level."""
        return cls(
            client if client is not None else redis_client(),
            ttl_seconds=settings.result_ttl_seconds,
            chunk_bytes=settings.result_chunk_bytes,
            zstd_level=settings.compress_zstd_level,
        )

    def _key(self, job_id: str, suffix: str) -> str:
        return f"{self.prefix}:{job_id}:{suffix}"

    def put(self, job_id: str, body: bytes, *, http_status: int = 200) -> int:
        """Store a serialized response and wake the waiter.

        Returns:
            The compressed size in bytes.
        """
        compressed = zstandard.ZstdCompressor(level=self.zstd_level).compress(body)
        chunks = [
            compressed[i : i + self.chunk_bytes]
            for i in range(0, len(compressed), self.chunk_bytes)
        ] or [b""]
        pipe = self._r.pipeline(transaction=False)
        for n, chunk in enumerate(chunks):
            pipe.set(self._key(job_id, f"chunk:{n}"), chunk, ex=self.ttl)
        meta = self._key(job_id, "meta")
        pipe.hset(
            meta,
            mapping={
                "chunks": len(chunks),
                "size": len(body),
                "compressed_size": len(compressed),
                "http_status": http_status,
                "content_encoding": "zstd",
            },
        )
        pipe.expire(meta, self.ttl)
        done = self._key(job_id, "done")
        pipe.rpush(done, b"1")
        pipe.expire(done, self.ttl)
        pipe.execute()
        return len(compressed)

    def wait(self, job_id: str, timeout: float) -> Optional[StoredResult]:
        """Block until the job's result is stored, or *timeout* seconds pass.

        Returns:
            The result, or None if it did not arrive in time.
        """
        key = self._key(job_id, "done")
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            # BLPOP's zero means "forever", so the shortest wait is one tick;
            # each wait stays under the socket timeout (blocking_slice).
            popped = self._r.blpop(
                [key], timeout=max(blocking_slice(self._r, remaining), 0.01)
            )
            if popped is not None:
                return self.get(job_id)

    def get(self, job_id: str) -> Optional[StoredResult]:
        """The stored result, or None if there is none (or it expired)."""
        meta = self._r.hgetall(self._key(job_id, "meta"))
        if not meta:
            return None
        n_chunks = int(meta[b"chunks"])
        chunks = self._r.mget(
            [self._key(job_id, f"chunk:{n}") for n in range(n_chunks)]
        )
        if any(chunk is None for chunk in chunks):
            return None
        return StoredResult(
            compressed=b"".join(chunks),
            size=int(meta[b"size"]),
            http_status=int(meta[b"http_status"]),
        )

    def delete(self, job_id: str) -> None:
        """Drop a collected result rather than wait for its TTL."""
        meta = self._r.hgetall(self._key(job_id, "meta"))
        n_chunks = int(meta[b"chunks"]) if meta else 0
        self._r.delete(
            self._key(job_id, "meta"),
            self._key(job_id, "done"),
            *[self._key(job_id, f"chunk:{n}") for n in range(n_chunks)],
        )


# ---------------------------------------------------------------------------
# Status: worker registry and job history
# ---------------------------------------------------------------------------


class WorkerRegistry:
    """What each worker last reported about itself.

    A worker writes a hash under ``<prefix>:<name>`` with a TTL and its name
    into a sorted set scored by the report time; a worker that stops
    reporting (it died, or was scaled away without a clean exit) drops out
    when its hash expires.

    Args:
        client: A ``redis.Redis`` returning bytes.
        prefix: Key prefix.
        ttl_seconds: How long a report stays valid without a refresh.
    """

    def __init__(
        self, client: redis.Redis, *, prefix: str = "gandalf:worker", ttl_seconds: float
    ):
        self._r = client
        self.prefix = prefix
        self.ttl = ttl_seconds
        self._index = f"{prefix}:index"

    @classmethod
    def from_settings(cls, client: Optional[redis.Redis] = None) -> "WorkerRegistry":
        """A registry whose reports outlive a few missed keepalives."""
        return cls(
            client if client is not None else redis_client(),
            ttl_seconds=settings.queue_claim_idle_seconds,
        )

    @property
    def client(self) -> redis.Redis:
        """The underlying Redis client."""
        return self._r

    def report(self, name: str, **fields: object) -> None:
        """Record *fields* as *name*'s current state (replacing the last report)."""
        now = time.time()
        mapping = {k: _field(v) for k, v in fields.items()}
        mapping["name"] = name
        mapping["last_seen"] = repr(now)
        key = f"{self.prefix}:{name}"
        pipe = self._r.pipeline(transaction=True)
        pipe.delete(key)
        pipe.hset(key, mapping=mapping)
        pipe.pexpire(key, int(self.ttl * 1000))
        pipe.zadd(self._index, {name: now})
        pipe.execute()

    def get(self, name: str) -> Optional[dict]:
        """*name*'s live report, or None if it has none (never ran, or expired)."""
        fields = self._r.hgetall(f"{self.prefix}:{name}")
        return _typed_report(_decode(fields)) if fields else None

    def record_last_words(self, name: str, text: str, ttl_seconds: int = 3600) -> None:
        """Keep why *name* is dying, for its next incarnation to report."""
        self._r.set(f"{self.prefix}:{name}:last_words", text, ex=ttl_seconds)

    def last_words(self, name: str) -> str:
        """What *name*'s previous incarnation recorded as it died, consumed."""
        words = self._r.getdel(f"{self.prefix}:{name}:last_words")
        return words.decode() if words else ""

    def forget(self, name: str) -> None:
        """Drop *name*'s report (a clean exit)."""
        pipe = self._r.pipeline(transaction=True)
        pipe.delete(f"{self.prefix}:{name}")
        pipe.zrem(self._index, name)
        pipe.execute()

    def workers(self) -> list[dict]:
        """Every worker with a live report, most recently seen first."""
        names = self._r.zrevrange(self._index, 0, -1)
        reports = []
        for raw_name in names:
            name = raw_name.decode()
            fields = self._r.hgetall(f"{self.prefix}:{name}")
            if not fields:
                self._r.zrem(self._index, name)  # expired: gone for good
                continue
            reports.append(_typed_report(_decode(fields)))
        return reports


def _field(value: object) -> str:
    """A report value as the string Redis stores."""
    if isinstance(value, (dict, list)):
        data: bytes = orjson.dumps(value)
        return data.decode()
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return str(value)


_NUMERIC_REPORT_FIELDS = frozenset(
    {
        "pid",
        "jobs_done",
        "started_at",
        "last_seen",
        "job_started_at",
        "rss_anon_kb",
        "startup_anon_kb",
        "max_jobs",
    }
)


def _typed_report(report: dict) -> dict:
    """Numbers back to numbers and the query summary back to a dict."""
    typed: dict = {}
    for key, value in report.items():
        if key in _NUMERIC_REPORT_FIELDS and value != "":
            typed[key] = float(value) if "." in value else int(value)
        elif key == "job_query" and value:
            typed[key] = orjson.loads(value)
        else:
            typed[key] = value
    return typed


class JobHistory:
    """The last few thousand finished jobs, as a capped Redis Stream.

    Args:
        client: A ``redis.Redis`` returning bytes.
        stream: Stream key.
        maxlen: Roughly how many records to keep.
    """

    def __init__(
        self, client: redis.Redis, *, stream: str = "gandalf:jobs:history", maxlen: int
    ):
        self._r = client
        self.stream = stream
        self.maxlen = maxlen

    @classmethod
    def from_settings(cls, client: Optional[redis.Redis] = None) -> "JobHistory":
        """A history of the configured length."""
        return cls(
            client if client is not None else redis_client(),
            maxlen=settings.history_maxlen,
        )

    def record(self, **fields: object) -> None:
        """Append one finished job."""
        self._r.xadd(
            self.stream,
            {k: _field(v) for k, v in fields.items()},
            maxlen=self.maxlen,
            approximate=True,
        )

    def recent(self, count: int) -> list[dict]:
        """The newest *count* records, newest first."""
        records = []
        for entry_id, fields in self._r.xrevrange(self.stream, count=count):
            record = _typed_history(_decode(fields))
            record["entry_id"] = entry_id.decode()
            records.append(record)
        return records


_NUMERIC_HISTORY_FIELDS = frozenset(
    {
        "http_status",
        "deliveries",
        "accepted_at",
        "started_at",
        "finished_at",
        "wait_s",
        "duration_s",
        "bytes",
    }
)


def _typed_history(record: dict) -> dict:
    typed: dict = {}
    for key, value in record.items():
        if key in _NUMERIC_HISTORY_FIELDS and value != "":
            typed[key] = float(value) if "." in value else int(value)
        elif key == "query" and value:
            typed[key] = orjson.loads(value)
        else:
            typed[key] = value
    return typed


# ---------------------------------------------------------------------------
# Keepalive
# ---------------------------------------------------------------------------


@contextmanager
def keepalive(
    queue: JobQueue,
    entry_id: bytes,
    consumer: str,
    interval_seconds: float,
    on_tick: Optional[Callable[[], None]] = None,
) -> Iterator[None]:
    """Refresh a running job's idle time every *interval_seconds*.

    Runs :meth:`JobQueue.touch` on a daemon thread for the duration of the
    block, and *on_tick* with it (the worker uses it to refresh its liveness
    heartbeat during a long query).  A thread rather than a check inside the
    query: the query code knows nothing about the queue.
    """
    stop = threading.Event()

    def _loop() -> None:
        while not stop.wait(interval_seconds):
            queue.touch(entry_id, consumer)
            if on_tick is not None:
                on_tick()

    thread = threading.Thread(target=_loop, daemon=True, name="gandalf-keepalive")
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()
