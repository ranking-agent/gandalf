"""Durable async-job queue backed by a Redis Stream + consumer group.

Why a stream (and not a plain list): Redis Streams give at-least-once delivery
out of the box. ``XREADGROUP`` moves an entry into the group's Pending Entries
List (PEL); it is removed only by an explicit ``XACK``. If a worker dies after
reading a job but before acking (kernel OOM, pod eviction, SIGKILL), the entry
stays pending and a live worker reclaims it via ``XAUTOCLAIM`` after an idle
timeout. That is the property that stops overflow queries from being dropped --
the failure mode this whole change exists to fix.

KEDA's ``redis-streams`` scaler reads the same stream/group directly (via
``XINFO``/``XPENDING``) to compute lag, so autoscaling needs no metrics
endpoint and no Prometheus -- just Redis, which lives in Gandalf's own chart.

The ``redis`` dependency is imported lazily so the core library and the
non-queue server path never require it.
"""

from __future__ import annotations

import logging
import os
import socket
from typing import Any, List, Optional, Tuple

import orjson

logger = logging.getLogger(__name__)

# The single stream field jobs are stored under. Value is orjson-encoded bytes.
_FIELD = b"data"


def default_consumer_name() -> str:
    """A per-process consumer name: ``<hostname>-<pid>``.

    Distinct per worker process so the group's pending list attributes each
    in-flight job to the exact consumer holding it, which is what makes
    reclaim-after-crash work.
    """
    return f"{socket.gethostname()}-{os.getpid()}"


class JobQueue:
    """Thin wrapper over a Redis Stream used as an async-job queue.

    Parameters mirror the ``queue_*`` settings in :mod:`gandalf.config`. Pass a
    ready-made ``client`` (e.g. ``fakeredis`` in tests) to bypass URL
    construction; otherwise a client is built from ``redis_url``.
    """

    def __init__(
        self,
        *,
        redis_url: str = "",
        stream: str = "gandalf:asyncquery",
        group: str = "gandalf-workers",
        consumer: str = "",
        max_len: int = 100_000,
        client: Any = None,
    ):
        if client is None:
            if not redis_url:
                raise ValueError("redis_url is required when no client is supplied")
            import redis  # lazy: only needed when the queue is actually used

            client = redis.Redis.from_url(redis_url)
        self._r = client
        self._stream = stream
        self._group = group
        self._consumer = consumer or default_consumer_name()
        self._max_len = max_len

    # -- lifecycle ---------------------------------------------------------

    def ensure_group(self) -> None:
        """Create the stream and consumer group if absent (idempotent).

        ``MKSTREAM`` creates the stream so producers and consumers can start in
        any order. ``id="0"`` means a freshly created group would replay any
        pre-existing entries; that only matters if the stream was populated
        before the group existed, which does not happen in normal operation.
        """
        try:
            self._r.xgroup_create(
                name=self._stream, groupname=self._group, id="0", mkstream=True
            )
            logger.info(
                "Created consumer group %s on stream %s", self._group, self._stream
            )
        except Exception as exc:  # redis.exceptions.ResponseError: BUSYGROUP
            if "BUSYGROUP" in str(exc):
                return
            raise

    # -- producer ----------------------------------------------------------

    def enqueue(self, job: dict) -> str:
        """Append a job to the stream. Returns the stream entry ID."""
        payload = orjson.dumps(job)
        entry_id = self._r.xadd(
            self._stream,
            {_FIELD: payload},
            maxlen=self._max_len,
            approximate=True,
        )
        return entry_id.decode() if isinstance(entry_id, bytes) else str(entry_id)

    # -- consumer ----------------------------------------------------------

    def read(self, count: int = 1, block_ms: int = 5_000) -> List[Tuple[str, dict]]:
        """Claim up to ``count`` new jobs for this consumer.

        Blocks up to ``block_ms`` for new entries. Returns ``(entry_id, job)``
        tuples; the entries are now in the pending list until :meth:`ack`.
        """
        resp = self._r.xreadgroup(
            groupname=self._group,
            consumername=self._consumer,
            streams={self._stream: ">"},
            count=count,
            block=block_ms,
        )
        return self._decode_stream_response(resp)

    def reclaim(self, min_idle_ms: int, count: int = 10) -> List[Tuple[str, dict, int]]:
        """Reclaim entries idle longer than ``min_idle_ms`` from dead workers.

        Uses ``XAUTOCLAIM`` to transfer ownership of stale pending entries to
        this consumer. Returns ``(entry_id, job, delivery_count)`` tuples;
        ``delivery_count`` lets the caller dead-letter poison jobs that keep
        killing their worker.

        Entries whose payload can no longer be parsed (or that vanished from
        the stream after a trim) are acked here and excluded from the result.
        """
        # XAUTOCLAIM returns (next_cursor, claimed_entries, deleted_ids) on
        # redis-py >= 4.2 / Redis >= 7; older Redis omits the deleted list.
        result = self._r.xautoclaim(
            name=self._stream,
            groupname=self._group,
            consumername=self._consumer,
            min_idle_time=min_idle_ms,
            count=count,
            justid=False,
        )
        entries = result[1] if isinstance(result, (list, tuple)) else []
        claimed = self._decode_stream_response([(self._stream, entries)])

        # Delivery counts come from XPENDING (XAUTOCLAIM does not return them).
        out: List[Tuple[str, dict, int]] = []
        for entry_id, job in claimed:
            delivered = self._delivery_count(entry_id)
            out.append((entry_id, job, delivered))
        return out

    def ack(self, entry_id: str) -> None:
        """Acknowledge and delete a finished entry.

        ``XACK`` removes it from the pending list; ``XDEL`` drops the payload
        from the stream so a completed job stops counting toward length/lag and
        does not accumulate. (KEDA's lag metric already excludes acked entries,
        but XDEL keeps the stream itself from growing unboundedly.)
        """
        self._r.xack(self._stream, self._group, entry_id)
        self._r.xdel(self._stream, entry_id)

    # -- observability -----------------------------------------------------

    def stats(self) -> dict:
        """Return a small snapshot for /queue_status and debugging."""
        length = int(self._r.xlen(self._stream))
        try:
            pending = self._r.xpending(self._stream, self._group)
            pending_count = int(pending["pending"]) if pending else 0
        except Exception:
            pending_count = 0
        return {
            "stream": self._stream,
            "group": self._group,
            "length": length,
            "pending": pending_count,
        }

    def _delivery_count(self, entry_id: str) -> int:
        try:
            details = self._r.xpending_range(
                self._stream, self._group, min=entry_id, max=entry_id, count=1
            )
            if details:
                return int(details[0]["times_delivered"])
        except Exception:
            pass
        return 1

    def _decode_stream_response(self, resp: Any) -> List[Tuple[str, dict]]:
        """Flatten a raw XREADGROUP/XAUTOCLAIM response into (id, job) tuples.

        Skips (and acks) entries whose payload is missing or unparseable so a
        single corrupt record can never wedge the consume loop.
        """
        out: List[Tuple[str, dict]] = []
        if not resp:
            return out
        for _stream_name, entries in resp:
            for entry_id, fields in entries:
                if isinstance(entry_id, bytes):
                    entry_id = entry_id.decode()
                if not fields:  # tombstone from a trimmed/deleted entry
                    self._r.xack(self._stream, self._group, entry_id)
                    continue
                raw = fields.get(_FIELD) or fields.get(_FIELD.decode())
                if raw is None:
                    self._r.xack(self._stream, self._group, entry_id)
                    continue
                try:
                    job = orjson.loads(raw)
                except orjson.JSONDecodeError:
                    logger.error("Dropping unparseable job %s", entry_id)
                    self.ack(entry_id)
                    continue
                out.append((entry_id, job))
        return out


def queue_from_settings(settings: Any, client: Any = None) -> JobQueue:
    """Build a :class:`JobQueue` from a Settings object."""
    return JobQueue(
        redis_url=settings.redis_url,
        stream=settings.queue_stream,
        group=settings.queue_group,
        consumer=settings.queue_consumer,
        max_len=settings.queue_max_len,
        client=client,
    )
