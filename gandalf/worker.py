"""Standalone consumer for the Redis-backed async-query queue.

Run one or more of these as their own Kubernetes Deployment and let KEDA's
``redis-streams`` scaler grow/shrink the replica count on consumer-group lag::

    python -m gandalf.worker

Each worker loads its own copy of the graph (there is no cross-pod COW sharing
regardless), then loops: reclaim stale jobs from crashed peers, read new jobs,
run each lookup, POST the result to its callback, and ack. SIGTERM (sent by
Kubernetes on scale-down/rollout) triggers a graceful drain: the loop stops
reading new jobs, finishes the in-flight batch, and exits. Anything not acked
by exit stays pending and is reclaimed by a surviving worker.

This deliberately reuses the same graph-load path and job semantics as the API
server so a query behaves identically whether it ran inline or on a worker.
"""

from __future__ import annotations

import logging
import os
import signal
import threading
from typing import Any, Optional

from gandalf import CSRGraph
from gandalf.biolink import make_toolkit
from gandalf.config import settings
from gandalf.jobs import execute_job
from gandalf.logging_config import configure_logging
from gandalf.queue import JobQueue, queue_from_settings

logger = logging.getLogger("gandalf.worker")

# A job whose worker is killed mid-run is retried via reclaim. If it keeps
# killing its worker (a genuine poison pill -- e.g. a query that OOMs), stop
# after this many deliveries: ack it and log, so it cannot take down the whole
# pool in a retry loop.
_MAX_DELIVERIES = 5


def load_graph_and_bmt() -> tuple[CSRGraph, Any]:
    """Load the graph + BMT exactly as the server's preload path does."""
    import gc

    logger.info(
        "Loading graph from %s (format=%s)...",
        settings.graph_path,
        settings.graph_format,
    )
    graph = CSRGraph.load_mmap(settings.graph_path)
    logger.info("Initializing Biolink Model Toolkit...")
    bmt = make_toolkit()
    # Same GC tuning as server.py: freeze the graph+BMT into a generation the
    # cyclic collector never scans, then relax thresholds for query-time churn.
    gc.collect()
    gc.freeze()
    gc.set_threshold(50_000, 50, 50)
    logger.info("Worker graph loaded (PID=%d).", os.getpid())
    return graph, bmt


class Worker:
    """Consume-and-execute loop over a :class:`JobQueue`."""

    def __init__(self, queue: JobQueue, graph: CSRGraph, bmt: Any):
        self._queue = queue
        self._graph = graph
        self._bmt = bmt
        self._stop = threading.Event()

    def request_stop(self, *_: Any) -> None:
        logger.info("Stop requested — draining current batch then exiting.")
        self._stop.set()

    def run(self) -> None:
        self._queue.ensure_group()
        logger.info("Worker consuming from queue.")
        while not self._stop.is_set():
            # Reclaim first: pick up jobs abandoned by workers that died mid-run
            # before touching the fresh backlog.
            self._drain_reclaimed()
            if self._stop.is_set():
                break
            batch = self._queue.read(
                count=settings.queue_batch, block_ms=settings.queue_block_ms
            )
            for entry_id, job in batch:
                self._process(entry_id, job)
        logger.info("Worker loop exited cleanly.")

    def _drain_reclaimed(self) -> None:
        try:
            reclaimed = self._queue.reclaim(
                min_idle_ms=settings.queue_reclaim_idle_ms, count=settings.queue_batch
            )
        except Exception:
            logger.exception("Reclaim failed; continuing")
            return
        for entry_id, job, delivered in reclaimed:
            if delivered > _MAX_DELIVERIES:
                logger.error(
                    "Dead-lettering job %s after %d deliveries (callback=%s)",
                    entry_id,
                    delivered,
                    job.get("callback"),
                )
                self._queue.ack(entry_id)
                continue
            logger.warning("Reclaimed stale job %s (delivery #%d)", entry_id, delivered)
            self._process(entry_id, job)

    def _process(self, entry_id: str, job: dict) -> None:
        # execute_job never raises (best-effort per job); acking after it
        # returns means application errors are not retried, while a worker that
        # is killed *inside* execute_job never reaches this ack and is reclaimed.
        execute_job(job, self._graph, self._bmt)
        self._queue.ack(entry_id)


def main() -> None:
    configure_logging(
        getattr(logging, settings.log_level, logging.INFO), fmt=settings.log_format
    )
    if not settings.queue_enabled:
        logger.warning(
            "GANDALF_QUEUE_ENABLED is false — worker will run but the API is not "
            "enqueuing. Set it true on both the API and worker deployments."
        )
    if not settings.redis_url:
        raise SystemExit("GANDALF_REDIS_URL must be set to run the queue worker.")

    graph, bmt = load_graph_and_bmt()
    queue = queue_from_settings(settings)
    worker = Worker(queue, graph, bmt)

    signal.signal(signal.SIGTERM, worker.request_stop)
    signal.signal(signal.SIGINT, worker.request_stop)
    worker.run()


if __name__ == "__main__":
    main()
