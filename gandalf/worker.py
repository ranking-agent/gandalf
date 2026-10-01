"""Queue worker: runs Gandalf jobs taken from the Redis Stream.

    python -m gandalf.worker

One process, one job at a time.  The process opens the graph like the API
does, joins the consumer group, and loops: take a job, run it, deliver the
answer (POST it to the client's callback, or store it for the ``/query``
that is waiting), acknowledge, repeat.  Horizontal scaling is more
processes; KEDA adds them when the stream's lag grows (see ``deploy/``).

Lifecycle, all driven by settings:

* ``SIGTERM`` (a scale-down or rollout) stops the loop after the current job,
  so the pod's ``terminationGracePeriodSeconds`` should cover one query.
* After ``worker_max_jobs`` jobs the process exits on its own and Kubernetes
  restarts it, returning the memory glibc holds after large queries.
* ``worker_heartbeat_file`` is touched on every loop iteration and every
  keepalive tick while a job runs; a liveness probe checks its age.
* ``worker_metrics_port`` serves Prometheus metrics (``gandalf.metrics``).
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import redis
from bmt.toolkit import Toolkit
from prometheus_client import start_http_server

from gandalf import CSRGraph
from gandalf.config import settings
from gandalf.execute import execute_to_bytes, load_runtime, post_callback
from gandalf.execute import serialize_response
from gandalf.logging_config import configure_logging, request_id_var
from gandalf.metrics import (
    JOB_DURATION,
    JOB_QUEUE_WAIT,
    JOB_RESULT_BYTES,
    JOBS,
    JOBS_INFLIGHT,
    PROCESS_RSS_ANON_BYTES,
    metrics_registry,
    rss_anon_kb,
)
from gandalf.queue import (
    Delivery,
    Job,
    JobQueue,
    ResultStore,
    default_consumer_name,
    keepalive,
)
from gandalf.trapi import error_response, timeout_response

logger = logging.getLogger(__name__)


class Worker:
    """The consume-execute-deliver loop.

    Args:
        graph: The loaded graph.
        bmt: The Biolink Model Toolkit.
        queue: The job stream.
        store: Where ``/query`` results go.
        consumer: This worker's name in the consumer group.
        max_jobs: Stop after this many jobs (0 = never).
        block_seconds: How long one wait for a job lasts before the loop
            checks for shutdown and refreshes the heartbeat.
        keepalive_seconds: How often a running job's idle time is refreshed.
        heartbeat_file: Touched to show the process is alive, or ``""``.
    """

    def __init__(
        self,
        graph: CSRGraph,
        bmt: Optional[Toolkit],
        queue: JobQueue,
        store: ResultStore,
        *,
        consumer: str,
        max_jobs: int = 0,
        block_seconds: float = 5.0,
        keepalive_seconds: float = 30.0,
        heartbeat_file: str = "",
    ):
        self.graph = graph
        self.bmt = bmt
        self.queue = queue
        self.store = store
        self.consumer = consumer
        self.max_jobs = max_jobs
        self.block_seconds = block_seconds
        self.keepalive_seconds = keepalive_seconds
        self.heartbeat_file = Path(heartbeat_file) if heartbeat_file else None
        self.jobs_done = 0
        self._stop = threading.Event()

    def stop(self) -> None:
        """Finish the current job, if any, then leave :meth:`run`."""
        self._stop.set()

    @property
    def stopping(self) -> bool:
        """Whether :meth:`stop` has been called."""
        return self._stop.is_set()

    def heartbeat(self) -> None:
        """Record that the process is alive."""
        if self.heartbeat_file is not None:
            self.heartbeat_file.touch()

    def run(self) -> int:
        """Consume jobs until stopped or ``max_jobs`` is reached.

        Returns:
            How many jobs this worker finished.
        """
        self.queue.ensure_group()
        logger.info(
            "Worker %s consuming %s as %s (PID=%d)",
            self.consumer,
            self.queue.stream,
            self.queue.group,
            os.getpid(),
        )
        while not self._stop.is_set():
            if self.max_jobs and self.jobs_done >= self.max_jobs:
                logger.info(
                    "Worker %s done %d jobs; recycling", self.consumer, self.jobs_done
                )
                break
            self.heartbeat()
            delivery = self.queue.receive(self.consumer, self.block_seconds)
            if delivery is not None:
                self.handle(delivery)
        return self.jobs_done

    def handle(self, delivery: Delivery) -> str:
        """Run one delivered job, deliver its answer and acknowledge it.

        Returns:
            The outcome recorded in ``gandalf_jobs_total``.
        """
        job = delivery.job
        request_id_var.set(job.request_id or job.job_id[:8])
        JOB_QUEUE_WAIT.observe(job.elapsed())
        logger.info(
            "job %s start mode=%s waited=%.1fs deliveries=%d",
            job.job_id,
            job.mode,
            job.elapsed(),
            delivery.deliveries,
        )
        if delivery.poisoned:
            body = serialize_response(
                error_response(
                    job.query,
                    f"Query was abandoned after {delivery.deliveries - 1} attempts "
                    "ended without a result (the worker running it died); it was "
                    "not run again.",
                )
            )
            http_status, outcome = 500, "poisoned"
        else:
            with (
                keepalive(
                    self.queue,
                    delivery.entry_id,
                    self.consumer,
                    self.keepalive_seconds,
                    on_tick=self.heartbeat,
                ),
                JOBS_INFLIGHT.track_inprogress(),
            ):
                body, http_status, outcome = self.execute(job)
        self.deliver(job, body, http_status)
        self.queue.ack(delivery.entry_id)
        self.jobs_done += 1
        JOBS.labels(outcome).inc()
        PROCESS_RSS_ANON_BYTES.set(rss_anon_kb() * 1024)
        logger.info(
            "job %s end outcome=%s status=%d bytes=%d",
            job.job_id,
            outcome,
            http_status,
            len(body),
        )
        return outcome

    def execute(self, job: Job) -> tuple[bytes, int, str]:
        """Run a job to a serialized Response.

        Returns:
            The body, the HTTP status a ``/query`` should answer with, and
            the outcome: ``ok``, ``timeout``, ``expired`` or ``error``.
        """
        deadline = job.deadline()
        if deadline.expired:
            # The budget ran out while the job waited in the queue.  Answer as
            # a lookup that overran would, without starting one.
            return (
                serialize_response(timeout_response(job.query, deadline, [])),
                200,
                "expired",
            )
        t0 = time.perf_counter()
        # The job boundary: one query that blows up (a bug, a resource limit)
        # is answered with an Error response and the worker lives on to take
        # the next job.  Nothing narrower would do -- lookup raises whatever
        # the graph or a plugin raises.
        try:
            body = execute_to_bytes(
                self.graph, self.bmt, job.query, profile=job.profile, deadline=deadline
            )
        except Exception as exc:
            logger.exception("job %s failed", job.job_id)
            body = serialize_response(
                error_response(job.query, f"{type(exc).__name__}: {exc}")
            )
            return body, 500, "error"
        JOB_DURATION.observe(time.perf_counter() - t0)
        JOB_RESULT_BYTES.observe(len(body))
        return body, 200, ("timeout" if deadline.expired else "ok")

    def deliver(self, job: Job, body: bytes, http_status: int) -> None:
        """Hand the answer to whoever is waiting for it."""
        if job.callback:
            post_callback(job.callback, body, job.trace_headers)
        else:
            self.store.put(job.job_id, body, http_status=http_status)


def main() -> int:
    """Entry point for ``python -m gandalf.worker``."""
    configure_logging(
        getattr(logging, settings.log_level, logging.INFO), fmt=settings.log_format
    )
    if not settings.queue_url:
        raise SystemExit("GANDALF_QUEUE_URL must be set to run a worker")

    graph, bmt = load_runtime()
    client = redis.Redis.from_url(settings.queue_url)
    worker = Worker(
        graph,
        bmt,
        JobQueue.from_settings(client),
        ResultStore.from_settings(client),
        consumer=settings.worker_name or default_consumer_name(),
        max_jobs=settings.worker_max_jobs,
        block_seconds=settings.queue_block_seconds,
        keepalive_seconds=settings.queue_keepalive_seconds,
        heartbeat_file=settings.worker_heartbeat_file,
    )
    if settings.worker_metrics_port:
        # The image sets PROMETHEUS_MULTIPROC_DIR for the API's gunicorn
        # workers; a worker is one process, and metrics_registry() serves
        # its metrics correctly either way.
        start_http_server(settings.worker_metrics_port, registry=metrics_registry())

    def _stop(signum: int, _frame: object) -> None:
        logger.info("Signal %d: stopping after the current job", signum)
        worker.stop()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    done = worker.run()
    logger.info("Worker exiting after %d jobs", done)
    return 0


if __name__ == "__main__":
    sys.exit(main())
