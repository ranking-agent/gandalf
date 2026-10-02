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
import socket
import sys
import threading
import time
import traceback
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
from gandalf.notify import Event, Notifier, mark_worker_leaving
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
from gandalf.jobs import (
    Delivery,
    Job,
    JobHistory,
    JobQueue,
    ResultStore,
    WorkerRegistry,
    default_consumer_name,
    keepalive,
    redis_client,
    summarize_query,
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
        registry: Where this worker reports its state for the status page.
        history: Where finished jobs are recorded for the status page.
        notifier: Where this worker's events (start, stop, a job retried,
            dead-lettered or failed) go.
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
        registry: Optional[WorkerRegistry] = None,
        history: Optional[JobHistory] = None,
        notifier: Optional[Notifier] = None,
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
        self.registry = registry
        self.history = history
        self.notifier = notifier
        self.jobs_done = 0
        self.started_at = time.time()
        self.last_outcome = ""
        self._current: dict = {}  # the running job's report fields
        self._stop = threading.Event()

    def stop(self) -> None:
        """Finish the current job, if any, then leave :meth:`run`."""
        self._stop.set()

    @property
    def stopping(self) -> bool:
        """Whether :meth:`stop` has been called."""
        return self._stop.is_set()

    def heartbeat(self) -> None:
        """Record that the process is alive: the file, and the registry."""
        if self.heartbeat_file is not None:
            self.heartbeat_file.touch()
        self.report()

    def report(self) -> None:
        """Tell the registry what this worker is doing right now."""
        if self.registry is None:
            return
        self.registry.report(
            self.consumer,
            host=socket.gethostname(),
            pid=os.getpid(),
            started_at=self.started_at,
            state="running" if self._current else "idle",
            jobs_done=self.jobs_done,
            max_jobs=self.max_jobs,
            last_outcome=self.last_outcome,
            rss_anon_kb=rss_anon_kb(),
            stopping=self.stopping,
            **self._current,
        )

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
        self.notify(self._start_event())
        self.report()
        recycled = False
        backoff = 1.0
        while not self._stop.is_set():
            if self.max_jobs and self.jobs_done >= self.max_jobs:
                logger.info(
                    "Worker %s done %d jobs; recycling", self.consumer, self.jobs_done
                )
                recycled = True
                break
            # Redis trouble (a timeout, an outage, a restart) is waited out
            # here rather than ending the process: the graph took seconds to
            # open, and a job that was not acknowledged is redelivered.
            try:
                self.heartbeat()
                delivery = self.queue.receive(self.consumer, self.block_seconds)
                if delivery is not None:
                    self.handle(delivery)
            except redis.RedisError as exc:
                logger.warning(
                    "Worker %s: Redis unavailable (%s: %s); retrying in %.0fs",
                    self.consumer,
                    type(exc).__name__,
                    exc,
                    backoff,
                )
                self._current = {}
                self._stop.wait(backoff)
                backoff = min(backoff * 2, 30.0)
                continue
            backoff = 1.0
        # Leaving on purpose: say so before the registry entry goes, or the
        # monitor reports a lost worker.
        if self.registry is not None:
            mark_worker_leaving(self.registry.client, self.consumer)
            self.registry.forget(self.consumer)
        pool = self._pool_size()
        self.notify(
            Event(
                "worker_recycled" if recycled else "worker_stopped",
                "Worker recycled" if recycled else "Worker stopped",
                f"Worker {self.consumer} exited after {self.jobs_done} jobs "
                + (
                    f"(its {self.max_jobs}-job limit); Kubernetes restarts it."
                    if recycled
                    else "on SIGTERM (a scale-down or a rollout)."
                )
                + f" {pool} alive.",
                {"worker": self.consumer, "jobs done": self.jobs_done, "pool": pool},
            )
        )
        return self.jobs_done

    def _start_event(self) -> Event:
        """ "Worker started", or "Worker restarted" when the last instance died.

        A report under this worker's name that is still live means the
        previous process with this name never exited cleanly (a clean exit
        removes it): it crashed, or was killed.  Its last words, if it
        managed any, say which.
        """
        previous = self.registry.get(self.consumer) if self.registry else None
        pool = self._pool_size() + (0 if previous else 1)
        if previous is None:
            return Event(
                "worker_started",
                "Worker started",
                f"Worker {self.consumer} is taking jobs; {pool} alive.",
                {"worker": self.consumer, "pool": pool},
            )
        now = time.time()
        last_words = self.registry.last_words(self.consumer) if self.registry else ""
        fields: dict = {
            "worker": self.consumer,
            "previous lifetime": f"{now - float(previous.get('started_at') or now):.0f}s",
            "jobs done": previous.get("jobs_done", 0),
            "last report": f"{now - float(previous.get('last_seen') or now):.0f}s ago",
        }
        if previous.get("state") == "running":
            query = previous.get("job_query") or {}
            fields["was running"] = (
                f"{str(previous.get('job_id', ''))[:8]} "
                f"({query.get('nodes', '?')} nodes / {query.get('edges', '?')} edges, "
                f"ids {', '.join(query.get('ids') or []) or 'none'})"
            )
            fields["anon RSS"] = f"{int(previous.get('rss_anon_kb', 0)) // 1024} MB"
        if last_words:
            fields["died with"] = last_words
            cause = "It died with the error below."
        else:
            cause = (
                "It recorded no error, so it was killed from outside: a kernel "
                "OOM kill or a liveness probe, usually."
            )
        return Event(
            "worker_restarted",
            "Worker restarted after dying",
            f"Worker {self.consumer} is back, but its previous instance never "
            f"exited cleanly. {cause} {pool} alive.",
            fields,
        )

    def notify(self, event: Event) -> None:
        """Emit *event* if a notifier is attached."""
        if self.notifier is not None:
            self.notifier.emit(event)

    def _pool_size(self) -> int:
        """How many workers currently report, this one included if it does."""
        return len(self.registry.workers()) if self.registry is not None else 0

    def handle(self, delivery: Delivery) -> str:
        """Run one delivered job, deliver its answer and acknowledge it.

        Returns:
            The outcome recorded in ``gandalf_jobs_total``.
        """
        job = delivery.job
        request_id_var.set(job.request_id or job.job_id[:8])
        started_at = time.time()
        wait_s = job.elapsed()
        JOB_QUEUE_WAIT.observe(wait_s)
        self._current = {
            "job_id": job.job_id,
            "job_mode": job.mode,
            "job_started_at": started_at,
            "job_query": summarize_query(job.query),
            "job_deliveries": delivery.deliveries,
        }
        self.report()
        logger.info(
            "job %s start mode=%s waited=%.1fs deliveries=%d",
            job.job_id,
            job.mode,
            job.elapsed(),
            delivery.deliveries,
        )
        summary = summarize_query(job.query)
        describe = (
            f"{summary['nodes']} nodes / {summary['edges']} edges, "
            f"ids {', '.join(summary['ids']) or 'none'}"
        )
        if delivery.poisoned:
            self.notify(
                Event(
                    "job_dead_lettered",
                    "Job dead-lettered",
                    f"Job {job.job_id[:8]} ended {delivery.deliveries - 1} workers "
                    "without a result and was not run again; the client gets an "
                    "Error response.",
                    {"job": job.job_id, "query": describe, "mode": job.mode},
                )
            )
        elif delivery.deliveries > 1:
            self.notify(
                Event(
                    "job_retried",
                    "Job retried",
                    f"Job {job.job_id[:8]} was abandoned by its previous worker "
                    f"and is being run again by {self.consumer} "
                    f"(attempt {delivery.deliveries}).",
                    {"job": job.job_id, "query": describe, "mode": job.mode},
                )
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
        if outcome == "error":
            self.notify(
                Event(
                    "job_failed",
                    "Job failed",
                    f"Job {job.job_id[:8]} raised on worker {self.consumer}; the "
                    "client gets an Error response. See the worker log for the "
                    "traceback.",
                    {"job": job.job_id, "query": describe, "mode": job.mode},
                )
            )
        self.deliver(job, body, http_status)
        self.queue.ack(delivery.entry_id)
        self.jobs_done += 1
        self.last_outcome = outcome
        self._current = {}
        JOBS.labels(outcome).inc()
        PROCESS_RSS_ANON_BYTES.set(rss_anon_kb() * 1024)
        finished_at = time.time()
        if self.history is not None:
            self.history.record(
                job_id=job.job_id,
                request_id=job.request_id,
                mode=job.mode,
                outcome=outcome,
                http_status=http_status,
                worker=self.consumer,
                deliveries=delivery.deliveries,
                accepted_at=job.accepted_at,
                started_at=started_at,
                finished_at=finished_at,
                wait_s=wait_s,
                duration_s=finished_at - started_at,
                bytes=len(body),
                query=summarize_query(job.query),
            )
        self.report()
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
    client = redis_client()
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
        registry=WorkerRegistry.from_settings(client),
        history=JobHistory.from_settings(client),
        notifier=Notifier.from_settings(client),
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

    try:
        done = worker.run()
    except Exception:
        # The last words: whatever killed the loop, kept for the next
        # instance of this worker to report (see Worker._start_event).  A
        # kill signal leaves none, which is itself the diagnosis.
        logger.exception("Worker %s died", worker.consumer)
        if worker.registry is not None:
            try:
                worker.registry.record_last_words(
                    worker.consumer, traceback.format_exc(limit=-3).strip()[-1500:]
                )
            except redis.RedisError:
                pass  # dying because Redis is gone: nowhere to leave them
        raise
    if worker.notifier is not None and worker.notifier.webhook is not None:
        worker.notifier.webhook.flush()
    logger.info("Worker exiting after %d jobs", done)
    return 0


if __name__ == "__main__":
    sys.exit(main())
