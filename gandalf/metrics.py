"""Prometheus metrics for the API and worker processes.

The API runs under gunicorn with several worker processes, so its metrics
are aggregated with ``prometheus_client``'s multiprocess mode when
``PROMETHEUS_MULTIPROC_DIR`` is set (``gunicorn.conf.py`` prepares the
directory and retires dead workers' files).  Without it, as in the dev
server and tests, the default single-process registry is served.  The
queue worker is one process and serves the default registry on its own
port (``GANDALF_WORKER_METRICS_PORT``).

Metric names are stable: dashboards and the KEDA ``ScaledObject`` in
``deploy/`` refer to them.
"""

from __future__ import annotations

import os

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    multiprocess,
)


def multiprocess_dir() -> str:
    """The ``PROMETHEUS_MULTIPROC_DIR`` in effect, or ``""`` when unset."""
    return os.environ.get("PROMETHEUS_MULTIPROC_DIR", "")


# In multiprocess mode the client opens a file under the directory as soon as
# a metric is defined -- the definitions below, at import -- so the directory
# has to exist first.  The Docker image sets the variable; this creates the
# directory.  A container starts with an empty /tmp, so no stale files from
# an earlier run need clearing.
if multiprocess_dir():
    os.makedirs(multiprocess_dir(), exist_ok=True)


#: Duration buckets spanning a sub-second one-hop to a multi-minute query.
_DURATION_BUCKETS = (
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
    60.0,
    120.0,
    300.0,
    600.0,
    1800.0,
)

#: Byte buckets from a few KB to a few GB.
_BYTES_BUCKETS = tuple(float(2**n) for n in range(10, 33, 2))

# --- API process -----------------------------------------------------------

HTTP_REQUESTS = Counter(
    "gandalf_http_requests_total",
    "HTTP requests handled, by route template and status code.",
    ["method", "route", "status"],
)
HTTP_DURATION = Histogram(
    "gandalf_http_request_duration_seconds",
    "Wall time to answer an HTTP request, including any wait on a worker.",
    ["method", "route"],
    buckets=_DURATION_BUCKETS,
)
JOBS_ENQUEUED = Counter(
    "gandalf_jobs_enqueued_total",
    "Jobs the API put on the queue, by delivery mode (sync: a waiting "
    "/query; callback: /asyncquery).",
    ["mode"],
)
SYNC_WAIT = Histogram(
    "gandalf_sync_wait_seconds",
    "How long a /query waited for its job's result, queue time included.",
    buckets=_DURATION_BUCKETS,
)
SYNC_WAIT_TIMEOUTS = Counter(
    "gandalf_sync_wait_timeouts_total",
    "/query requests that gave up waiting for a worker (HTTP 504).",
)
QUEUE_LAG = Gauge(
    "gandalf_queue_lag",
    "Jobs on the stream not yet delivered to any worker (read at scrape).",
    multiprocess_mode="max",
)
QUEUE_PENDING = Gauge(
    "gandalf_queue_pending",
    "Jobs delivered to a worker and not yet acknowledged (read at scrape).",
    multiprocess_mode="max",
)

# --- Worker process --------------------------------------------------------

JOBS = Counter(
    "gandalf_jobs_total",
    "Jobs finished, by outcome: ok, timeout, expired (budget spent before "
    "the worker started it), error, poisoned (dead-lettered).",
    ["outcome"],
)
JOB_DURATION = Histogram(
    "gandalf_job_duration_seconds",
    "Wall time a worker spent executing one job.",
    buckets=_DURATION_BUCKETS,
)
JOB_QUEUE_WAIT = Histogram(
    "gandalf_job_queue_wait_seconds",
    "Time from enqueue to a worker starting the job.",
    buckets=_DURATION_BUCKETS,
)
JOB_RESULT_BYTES = Histogram(
    "gandalf_job_result_bytes",
    "Serialized (uncompressed) size of a job's response.",
    buckets=_BYTES_BUCKETS,
)
JOBS_INFLIGHT = Gauge(
    "gandalf_jobs_inflight",
    "Jobs this process is executing right now.",
    multiprocess_mode="livesum",
)
PROCESS_RSS_ANON_BYTES = Gauge(
    "gandalf_process_rss_anon_bytes",
    "Anonymous resident memory of the process after its last job: the part "
    "the kernel cannot reclaim, so the OOM-kill risk.",
    multiprocess_mode="max",
)


def metrics_registry() -> CollectorRegistry:
    """The registry to serve: every process's metrics in multiprocess mode.

    In multiprocess mode the values live in the files under
    ``PROMETHEUS_MULTIPROC_DIR`` and a ``MultiProcessCollector`` reads them
    all at scrape time; otherwise the default in-process registry is it.
    """
    if multiprocess_dir():
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        return registry
    return REGISTRY


def metrics_payload() -> tuple[bytes, str]:
    """Render every metric in the Prometheus text format.

    Returns:
        The body and its ``Content-Type``.
    """
    return generate_latest(metrics_registry()), CONTENT_TYPE_LATEST


# --- Process memory (gandalf.memory, re-exported) --------------------------

from gandalf.memory import (  # noqa: E402,F401
    rss_anon_kb,
    rss_kb,
    trim_heap,
    trim_heap_if_large,
)
