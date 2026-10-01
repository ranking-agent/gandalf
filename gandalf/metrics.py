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

import psutil
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


def multiprocess_dir() -> str:
    """The ``PROMETHEUS_MULTIPROC_DIR`` in effect, or ``""`` when unset."""
    return os.environ.get("PROMETHEUS_MULTIPROC_DIR", "")


def metrics_payload() -> tuple[bytes, str]:
    """Render every metric in the Prometheus text format.

    Returns:
        The body and its ``Content-Type``.
    """
    if multiprocess_dir():
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
    else:
        registry = REGISTRY
    return generate_latest(registry), CONTENT_TYPE_LATEST


# --- Process memory --------------------------------------------------------


def rss_kb() -> int:
    """Current resident set size in KB. Cross-platform via psutil.

    Constructs psutil.Process() per call rather than caching, so that with
    gunicorn preload_app=True each worker reads its own RSS instead of the
    master's PID it inherited at fork time.
    """
    return int(psutil.Process().memory_info().rss) // 1024


def rss_anon_kb() -> int:
    """Anonymous (private, non-file-backed) RSS in KB.

    Reads RssAnon from /proc/self/status on Linux -- this is the precise
    metric for OOM risk, since file-backed pages (LMDB, .so files) are
    reclaimable but anon pages are not. Falls back to psutil's USS on
    non-Linux; USS additionally includes private file mappings, but it's
    the closest cross-platform approximation.
    """
    try:
        with open("/proc/self/status", "rb") as f:
            for line in f:
                if line.startswith(b"RssAnon:"):
                    return int(line.split()[1])
    except OSError:
        pass
    try:
        return int(psutil.Process().memory_full_info().uss) // 1024
    except Exception:
        return -1
