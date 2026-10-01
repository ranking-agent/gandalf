"""Gunicorn configuration for GANDALF."""

import os
import shutil

from gandalf.config import settings
from gandalf.metrics import multiprocess_dir
from gandalf.otel import init_otel
from gandalf.server import APP

# Bind to all interfaces on port 6429
bind = "0.0.0.0:6429"

# Worker count — keep low due to large graph memory footprint.
# COW sharing is most effective with fewer workers.
workers = settings.workers

# Use Uvicorn's ASGI worker for FastAPI compatibility
worker_class = "uvicorn.workers.UvicornWorker"

# Preload the application so graph loading happens once in the master
# process. Workers inherit the loaded graph via fork COW.
preload_app = True

# Timeout for worker responsiveness (seconds)
timeout = 300

# Graceful shutdown timeout
graceful_timeout = 30

# Recycle workers after this many requests (plus jitter) to release memory
# that Python/glibc allocators hold onto after large queries.
max_requests = 500
max_requests_jitter = 100

# Access log to stdout (structured logging handled by app middleware)
accesslog = "-"

# Forward all logs to stderr
errorlog = "-"
loglevel = settings.log_level.lower()


# Prometheus metrics are aggregated across the workers through files in
# PROMETHEUS_MULTIPROC_DIR (gandalf.metrics).  Start from an empty directory
# so a previous run's counters do not survive a restart.
def on_starting(server):
    directory = multiprocess_dir()
    if directory:
        shutil.rmtree(directory, ignore_errors=True)
        os.makedirs(directory, exist_ok=True)


# The OpenTelemetry SDK (OTLP/gRPC exporter channel + batch export thread)
# must be (re)built post-fork per worker because gRPC channels are not fork-safe
# so an exporter inherited from one master process won't work with multiple workers.
def post_fork(server, worker):
    init_otel(APP)


# child_exit runs in the master after waitpid() reaps the worker, so it fires
# even on SIGKILL (kernel OOM) — unlike worker_exit, which runs in the worker
# itself and is skipped when SIGKILL bypasses the worker's finally block.
def child_exit(server, worker):
    server.log.warning("child_exit pid=%s age=%s", worker.pid, worker.age)
    if multiprocess_dir():
        # Retire the dead worker's gauge files so its last values stop
        # being reported; counters and histograms are kept and summed.
        from prometheus_client import multiprocess

        multiprocess.mark_process_dead(worker.pid)


def worker_abort(worker):
    # Fires inside the worker on SIGABRT (gunicorn's pre-SIGKILL on timeout).
    # Does NOT fire on kernel OOM SIGKILL — nothing runs in the worker then.
    worker.log.warning(
        "worker_abort pid=%s — about to be killed by gunicorn", worker.pid
    )
