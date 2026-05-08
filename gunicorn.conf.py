"""Gunicorn configuration for GANDALF."""

from gandalf.config import settings

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

# Access log to stdout (structured logging handled by app middleware)
accesslog = "-"

# Forward all logs to stderr
errorlog = "-"
loglevel = settings.log_level.lower()


# Hooks run in the master process, so they survive worker SIGKILL (OOM, timeout)
# and let us correlate a dead PID with the in-flight request_id seen in the
# worker's "request start" log line.
def worker_exit(server, worker):
    server.log.warning("worker_exit pid=%s age=%s", worker.pid, worker.age)


def worker_abort(worker):
    # Fires inside the worker on SIGABRT (gunicorn's pre-SIGKILL on timeout).
    # Does NOT fire on kernel OOM SIGKILL — nothing runs in the worker then.
    worker.log.warning("worker_abort pid=%s — about to be killed by gunicorn", worker.pid)
