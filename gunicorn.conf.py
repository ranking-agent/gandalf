"""Gunicorn configuration for GANDALF."""
import os

# Bind to all interfaces on port 6429
bind = "0.0.0.0:6429"

# Worker count — keep low due to large graph memory footprint.
# COW sharing is most effective with fewer workers.
workers = int(os.getenv("GANDALF_WORKERS", "2"))

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
loglevel = os.getenv("GANDALF_GUNICORN_LOG_LEVEL", "info")
