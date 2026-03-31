"""Logging configuration for the gandalf package."""

import json
import logging
import sys
from contextvars import ContextVar
from datetime import datetime, timezone

# Context variable for per-request ID propagation.
request_id_var: ContextVar[str] = ContextVar("request_id", default="")


class _JSONFormatter(logging.Formatter):
    """Emit each log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        req_id = request_id_var.get("")
        if req_id:
            entry["request_id"] = req_id
        if record.exc_info and record.exc_info[0] is not None:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry)


_TRAPI_LEVELS = {"ERROR", "WARNING", "INFO", "DEBUG"}


class TRAPILogCollector(logging.Handler):
    """A logging handler that collects log entries as TRAPI-spec LogEntry dicts.

    Attach to the ``gandalf`` logger for the duration of a query, then call
    :meth:`get_logs` to retrieve the accumulated entries.
    """

    def __init__(self, level: int = logging.DEBUG):
        super().__init__(level)
        self._entries: list[dict] = []

    def emit(self, record: logging.LogRecord) -> None:
        level_name = record.levelname
        self._entries.append(
            {
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "level": level_name if level_name in _TRAPI_LEVELS else None,
                "message": record.getMessage(),
            }
        )

    def get_logs(self) -> list[dict]:
        """Return collected log entries in chronological order."""
        return list(self._entries)


def configure_logging(level=logging.INFO, fmt: str = "text"):
    """Configure logging for the gandalf package.

    Sets up a StreamHandler on stderr with a standard format.
    All gandalf.* loggers inherit this configuration.

    Args:
        level: Logging level (default: logging.INFO).
        fmt: Log format — ``"text"`` (human-readable) or ``"json"``
            (structured, one JSON object per line).
    """
    handler = logging.StreamHandler(sys.stderr)

    formatter: logging.Formatter
    if fmt == "json":
        formatter = _JSONFormatter(datefmt="%Y-%m-%dT%H:%M:%S")
    else:
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    handler.setFormatter(formatter)

    root_logger = logging.getLogger("gandalf")
    root_logger.setLevel(level)
    root_logger.addHandler(handler)
