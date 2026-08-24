"""Logging configuration for the gandalf package."""

import json
import logging
import sys
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Optional

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


def make_query_logger(
    query_id: str, parent: Optional[logging.Logger] = None
) -> logging.Logger:
    """Create a logger private to a single query.

    A :class:`TRAPILogCollector` attached to the shared ``gandalf`` logger would
    also receive records from every other query running concurrently in the same
    worker process, because ``logging.getLogger`` returns a process-global
    singleton and handlers fire for every record that propagates through it.
    Attaching the collector to a logger owned by one query instead keeps each
    query's TRAPI ``logs`` to its own records.

    The logger is constructed directly rather than via ``logging.getLogger`` so
    that it is *not* interned in the global logger registry -- that registry
    never evicts, so a name-per-query would leak an entry for the lifetime of
    the process. This logger is garbage-collected along with the request.

    Records still propagate to *parent* (the shared ``gandalf`` logger by
    default), so stderr output is unaffected.

    Args:
        query_id: Short identifier distinguishing this query's logger.
        parent: Logger to propagate records to. Defaults to ``gandalf``.

    Returns:
        A logger whose level and handlers are private to a single query.

    Example:
        >>> log = make_query_logger("abc123")
        >>> log.name
        'gandalf.query.abc123'
        >>> log.parent.name
        'gandalf'
        >>> "gandalf.query.abc123" in logging.root.manager.loggerDict
        False
    """
    log = logging.Logger(f"gandalf.query.{query_id}")
    log.parent = parent if parent is not None else logging.getLogger("gandalf")
    return log


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
