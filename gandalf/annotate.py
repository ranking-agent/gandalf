"""Run registered response annotators against a TRAPI response.

Annotators are registered by plugin modules (see
``gandalf/plugins/``) into the registry in
``gandalf.search.response_annotators``. This module exposes the runner
that the server invokes after ``lookup()`` returns:

    response = lookup(graph, query, ...)
    annotate_response(response, graph, annotator_config)
    return response

The runner walks the registry in registration order, asks each factory for
an active annotator (factories return ``None`` when their config key is
missing), and calls each active annotator with ``(response, graph)``.

Each annotator runs inside a ``try``/``except``: a failing annotator logs
an error TRAPI ``LogEntry`` into ``response["logs"]`` and is skipped.
Subsequent annotators still run; the response is returned to the caller
either way.
"""

from __future__ import annotations

import logging
import time
import traceback
from datetime import datetime, timezone
from typing import Any

from gandalf.search.response_annotators import build_response_annotators

logger = logging.getLogger(__name__)


def _log_entry(level: str, message: str, code: str | None = None) -> dict:
    entry = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "level": level,
        "code": code,
        "message": message,
    }
    return entry


def annotate_response(response: dict, graph: Any, annotator_config: dict) -> dict:
    """Run all active annotators against *response* in place.

    Args:
        response: A TRAPI response dict (the value returned by
            ``lookup()``). Mutated in place.
        graph: The :class:`CSRGraph` the response was built from.
        annotator_config: Per-request opt-in dict. Each registered
            annotator pulls its own key out of this dict; an annotator
            whose key is missing or ``None`` does not run.

    Returns:
        The same *response* dict, for caller convenience.
    """
    if not annotator_config:
        return response

    active = build_response_annotators(annotator_config)
    if not active:
        return response

    logs = response.setdefault("logs", [])

    for name, annotator in active:
        t0 = time.perf_counter()
        try:
            annotator(response, graph)
        except Exception:
            tb = traceback.format_exc()
            logger.exception("Annotator %r failed", name)
            logs.append(
                _log_entry(
                    "ERROR",
                    f"Annotator {name!r} failed: {tb}",
                    code="AnnotatorError",
                )
            )
            continue
        dt_ms = (time.perf_counter() - t0) * 1000
        logs.append(
            _log_entry(
                "DEBUG",
                f"Annotator {name!r} ran in {dt_ms:.1f}ms",
            )
        )

    return response
