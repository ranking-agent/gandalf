"""Execution of async TRAPI jobs, decoupled from the HTTP layer.

The logic here used to live inside ``server._async_lookup`` as a FastAPI
``BackgroundTask``. It is factored out so it can be driven by two callers with
no FastAPI dependency between them:

* the in-process fallback path in ``/asyncquery`` (queue disabled), and
* the standalone Redis-Stream consumer (``gandalf.worker``).

A "job" is a plain dict -- the unit that is enqueued and later consumed:

    {
        "callback": "https://client/callback",   # where to POST the result
        "query": { ...raw TRAPI request dict... },
        "trace_headers": { "traceparent": ... },  # W3C trace context (optional)
        "profile": false,                          # emit per-stage timings
    }

Keeping the payload a dict (not a Pydantic model) mirrors the server's default
non-validating path and lets the same body ride through orjson untouched.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import httpx
import orjson

from gandalf import annotate_response, enrich_knowledge_graph, lookup

logger = logging.getLogger(__name__)


def _orjson_default(obj: Any) -> Any:
    if isinstance(obj, set):
        return list(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def run_lookup(graph: Any, query: dict, bmt: Any = None, profile: bool = False) -> dict:
    """Run a lookup (or rehydration) and return the TRAPI response dict.

    This is the pure compute half of a job -- no network egress. Mirrors the
    branching in ``server._async_lookup`` so behaviour is identical whether the
    job runs in-process or on a worker.
    """
    params = query.get("parameters", {})

    # Rehydration: skip lookup entirely, only enrich the supplied graph.
    if params.get("rehydrate") is not None:
        enrich_knowledge_graph(query, graph)
        return {"message": query["message"]}

    subclass = params.get("subclass", True)
    subclass_depth = params.get("subclass_depth", 1)
    log_level = query.pop("log_level", None)
    dehydrated = params.get("dehydrated")
    filter_config = params.get("filter_config")
    annotator_config = params.get("annotator_config") or {}
    response: dict = lookup(
        graph,
        query,
        bmt=bmt,
        subclass=subclass,
        subclass_depth=subclass_depth,
        filter_config=filter_config,
        log_level=log_level,
        dehydrated=dehydrated,
        profile=profile,
    )
    if annotator_config:
        annotate_response(response, graph, annotator_config)
    return response


def post_callback(
    callback_url: str, response: dict, trace_headers: Optional[dict] = None
) -> None:
    """POST a TRAPI response to the client's callback URL.

    Serialises with orjson rather than httpx's stdlib-json ``json=`` path,
    which is markedly slower for large result sets. Raises on transport error
    or a non-2xx callback response so the caller can decide whether to ack.
    """
    body = orjson.dumps(
        response, default=_orjson_default, option=orjson.OPT_SERIALIZE_NUMPY
    )
    headers = dict(trace_headers or {})
    headers["Content-Type"] = "application/json"
    with httpx.Client(timeout=httpx.Timeout(timeout=600.0)) as client:
        res = client.post(callback_url, content=body, headers=headers)
        res.raise_for_status()
        logger.info("Posted to %s with code %s", callback_url, res.status_code)


def execute_job(job: dict, graph: Any, bmt: Any = None) -> None:
    """Run one async job end-to-end: lookup then POST to the callback.

    Exceptions are caught and logged (never re-raised) so a single bad job --
    a failing lookup or an unreachable callback -- does not crash the worker
    loop or the background task. This preserves the original
    ``_async_lookup`` contract: best-effort delivery, failures land in the log.
    """
    callback = job.get("callback")
    if not isinstance(callback, str):
        logger.error("Async job missing a callback URL; dropping: %r", callback)
        return
    query = job.get("query", {})
    trace_headers = job.get("trace_headers") or {}
    profile = bool(job.get("profile", False))
    try:
        response = run_lookup(graph, query, bmt=bmt, profile=profile)
        post_callback(callback, response, trace_headers)
    except Exception:
        logger.exception("Async job failed (callback=%s)", callback)
