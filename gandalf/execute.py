"""Query execution shared by the HTTP server and the queue worker.

The server's ``/query`` handler, its in-process ``/asyncquery`` background
task and the queue worker (:mod:`gandalf.worker`) all run the same steps on a
prepared TRAPI request: rehydrate or look up, run the response annotators,
turn stored JSON into fragments and serialize.  They live here, once, so the
bytes a client receives do not depend on which process produced them.

Loading the graph and the Biolink Model Toolkit is here too, for the same
reason: the API process and the worker process open the same graph the same
way.
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any, Optional

import httpx
import orjson
from bmt.toolkit import Toolkit
from translator_tom.model_dicts import QueryDict, ResponseDict

from gandalf import CSRGraph, annotate_response, enrich_knowledge_graph, lookup
from gandalf.biolink import make_toolkit
from gandalf.config import settings
from gandalf.search.gc_utils import gc_disabled
from gandalf.trapi import Deadline, finalize_response, query_parameters, to_fragments

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_graph(path: str, format: str = "auto") -> CSRGraph:
    """Load graph from disk.

    Args:
        path: Path to graph directory (mmap format)
        format: "auto" (detect from path) or "mmap"

    Returns:
        Loaded CSRGraph
    """
    resolved_path = Path(path)

    if format == "auto":
        if resolved_path.is_dir():
            format = "mmap"
        else:
            raise ValueError(
                f"Cannot auto-detect format for: {resolved_path}. Expected a directory."
            )

    if format == "mmap":
        graph: CSRGraph = CSRGraph.load_mmap(resolved_path)
        return graph
    else:
        raise ValueError(f"Unknown format: {format}")


def load_runtime() -> tuple[CSRGraph, Toolkit]:
    """Open the configured graph and build the Biolink Model Toolkit.

    Everything allocated by then (the graph and BMT) is frozen into a
    permanent GC generation that the cyclic collector never scans again, so
    Gen 2 collections stay cheap at query time, and the thresholds are raised
    so those collections are rare for the (now small) unfrozen object set.

    Returns:
        The loaded graph and toolkit.
    """
    logger.info(
        "Loading graph from %s (format=%s)...",
        settings.graph_path,
        settings.graph_format,
    )
    graph = load_graph(settings.graph_path, settings.graph_format)
    logger.info("Initializing Biolink Model Toolkit...")
    bmt = make_toolkit()

    gc.collect()
    gc.freeze()
    gc.set_threshold(50_000, 50, 50)
    return graph, bmt


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def orjson_default(obj: Any) -> Any:
    """``orjson.dumps`` fallback for the types a response may still carry."""
    if isinstance(obj, set):
        return list(obj)
    if isinstance(obj, bytes):
        # Edge attributes' stored JSON, or results written as JSON.  Correct,
        # but slow per entry: responses convert these up front with
        # to_fragments; this only catches one that was missed.
        return orjson.Fragment(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def serialize_response(response: ResponseDict) -> bytes:
    """Serialize a response dict exactly as the server's response class does."""
    data: bytes = orjson.dumps(
        response, default=orjson_default, option=orjson.OPT_SERIALIZE_NUMPY
    )
    return data


def run_query(
    graph: CSRGraph,
    bmt: Optional[Toolkit],
    raw: QueryDict,
    *,
    profile: bool = False,
    deadline: Optional[Deadline] = None,
    as_json: bool = True,
) -> ResponseDict:
    """Execute a prepared TRAPI request and return its complete Response dict.

    *raw* must already have been through the server's request preparation
    (``server._prepare_query``), which validates and normalizes the query
    graph in place; nothing here re-checks it.

    The caller holds the GC pause (``gc_disabled``) around this call and
    frees the returned dict before releasing it -- see "GC stays paused until
    the response is freed" in CLAUDE.MD.

    Args:
        graph: The loaded graph.
        bmt: The Biolink Model Toolkit, or None to let ``lookup`` build one.
        raw: The request body as a dict.
        profile: Emit per-stage timings into ``message.logs``.
        deadline: The query's time budget, if any.
        as_json: Carry edge attributes and single-path results as stored
            JSON bytes for an orjson caller (the production path).  False
            yields plain Python objects, which FastAPI's response validation
            needs.

    Returns:
        The finished Response dict, fragments converted and ready for
        :func:`serialize_response`.
    """
    params = query_parameters(raw)

    # Rehydration: skip lookup entirely, only enrich the supplied knowledge graph.
    if params.get("rehydrate") is not None:
        enrich_knowledge_graph(raw, graph)
        return finalize_response({"message": raw["message"]}, raw)

    annotator_config = params.get("annotator_config") or {}
    response = lookup(
        graph,
        raw,
        bmt=bmt,
        subclass=params.get("subclass", True),
        subclass_depth=params.get("subclass_depth", 1),
        filter_config=params.get("filter_config"),
        log_level=params.get("log_level"),
        dehydrated=params.get("dehydrated"),
        profile=profile,
        deadline=deadline,
        attributes_as_json=as_json,
        results_as_json=as_json,
    )
    if annotator_config:
        annotate_response(response, graph, annotator_config)
    to_fragments(response)
    return response


def execute_to_bytes(
    graph: CSRGraph,
    bmt: Optional[Toolkit],
    raw: QueryDict,
    *,
    profile: bool = False,
    deadline: Optional[Deadline] = None,
) -> bytes:
    """Run :func:`run_query` and serialize the Response, under one GC pause.

    The pause covers serialization and the freeing of the dict, so no
    collection ever scans the millions of objects a large response holds.

    Returns:
        The serialized TRAPI Response.
    """
    with gc_disabled():
        response = run_query(graph, bmt, raw, profile=profile, deadline=deadline)
        body = serialize_response(response)
        del response
    return body


# ---------------------------------------------------------------------------
# Delivery of an async result
# ---------------------------------------------------------------------------


def post_callback(
    callback_url: str,
    body: bytes,
    trace_headers: Optional[dict[str, str]] = None,
    timeout_seconds: float = 600.0,
) -> bool:
    """POST a serialized Response to an ``/asyncquery`` client's callback.

    *trace_headers* carries the W3C trace context (``traceparent`` /
    ``tracestate``) captured at the original request so the callback stays
    linked to the originating trace; the httpx client is not
    auto-instrumented and neither a background thread nor a worker process
    inherits the request's context.

    Returns:
        Whether the callback accepted the result.  A failure is logged, not
        raised: there is nobody left to raise it to.
    """
    headers = dict(trace_headers or {})
    headers["Content-Type"] = "application/json"
    try:
        with httpx.Client(timeout=httpx.Timeout(timeout=timeout_seconds)) as client:
            res = client.post(callback_url, content=body, headers=headers)
            res.raise_for_status()
    except Exception:
        logger.exception("Callback to %s failed", callback_url)
        return False
    logger.info("Posted to %s with code %s", callback_url, res.status_code)
    return True
