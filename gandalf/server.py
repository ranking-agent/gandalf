"""GANDALF — Plater-compatible TRAPI server."""

import gc
import logging
import os
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import httpx
import orjson
from bmt.toolkit import Toolkit
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import (
    get_swagger_ui_html,
)
from fastapi.staticfiles import StaticFiles
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse

from gandalf import CSRGraph, lookup
from gandalf.logging_config import configure_logging, request_id_var
from gandalf.models import (
    AsyncTRAPIQuery,
    EdgesResponse,
    EdgeSummaryResponse,
    MetadataResponse,
    NodeResponse,
    TRAPIQuery,
    TRAPIResponse,
    WorkflowStep,
)

# ---------------------------------------------------------------------------
# Configuration via environment variables
# ---------------------------------------------------------------------------

GRAPH_PATH = os.environ.get("GANDALF_GRAPH_PATH", "../02_26_2026/gandalf_mmap")
GRAPH_FORMAT = os.environ.get("GANDALF_GRAPH_FORMAT", "auto")  # "auto" or "mmap"
LOG_LEVEL = os.environ.get("GANDALF_LOG_LEVEL", "INFO").upper()
LOG_FORMAT = os.environ.get("GANDALF_LOG_FORMAT", "text")  # "text" or "json"
CORS_ORIGINS = os.environ.get("GANDALF_CORS_ORIGINS", "*")
MAX_REQUEST_SIZE_MB = int(os.environ.get("GANDALF_MAX_REQUEST_SIZE_MB", "10"))
RATE_LIMIT_PER_MINUTE = int(os.environ.get("GANDALF_RATE_LIMIT", "0"))

configure_logging(getattr(logging, LOG_LEVEL, logging.INFO), fmt=LOG_FORMAT)
logger = logging.getLogger(__name__)

GRAPH = None
BMT = None


# ---------------------------------------------------------------------------
# orjson-based response class (3-10x faster than stdlib json)
# ---------------------------------------------------------------------------


def _orjson_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class CustomORJSONResponse(JSONResponse):
    media_type = "application/json"

    def render(self, content: Any) -> bytes:
        t0 = time.perf_counter()
        data: bytes = orjson.dumps(content, default=_orjson_default)
        dt_ms = (time.perf_counter() - t0) * 1000
        size_kb = len(data) / 1024
        logger.debug("orjson serialization: %.2f ms, %.1f KB", dt_ms, size_kb)
        return data


# ---------------------------------------------------------------------------
# Rate limiting middleware (token bucket per client IP)
# ---------------------------------------------------------------------------


class _TokenBucket:
    """Simple in-process token bucket rate limiter."""

    __slots__ = ("_buckets", "_rate", "_capacity")

    def __init__(self, rate_per_minute: int):
        self._buckets: dict[str, tuple[float, float]] = {}
        self._rate = rate_per_minute / 60.0
        self._capacity = float(rate_per_minute)

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        tokens, last = self._buckets.get(key, (self._capacity, now))
        tokens = min(self._capacity, tokens + (now - last) * self._rate)
        if tokens >= 1.0:
            self._buckets[key] = (tokens - 1.0, now)
            return True
        self._buckets[key] = (tokens, now)
        return False


_rate_limiter = (
    _TokenBucket(RATE_LIMIT_PER_MINUTE) if RATE_LIMIT_PER_MINUTE > 0 else None
)


# ---------------------------------------------------------------------------
# Graph loading
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


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle graph and BMT loading on startup."""
    global GRAPH, BMT
    logger.info("Loading graph from %s (format=%s)...", GRAPH_PATH, GRAPH_FORMAT)
    GRAPH = load_graph(GRAPH_PATH, GRAPH_FORMAT)
    logger.info("Initializing Biolink Model Toolkit...")
    BMT = Toolkit()

    # Freeze all objects allocated so far (graph + BMT) into a permanent
    # generation that the cyclic GC will never scan.  This makes Gen 2
    # collections cheap because they skip the large CSR arrays.
    gc.collect()
    gc.freeze()
    # Raise thresholds so Gen 2 collections are less frequent even for
    # the (now-small) unfrozen query-time object set.
    gc.set_threshold(50_000, 50, 50)
    logger.info("Server ready!")
    yield
    logger.info("Shutting down — releasing resources...")
    if (
        GRAPH is not None
        and hasattr(GRAPH, "lmdb_store")
        and GRAPH.lmdb_store is not None
    ):
        GRAPH.lmdb_store.close()
    GRAPH = None
    BMT = None
    logger.info("Shutdown complete.")


APP = FastAPI(
    title="GANDALF",
    lifespan=lifespan,
    docs_url=None,
    default_response_class=CustomORJSONResponse,
)

# Parse CORS origins from env var (comma-separated)
_cors_origins = [o.strip() for o in CORS_ORIGINS.split(",")]
APP.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials="*" not in _cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent.parent / "static"
if STATIC_DIR.exists():
    APP.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Middleware: request ID, access logging, request size limit, rate limiting
# ---------------------------------------------------------------------------


@APP.middleware("http")
async def request_middleware(request: Request, call_next):
    """Add request ID, enforce size limits, rate limiting, and access logging."""
    # Request ID — use incoming header or generate one
    req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())[:8]
    request_id_var.set(req_id)

    # Request size limit (skip for GET/HEAD/OPTIONS)
    if request.method in ("POST", "PUT", "PATCH"):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_REQUEST_SIZE_MB * 1024 * 1024:
            return JSONResponse(
                status_code=413,
                content={
                    "detail": f"Request body too large (max {MAX_REQUEST_SIZE_MB}MB)"
                },
            )

    # Rate limiting
    if _rate_limiter is not None:
        client_ip = request.client.host if request.client else "unknown"
        if not _rate_limiter.allow(client_ip):
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests. Try again later."},
                headers={"Retry-After": "60"},
            )

    t_start = time.monotonic()
    response: Response = await call_next(request)
    duration_ms = (time.monotonic() - t_start) * 1000

    response.headers["X-Request-ID"] = req_id
    logger.info(
        "%s %s %s %.1fms",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


# ---------------------------------------------------------------------------
# Global exception handler
# ---------------------------------------------------------------------------


@APP.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch unhandled exceptions and return 500 with request ID."""
    req_id = request_id_var.get("")
    logger.exception("Unhandled exception [request_id=%s]", req_id)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "request_id": req_id,
        },
    )


# ---------------------------------------------------------------------------
# Root redirect
# ---------------------------------------------------------------------------


@APP.get("/", include_in_schema=False)
async def root():
    """Redirect to API documentation."""
    return RedirectResponse(url="/docs")


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@APP.get("/health")
def health():
    """Health check endpoint for load balancers and monitoring."""
    graph_loaded = GRAPH is not None
    info: dict[str, Any] = {"status": "ok" if graph_loaded else "degraded"}
    info["graph_loaded"] = graph_loaded
    if graph_loaded:
        info["node_count"] = GRAPH.num_nodes
        info["edge_count"] = GRAPH.num_edges
    return info


# ---------------------------------------------------------------------------
# Documentation
# ---------------------------------------------------------------------------


@APP.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html(req: Request) -> HTMLResponse:
    """Customize Swagger UI."""
    root_path = req.scope.get("root_path", "").rstrip("/")
    openapi_url = root_path + APP.openapi_url
    swagger_favicon_url = root_path + "/static/favicon.png"
    return get_swagger_ui_html(
        openapi_url=openapi_url,
        title=APP.title + " - Swagger UI",
        swagger_favicon_url=swagger_favicon_url,
    )


# ---------------------------------------------------------------------------
# Meta Knowledge Graph (Plater-compatible)
# ---------------------------------------------------------------------------


@APP.get("/meta_knowledge_graph")
def meta_knowledge_graph():
    """Return the meta knowledge graph.

    Returns the union of the most specific categories and predicates
    present in the knowledge graph, with edge counts.
    """
    if GRAPH is None:
        raise HTTPException(503, "Graph not loaded")
    return GRAPH.meta_kg


# ---------------------------------------------------------------------------
# SRI Testing Data (Plater-compatible)
# ---------------------------------------------------------------------------


@APP.get("/sri_testing_data")
def sri_testing_data():
    """Return representative example edges for the SRI Testing Harness."""
    if GRAPH is None:
        raise HTTPException(503, "Graph not loaded")
    return GRAPH.sri_testing_data


# ---------------------------------------------------------------------------
# Metadata (Plater-compatible)
# ---------------------------------------------------------------------------


@APP.get("/metadata", responses={200: {"model": MetadataResponse}})
def metadata():
    """Return knowledge graph metadata and statistics."""
    if GRAPH is None:
        raise HTTPException(503, "Graph not loaded")
    return GRAPH.graph_metadata


# ---------------------------------------------------------------------------
# Node lookup (Plater-compatible)
# ---------------------------------------------------------------------------


@APP.get("/node/{curie:path}", responses={200: {"model": NodeResponse}})
def get_node(curie: str):
    """Retrieve node information by CURIE identifier."""
    if GRAPH is None:
        raise HTTPException(503, "Graph not loaded")

    node_idx = GRAPH.get_node_idx(curie)
    if node_idx is None:
        raise HTTPException(404, f"Node not found: {curie}")

    props = GRAPH.get_all_node_properties(node_idx)
    return {"id": curie, **props}


# ---------------------------------------------------------------------------
# Edge lookup (Plater-compatible)
# ---------------------------------------------------------------------------


@APP.get("/edges/{curie:path}", responses={200: {"model": EdgesResponse}})
def get_edges(
    curie: str,
    category: Optional[str] = Query(None, description="Filter by target node category"),
    predicate: Optional[str] = Query(None, description="Filter by edge predicate"),
    limit: Optional[int] = Query(None, ge=0, description="Maximum number of results"),
    offset: int = Query(0, ge=0, description="Result offset for pagination"),
    count_only: bool = Query(
        False, description="Return only the count of matching edges"
    ),
):
    """Retrieve edges connected to a node with optional filtering."""
    if GRAPH is None:
        raise HTTPException(503, "Graph not loaded")

    node_idx = GRAPH.get_node_idx(curie)
    if node_idx is None:
        raise HTTPException(404, f"Node not found: {curie}")

    # Build predicate filter set
    pred_filter = None
    if predicate is not None:
        if predicate not in GRAPH.predicate_to_idx:
            return CustomORJSONResponse({"query_curie": curie, "edges": []})
        pred_filter = [predicate]

    # Collect outgoing edges
    edges = []
    for tgt_idx, pred_str, props, fwd_idx in GRAPH.neighbors_with_properties(
        node_idx, pred_filter
    ):
        tgt_id = GRAPH.get_node_id(tgt_idx)
        tgt_props = GRAPH.get_all_node_properties(tgt_idx)
        tgt_cats = tgt_props.get("categories", ["biolink:NamedThing"])

        if category is not None and category not in tgt_cats:
            continue

        edge_id = GRAPH.get_edge_id(fwd_idx)
        edges.append(
            {
                "subject": curie,
                "object": tgt_id,
                "predicate": pred_str,
                "edge_id": edge_id,
                "sources": props.get("sources", []),
                "qualifiers": props.get("qualifiers", []),
            }
        )

    # Collect incoming edges
    for src_idx, pred_str, props, fwd_idx in GRAPH.incoming_neighbors_with_properties(
        node_idx, pred_filter
    ):
        src_id = GRAPH.get_node_id(src_idx)
        src_props = GRAPH.get_all_node_properties(src_idx)
        src_cats = src_props.get("categories", ["biolink:NamedThing"])

        if category is not None and category not in src_cats:
            continue

        edge_id = GRAPH.get_edge_id(fwd_idx)
        edges.append(
            {
                "subject": src_id,
                "object": curie,
                "predicate": pred_str,
                "edge_id": edge_id,
                "sources": props.get("sources", []),
                "qualifiers": props.get("qualifiers", []),
            }
        )

    if count_only:
        return {"query_curie": curie, "count": len(edges)}

    # Apply pagination
    if offset > 0:
        edges = edges[offset:]
    if limit is not None:
        edges = edges[:limit]

    return {"query_curie": curie, "edges": edges}


# ---------------------------------------------------------------------------
# Edge summary (Plater-compatible)
# ---------------------------------------------------------------------------


@APP.get("/edge_summary/{curie:path}", responses={200: {"model": EdgeSummaryResponse}})
def edge_summary(curie: str):
    """Summarize edge types connected to a node.

    Returns a list of [predicate, category, count] triples.
    """
    if GRAPH is None:
        raise HTTPException(503, "Graph not loaded")

    node_idx = GRAPH.get_node_idx(curie)
    if node_idx is None:
        raise HTTPException(404, f"Node not found: {curie}")

    summary: defaultdict[tuple[str, str], int] = defaultdict(int)

    # Outgoing edges: group by (predicate, target_category)
    fwd_start = int(GRAPH.fwd_offsets[node_idx])
    fwd_end = int(GRAPH.fwd_offsets[node_idx + 1])
    for pos in range(fwd_start, fwd_end):
        tgt_idx = int(GRAPH.fwd_targets[pos])
        pred_id = int(GRAPH.fwd_predicates[pos])
        pred = GRAPH.id_to_predicate[pred_id]
        tgt_props = GRAPH.get_all_node_properties(tgt_idx)
        tgt_cat = (tgt_props.get("categories") or ["biolink:NamedThing"])[0]
        summary[(pred, tgt_cat)] += 1

    # Incoming edges: group by (predicate, source_category)
    rev_start = int(GRAPH.rev_offsets[node_idx])
    rev_end = int(GRAPH.rev_offsets[node_idx + 1])
    for pos in range(rev_start, rev_end):
        src_idx = int(GRAPH.rev_sources[pos])
        pred_id = int(GRAPH.rev_predicates[pos])
        pred = GRAPH.id_to_predicate[pred_id]
        src_props = GRAPH.get_all_node_properties(src_idx)
        src_cat = (src_props.get("categories") or ["biolink:NamedThing"])[0]
        summary[(pred, src_cat)] += 1

    return {
        "query_curie": curie,
        "edge_summary": [
            [pred, cat, count] for (pred, cat), count in sorted(summary.items())
        ],
    }


# ---------------------------------------------------------------------------
# Simple spec (Plater-compatible)
# ---------------------------------------------------------------------------


@APP.get("/simple_spec")
def simple_spec(
    source: Optional[str] = Query(None, description="Filter by source category"),
    target: Optional[str] = Query(None, description="Filter by target category"),
):
    """Return the one-hop connection schema.

    Without parameters, returns the full schema. With source/target,
    returns only matching connections.
    """
    if GRAPH is None:
        raise HTTPException(503, "Graph not loaded")

    spec = GRAPH.simple_spec

    if source is None and target is None:
        return spec

    # Filter by source and/or target
    filtered: dict[str, dict] = {}
    for src_cat, targets in spec.items():
        if source is not None and src_cat != source:
            continue
        for tgt_cat, preds in targets.items():
            if target is not None and tgt_cat != target:
                continue
            if src_cat not in filtered:
                filtered[src_cat] = {}
            filtered[src_cat][tgt_cat] = preds

    return filtered


# ---------------------------------------------------------------------------
# TRAPI query (Plater-compatible with query params)
# ---------------------------------------------------------------------------


@APP.post("/query", responses={200: {"model": TRAPIResponse}})
def sync_lookup(
    request: TRAPIQuery,
    subclass: Optional[bool] = Query(
        None, description="Enable biolink subclass inference"
    ),
):
    """Execute a TRAPI query against the knowledge graph.

    Supports the 'lookup' workflow operation.
    """
    if GRAPH is None:
        raise HTTPException(503, "Graph not loaded")

    raw = request.model_dump(exclude_none=True)
    log_level = raw.pop("log_level", None)

    # Query params take precedence, fall back to request body
    sc = subclass if subclass is not None else raw.get("subclass", True)
    subclass_depth = raw.get("subclass_depth", 1)

    return lookup(
        GRAPH,
        raw,
        bmt=BMT,
        subclass=sc,
        subclass_depth=subclass_depth,
        log_level=log_level,
    )


# ---------------------------------------------------------------------------
# Async query
# ---------------------------------------------------------------------------


def _async_lookup(callback_url: str, query: dict):
    """Execute lookup and POST results to callback URL."""
    subclass = query.get("subclass", True)
    subclass_depth = query.get("subclass_depth", 1)
    log_level = query.pop("log_level", None)
    response = lookup(
        GRAPH,
        query,
        bmt=BMT,
        subclass=subclass,
        subclass_depth=subclass_depth,
        log_level=log_level,
    )

    try:
        with httpx.Client(timeout=httpx.Timeout(timeout=600.0)) as client:
            res = client.post(callback_url, json=response)
            res.raise_for_status()
            logger.info("Posted to %s with code %s", callback_url, res.status_code)
    except Exception:
        logger.exception("Callback to %s failed", callback_url)


@APP.post("/asyncquery")
def async_query(
    background_tasks: BackgroundTasks,
    query: AsyncTRAPIQuery,
):
    """Handle asynchronous query."""
    raw = query.model_dump(exclude_none=True)
    callback = query.callback

    # Validate callback URL scheme
    if not callback.startswith(("http://", "https://")):
        raise HTTPException(400, "callback must be an http:// or https:// URL")

    # parse requested workflow
    workflow = query.workflow or [WorkflowStep(id="lookup", parameters=None)]
    workflow_dicts = [w.model_dump(exclude_none=True) for w in workflow]

    if len(workflow_dicts) != 1:
        raise HTTPException(400, "workflow must contain exactly 1 operation")
    if workflow_dicts[0]["id"] == "filter_results_top_n":
        params = workflow_dicts[0].get("parameters", {})
        max_results = params.get("max_results")
        if max_results is None:
            raise HTTPException(
                400, "filter_results_top_n requires parameters.max_results"
            )
        results = raw.get("message", {}).get("results", [])
        if max_results < len(results):
            raw["message"]["results"] = results[:max_results]
        return raw
    if workflow_dicts[0]["id"] != "lookup":
        raise HTTPException(400, "operations must have id 'lookup'")

    if (query.set_interpretation or "BATCH") == "MANY":
        raise HTTPException(422, "set_interpretation MANY not supported.")

    logger.info("Doing async lookup for %s", callback)
    background_tasks.add_task(_async_lookup, callback, raw)

    return {"status": "accepted", "callback": callback}
