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
from gandalf.config import settings
from gandalf.heartbeat import start_heartbeat
from gandalf.openapi import construct_open_api_schema

configure_logging(
    getattr(logging, settings.log_level, logging.INFO), fmt=settings.log_format
)
logger = logging.getLogger(__name__)

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


_rate_limiter = _TokenBucket(settings.rate_limit) if settings.rate_limit > 0 else None


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
# Module-level graph loading (runs once in master with gunicorn --preload,
# so every forked worker shares graph RAM via Copy-on-Write).
# ---------------------------------------------------------------------------

_SKIP_PRELOAD = settings.skip_preload

GRAPH: Optional[CSRGraph] = None
BMT: Optional[Toolkit] = None

if not _SKIP_PRELOAD:
    logger.info(
        "Loading graph from %s (format=%s)...",
        settings.graph_path,
        settings.graph_format,
    )
    GRAPH = load_graph(settings.graph_path, settings.graph_format)
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
    logger.info("Graph and BMT loaded at module level (PID=%d).", os.getpid())


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Worker lifecycle — handles heartbeat and shutdown cleanup."""
    logger.info("Worker started (PID=%d).", os.getpid())
    heartbeat_stop = None
    if settings.automat_host:
        heartbeat_stop = start_heartbeat(settings)
    yield
    logger.info("Shutting down — releasing resources...")
    if heartbeat_stop is not None:
        heartbeat_stop.set()
    if (
        GRAPH is not None
        and hasattr(GRAPH, "lmdb_store")
        and GRAPH.lmdb_store is not None
    ):
        GRAPH.lmdb_store.close()
    logger.info("Shutdown complete.")


APP = FastAPI(
    title="GANDALF",
    lifespan=lifespan,
    docs_url=None,
    default_response_class=CustomORJSONResponse,
)

# Parse CORS origins from env var (comma-separated)
_cors_origins = [o.strip() for o in settings.cors_origins.split(",")]
APP.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials="*" not in _cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    APP.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ---------------------------------------------------------------------------
# OpenTelemetry / Jaeger tracing (Plater-compatible)
# ---------------------------------------------------------------------------

if settings.otel_enabled:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    _otel_resource = Resource(attributes={SERVICE_NAME: settings.otel_service_name})

    _otel_exporter: SpanExporter

    if settings.otel_use_console_exporter:
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter

        _otel_exporter = ConsoleSpanExporter()
    else:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )

        _otel_exporter = OTLPSpanExporter(
            endpoint=f"{settings.jaeger_host}:{settings.jaeger_port}",
        )

    _otel_provider = TracerProvider(resource=_otel_resource)
    _otel_provider.add_span_processor(BatchSpanProcessor(_otel_exporter))
    trace.set_tracer_provider(_otel_provider)
    FastAPIInstrumentor.instrument_app(
        APP,
        tracer_provider=_otel_provider,
        excluded_urls="docs,openapi.json",
    )
    logger.info(
        "OpenTelemetry tracing enabled (service=%s).", settings.otel_service_name
    )


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
        if (
            content_length
            and int(content_length) > settings.max_request_size_mb * 1024 * 1024
        ):
            return JSONResponse(
                status_code=413,
                content={
                    "detail": f"Request body too large (max {settings.max_request_size_mb}MB)"
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
# Documentation
# ---------------------------------------------------------------------------


@APP.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html(req: Request) -> HTMLResponse:
    """Customize Swagger UI."""
    root_path = req.scope.get("root_path", "").rstrip("/")
    openapi_url = root_path + APP.openapi_url
    swagger_favicon_url = root_path + "/static/gandalf.png"
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
# TRAPI query (Plater-compatible with query params)
# ---------------------------------------------------------------------------


@APP.post("/query", responses={200: {"model": TRAPIResponse}})
def sync_lookup(
    request: TRAPIQuery,
    subclass: Optional[bool] = Query(
        None, description="Enable biolink subclass inference"
    ),
    dehydrated: Optional[bool] = Query(
        None,
        description="Return a dehydrated response (skip edge attribute enrichment)",
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
    dehydrated_param = dehydrated if dehydrated is not None else raw.get("dehydrated")

    return lookup(
        GRAPH,
        raw,
        bmt=BMT,
        subclass=sc,
        subclass_depth=subclass_depth,
        log_level=log_level,
        dehydrated=dehydrated_param,
    )


# ---------------------------------------------------------------------------
# Async query
# ---------------------------------------------------------------------------


def _async_lookup(callback_url: str, query: dict):
    """Execute lookup and POST results to callback URL."""
    subclass = query.get("subclass", True)
    subclass_depth = query.get("subclass_depth", 1)
    log_level = query.pop("log_level", None)
    dehydrated = query.get("dehydrated")
    response = lookup(
        GRAPH,
        query,
        bmt=BMT,
        subclass=subclass,
        subclass_depth=subclass_depth,
        log_level=log_level,
        dehydrated=dehydrated,
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


APP.openapi_schema = construct_open_api_schema(
    APP,
)
