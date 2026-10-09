"""GANDALF — Plater-compatible TRAPI server."""

import logging
import os
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional, cast

import orjson
import redis
from bmt.toolkit import Toolkit
from fastapi import (
    BackgroundTasks,
    Body,
    FastAPI,
    HTTPException,
    Query,
    Request,
    Response,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import (
    get_swagger_ui_html,
)
from fastapi.staticfiles import StaticFiles
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse
from starlette.routing import Match

from gandalf import CSRGraph
from gandalf import otel
from gandalf.compression import ZstdCompressionMiddleware
from gandalf.execute import (
    execute_to_bytes,
    load_runtime,
    orjson_default,
    post_callback,
    run_query,
)
from gandalf.logging_config import configure_logging, request_id_var
from gandalf.metrics import (
    HTTP_DURATION,
    HTTP_REQUESTS,
    JOBS_ENQUEUED,
    QUEUE_LAG,
    QUEUE_PENDING,
    SYNC_WAIT,
    SYNC_WAIT_TIMEOUTS,
    metrics_payload,
    rss_anon_kb,
    rss_kb,
    trim_heap_if_large,
)
from gandalf.jobs import (
    Job,
    JobHistory,
    JobQueue,
    ResultStore,
    StoredResult,
    WorkerRegistry,
    redis_client,
)
from gandalf.notify import Monitor, Notifier
from gandalf.status import snapshot
from translator_tom import TOMBase
from translator_tom.model_dicts import QueryDict

from gandalf.models import (
    AsyncTRAPIQuery,
    EdgesResponse,
    EdgeSummaryResponse,
    MetadataResponse,
    NodeDegreeResponse,
    NodeResponse,
    TRAPIQuery,
    TRAPIResponse,
)
from gandalf.config import settings
from gandalf.heartbeat import start_heartbeat

_validate = settings.validate_responses
from gandalf.openapi import construct_open_api_schema
from gandalf.request_validation import (
    normalize_query_graph,
    reject_retired_trapi_fields,
    require_query_graph,
    validate_edge_node_references,
    validate_query_graph_is_executable,
    validate_set_interpretation,
)
from gandalf.search.edge_constraints import ConstraintError, EdgeConstraints
from gandalf.search.gc_utils import gc_disabled
from gandalf.trapi import (
    Deadline,
    TimeoutNotSatisfiable,
    finalize_response,
    query_parameters,
    resolve_timeout,
)

configure_logging(
    getattr(logging, settings.log_level, logging.INFO), fmt=settings.log_format
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# orjson-based response class (3-10x faster than stdlib json)
# ---------------------------------------------------------------------------


class CustomORJSONResponse(JSONResponse):
    media_type = "application/json"

    def render(self, content: Any) -> bytes:
        t0 = time.perf_counter()
        data: bytes = orjson.dumps(
            content, default=orjson_default, option=orjson.OPT_SERIALIZE_NUMPY
        )
        dt_ms = (time.perf_counter() - t0) * 1000
        size_kb = len(data) / 1024
        logger.debug("orjson serialization: %.2f ms, %.1f KB", dt_ms, size_kb)
        return data


def _trapi_response(content: Any) -> Any:
    """Wrap a TRAPI response body for return from an endpoint.

    On the default (non-validating) path this returns a pre-rendered
    ``CustomORJSONResponse`` so FastAPI skips ``serialize_response()`` and its
    ``jsonable_encoder()`` -- a pure-Python recursive re-walk of the entire
    response that, for large result sets (hundreds of thousands of results),
    dominates request time (~30x slower than orjson alone) even though our
    renderer then serializes the result a second time.

    When ``validate_responses`` is enabled (dev/testing) we return the raw
    dict so FastAPI still validates it against the declared response_model.
    """
    if _validate:
        return content
    return CustomORJSONResponse(content)


def _request_dict(body: dict, model: type[TOMBase]) -> QueryDict:
    """Return the request body as a plain dict for downstream handling.

    On the default (non-validating) path the JSON body -- already parsed into
    a dict by Starlette -- is used as-is, skipping Pydantic validation and the
    ``model_dump()`` re-walk that follows it.  For large incoming messages
    (e.g. ``rehydrate`` payloads carrying a full knowledge graph) that pair of
    passes can cost seconds; the handlers below read everything via ``dict``
    access with defaults, so the model is not needed.

    When ``validate_responses`` is enabled (dev/testing) the body is validated
    against *model* and normalized via TOM's own ``to_dict()`` so malformed
    requests are still rejected.  Gating inbound validation on the same flag
    keeps a single switch for "strict" mode.

    ``to_dict()`` excludes both nulls and unset defaults, which matters for the
    ``parameters`` echo: TRAPI 2.0 asks the server to repeat the parameters it
    was *given*, and a plain ``exclude_none`` dump would add TOM's
    ``bypass_cache=False`` default to a request that never mentioned it.
    """
    if _validate:
        validated = model.model_validate(body).to_dict()
        return cast("QueryDict", validated)
    return cast("QueryDict", body)


def _prepare_query(raw: QueryDict) -> Deadline:
    """Validate a TRAPI request's query graph and read its query parameters.

    Shared by ``/query`` and ``/asyncquery`` so both reject the same malformed
    requests with the same status codes.

    Args:
        raw: The request body, normalized in place.

    Returns:
        The :class:`~gandalf.trapi.Deadline` for this query.

    Raises:
        HTTPException: 400 for a malformed or non-executable query graph, or a
            malformed constraints object,
            409 when the requested ``parameters.timeout`` is below what this
            server can answer within, 422 for an unsupported
            ``set_interpretation``.
    """
    query_graph = require_query_graph(raw.get("message"))
    normalize_query_graph(query_graph)
    reject_retired_trapi_fields(query_graph)
    validate_query_graph_is_executable(query_graph)
    validate_set_interpretation(query_graph)
    validate_edge_node_references(query_graph)

    # Parse each QEdge's constraints up front so a malformed constraint is a
    # 400 here rather than an opaque 500 from deep inside the search.
    for qedge_id, qedge in (query_graph.get("edges") or {}).items():
        try:
            EdgeConstraints.parse(qedge)
        except ConstraintError as exc:
            raise HTTPException(400, f"edge '{qedge_id}': {exc}") from exc

    try:
        return Deadline(resolve_timeout(raw.get("parameters")))
    except TimeoutNotSatisfiable as exc:
        # TRAPI 2.0 gives 409 to a conflict between the client's parameters
        # and the server's capabilities.
        raise HTTPException(409, str(exc)) from exc


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
# Module-level graph loading (runs once in master with gunicorn --preload,
# so every forked worker shares graph RAM via Copy-on-Write).
# ---------------------------------------------------------------------------

_SKIP_PRELOAD = settings.skip_preload

GRAPH: Optional[CSRGraph] = None
BMT: Optional[Toolkit] = None

if not _SKIP_PRELOAD:
    GRAPH, BMT = load_runtime()
    logger.info("Graph and BMT loaded at module level (PID=%d).", os.getpid())

# Queue mode (GANDALF_QUEUE_URL): the job stream and result store this
# process enqueues on and waits on.  Opened per worker process in the
# lifespan rather than at import, so no Redis connection crosses gunicorn's
# fork.  Both stay None in in-process mode.
QUEUE: Optional[JobQueue] = None
RESULTS: Optional[ResultStore] = None
WORKERS: Optional[WorkerRegistry] = None
HISTORY: Optional[JobHistory] = None
NOTIFIER: Optional[Notifier] = None
MONITOR: Optional[Monitor] = None


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------


def open_queue() -> None:
    """Connect this process to the configured job queue (no-op without one)."""
    global QUEUE, RESULTS, WORKERS, HISTORY, NOTIFIER, MONITOR
    if not settings.queue_url or QUEUE is not None:
        return
    client = redis_client()
    QUEUE = JobQueue.from_settings(client)
    RESULTS = ResultStore.from_settings(client)
    WORKERS = WorkerRegistry.from_settings(client)
    HISTORY = JobHistory.from_settings(client)
    NOTIFIER = Notifier.from_settings(client)
    # Every API process runs a monitor thread; the Redis lock inside makes
    # exactly one of them, across all pods, the one that looks and alerts.
    MONITOR = Monitor(
        client,
        QUEUE,
        WORKERS,
        NOTIFIER,
        interval_seconds=settings.monitor_interval_seconds,
        lag_per_worker=settings.alert_queue_lag_per_worker,
        backlog_seconds=settings.alert_queue_backlog_seconds,
        stuck_seconds=settings.alert_stuck_seconds,
    )
    MONITOR.start()
    QUEUE.ensure_group()
    logger.info(
        "Queue mode: jobs go to %s on %s", settings.queue_stream, settings.queue_url
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Worker lifecycle — handles OTel init, heartbeat and shutdown cleanup."""
    logger.info("Worker started (PID=%d).", os.getpid())
    # Initialize OTel here (per-worker, post-fork) rather than at import; see
    # gandalf.otel.init_otel for why this must not run in a preloaded master.
    otel.init_otel(app)
    open_queue()
    heartbeat_stop = None
    if settings.automat_host:
        heartbeat_stop = start_heartbeat(settings)
    yield
    logger.info("Shutting down — releasing resources...")
    if heartbeat_stop is not None:
        heartbeat_stop.set()
    if MONITOR is not None:
        MONITOR.stop()
    if QUEUE is not None:
        QUEUE.client.close()
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
# Middleware: request ID, access logging, request size limit, rate limiting
# ---------------------------------------------------------------------------


def _route_template(request: Request) -> str:
    """The matched route's path template (``/node/{curie}``), for metric labels.

    A per-CURIE path would be one label value per node; the template keeps
    the cardinality to the number of routes.
    """
    for route in APP.router.routes:
        match, _ = route.matches(request.scope)
        if match == Match.FULL:
            return str(getattr(route, "path", request.url.path))
    return "unmatched"


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

    pid = os.getpid()
    rss_start_kb = rss_kb()
    anon_start_kb = rss_anon_kb()
    logger.info(
        "request start pid=%s rss_kb=%s anon_kb=%s %s %s",
        pid,
        rss_start_kb,
        anon_start_kb,
        request.method,
        request.url.path,
    )

    t_start = time.monotonic()
    response: Response = await call_next(request)
    duration_s = time.monotonic() - t_start
    duration_ms = duration_s * 1000
    route = _route_template(request)
    HTTP_REQUESTS.labels(request.method, route, str(response.status_code)).inc()
    HTTP_DURATION.labels(request.method, route).observe(duration_s)
    rss_end_kb = rss_kb()
    anon_end_kb = rss_anon_kb()

    def _delta(start: int, end: int) -> int:
        return end - start if start >= 0 and end >= 0 else -1

    rss_delta_kb = _delta(rss_start_kb, rss_end_kb)
    anon_delta_kb = _delta(anon_start_kb, anon_end_kb)

    response.headers["X-Request-ID"] = req_id
    logger.info(
        "request end pid=%s rss_kb=%s rss_delta_kb=%s anon_kb=%s anon_delta_kb=%s "
        "%s %s %s %.1fms",
        pid,
        rss_end_kb,
        rss_delta_kb,
        anon_end_kb,
        anon_delta_kb,
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


# Zstandard compression/decompression. Added last so it is the outermost
# middleware: on the way in it decompresses zstd request bodies (and rewrites
# content-length, so request_middleware's size limit sees the decompressed
# size); on the way out it compresses responses for clients that advertise
# Accept-Encoding: zstd.
if settings.compression_enabled:
    APP.add_middleware(
        ZstdCompressionMiddleware,
        minimum_size=settings.compress_minimum_size,
        level=settings.compress_zstd_level,
        max_request_size_mb=settings.max_request_size_mb,
        decompress_requests=settings.decompress_request_enabled,
        compress_responses=settings.compress_response_enabled,
    )


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


@APP.head("/", include_in_schema=False)
@APP.head("/health", include_in_schema=False)
async def head_alive() -> Response:
    """Liveness for checkers that send a bare ``HEAD``: 200, no body."""
    return Response(status_code=200)


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
# Health, readiness, metrics
# ---------------------------------------------------------------------------


@APP.get("/health", include_in_schema=False)
async def health() -> dict:
    """Liveness: the process answers HTTP.

    Deliberately checks nothing else.  A worker busy compressing a
    multi-gigabyte response answers this late, not wrong, so the probe that
    calls it needs a generous timeout rather than a stricter check here.
    """
    return {"status": "ok"}


@APP.get("/ready", include_in_schema=False)
def ready() -> JSONResponse:
    """Readiness: this process can take a query right now.

    The graph is loaded and, in queue mode, Redis answers.  503 takes the
    pod out of the Service until it does.
    """
    if GRAPH is None:
        return JSONResponse(status_code=503, content={"status": "graph not loaded"})
    if settings.queue_url:
        # Readiness is the one place a Redis failure must become a status
        # code rather than an exception: the probe reads the code.
        try:
            reachable = QUEUE is not None and QUEUE.ping()
        except redis.RedisError as exc:
            logger.warning("Readiness: queue unreachable: %s", exc)
            reachable = False
        if not reachable:
            return JSONResponse(
                status_code=503, content={"status": "queue unreachable"}
            )
    return JSONResponse(content={"status": "ready"})


@APP.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    """Prometheus metrics for this process (all gunicorn workers, aggregated)."""
    if QUEUE is not None:
        # A Redis outage must not fail the scrape: the other metrics are
        # most wanted exactly then, and /ready already reports the outage.
        try:
            stats = QUEUE.stats()
        except redis.RedisError as exc:
            logger.warning("Metrics: queue stats unavailable: %s", exc)
        else:
            QUEUE_LAG.set(stats.lag)
            QUEUE_PENDING.set(stats.pending)
    body, content_type = metrics_payload()
    return Response(content=body, media_type=content_type)


# ---------------------------------------------------------------------------
# Status page
# ---------------------------------------------------------------------------

_STATUS_PAGE = (STATIC_DIR / "status.html").read_text(encoding="utf-8")


@APP.get("/status", include_in_schema=False)
async def status_page() -> HTMLResponse:
    """The live status page: queue, workers, recent jobs, this pod.

    Self-contained HTML that polls ``status.json`` (relative, so it works
    behind any path prefix) and needs nothing outside this server.
    """
    return HTMLResponse(_STATUS_PAGE)


@APP.get("/status.json", include_in_schema=False)
def status_json(recent: int = Query(50, ge=1, le=500)) -> dict:
    """The data behind ``/status``; see :mod:`gandalf.status`."""
    try:
        return snapshot(GRAPH, QUEUE, WORKERS, HISTORY, NOTIFIER, recent=recent)
    except redis.RedisError as exc:
        raise HTTPException(503, f"Queue unavailable: {type(exc).__name__}") from exc


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


@APP.get(
    "/metadata",
    response_model=MetadataResponse if _validate else None,
    responses={200: {"model": MetadataResponse}},
)
def metadata():
    """Return knowledge graph metadata and statistics."""
    if GRAPH is None:
        raise HTTPException(503, "Graph not loaded")
    return GRAPH.graph_metadata


# ---------------------------------------------------------------------------
# Node degree
# ---------------------------------------------------------------------------


@APP.get(
    "/node_degree/{curie}",
    response_model=NodeDegreeResponse if _validate else None,
    responses={200: {"model": NodeDegreeResponse}},
)
def node_degree(curie: str):
    """Return the total degree (incoming + outgoing edges) of a node."""
    if GRAPH is None:
        raise HTTPException(503, "Graph not loaded")

    node_idx = GRAPH.get_node_idx(curie)
    if node_idx is None:
        raise HTTPException(404, f"Node not found: {curie}")

    out_deg = int(GRAPH.fwd_offsets[node_idx + 1] - GRAPH.fwd_offsets[node_idx])
    in_deg = int(GRAPH.rev_offsets[node_idx + 1] - GRAPH.rev_offsets[node_idx])
    return {"id": curie, "degree": out_deg + in_deg}


# ---------------------------------------------------------------------------
# TRAPI query (Plater-compatible with query params)
# ---------------------------------------------------------------------------


#: Documented for both TRAPI query endpoints: a client's parameters.timeout
#: can conflict with what this server is able to do.
_CONFLICT_RESPONSE: dict[int | str, dict[str, Any]] = {
    409: {
        "description": "There is a conflict between client-given TRAPI "
        "parameters and server capabilities."
    }
}


@APP.post(
    "/query",
    response_model=TRAPIResponse if _validate else None,
    # TRAPI 2.0 dropped `nullable` throughout, so an absent optional property
    # must be absent rather than null.  Without this, strict mode would emit
    # `"description": null` and a `parameters` object padded out with nulls,
    # neither of which validates -- and neither of which the default
    # (non-validating) path produces.
    response_model_exclude_none=True,
    responses={200: {"model": TRAPIResponse}, **_CONFLICT_RESPONSE},
)
def sync_lookup(
    request: Request,
    body: dict = Body(...),
    profile: Optional[bool] = Query(
        None,
        description="Emit per-stage timings into message.logs as ProfileStage / ProfileSummary entries",
    ),
):
    """Execute a TRAPI 2.0 query against the knowledge graph.

    Supports the 'lookup' workflow operation. All request configuration other
    than ``profile`` is read from the body's ``parameters`` object, including
    TRAPI's own ``timeout``, ``log_level`` and ``bypass_cache``.

    The body is taken as a raw dict and not run through Pydantic validation
    unless ``validate_responses`` is enabled -- see ``_request_dict``.

    In queue mode the query becomes a job for a worker process and this
    handler waits for its result; otherwise it runs here, on the thread pool.
    """
    otel.record_baggage()
    if GRAPH is None:
        raise HTTPException(503, "Graph not loaded")
    # The previous response's pages are free by now; return them before
    # building (or receiving) the next one.
    trim_heap_if_large(settings.heap_trim_threshold_mb)

    raw = _request_dict(body, TRAPIQuery)

    # Rehydration enriches the supplied knowledge graph and runs no lookup,
    # so it has no query graph to validate and no time budget.
    rehydrate = query_parameters(raw).get("rehydrate") is not None
    deadline = None if rehydrate else _prepare_query(raw)

    if QUEUE is not None:
        accept_encoding = request.headers.get("accept-encoding", "")
        return _queued_lookup(raw, bool(profile), deadline, accept_encoding)

    # Keep GC paused until the response has been serialized and freed.
    # lookup() pauses it too, but lets it resume as it returns; the next
    # allocation would then trigger a collection over every object of the
    # response (millions, several seconds on a large query, with the GIL
    # held) just before the response is thrown away.  Freed while paused, it
    # never gets scanned: it holds no reference cycles, so dropping it frees
    # it, and the collection that runs on resuming finds little left.
    with gc_disabled():
        # Edge attributes served straight from the graph's JSON, and results
        # written straight to JSON, unless FastAPI is to validate the
        # response, which needs them as Python objects.
        response = run_query(
            GRAPH,
            BMT,
            raw,
            profile=bool(profile),
            deadline=deadline,
            as_json=not _validate,
        )
        rendered = _trapi_response(response)
        del response
    return rendered


def _queued_lookup(
    raw: QueryDict,
    profile: bool,
    deadline: Optional[Deadline],
    accept_encoding: str,
) -> Response:
    """Enqueue a ``/query`` and wait for a worker's answer.

    The wait is the client's time budget plus a grace period for the
    worker's own Timeout response to arrive, or the configured maximum for a
    query with no budget.  A result that never comes is a 504: the queue is
    deeper than the budget allows, which is what the queue lag metric and
    the autoscaler are there to prevent.

    The result is stored zstd-compressed; a client that accepts zstd gets
    those bytes as they are, anyone else gets them decompressed here.
    """
    assert QUEUE is not None and RESULTS is not None
    job = Job.new(
        raw,
        profile=profile,
        budget=deadline.budget if deadline is not None else None,
        request_id=request_id_var.get(""),
    )
    wait = job.wait_seconds()
    t0 = time.monotonic()
    # Redis down is a 503, not a 500: the client can retry, and /ready is
    # already telling the load balancer the same thing.
    try:
        QUEUE.enqueue(job)
        JOBS_ENQUEUED.labels("sync").inc()
        logger.info("job %s enqueued; waiting up to %.0fs", job.job_id, wait)
        result = RESULTS.wait(job.job_id, timeout=wait)
    except redis.RedisError as exc:
        logger.error("job %s: queue unavailable: %s", job.job_id, exc)
        raise HTTPException(503, f"Queue unavailable: {type(exc).__name__}") from exc
    SYNC_WAIT.observe(time.monotonic() - t0)
    if result is None:
        SYNC_WAIT_TIMEOUTS.inc()
        raise HTTPException(
            504,
            f"Query {job.job_id} did not complete within {wait:.0f}s; "
            "the queue may be deeper than the requested timeout allows.",
        )
    RESULTS.delete(job.job_id)
    return _stored_response(result, accept_encoding)


def _stored_response(result: StoredResult, accept_encoding: str) -> Response:
    """Answer with a stored result, compressed when the client takes zstd."""
    if settings.compress_response_enabled and "zstd" in accept_encoding.lower():
        # The compression middleware leaves an already-encoded body alone.
        return Response(
            content=result.compressed,
            status_code=result.http_status,
            media_type="application/json",
            headers={"Content-Encoding": "zstd"},
        )
    return Response(
        content=result.decompress(),
        status_code=result.http_status,
        media_type="application/json",
    )


# ---------------------------------------------------------------------------
# Async query
# ---------------------------------------------------------------------------


def _async_lookup(
    callback_url: str,
    query: QueryDict,
    trace_headers: Optional[dict] = None,
    profile: bool = False,
    deadline: Optional[Deadline] = None,
):
    """Execute lookup in this process and POST results to callback URL.

    The in-process ``/asyncquery`` path (no queue configured): a FastAPI
    background task on the thread pool.  ``trace_headers`` carries the W3C
    trace context captured from the original request, since the thread does
    not inherit the request's contextvars; ``deadline`` carries the client's
    ``parameters.timeout`` budget, measured from when the request was
    accepted.
    """
    if GRAPH is None:
        raise HTTPException(503, "Graph not loaded")
    # GC stays paused until the response is serialized and freed, as in
    # sync_lookup; the serialized bytes are posted with GC back on.
    body = execute_to_bytes(GRAPH, BMT, query, profile=profile, deadline=deadline)
    post_callback(callback_url, body, trace_headers)


def _async_accepted(callback: str, job_id: Optional[str] = None) -> dict:
    """Build the TRAPI AsyncQueryResponse for an accepted job.

    TRAPI 2.0 requires ``job_id``; gandalf POSTs the result to the callback
    rather than exposing an ``/asyncquery_status`` endpoint, so the id
    identifies the job in this server's (and, in queue mode, the worker's)
    logs.
    """
    return {
        "status": "Accepted",
        "description": "Query has been queued.",
        "job_id": job_id or request_id_var.get("") or str(uuid.uuid4())[:8],
        "callback": callback,
    }


def _dispatch_async(
    background_tasks: BackgroundTasks,
    callback: str,
    raw: QueryDict,
    profile: bool,
    deadline: Optional[Deadline],
) -> dict:
    """Hand an accepted ``/asyncquery`` to a worker or a background task.

    The OTel trace context is captured now, while still inside the request
    span, so whoever runs the query can propagate it to the callback.  When
    OTel is disabled the carrier stays empty.
    """
    trace_headers: dict[str, str] = {}
    otel.inject_headers(trace_headers)
    if QUEUE is not None:
        job = Job.new(
            raw,
            profile=profile,
            budget=deadline.budget if deadline is not None else None,
            callback=callback,
            trace_headers=trace_headers,
            request_id=request_id_var.get(""),
        )
        try:
            QUEUE.enqueue(job)
        except redis.RedisError as exc:
            logger.error("job %s: queue unavailable: %s", job.job_id, exc)
            raise HTTPException(
                503, f"Queue unavailable: {type(exc).__name__}"
            ) from exc
        JOBS_ENQUEUED.labels("callback").inc()
        logger.info("job %s enqueued for callback %s", job.job_id, callback)
        return _async_accepted(callback, job.job_id)
    logger.info("Doing async lookup for %s", callback)
    background_tasks.add_task(
        _async_lookup, callback, raw, trace_headers, profile, deadline
    )
    return _async_accepted(callback)


@APP.post("/asyncquery", responses=_CONFLICT_RESPONSE)
def async_query(
    background_tasks: BackgroundTasks,
    query: dict = Body(...),
    profile: Optional[bool] = Query(
        None,
        description="Emit per-stage timings into message.logs as ProfileStage / ProfileSummary entries",
    ),
):
    """Handle asynchronous query.

    The body is taken as a raw dict and not run through Pydantic validation
    unless ``validate_responses`` is enabled -- see ``_request_dict``.
    """
    otel.record_baggage()
    if GRAPH is None:
        raise HTTPException(503, "Graph not loaded")
    raw = _request_dict(query, AsyncTRAPIQuery)
    callback = raw.get("callback")

    # Validate callback URL is present and uses an http(s) scheme
    if not isinstance(callback, str) or not callback.startswith(
        ("http://", "https://")
    ):
        raise HTTPException(400, "callback must be an http:// or https:// URL")

    # Rehydration: skip lookup/workflow validation, only enrich the supplied
    # knowledge graph in the background and POST it to the callback.
    if query_parameters(raw).get("rehydrate") is not None:
        return _dispatch_async(background_tasks, callback, raw, bool(profile), None)

    # Parse the requested workflow.  TRAPI models Operation as a union of ~30
    # per-operation types; this server implements two op ids, so the branch
    # below reads them as plain dicts rather than discriminating that union.
    workflow_dicts: list[dict[str, Any]] = cast(
        "list[dict[str, Any]]", raw.get("workflow")
    ) or [{"id": "lookup", "parameters": None}]

    if len(workflow_dicts) != 1:
        raise HTTPException(400, "workflow must contain exactly 1 operation")
    if workflow_dicts[0].get("id") == "filter_results_top_n":
        params = workflow_dicts[0].get("parameters") or {}
        max_results = params.get("max_results")
        if max_results is None:
            raise HTTPException(
                400, "filter_results_top_n requires parameters.max_results"
            )
        results = raw["message"].get("results") or []
        if int(max_results) < len(results):
            raw["message"]["results"] = results[: int(max_results)]
        return _trapi_response(finalize_response({"message": raw["message"]}, raw))
    if workflow_dicts[0].get("id") != "lookup":
        raise HTTPException(400, "operations must have id 'lookup'")

    if (raw.get("set_interpretation") or "BATCH") == "MANY":
        raise HTTPException(422, "set_interpretation MANY not supported.")

    deadline = _prepare_query(raw)
    return _dispatch_async(background_tasks, callback, raw, bool(profile), deadline)


def _custom_openapi() -> dict:
    """Return the TRAPI-customised OpenAPI schema, building it once.

    Override ``APP.openapi`` rather than assigning ``APP.openapi_schema``
    directly: since FastAPI 0.138 the default ``openapi()`` regenerates the
    schema whenever its cached routes-version marker is stale, which would
    discard our injected request-body refs (see openapi._inject_request_schemas).
    """
    if APP.openapi_schema is None:
        APP.openapi_schema = construct_open_api_schema(APP)
    return APP.openapi_schema


APP.openapi = _custom_openapi  # type: ignore[method-assign]
_custom_openapi()  # build eagerly so import-time readers share one schema
