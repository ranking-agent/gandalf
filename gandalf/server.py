"""GANDALF — Plater-compatible TRAPI server."""

import gc
import logging
import os
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import httpx
import orjson
from bmt.toolkit import Toolkit
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import (
    get_swagger_ui_html,
)
from fastapi.staticfiles import StaticFiles
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse

from gandalf import CSRGraph, lookup
from gandalf.logging_config import configure_logging

configure_logging(logging.INFO)
logger = logging.getLogger(__name__)

GRAPH = None
BMT = None

# Configuration via environment variables
GRAPH_PATH = os.environ.get("GANDALF_GRAPH_PATH", "../12_17_2025/gandalf_mmap")
GRAPH_FORMAT = os.environ.get("GANDALF_GRAPH_FORMAT", "auto")  # "auto" or "mmap"


# ---------------------------------------------------------------------------
# orjson-based response class (3-10x faster than stdlib json)
# ---------------------------------------------------------------------------

def _orjson_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class CustomORJSONResponse(JSONResponse):
    media_type = "application/json"

    def render(self, content) -> bytes:
        return orjson.dumps(content, default=_orjson_default)


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
    path = Path(path)

    if format == "auto":
        if path.is_dir():
            format = "mmap"
        else:
            raise ValueError(f"Cannot auto-detect format for: {path}. Expected a directory.")

    if format == "mmap":
        return CSRGraph.load_mmap(path)
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
    GRAPH = None
    BMT = None


APP = FastAPI(
    title="GANDALF",
    lifespan=lifespan,
    docs_url=None,
    default_response_class=CustomORJSONResponse,
)

APP.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent.parent / "static"
if STATIC_DIR.exists():
    APP.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


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

@APP.get("/metadata")
def metadata():
    """Return knowledge graph metadata and statistics."""
    if GRAPH is None:
        raise HTTPException(503, "Graph not loaded")
    return GRAPH.graph_metadata


# ---------------------------------------------------------------------------
# Node lookup (Plater-compatible)
# ---------------------------------------------------------------------------

@APP.get("/node/{curie:path}")
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

@APP.get("/edges/{curie:path}")
def get_edges(
    curie: str,
    category: Optional[str] = Query(None, description="Filter by target node category"),
    predicate: Optional[str] = Query(None, description="Filter by edge predicate"),
    limit: Optional[int] = Query(None, description="Maximum number of results"),
    offset: int = Query(0, description="Result offset for pagination"),
    count_only: bool = Query(False, description="Return only the count of matching edges"),
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
    for tgt_idx, pred_str, props, fwd_idx in GRAPH.neighbors_with_properties(node_idx, pred_filter):
        tgt_id = GRAPH.get_node_id(tgt_idx)
        tgt_props = GRAPH.get_all_node_properties(tgt_idx)
        tgt_cats = tgt_props.get("categories", ["biolink:NamedThing"])

        if category is not None and category not in tgt_cats:
            continue

        edge_id = GRAPH.get_edge_id(fwd_idx)
        edges.append({
            "subject": curie,
            "object": tgt_id,
            "predicate": pred_str,
            "edge_id": edge_id,
            "sources": props.get("sources", []),
            "qualifiers": props.get("qualifiers", []),
        })

    # Collect incoming edges
    for src_idx, pred_str, props, fwd_idx in GRAPH.incoming_neighbors_with_properties(node_idx, pred_filter):
        src_id = GRAPH.get_node_id(src_idx)
        src_props = GRAPH.get_all_node_properties(src_idx)
        src_cats = src_props.get("categories", ["biolink:NamedThing"])

        if category is not None and category not in src_cats:
            continue

        edge_id = GRAPH.get_edge_id(fwd_idx)
        edges.append({
            "subject": src_id,
            "object": curie,
            "predicate": pred_str,
            "edge_id": edge_id,
            "sources": props.get("sources", []),
            "qualifiers": props.get("qualifiers", []),
        })

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

@APP.get("/edge_summary/{curie:path}")
def edge_summary(curie: str):
    """Summarize edge types connected to a node.

    Returns a list of [predicate, category, count] triples.
    """
    if GRAPH is None:
        raise HTTPException(503, "Graph not loaded")

    node_idx = GRAPH.get_node_idx(curie)
    if node_idx is None:
        raise HTTPException(404, f"Node not found: {curie}")

    summary = defaultdict(int)

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
            [pred, cat, count]
            for (pred, cat), count in sorted(summary.items())
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
    filtered = {}
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

@APP.post("/query")
def sync_lookup(
    request: dict,
    subclass: Optional[bool] = Query(None, description="Enable biolink subclass inference"),
):
    """Execute a TRAPI query against the knowledge graph.

    Supports the 'lookup' workflow operation.
    """
    if GRAPH is None:
        raise HTTPException(503, "Graph not loaded")

    # Query params take precedence, fall back to request body
    sc = subclass if subclass is not None else request.get("subclass", True)
    subclass_depth = request.get("subclass_depth", 1)

    response = lookup(GRAPH, request, bmt=BMT, subclass=sc, subclass_depth=subclass_depth)

    return response


# ---------------------------------------------------------------------------
# Async query
# ---------------------------------------------------------------------------

def async_lookup(callback_url: str, query: dict):
    """Do an async lookup."""
    subclass = query.get("subclass", True)
    subclass_depth = query.get("subclass_depth", 1)
    response = lookup(GRAPH, query, bmt=BMT, subclass=subclass, subclass_depth=subclass_depth)

    try:
        with httpx.Client(
            timeout=httpx.Timeout(timeout=600.0)
        ) as client:
            res = client.post(callback_url, json=response)
            res.raise_for_status()
            logger.info("Posted to %s with code %s", callback_url, res.status_code)
    except Exception as e:
        logger.error("Callback to %s failed with: %s", callback_url, e)


@APP.post("/asyncquery")
def async_query(
    background_tasks: BackgroundTasks,
    query: dict,
):
    """Handle asynchronous query."""
    # parse requested workflow
    callback = query["callback"]
    workflow = query.get("workflow", None) or [{"id": "lookup"}]
    if not isinstance(workflow, list):
        raise HTTPException(400, "workflow must be a list")
    if not len(workflow) == 1:
        raise HTTPException(400, "workflow must contain exactly 1 operation")
    if "id" not in workflow[0]:
        raise HTTPException(400, "workflow must have property 'id'")
    if workflow[0]["id"] == "filter_results_top_n":
        max_results = workflow[0]["parameters"]["max_results"]
        if max_results < len(query["message"]["results"]):
            query["message"]["results"] = query["message"]["results"][
                :max_results
            ]
        return query
    if not workflow[0]["id"] == "lookup":
        raise HTTPException(400, "operations must have id 'lookup'")

    if (query.get("set_interpretation", None) or "BATCH") == "MANY":
        raise HTTPException(422, "set_interpretation MANY not supported.")

    logger.info("Doing async lookup for %s", callback)
    background_tasks.add_task(async_lookup, callback, query)

    return
