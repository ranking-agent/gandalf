"""GANDALF."""

import gc
import os
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from bmt.toolkit import Toolkit
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import (
    get_swagger_ui_html,
)
from fastapi.staticfiles import StaticFiles
from starlette.responses import HTMLResponse

from gandalf import CSRGraph, lookup

GRAPH = None
BMT = None

# Configuration via environment variables
GRAPH_PATH = os.environ.get("GANDALF_GRAPH_PATH", "../12_17_2025/gandalf_mmap")
GRAPH_FORMAT = os.environ.get("GANDALF_GRAPH_FORMAT", "auto")  # "auto", "pickle", or "mmap"


def load_graph(path: str, format: str = "auto") -> CSRGraph:
    """
    Load graph from disk.

    Args:
        path: Path to graph file (pickle) or directory (mmap)
        format: "auto" (detect from path), "pickle", or "mmap"

    Returns:
        Loaded CSRGraph
    """
    path = Path(path)

    # Auto-detect format
    if format == "auto":
        if path.is_dir():
            format = "mmap"
        elif path.suffix == ".pkl":
            format = "pickle"
        else:
            raise ValueError(f"Cannot auto-detect format for: {path}")

    if format == "mmap":
        return CSRGraph.load_mmap(path)
    elif format == "pickle":
        return CSRGraph.load(path)
    else:
        raise ValueError(f"Unknown format: {format}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle graph and BMT loading on startup."""
    global GRAPH, BMT
    print(f"Loading graph from {GRAPH_PATH} (format={GRAPH_FORMAT})...")
    GRAPH = load_graph(GRAPH_PATH, GRAPH_FORMAT)
    print("Initializing Biolink Model Toolkit...")
    BMT = Toolkit()

    # Freeze all objects allocated so far (graph + BMT) into a permanent
    # generation that the cyclic GC will never scan.  This makes Gen 2
    # collections cheap because they skip the large CSR arrays.
    gc.collect()
    gc.freeze()
    # Raise thresholds so Gen 2 collections are less frequent even for
    # the (now-small) unfrozen query-time object set.
    gc.set_threshold(50_000, 50, 50)
    print("Server ready!")
    yield
    GRAPH = None
    BMT = None


APP = FastAPI(
    title="GANDALF",
    lifespan=lifespan,
    docs_url=None,
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

@APP.post("/query")
def sync_lookup(request: dict):
    """Do a lookup."""
    subclass = request.get("subclass", True)
    subclass_depth = request.get("subclass_depth", 1)
    response = lookup(GRAPH, request, bmt=BMT, subclass=subclass, subclass_depth=subclass_depth)

    return response


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
            print(f"Posted to {callback_url} with code {res.status_code}")
    except Exception as e:
        print(f"Callback to {callback_url} failed with: {e}")


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

    print(f"Doing async lookup for {callback}")
    background_tasks.add_task(async_lookup, callback, query)

    return

# APP.openapi_schema = construct_open_api_schema(
#     APP,
#     infores="infores:shepherd",
# )
