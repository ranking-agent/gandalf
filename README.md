# GANDALF

Graph Analysis Navigator for Discovery And Link Finding

A high-performance Python library and [Translator](https://ncats.nih.gov/translator)-compatible TRAPI server for fast path finding in large biomedical knowledge graphs.

## Features

- **Compressed Sparse Row (CSR)** graph representation for memory-efficient storage of 10M+ node, 38M+ edge graphs
- **Bidirectional search** for optimal path-finding performance
- **O(1) property lookups** via hash indexing
- **Predicate filtering** to reduce path explosion
- **Qualifier filtering** for advanced edge constraints (aspect, direction, mechanism)
- **Subclass expansion** via Biolink Model Toolkit with configurable depth
- **Batch property enrichment** — enrich only final paths, not intermediate results
- **Diagnostic tools** to understand path counts and explosion
- **TRAPI 1.5 compatible** REST API with Plater-compatible endpoints
- **Async query support** with callback URLs
- **Dehydrated mode** for lightweight responses that skip edge and node attribute enrichment
- **OpenTelemetry tracing** with Jaeger integration

## Installation

**Recommended: Use a virtual environment**

Some transitive dependencies (e.g., `stringcase`, `pytest-logging`) require modern pip/setuptools to build correctly. Using a virtual environment ensures you have updated tools.

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Upgrade pip and setuptools (important for building dependencies)
pip install --upgrade pip setuptools wheel

# Install the core package
pip install -e .

# Install with server dependencies (FastAPI, uvicorn, etc.)
pip install -e ".[server]"

# Install with dev dependencies (pytest, black, mypy)
pip install -e ".[dev]"
```

## Quick Start

### Unzipping a full Translator KGX

```bash
tar -xvf translator_kg.tar.zst
```

This will output a `nodes.jsonl` and `edges.jsonl` file.

### Build a graph from JSONL

```python
from gandalf import build_graph_from_jsonl

# Build with ontology filtering
graph = build_graph_from_jsonl(
    edges_path="data/raw/edges.jsonl",
    nodes_path="data/raw/nodes.jsonl",
)

# Save for fast loading
graph.save_mmap("data/processed/gandalf_mmap")
```

### Query paths (TRAPI format)

```python
from gandalf import CSRGraph, lookup

# Load graph (takes ~1-2 seconds)
graph = CSRGraph.load_mmap("data/processed/gandalf_mmap")

# Execute a TRAPI query
response = lookup(
    graph,
    {
        "message": {
            "query_graph": {
                "nodes": {
                    "n0": {"ids": ["CHEBI:45783"]},
                    "n1": {"categories": ["biolink:Gene"]},
                    "n2": {"categories": ["biolink:Disease"]}
                },
                "edges": {
                    "e0": {"subject": "n0", "object": "n1", "predicates": ["biolink:affects"]},
                    "e1": {"subject": "n1", "object": "n2"}
                }
            }
        }
    },
    subclass=True,
    subclass_depth=1,
)

print(f"Found {len(response['message']['results'])} paths")
```

## Architecture

The package uses a three-stage pipeline:

1. **Topology Search** (fast) - Find all paths using indices only
2. **Filtering** (medium) - Apply business logic on necessary node or edge properties
3. **Enrichment** (batch) - Load all properties for final paths only

This separation allows filtering millions of paths before expensive property lookups.

## REST API

The server exposes Plater-compatible TRAPI endpoints on port 6429.

**Run the development server:**

```bash
python gandalf/main.py
```

**Run the production server:**

```bash
gunicorn gandalf.server:APP -c gunicorn.conf.py
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Redirect to `/docs` |
| `GET` | `/docs` | Swagger UI documentation |
| `GET` | `/metadata` | Graph statistics and metadata |
| `GET` | `/meta_knowledge_graph` | Meta KG with predicates, categories, and counts |
| `GET` | `/sri_testing_data` | Representative edges for SRI Testing Harness |
| `POST` | `/query` | Synchronous TRAPI query |
| `POST` | `/asyncquery` | Async TRAPI query with callback URL |

The `/query` endpoint accepts optional query parameters:
- `?subclass=true` — Enable biolink subclass inference
- `?dehydrated=true` — Skip edge attribute enrichment for faster, lighter responses

## CLI Commands

```bash
# Build a CSR graph from JSONL node/edge files
gandalf-build --edges data/edges.jsonl --nodes data/nodes.jsonl --output data/graph_mmap/

# Query paths from the command line
gandalf-query --graph data/graph_mmap/ --start "CHEBI:45783" --end "MONDO:0004979"

# Diagnose path explosion between two nodes
gandalf-diagnose --graph data/graph_mmap/ --start "CHEBI:45783" --end "MONDO:0004979"
```

## Configuration

The server is configured via environment variables (prefixed with `GANDALF_`):

### Core

| Variable | Default | Description |
|----------|---------|-------------|
| `GANDALF_GRAPH_PATH` | `/data/graph` | Path to the mmap graph directory |
| `GANDALF_GRAPH_FORMAT` | `auto` | Graph format (`auto` or `mmap`) |
| `GANDALF_LOAD_MMAPS_INTO_MEMORY` | `false` | Load memory-mapped arrays fully into RAM |
| `GANDALF_LOG_LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `GANDALF_LOG_FORMAT` | `text` | Log format (`text` for human-readable, `json` for structured) |
| `GANDALF_CORS_ORIGINS` | `*` | Comma-separated list of allowed CORS origins |
| `GANDALF_MAX_REQUEST_SIZE_MB` | `10` | Maximum request body size in MB |
| `GANDALF_RATE_LIMIT` | `0` | Max requests per minute per client IP (0 = disabled) |
| `GANDALF_SKIP_PRELOAD` | `false` | Skip module-level graph loading |
| `GANDALF_WORKERS` | `2` | Gunicorn worker count |

### Search Tuning

| Variable | Default | Description |
|----------|---------|-------------|
| `GANDALF_LARGE_RESULT_THRESHOLD` | `50000` | Path count threshold for auto-dehydrated responses |
| `GANDALF_MAX_PATH_LIMIT` | `0` | Max intermediate paths during joins (0 = unlimited) |
| `GANDALF_DEBUG_PATHS_TSV` | _(empty)_ | File path to write debug TSV of reconstructed paths |

### Server Identity

| Variable | Default | Description |
|----------|---------|-------------|
| `GANDALF_SERVER_URL` | `http://localhost:6429` | Public URL of this instance |
| `GANDALF_SERVER_MATURITY` | `development` | Maturity level for TRAPI metadata |
| `GANDALF_SERVER_LOCATION` | `RENCI` | Server location for TRAPI metadata |
| `GANDALF_INFORES` | `infores:gandalf` | Translator infores identifier |

### Automat Heartbeat

| Variable | Default | Description |
|----------|---------|-------------|
| `GANDALF_AUTOMAT_HOST` | _(empty, disabled)_ | Automat cluster URL for registration |
| `GANDALF_HEART_RATE` | `30` | Seconds between heartbeats |
| `GANDALF_SERVICE_ADDRESS` | _(empty)_ | Reachable address of this instance |
| `GANDALF_WEB_PORT` | `8080` | Port for heartbeat registration |

### Observability

| Variable | Default | Description |
|----------|---------|-------------|
| `GANDALF_OTEL_ENABLED` | `true` | Enable OpenTelemetry tracing |
| `GANDALF_OTEL_SERVICE_NAME` | `gandalf` | Service name for traces |
| `GANDALF_JAEGER_HOST` | `http://jaeger` | Jaeger collector host |
| `GANDALF_JAEGER_PORT` | `4317` | Jaeger collector gRPC port |

## Docker

```bash
# Build the image
docker build -t gandalf .

# Run with a graph volume
docker run -p 6429:6429 \
  -v /path/to/graph:/data/graph \
  -e GANDALF_GRAPH_PATH=/data/graph \
  gandalf
```

## Verifying the Server

```bash
# Check graph metadata
curl http://localhost:6429/metadata

# Browse the API docs
open http://localhost:6429/docs
```

## Releases
- Make a release in GitHub to run a GitHub Action that pushes a gandalf to ghcr
- Run this on the mmap folder: `tar -czvf gandalf_mmap_<date>.tar.gz gandalf_mmap`
- Upload the tar.gz file to a public file server
- Update any helm charts and deploy
