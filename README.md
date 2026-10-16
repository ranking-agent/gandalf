# GANDALF

Graph Analysis Navigator for Discovery And Link Finding

A high-performance Python library and [Translator](https://ncats.nih.gov/translator)-compatible TRAPI server for fast path finding in large biomedical knowledge graphs.

## Features

- **Compressed Sparse Row (CSR)** graph representation for memory-efficient storage of 10M+ node, 38M+ edge graphs
- **Bidirectional search** for optimal path-finding performance
- **O(1) property lookups** via hash indexing
- **Predicate filtering** to reduce path explosion
- **Qualifier filtering** for advanced edge constraints (aspect, direction, mechanism)
- **Attribute constraints** on edges and nodes, including filtering edges by specific PubMed IDs
- **Subclass expansion** via Biolink Model Toolkit with configurable depth
- **Batch property enrichment** — enrich only final paths, not intermediate results
- **Diagnostic tools** to understand path counts and explosion
- **TRAPI 2.0 compatible** REST API with Plater-compatible endpoints, modelled
  with [`translator_tom`](https://github.com/NCATSTranslator/TRAPIObjectModeling)
- **Async query support** with callback URLs
- **Dehydrated mode** for lightweight responses that skip edge and node attribute enrichment
- **OpenTelemetry tracing** with Jaeger integration
- **Queue mode** — queries run in a pool of worker processes fed by a Redis Stream, with Prometheus metrics and a KEDA-ready scaling signal

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

# Install the optional node annotation dependency (biothings_annotator)
pip install -r requirements-annotate.txt
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

> **Upgrading to TRAPI 2.0 requires a rebuild.** Some of what 2.0 mandates is
> baked into the serialized graph, so a graph built by an earlier version
> cannot serve a conformant response. `CSRGraph.load_mmap` refuses such a
> graph outright with a `GraphFormatError` naming the rebuild, rather than
> starting up and serving edges with no `knowledge_level` / `agent_type`.
>
> Baked in at build time (a rebuild is the only way to change these):
> `knowledge_level` / `agent_type` on every edge, `Node.name` omitted when
> unknown, `Node.categories` defaulted to `biolink:NamedThing`,
> `RetrievalSource.upstream_resource_ids` omitted when empty, and the
> persisted `meta_kg.json` / `sri_testing_data.json`.
>
> **Graphs built before edge attributes were stored as JSON also need a
> rebuild.** Each edge's attributes are now stored as the JSON array a response
> carries, so the server can copy them into a response without decoding them;
> `load_mmap` refuses a graph that still stores them as msgpack.
>
> **So do graphs that store their edge IDs in LMDB.** Edge IDs are now a
> memory-mapped blob of their JSON (`edge_ids.bin` + `edge_id_offsets.npy`),
> shared by every worker, and `load_mmap` refuses a graph with
> `edge_ids.lmdb` instead. The same rebuild also sorts each edge's
> qualifiers by type, so their order no longer varies from build to build.
>
> Everything else 2.0 changed is computed per query and takes effect on
> deploy: binding shapes, `QEdge.constraints`, the `parameters` object and its
> timeout, the response envelope, and the remaining null / empty-container
> rules.

> **Rebuild to shrink memory if your edges carry `source_record_urls`.**
> A record URL is unique to its edge, and a graph built before this release
> kept it inside the interned source lists, making one list per edge: 2.8 GB
> of private memory in every process on the Translator graph.  The loader
> now stores the URLs per edge in `edge_source_urls.lmdb` and full
> responses are unchanged.  An old graph still loads and serves correctly;
> the load warns when its source pool is oversized.

### Annotating nodes at build time

Passing `annotate_nodes=True` resolves every node through the Translator
Annotator service ([`biothings_annotator`](https://github.com/biothings/biothings_annotator))
and stores whatever it returns on the node, so queries never pay for the
lookup:

```python
graph = build_graph_from_jsonl(
    edges_path="data/raw/edges.jsonl",
    nodes_path="data/raw/nodes.jsonl",
    annotate_nodes=True,
)
```

Annotations are stored as a single TRAPI node attribute with
`attribute_type_id: "biothings_annotations"` — the same shape the Annotator
service produces — and are returned with every node in the knowledge graph:

```json
{
  "id": "MONDO:0005148",
  "name": "type 2 diabetes mellitus",
  "attributes": [
    {
      "attribute_type_id": "biothings_annotations",
      "value": {"mondo": {"label": "type 2 diabetes mellitus"}}
    }
  ]
}
```

Only CURIE prefixes the Annotator knows (NCBIGene, CHEBI, MONDO, HP, ...) are
sent to it; everything else is skipped without a request. Nodes the Annotator
has nothing for are left untouched rather than given an empty attribute.

This requires network access and the optional dependency:

```bash
pip install -r requirements-annotate.txt
```

The Annotator fans each call out to the BioThings APIs (mygene.info,
mydisease.info, ...), and any of them can answer 500 for a while or for one
particular CURIE.  A whole-graph run survives both: each batch is retried
with backoff (`--annotate-attempts`, default 5: waits of 2, 4, 8, 16 s), a
batch that keeps failing is split in half until the CURIEs the service
rejects are isolated and left unannotated (logged, with a count at the
end), and a service that fails everything even one CURIE at a time aborts
the build with a clear error rather than quietly producing an unannotated
graph.  Give the build an annotation cache so an aborted run, or a later
rebuild of the same nodes, fetches only what it does not have yet:

```bash
gandalf-build --edges data/edges.jsonl --nodes data/nodes.jsonl --output graph_mmap/ \
    --annotate --annotation-cache data/annotations.jsonl
```

The cache is a JSON-lines file of every answer so far, including "nothing
available"; CURIEs the service failed on are not recorded and are tried
again next run.

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

### Constraining query edges

TRAPI 2.0 gathers every constraint on a query edge into one `constraints`
object. All constraints given must hold:

```python
"edges": {
    "e0": {
        "subject": "n0",
        "object": "n1",
        "predicates": ["biolink:affects"],
        "constraints": {
            "knowledge_level": {
                "behavior": "ALLOW",
                "values": ["knowledge_assertion"],
            },
            "agent_type": {"behavior": "DENY", "values": ["text_mining_agent"]},
            "sources": {
                "behavior": "ALLOW",
                "values": ["infores:ctd"],
                "primary_only": True,
            },
            "qualifiers": [
                {"biolink:object_aspect_qualifier": "activity"},
            ],
            "attributes": [
                {"id": "biolink:publications", "operator": "==", "value": [...]},
            ],
        },
    }
}
```

- `knowledge_level` / `agent_type` — allow or deny Biolink values on the bound
  edges. `ALLOW` needs at least one listed value to be present; `DENY` needs
  none of them to be.
- `sources` — the same allow/deny over the infores CURIEs in an edge's
  `sources`. `primary_only` narrows the check to the source whose role is
  `primary_knowledge_source`, so a constraint can ignore aggregators.
- `qualifiers` — a list of qualifier mappings. AND within one mapping, OR
  between them. Values expand through the Biolink hierarchy, so a query for a
  parent value also matches edges carrying a child value.
- `attributes` — attribute constraints, evaluated against the edge's
  attributes. Query nodes accept the same list directly under `constraints`.
  `"not": true` negates one.

Migrating from TRAPI 1.x: `qualifier_constraints` is now
`constraints.qualifiers` (and its `qualifier_set` list of
`qualifier_type_id`/`qualifier_value` pairs collapses into a single mapping),
and `attribute_constraints` is now `constraints.attributes`. Gandalf rejects
the old field names with a 400 rather than ignoring them, so a stale client
never silently receives unfiltered results.

#### Filtering edges by PubMed ID

Attribute values are often lists — `publications` above all — and every
operator except `===` is applied to each member, so `==` reads as "contains".
Filtering an edge down to specific PubMed IDs is therefore plain equality:

```python
"constraints": {
    "attributes": [
        {
            "id": "biolink:publications",
            "operator": "==",
            "value": ["PMID:23456789", "PMID:11111111"],
        }
    ]
}
```

Only edges citing at least one of those PMIDs survive. A list `value` means
"any of"; a single string constrains to one publication. Publication
identifiers are compared canonically, so `PMID:23456789`, `pubmed:23456789`,
`https://pubmed.ncbi.nlm.nih.gov/23456789` and the bare `23456789` all select
the same article — unlike the `matches` operator, which does a substring
regex and would also accept `PMID:234567890`.

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
| `GET` | `/node_degree/{curie}` | Total degree (in + out) of a node |
| `GET` | `/meta_knowledge_graph` | Meta KG with predicates, categories, and counts |
| `GET` | `/sri_testing_data` | Representative edges for SRI Testing Harness |
| `POST` | `/query` | Synchronous TRAPI query |
| `POST` | `/asyncquery` | Async TRAPI query with callback URL |
| `GET` | `/health` | Liveness: the process answers |
| `GET` | `/ready` | Readiness: the graph is open and, in queue mode, Redis answers (503 otherwise) |
| `GET` | `/metrics` | Prometheus metrics |
| `GET` | `/status` | Live status page: queue, workers, recent jobs, this pod (`/status.json` has the data) |

Both `/query` and `/asyncquery` accept a single optional query parameter:
- `?profile=true` — Emit per-stage timing diagnostics into `message.logs`

All other request configuration lives under the body's `parameters` object,
which TRAPI 2.0 defines for query-time settings that do not change what a query
means. The server repeats it back in the response, as the spec requires.

```json
{
  "message": { "query_graph": { ... } },
  "parameters": {
    "timeout": 60,
    "log_level": "INFO",
    "bypass_cache": false,
    "subclass": true,
    "subclass_depth": 1,
    "dehydrated": false,
    "filter_config": { "max_node_degree": 50 },
    "annotator_config": {}
  }
}
```

Responses never serialize a null, and never serialize an empty container for a
property whose schema forbids one (`Edge.qualifiers`,
`RetrievalSource.upstream_resource_ids`, `message.auxiliary_graphs`, `logs`, …)
— TRAPI 2.0 is OpenAPI 3.1 and dropped `nullable`, so an absent value is an
absent property. `message.results` is deliberately still `[]` when a query
matched nothing, which is what 2.0 asks for.

TRAPI's own parameters:

- `timeout` (number): Seconds the client is willing to wait. When the budget is
  spent the query stops and the response carries a `Timeout` status with the
  logs from the work done. A negative value disables the server's default
  timeout (`GANDALF_QUERY_TIMEOUT`). A value below `GANDALF_MIN_QUERY_TIMEOUT`
  is refused up front with HTTP 409, since the server knows it cannot answer
  that fast.
- `log_level` (string): Least critical level of logs to return — `ERROR`,
  `WARNING`, `INFO` or `DEBUG`. (This moved here from the request's top level
  in TRAPI 2.0.)
- `bypass_cache` (bool): Accepted and has no effect — gandalf answers from its
  own graph and holds no query cache.

Gandalf's own parameters:

- `subclass` (bool): Enable biolink subclass inference (default `true`)
- `subclass_depth` (int): Maximum `subclass_of` hops (default `1`)
- `dehydrated` (bool): Return the smallest useful response — edges carry only
  subject, object, predicate, `knowledge_level` and `agent_type`, with no
  attributes and no `sources` (auto-enabled for very large result sets).
  TRAPI 2.0 requires `sources`, so a dehydrated response is deliberately not
  schema-valid: the mode trades conformance for size, and rehydrating one
  (see `rehydrate`) restores a conformant response
- `rehydrate` (bool): When true, the server skips the graph lookup and **only** enriches the `knowledge_graph` already supplied in `message` — used to re-enrich a previously dehydrated response
- `filter_config` (object): Plugin-defined node filter settings (each NodeFilter plugin reads its own key)
- `annotator_config` (object): Per-request opt-in response-annotator settings (each key activates one annotator plugin)

## CLI Commands

```bash
# Build a CSR graph from JSONL node/edge files
gandalf-build --edges data/edges.jsonl --nodes data/nodes.jsonl --output data/graph_mmap/

# ... and add node annotations from the Translator Annotator service
gandalf-build --edges data/edges.jsonl --nodes data/nodes.jsonl \
    --output data/graph_mmap/ --annotate

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
| `GANDALF_QUEUE_URL` | _(empty)_ | Redis URL; set it to run queries in worker processes (see [Queue mode](#queue-mode-workers-and-autoscaling)) |
| `GANDALF_QUEUE_SOCKET_TIMEOUT_SECONDS` | `30` | Redis socket timeout; a `?socket_timeout=` on the URL overrides it. Blocking waits are sliced to stay under it |
| `GANDALF_QUEUE_CONNECT_TIMEOUT_SECONDS` | `5` | Redis connect timeout |
| `GANDALF_QUEUE_STREAM` | `gandalf:jobs` | Redis Stream the jobs go on |
| `GANDALF_QUEUE_GROUP` | `gandalf-workers` | Consumer group the workers read through |
| `GANDALF_QUEUE_DEAD_STREAM` | `gandalf:jobs:dead` | Where jobs over the delivery limit are moved |
| `GANDALF_QUEUE_MAX_DELIVERIES` | `2` | Attempts a job gets before it is dead-lettered |
| `GANDALF_QUEUE_CLAIM_IDLE_SECONDS` | `120` | A job idle this long is taken over as abandoned |
| `GANDALF_QUEUE_KEEPALIVE_SECONDS` | `30` | How often a running worker refreshes its job |
| `GANDALF_QUEUE_SYNC_MAX_WAIT_SECONDS` | `1800` | How long `/query` waits for a worker when the query has no timeout |
| `GANDALF_QUEUE_SYNC_GRACE_SECONDS` | `30` | Added to the query's timeout for the worker's own Timeout response to arrive |
| `GANDALF_RESULT_TTL_SECONDS` | `900` | How long an uncollected `/query` result lives in Redis |
| `GANDALF_RESULT_CHUNK_BYTES` | `67108864` | Largest value written to one Redis key |
| `GANDALF_HISTORY_MAXLEN` | `2000` | Finished jobs the status page's history keeps |
| `GANDALF_HEAP_TRIM_THRESHOLD_MB` | `512` | Return freed heap to the OS after a job (worker) or before a query (API) when the process's anonymous RSS is above this; 0 disables |
| `GANDALF_SLACK_WEBHOOK_URL` | _(empty)_ | Slack incoming-webhook URL; set it to post alerts to its channel |
| `GANDALF_SLACK_EVENTS` | all but `worker_recycled` | Comma-separated event kinds Slack receives, or `all` |
| `GANDALF_SLACK_ENVIRONMENT` | _(empty)_ | Label in front of every Slack title, e.g. `prod` |
| `GANDALF_SLACK_STATUS_URL` | `<GANDALF_SERVER_URL>/status` | Linked from every Slack message |
| `GANDALF_SLACK_THROTTLE_SECONDS` | `300` | Minimum gap between Slack messages of one noisy kind (`job_failed`, `job_retried`) |
| `GANDALF_MONITOR_INTERVAL_SECONDS` | `15` | How often the (single, elected) monitor looks |
| `GANDALF_ALERT_QUEUE_LAG_PER_WORKER` | `50` | Jobs waiting per live worker (at least one) at which the queue counts as backed up |
| `GANDALF_ALERT_QUEUE_BACKLOG_SECONDS` | `60` | How long the backlog must stay over that threshold before `queue_backlog` fires |
| `GANDALF_ALERT_STUCK_SECONDS` | `1800` | A job delivered and unfinished this long fires `queue_stuck` |
| `GANDALF_ALERT_LOG_MAXLEN` | `500` | Alerts the status page keeps |
| `GANDALF_WORKER_NAME` | `<hostname>-<pid>` | The worker's consumer name |
| `GANDALF_WORKER_MAX_JOBS` | `500` | A worker exits after this many jobs (0 = never) |
| `GANDALF_WORKER_METRICS_PORT` | `9100` | Worker Prometheus port (0 = off) |
| `GANDALF_WORKER_HEARTBEAT_FILE` | `/tmp/gandalf-worker-heartbeat` | Touched while the worker is alive, for a liveness probe |

### Search Tuning

| Variable | Default | Description |
|----------|---------|-------------|
| `GANDALF_LARGE_RESULT_THRESHOLD` | `50000` | Path count threshold for auto-dehydrated responses |
| `GANDALF_MAX_PATH_LIMIT` | `0` | Max intermediate paths during joins (0 = unlimited) |
| `GANDALF_DEBUG_PATHS_TSV` | _(empty)_ | File path to write debug TSV of reconstructed paths |

### TRAPI

| Variable | Default | Description |
|----------|---------|-------------|
| `GANDALF_QUERY_TIMEOUT` | `0` | Server default for `parameters.timeout`, in seconds (0 = no timeout) |
| `GANDALF_MIN_QUERY_TIMEOUT` | `1.0` | Shortest `parameters.timeout` the server accepts; below this it answers HTTP 409 |
| `GANDALF_DATA_RELEASE_VERSIONS` | _(empty)_ | JSON object of source data versions reported as `Response.data_release_versions`, e.g. `{"translator_kg": "2026_06_21"}` |
| `GANDALF_BIOLINK_VERSION` | `4.3.2` | Biolink Model version reported and used for predicate/qualifier expansion |

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

### Docker Compose

`compose.yml` runs Gandalf on Shepherd's Docker network (`shepherd_default`), so
it sends traces to Shepherd's Jaeger and is reachable from Shepherd and the
retriever at `http://gandalf:6429`. Start Shepherd first (or run
`docker network create shepherd_default` to run Gandalf on its own).

```bash
# Graph read from ./graph by default
GANDALF_GRAPH_DIR=/path/to/graph docker compose up --build
```

| Variable | Default | Description |
|----------|---------|-------------|
| `GANDALF_GRAPH_DIR` | `./graph` | Host directory mounted read-only at `/data/graph` |
| `SHEPHERD_NETWORK` | `shepherd_default` | External Docker network to join |
| `GANDALF_OTEL_ENABLED` | `true` | Set to `false` when no Jaeger is running |
| `GANDALF_JAEGER_HOST` / `GANDALF_JAEGER_PORT` | `http://jaeger` / `4317` | OTLP gRPC collector |

## Queue mode: workers and autoscaling

By default a Gandalf process runs every query itself.  With
`GANDALF_QUEUE_URL` set, the HTTP process instead validates each `/query`
and `/asyncquery`, puts it on a Redis Stream as a job, and separate worker
processes run it:

```bash
# the API (any number of replicas)
GANDALF_QUEUE_URL=redis://redis:6379/0 gunicorn gandalf.server:APP -c gunicorn.conf.py

# a worker (as many as the load needs; one job at a time each)
GANDALF_QUEUE_URL=redis://redis:6379/0 python -m gandalf.worker
```

Both open the same graph.  A `/query` waits for its job's result, which the
worker stores zstd-compressed in Redis and the API streams back (as is, to a
client that accepts zstd).  An `/asyncquery` returns immediately and the
worker POSTs the result to the callback itself.

**What a worker costs in memory.**  The graph is memory-mapped, so opening
it costs a worker no private memory at all: its arrays and LMDB stores live
in the page cache, shared by every process on the node that maps the same
files.  A worker's private memory is a ~100 MB Python baseline plus whatever
the *last large query* left behind: building a response allocates
hundreds of megabytes to gigabytes, and glibc keeps the freed pages for
reuse rather than returning them.  A worker therefore hands them back with
`malloc_trim` after each job once its anonymous RSS is above
`GANDALF_HEAP_TRIM_THRESHOLD_MB` (measured: a 373 MB response left 560 MB
behind, 200 MB after the trim, in 10 ms), and the API does the same before
each query.  Size a worker for one query's peak, not for the graph.

A worker that is large *before its first query* is loading something it
should not (the one case seen so far: a graph built before
`source_record_urls` were stored off the hot path, 2.8 GB of interned
source lists; a rebuild fixes it).  Every load logs where its private memory went, stage by
stage (`Private memory taken by the load: +N MB (...)`), and the status
page shows each worker's memory at start; the usual culprits are a graph
in a legacy format (`source_record_urls` kept in the interned source
lists, node data in `metadata.pkl` instead of `node_store.lmdb`, a missing
`rev_to_fwd.npy`), each of which the load warns about and a rebuild with
`gandalf-build` fixes, and the metadata JSONs, which only the API needs and
a worker no longer loads.  A graph whose edge IDs predate the memory-mapped
blob is refused outright with the same rebuild message.

**One query at a time, by design.**  A query is single-threaded Python, so
a worker uses one core and a second CPU would go unused; concurrent queries
in one process would share the GIL and, worse, add their memory peaks
together.  Concurrency is more workers: on one node they share the graph's
page cache, so each extra worker costs only its private memory above.

What the queue gives, beyond "more workers":

- **Nothing runs unbounded.**  A worker runs one query at a time, so its
  memory is one query's peak, and a burst waits in the stream instead of
  running all at once and getting the process OOM-killed.
- **A dead worker's job is not lost.**  Jobs are acknowledged only after
  their result is delivered.  One whose worker died is taken over by another
  worker (`GANDALF_QUEUE_CLAIM_IDLE_SECONDS`); one that has killed a worker
  twice (`GANDALF_QUEUE_MAX_DELIVERIES`) is moved to the dead-letter stream
  and answered with an `Error` response rather than run again.
- **The query's timeout covers the wait.**  `parameters.timeout` runs from
  when the API accepted the request; a job whose budget is spent in the
  queue is answered with a `Timeout` response without being run.
- **One scaling signal.**  The stream's lag (jobs no worker has taken) is
  what [KEDA](https://keda.sh)'s `redis-streams` scaler reads.
  `deploy/k8s/` has reference manifests for the API and worker Deployments
  and the `ScaledObject`, with sizing notes.

Metrics are on `/metrics` of the API (aggregated across gunicorn workers via
`PROMETHEUS_MULTIPROC_DIR`, which the Docker image sets) and on port 9100 of
each worker: request counts and latencies by route, jobs by outcome, job
duration, queue wait, result size, the worker's anonymous RSS, and the
stream's lag and pending counts.

### The status page

`GET /status` on any API pod is a live view of the whole deployment, built
from what the processes report into Redis rather than from Kubernetes,
Prometheus or Jaeger -- so it works for anyone who can reach the API:

- how many jobs are waiting for a worker (the scaling signal), running, and
  dead-lettered;
- every worker: idle or running what, for how long, jobs done, last
  outcome, memory, last seen;
- the recent jobs with outcome, queue wait, run time and size, and a
  jobs-per-minute chart with p50/p95 latencies over the last 5 minutes,
  hour and day;
- this pod's request counts by route and its memory.

`GET /status.json` is the data behind it.  The page is one self-contained
HTML file (`gandalf/static/status.html`) that polls `status.json` every
five seconds; nothing is fetched from outside the server.  Without a queue
it shows the in-process sections only.

### Alerts and Slack

Things worth a message are *events*.  Every event goes to the alert log
shown on `/status`; with `GANDALF_SLACK_WEBHOOK_URL` set to a Slack
[incoming webhook](https://api.slack.com/messaging/webhooks), the kinds in
`GANDALF_SLACK_EVENTS` are posted to the webhook's channel as well, with a
link back to the status page:

| Kind | Severity | When |
|---|---|---|
| `worker_started` | info | A worker joined the pool (a scale-up, a rollout, a restart) |
| `worker_stopped` | info | A worker left on SIGTERM (a scale-down or a rollout) |
| `worker_recycled` | info | A worker exited after its `GANDALF_WORKER_MAX_JOBS` (routine; off by default) |
| `worker_restarted` | critical | A worker came back under a name whose last instance never exited cleanly. Says how long it lived, what it was running, and the error it died with, or that it recorded none and was killed from outside (an OOM kill, a probe) |
| `worker_lost` | critical | A worker stopped reporting without a clean exit: an OOM kill, usually. Says what it was running |
| `job_retried` | warning | A job abandoned by a dead worker is being run again |
| `job_dead_lettered` | critical | A job ended two workers and was answered with an error instead of a third try |
| `job_failed` | warning | A query raised; the client got an Error response |
| `queue_backlog` / `queue_backlog_cleared` | warning / info | Jobs waiting for a worker stayed at or over `GANDALF_ALERT_QUEUE_LAG_PER_WORKER` × the live workers for `GANDALF_ALERT_QUEUE_BACKLOG_SECONDS`, then dropped to zero |
| `queue_stuck` | warning | A job has been with one worker longer than `GANDALF_ALERT_STUCK_SECONDS` |

A worker never dies of Redis trouble: a timeout, an outage or a restart
of Redis is logged and retried with backoff, and a job that was not
acknowledged is redelivered.  When the API cannot reach Redis, `/query`,
`/asyncquery` and `/status.json` answer 503 and `/ready` reports it.

Each event is emitted exactly once however many pods run.  The worker
events come from the worker that saw them.  The threshold events
(`worker_lost`, `queue_backlog`, `queue_stuck`) come from a monitor thread
that every API process runs but only one, elected through a Redis lock, is
active; if that pod dies another takes over within a few intervals.
`job_failed` and `job_retried` can come in bursts, so Slack gets at most
one of each per `GANDALF_SLACK_THROTTLE_SECONDS`, and the next message says
how many were suppressed.  Scaling itself shows up as `worker_started` and
`worker_stopped`, each with the pool size after the change.

`docker compose --profile queue up --build` runs the API, a Redis and one
worker locally with `GANDALF_QUEUE_URL=redis://redis:6379/0`.

The Redis-backed tests need a server and are deselected by default:

```bash
python -m pytest -m integration tests/test_queue_redis.py   # starts a local redis-server
GANDALF_TEST_REDIS_URL=redis://localhost:6379/0 python -m pytest -m integration tests/test_queue_redis.py
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
