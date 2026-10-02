# CLAUDE.MD

## Project Overview

GANDALF (Graph Analysis Navigator for Discovery And Link Finding) is a Python library for fast 3-hop path finding in large biomedical knowledge graphs. It's designed for the Translator project to efficiently query biological relationships (e.g., drug → protein → disease).

## Tech Stack

- **Python 3.10+** (supports 3.10-3.12; the floor comes from `translator_tom`)
- **FastAPI** - REST API server
- **NumPy** - Sparse matrix operations
- **BMT** (Biolink Model Toolkit) - Biolink ontology handling
- **translator_tom** - the Translator-wide TRAPI object model (request models, OpenAPI schema, and TypedDicts for internal typing)
- **pytest** - Testing framework
- **black** - Code formatting

## Installation

**Important:** An up-to-date `setuptools` (>=69) is required before installing
dependencies.  BMT transitively depends on legacy packages (`stringcase`,
`pytest-logging`) whose `setup.py` installs fail with older setuptools due to a
`distutils` `install_layout` bug.  Always upgrade setuptools first.

```bash
pip install -U pip setuptools wheel
pip install -e ".[dev]"
```

## Quick Commands

```bash
# Build graph from JSONL
gandalf-build --edges data/edges.jsonl --nodes data/nodes.jsonl --output graph_mmap/

# Build graph with node annotations from the Translator Annotator
gandalf-build --edges data/edges.jsonl --nodes data/nodes.jsonl --output graph_mmap/ --annotate

# Query paths
gandalf-query --graph graph_mmap/ --start "CHEBI:45783" --end "MONDO:0004979"

# Diagnose path explosion
gandalf-diagnose --graph graph_mmap/ --start ID1 --end ID2

# Run development server (port 6429)
python gandalf/main.py

# Run tests
pytest
pytest --cov=gandalf

# Run the tests that hit live external services (deselected by default)
pytest -m integration
```

## Project Structure

```
gandalf/
├── graph.py          # CSRGraph class (core data structure)
├── loader.py         # JSONL to graph conversion
├── node_annotations.py  # Build-time node annotation (biothings_annotator)
├── trapi.py          # TRAPI 2.0 envelope, query parameters, timeouts
├── search/           # Search & path finding (package)
│   ├── __init__.py       # Re-exports public API
│   ├── lookup.py         # Main TRAPI lookup entry point
│   ├── expanders.py      # PredicateExpander, QualifierExpander
│   ├── qualifiers.py     # Qualifier constraint matching
│   ├── edge_constraints.py  # QEdge.constraints parsing & matching
│   ├── query_edge.py     # Edge query with predicate/constraint filtering
│   ├── reconstruct.py    # Path reconstruction & join ordering
│   ├── path_arrays.py    # Compact numpy path representation
│   ├── path_finder.py    # 3-hop path finding functions
│   └── gc_utils.py       # GC monitoring & control
├── query_planner.py  # Query optimization
├── diagnostics.py    # Path explosion analysis
├── execute.py        # Query execution shared by the server and the worker
├── jobs.py           # Jobs: Redis Streams queue, result store, worker registry, history
├── worker.py         # Queue worker entry point (python -m gandalf.worker)
├── metrics.py        # Prometheus metrics for both processes
├── status.py         # /status.json snapshot: queue, workers, job history, this pod
├── notify.py         # Events, the alert log, Slack webhook, the elected monitor
├── static/status.html  # The /status page (self-contained; polls status.json)
├── server.py         # FastAPI application
└── main.py           # Dev server entry point
scripts/              # CLI entry points (gandalf-build, gandalf-query, gandalf-diagnose)
```

## Code Conventions

### Naming
- Classes: `PascalCase` (e.g., `CSRGraph`)
- Functions: `snake_case` (e.g., `find_3hop_paths_filtered`)
- Private functions: `_underscore_prefix`
- Constants: `UPPER_SNAKE_CASE`
- Biolink predicates: `"biolink:treats"` format

### Style
- Use type hints for function signatures
- Docstrings in Google/NumPy style with Args, Returns, Examples
- Black formatting (88-char line width)
- Prefer NumPy arrays for performance-critical code
- Use dictionary indexing for O(1) lookups

### Patterns
- Class methods for serialization (`CSRGraph.save_mmap()`, `CSRGraph.load_mmap()`)
- Bidirectional search for graph traversal optimization
- Generator/list comprehensions for data transformation
- Batch property enrichment (enrich only final paths, not intermediates)

## Architecture Notes

- **CSRGraph**: Compressed Sparse Row matrix representation for memory efficiency
- **Bidirectional search**: Expands from both start and end nodes to minimize traversal
- **Property storage**: Hash dicts keyed by `(src_idx, dst_idx, pred_idx)` tuples
- **Node indexing**: Original string IDs mapped to integer indices for NumPy operations

## The TRAPI object model (`translator_tom`)

Everything TRAPI-shaped comes from `translator_tom` (TOM), the Translator-wide
TRAPI object model, rather than being re-derived here.  It is used in three
ways, and the split between them is deliberate:

1. **Request models + OpenAPI** (`gandalf/models.py`): `TRAPIQuery` /
   `AsyncTRAPIQuery` subclass TOM's `Query` / `AsyncQuery`, and
   `QueryParameters` subclasses TOM's to add gandalf's own settings under the
   `additionalProperties` that 2.0 gives that object.  A request body is small,
   so full Pydantic validation there costs microseconds.
2. **TypedDicts for internals** (`translator_tom.model_dicts`): gandalf keeps
   building plain dicts and serializing them with orjson, but the dicts are
   *typed*.  A `TypedDict` is free at runtime, so this is static checking at
   almost no cost.  (`typing.cast` is not free, though: it is a real function
   call, so keep it out of per-result and per-path loops -- an annotated
   assignment types a value without one.)  mypy (which CI runs) catches
   binding-shape mistakes, nulls where 2.0 admits none, and misspelled
   properties.  Two gandalf-specific subclasses live in `gandalf/trapi.py`:
   `InFlightEdge` (an Edge plus the `_edge_id` / `_query_*` markers stripped
   before serialization) and `GandalfParametersDict`.
3. **Conformance tests** (`tests/test_trapi_conformance.py`): every response
   shape except dehydrated (see below) is validated against TOM *and* checked
   to round-trip byte-identically through `to_json()`.  The round-trip is the
   part that catches nulls and empty containers, since TOM serializes with
   `exclude_none` + `exclude_defaults`.

Pydantic validation is **off by default on both directions of every request**
and runs only when `validate_responses` is set: `_request_dict` and
`_trapi_response` are both gated on it, and the routes' `response_model` is
bound to `None` without it.  Two tests in
`tests/test_query_parameters.py::TestResponseFastPath` hold that line -- one
asserts the request is never run through `TRAPIQuery.model_validate`, the
other that no TOM model is constructed for a response at all.

**Dehydrated responses are deliberately not conformant.**  That mode exists to
make the payload as small as possible, and it omits `Edge.sources` as well as
the cold-path `attributes` -- sources are the largest thing left on an edge
once attributes are gone (~210 bytes, +36% on a dehydrated payload).  TRAPI 2.0
requires `sources`, so a dehydrated response is knowingly not schema-valid and
is excluded from the conformance tests; validating it would only assert
something gandalf has chosen not to do.  `gandalf.trapi.InFlightEdge` is the
type that records the exception (it makes `sources` optional), and
`tests/test_trapi_conformance.py::TestDehydratedResponsesTradeConformanceForSize`
pins the exact shape so the omission cannot quietly widen.  Everything else
still holds there, nulls included.

**Responses are deliberately NOT built as TOM models.**  Measured on 50k
results: constructing TOM models and serializing is ~4.4x the CPU and ~2.2x
the peak RSS of building dicts and calling `orjson.dumps` (and going through
`Response.from_dict` on an already-built dict is ~50x on serialization alone).
Gandalf's `large_result_threshold` is 10M paths, and `server._trapi_response` /
`_request_dict` exist precisely to keep per-result object materialization off
the hot path.  Use the TypedDicts there instead.

**Single-path results skip the general per-group loop.**  In
`_build_response`, a result built from one path whose subclass matches are all
identities (child == parent) needs no edge dedup and no composite inferred
edge, so it binds each column's edge directly.  On the real Translator graph
94-100% of xl/xxl results take this path, and it makes response building
1.4-1.5x faster.  The output must stay byte-identical to the general loop's,
including result, binding and knowledge-graph key order: the benchmark
fingerprint ignores order, so `tests/test_single_path_fast_path.py` runs each
query with `_SINGLE_PATH_FAST` off and on and compares the serialized bytes.
`scripts/benchmarks/fast_path_ab.py` does the same comparison, with timings,
on any graph and query set.

**Interned source lists hold nothing per-edge.**  `EdgePropertyStore`
keeps one Python list per distinct `sources` value and relies on there
being a few thousand of them.  `source_record_urls` are unique per edge,
so the loader splits them out (`gandalf.trapi.split_source_record_urls`)
into a sparse cold store, `edge_source_urls.lmdb` (`graph.source_urls_store`),
and `_full_edge` / `enrich_knowledge_graph` put them back
(`with_source_record_urls`) on a *copy* of the interned list, key last, so
output is byte-identical to a graph that kept them in the pool.  Anything
else per-edge that lands on a source must take the same route, never the
pool; `load_mmap` warns when the pool exceeds `_POOL_SIZE_WARNING`.

**Bindings are shared objects.**  A single-ID node binding, or a fast-path
edge binding, is one `{"ids": [x]}` dict per ID, reused by every result that
binds x: on a 2M-result query that is millions fewer objects to build,
garbage-collect and free.  So a binding must never be changed in place once
it is in a result; build a new one instead.
`tests/test_single_path_fast_path.py::test_single_id_bindings_are_shared`
pins the sharing.

**Edge attributes reach the server's responses undecoded.**  Each edge's
attributes are stored as the JSON array a response carries.  The server calls
`lookup(..., attributes_as_json=True)`, so a real edge's `attributes` is
`bytes` (the stored JSON) rather than a list.  Just before serializing,
`to_fragments` turns those into `orjson.Fragment`, which orjson copies into
its output as they are.  Decoding
them and encoding them again was the costliest step of a large full response.
Two rules follow:
- **Read or change an edge's attributes through
  `gandalf.trapi.edge_attributes(edge)`** (response annotator plugins
  included), which decodes them in place.  A Fragment cannot be read back, so
  nothing may need them after `to_fragments`.
- **Output stays byte-identical to decoding them:** the stored bytes are
  exactly what orjson makes of the decoded list, and the attributes key keeps
  its place in the edge.  `tests/test_edge_attributes_json.py` checks both
  modes serialize to the same bytes.

**Single-path results are written straight to JSON.**  The server also calls
`lookup(..., results_as_json=True)`.  Nearly every result on a large query
comes from one path whose subclass matches are identities (the single-path
fast path above); `_single_path_json` writes all of those at once rather
than building them one by one: it picks them out with NumPy masks, builds
each unique edge's KG entry once, and fills a JSON template fixed by the
query graph with each result's IDs, encoded once each.  Runs of consecutive
results become single `bytes` entries of `message["results"]`, among the
results the per-group loop still builds as dicts (multi-path, inferred edges,
edges without IDs, ALL/COLLATE queries); `to_fragments` hands them to orjson
as Fragments.  The rules:
- **Read or change results through `gandalf.trapi.response_results(response)`**
  (response annotator plugins included), which decodes them in place.
- **Output stays byte-identical to the dict path**, results and
  knowledge-graph key order included: the loop files its KG entries into
  dicts of its own, merged with the JSON results' entries in the order the
  loop alone would have filed them (`_file_in_first_seen_order`).  A change to
  the Result shape the loop builds must change the template in
  `_single_path_json` too; `tests/test_results_as_json.py` compares the two
  byte for byte across the query shapes.

Library callers get plain lists by default, and so does the server when
`validate_responses` is on, since FastAPI's validation needs them.

**GC stays paused until the response is freed.**  `lookup()` pauses the
cyclic GC while it runs (`gandalf.search.gc_utils.pause_gc`), and the server's
query handlers hold the same pause until the response has been serialized
and freed.  Were GC to resume as `lookup()` returns, the next allocation
would trigger a collection over every object of the response -- measured at
4-6s on the 1-2M-result queries, with the GIL held -- just before the
response is thrown away.  The response holds no reference cycles, so freeing
it needs no collection.  The pause is counted, so concurrent queries share
it; never call `gc.disable()`/`gc.enable()` around a lookup directly, or the
first query to finish turns GC back on under the others.

TOM requires Python >= 3.10, which is why gandalf's floor is 3.10.

## TRAPI version

The server implements **TRAPI 2.0** (`x-trapi.version` in
`gandalf/openapi-config.yaml`). Things 2.0 changed that the code depends on:

- A NodeBinding / EdgeBinding / PathBinding is one object with an `ids` array,
  not a list of objects each with an `id`.
- `knowledge_level` and `agent_type` are required top-level Edge properties.
  They are interned into `EdgePropertyStore` at build time alongside sources
  and qualifiers, so a graph built before this cannot be loaded — `load_mmap`
  refuses it with a `GraphFormatError` naming the rebuild.
- A QEdge's constraints all live in one `constraints` object
  (`gandalf/search/edge_constraints.py`), replacing `qualifier_constraints`
  and `attribute_constraints`. A QualifierSetConstraint is a
  `{qualifier_type_id: qualifier_value}` mapping.
- `log_level` and `bypass_cache` moved into `parameters`, which also gained
  `timeout`; the server must repeat `parameters` in its Response.
- `Response.logs` has `minItems: 1`, so an empty log list is omitted rather
  than sent as `[]`.

### Nulls and empty containers

2.0 is an `openapi: 3.1.2` document and dropped every `nullable: true`, so a
typed property does not admit null: **an absent value must be an absent
property**. The only exceptions are `Attribute.value` /
`AttributeConstraint.value` (no declared `type`) and anything under an
`additionalProperties: true`.

Empty containers are *per-property*, not a blanket rule, and there are three
cases -- see the `gandalf.trapi` module docstring, which lists them with the
affected properties:

1. `minItems`/`minProperties` of 1 on an **optional** property -> omit it when
   empty (`Message.auxiliary_graphs`, `Edge.qualifiers`,
   `RetrievalSource.upstream_resource_ids`, `Response.logs`, ...).
2. The same on a **required** property -> omitting does not help, so an empty
   value is a data problem to fix at its source. `Node.categories` is the one
   that bites: a KGX record with no category is normalized to
   `biolink:NamedThing` (`gandalf.biolink.NAMED_THING`).
3. No minimum at all -> empty is valid and must be left alone.
   `Message.results` goes further: 2.0 says an expected-but-empty result set
   SHOULD be `[]`, so emptying it out would be the violation.

Because of case 3 there is no blanket "strip falsy values" pass, and because a
response can carry millions of results there is no recursive walk of one
either: each value is kept valid where it is produced -- in `normalize.py` and
the loader at build time, and in `reconstruct.py` / `trapi.prune_edge` /
`trapi.finalize_response` per distinct node, edge and response.

The 1.x spellings of renamed request fields are rejected with a 400 rather
than ignored (see `gandalf/request_validation.py`): silently dropping a filter
a client asked for is worse than an error.  So is a query graph with no
`edges` to execute, which used to KeyError behind a 500.

### Build-time vs query-time

Some 2.0 requirements are satisfied in the serialized graph and only change
on a rebuild: `knowledge_level`/`agent_type` interning, `Node.name` /
`Node.categories` normalization, `RetrievalSource.upstream_resource_ids`
pruning (done in the loader so it covers both graph sources), the persisted
`meta_kg.json` / `sri_testing_data.json`, and the storage format of edge
attributes (JSON arrays in `edge_properties.lmdb`, recorded as
`edge_attributes_format` in `metadata.pkl`; `load_mmap` refuses a graph
without it).  Everything else is
computed per query.  When changing a serialization rule, check which side it
lands on -- a fix applied only in the loader will not reach an already-built
graph.

## Queue mode

With `GANDALF_QUEUE_URL` set, `server.py` does not run queries: it
validates them (`_prepare_query`, unchanged) and enqueues a
`gandalf.jobs.Job` on a Redis Stream; `gandalf.worker` processes consume
through a consumer group and run them.  Rules that follow:

- **One execution path.**  `gandalf.execute.run_query` /
  `execute_to_bytes` is what the in-process `/query`, the in-process
  `/asyncquery` background task and the worker all call.  A change to how
  a query runs goes there, never into one of the callers, so the bytes a
  client gets do not depend on which process produced them.
- **The job carries an absolute deadline.**  `Job.accepted_at` plus
  `budget` -- the worker rebuilds the `Deadline` with
  `Deadline.started_ago`, so queue wait counts against
  `parameters.timeout`, and a job that expires in the queue is answered
  with `timeout_response` without being run.
- **Ack after delivery, never before.**  `Worker.handle` acknowledges the
  stream entry only once the result is stored or the callback POSTed.  A
  dead worker's job is re-delivered (`XAUTOCLAIM` after
  `queue_claim_idle_seconds`; a running worker `touch`es its job every
  `queue_keepalive_seconds`), and one over `queue_max_deliveries` is
  dead-lettered and answered with `error_response` -- the poison case, a
  query that OOM-kills whatever runs it.
- **Results do not go through the stream.**  `/query` results go to the
  `ResultStore` (zstd, chunked under Redis's 512 MB value cap, with a TTL);
  async results go straight to the callback.
- **Blocking Redis reads are sliced.**  redis-py applies the socket
  timeout to blocking commands, so `XREADGROUP BLOCK` and `BLPOP` longer
  than it raise `TimeoutError`.  `JobQueue.receive` and `ResultStore.wait`
  wait in slices of at most half the socket timeout (`blocking_slice`)
  and loop; make any new blocking wait do the same.  The worker loop
  catches `redis.RedisError` and retries with backoff rather than exiting;
  the API turns it into a 503.
- **A worker explains its own restart.**  A live registry report under a
  starting worker's own name means the previous instance never exited
  cleanly; the start is reported as `worker_restarted` with what it was
  running and its last words (`WorkerRegistry.record_last_words`, written
  by `main()` on an unhandled exception; none means a kill from outside).
- **Private memory is retained heap, not the graph.**  Opening a graph
  costs no anonymous memory (mmap + LMDB); a large response does, and glibc
  keeps the freed pages.  `gandalf.metrics.trim_heap_if_large` hands them
  back (`malloc_trim`) after each worker job and before each API query,
  above `heap_trim_threshold_mb`.  Measure with `RssAnon`, not RSS.
  `CSRGraph.load_mmap` keeps a `gandalf.memory.MemoryLedger` of what each
  load stage took (`graph.load_memory`, logged at the end of the load) and
  warns about legacy file formats that cost private memory; a worker loads
  with `serve_metadata=False`, since only the API serves the meta knowledge
  graph and SRI testing data.
- **Workers report, jobs are recorded.**  `Worker.report` writes the
  worker's state to the `WorkerRegistry` (a hash with a TTL, refreshed on
  every poll and keepalive tick, so a dead worker drops off the page), and
  `Worker.handle` appends each finished job to the `JobHistory` (a capped
  stream).  `/status.json` is built from those plus `XINFO`/`XPENDING`;
  the page never needs more than the API's own port.
- **Events fire once.**  `gandalf.notify.Event`s go through
  `Notifier.emit` (alert log in Redis, Slack when configured).  The worker
  emits what only it sees (its start/stop, a job it retried, dead-lettered
  or that raised); the `Monitor` in the API emits what needs a view of the
  whole deployment (a lost worker, the backlog, a stuck job) and holds a
  Redis lock so one instance across all pods is active.  A clean worker
  exit calls `mark_worker_leaving` before `registry.forget`, or the
  monitor reports it lost.  New kinds go in `KINDS` with a severity.
- **Tests against a real Redis** live in `tests/test_queue_redis.py`
  under the `integration` marker (they start a local `redis-server`, or
  use `GANDALF_TEST_REDIS_URL`); CI runs them with a Redis service.  The
  hermetic suite covers everything that does not need Redis
  (`tests/test_health_metrics.py`).

## API Endpoints

- `POST /query` - Synchronous path finding (TRAPI format)
- `POST /asyncquery` - Async query with callback
- `GET /health`, `GET /ready` - liveness / readiness (503 until the graph is open and, in queue mode, Redis answers)
- `GET /metrics` - Prometheus metrics (`gandalf/metrics.py`)
- `GET /status`, `GET /status.json` - application-level status page and its data (`gandalf/status.py`)
- `GET /docs` - Swagger UI

## Data Formats

Uses Translator Knowledge Graph Exchange (KGX) format:
- Edges: JSONL with `subject`, `object`, `predicate`, and optional properties
- Nodes: JSONL with `id` and optional properties

## Testing

Tests use pytest.  Use `python -m pytest` (not bare `pytest`) so the
system-installed packages (numpy, etc.) are on the path:
```bash
python -m pytest
python -m pytest --cov=gandalf --cov-report=html
```

BMT is mocked in tests via `MockBMT` in `tests/conftest.py` to avoid the
external API call that `bmt.Toolkit()` makes on initialisation.  If you add new
predicates or qualifier values to the test fixtures, update the corresponding
lookup tables in `MockBMT`.

Tests marked `@pytest.mark.integration` call live external services and are
deselected by default (see `[tool.pytest.ini_options]` in `pyproject.toml`).
Run them explicitly with `python -m pytest -m integration`.

## Known Considerations

- Symmetric predicates are handled via BMT toolkit
- Node annotation (`--annotate`) is optional: it needs network access and the
  `biothings_annotator` package (`pip install -r requirements-annotate.txt`),
  which is not on PyPI and so is deliberately not a pyproject extra — PyPI
  rejects distributions with direct-URL dependencies
- CI/CD pipeline configured via `.github/workflows/ci.yml` (lint, test, typecheck)
- Dockerfile available for containerized deployment
- Server configurable via environment variables (see README.md)
- **Response compression stays on the event loop, on one thread (decided).**
  `ZstdCompressionMiddleware` compresses each response inline, so a worker
  can't accept or serve anything else while it compresses (about 4s for a
  2.3 GB response at ~600 MB/s, level 4).  Both alternatives were weighed and
  turned down:
  - *zstd's own threads* (`ZstdCompressor(threads=...)`): about 3x faster,
    but production runs 7 gunicorn workers on 8 CPUs, so extra compression
    threads would take cores from queries running in other workers.
  - *Compressing on a worker thread* (`anyio.to_thread.run_sync`; zstd
    releases the GIL): no extra CPU, and the worker stays responsive, but it
    could then start a new query while the old response is still in memory
    (the serialized body plus its compressed copy, several GB on the largest
    queries).  Those queries peak around 11 GB each, so overlapping two in
    one worker makes an OOM kill more likely.  Today's blocking loop means
    a worker never accepts a new query until the previous body is compressed
    and freed.
  Revisit only if workers stall in practice (e.g. health checks failing
  during large queries).  In that case take the worker-thread version with
  `threads=0`.

## Best practice:

- Use doctests liberally - these serve as both explanatory examples for humans and as unit tests
- For longer examples, write pytest tests
- always write pytest functional style rather than unittest OO style
- use modern pytest idioms, including @pytest.mark.parametrize to test for combinations of inputs
- NEVER write mock tests unless requested. I need to rely on tests to know if something breaks
- For tests that have external dependencies, you can do @pytest.mark.integration
- Do not "fix" issues by changing or weakening test conditions. Try harder, or ask questions if a test fails.
- Avoid try/except blocks, these can mask bugs
- Fail fast is a good principle
- Follow the DRY principle
- Avoid repeating chunks of code, but also avoid premature over-abstraction
- Declarative principles are favored
- Always use type hints, always document methods and classes
- DO NOT FORCE PUSH TO GITHUB
