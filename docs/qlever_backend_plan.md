# QLever Backend Integration Plan

## Goal

Add QLever as an alternate execution backend inside `gandalf` while keeping
`gandalf` as the source of truth for:

- TRAPI request semantics
- TRAPI response shape
- provenance/source normalization
- subclass behavior
- dehydrated mode behavior
- metadata endpoints
- Python compatibility and coding style

This is a backend swap, not a second product.

## Decisions Already Made

- When `gandalf` and `qlever_trapi` differ, follow `gandalf`.
- The first implementation should be as close to a minimal backend swap as
  possible.
- Leave out QLever-only features and utilities.
- Runtime loading must be read-only. Backends open already-built artifacts and
  do not rewrite them.
- Use a shared on-disk canonical property cache for node and edge records.
- Do not add a separate canonical JSONL artifact unless forced to later.
- Python implementation details should follow `gandalf`'s current standards and
  compatibility targets, not `qlever_trapi`'s newer baseline.

## Non-Goals

These are explicitly out of scope for the first implementation:

- A separate QLever FastAPI application
- Async job polling/status APIs beyond current `gandalf` behavior
- Local stack-management scripts
- Porting `find_paths.py`
- Reverse-traversal alias experiments
- Benchmarking/timing-table/report-generation tools
- QLever-only request features that `gandalf` does not already support

## High-Level Architecture

The final shape should be:

1. Shared ingest normalization logic converts raw KGX rows into canonical
   `gandalf` node and edge records.
2. Build steps write:
   - a shared canonical property cache
   - backend-specific execution artifacts
3. Runtime loads:
   - the selected backend implementation
   - the shared canonical property cache
   - precomputed metadata files
4. `gandalf.server` continues to expose the current `gandalf` API surface.

The main distinction is:

- **Shared canonical property cache**
  - normalized node and edge records
  - same logical contents regardless of backend
- **Backend-specific execution artifacts**
  - CSR arrays and stores for the CSR backend
  - RDF and QLever index files for the QLever backend

## Shared Canonical Record Model

`gandalf` already defines the semantics we want, but not yet as a single
explicit record layer.

Today:

- node payload shaping is implicit in `gandalf.loader`
- edge payload shaping is split across:
  - CSR arrays for `subject`, `object`, `predicate`
  - `EdgePropertyStore` for `sources` and `qualifiers`
  - `LMDBPropertyStore` for cold-path `attributes`
  - edge ID storage

We will make this explicit.

### Canonical node record

The canonical node record should match current `gandalf` behavior:

```python
{
    "id": "CURIE",
    "name": "...",
    "categories": [...],
    "attributes": [...],
}
```

Notes:

- `categories` should use the same source fields and defaults as current
  `gandalf`.
- node attributes should be built exactly the way
  `_extract_node_attributes()` currently does.

### Canonical edge record

The canonical edge record should match current `gandalf` behavior:

```python
{
    "id": "edge-id-or-none",
    "subject": "CURIE",
    "predicate": "biolink:...",
    "object": "CURIE",
    "sources": [...],
    "qualifiers": [...],
    "attributes": [...],
}
```

Notes:

- `sources` must use `gandalf` normalization exactly.
- `qualifiers` must use current `gandalf` extraction semantics.
- `attributes` must be the current `gandalf` cold-path TRAPI attribute list.

## Shared Provenance Rules

This is one of the most important parity areas.

### Current `gandalf` behavior

`gandalf.loader._extract_sources()` currently:

- normalizes modern KGX `sources`
- normalizes legacy `primary_knowledge_source` /
  `aggregator_knowledge_source` forms
- guarantees `upstream_resource_ids`
- prepends `settings.infores` as an
  `aggregator_knowledge_source`

Tests already assert this behavior.

### Required QLever behavior

The QLever backend must not reconstruct sources using different service-layer
rules. Instead:

- shared normalization code should build canonical edge `sources`
- the shared property cache should persist those normalized `sources`
- the QLever backend should read those exact normalized `sources`

This means:

- regular edge provenance should come from the canonical property cache
- the QLever backend should not append its own service provenance for regular
  edges
- inferred subclass edges should continue to synthesize provenance exactly the
  way current `gandalf` does

### Important clarification

The parity target is **logical equality of decoded records**, not byte-for-byte
LMDB identity across separate builds.

## Shared Canonical Property Cache

Add a new shared LMDB-backed property cache module under `gandalf`.

It should be adapted from `qlever_trapi`'s property cache approach, but store
canonical `gandalf` records, not raw KGX rows.

### Required behavior

- persisted to disk during build
- opened read-only at runtime
- no rewrite during runtime load
- stores complete normalized node and edge records
- supports point lookups and batched lookups by resource ID / edge ID

### Suggested layout

- named DB `nodes`
- named DB `edges`
- named DB `meta`

Suggested keys:

- nodes: node CURIE
- edges: edge ID if present; otherwise a stable generated backend-independent
  edge key

The edge key strategy must be stable and deterministic across builds from the
same input. If the input always contains an edge `id`, use that. If not, define
one deterministic fallback and use it everywhere.

### Important note

This shared canonical property cache is separate from CSR's current
`LMDBPropertyStore`, which stores only cold-path edge attributes by CSR edge
index. The CSR-specific store remains for performance.

## Backend-Specific Artifacts

### CSR backend

Keep existing CSR artifact behavior:

- CSR arrays
- `EdgePropertyStore`
- CSR-aligned edge IDs storage
- CSR cold-path attribute LMDB
- precomputed metadata JSON files

Add:

- the shared canonical property cache

### QLever backend

Build:

- RDF export
- QLever index
- the shared canonical property cache
- precomputed metadata JSON files
- a small manifest file describing the backend artifact paths

## Artifact Layout

For QLever output directories, use a self-describing layout under the user
provided output path.

Suggested shape:

```text
output/
├── backend.json
├── property_cache.lmdb/
├── meta_kg.json
├── sri_testing_data.json
├── graph_metadata.json
├── rdf/
│   └── dataset.nt.zst
└── qlever/
    └── dataset/
        ├── dataset.index.spo
        ├── dataset.settings.json
        └── ...
```

For CSR output, keep current layout and add:

```text
output/
├── property_cache.lmdb/
├── meta_kg.json
├── sri_testing_data.json
├── graph_metadata.json
└── existing CSR files...
```

CSR does not need a new manifest immediately if current format detection is
simple enough, but adding one later is fine if it simplifies backend loading.

## Build Flow

## Phase 1: Shared normalization layer

Create shared normalization helpers under `gandalf`, for example:

- `gandalf/canonical_records.py`
- `gandalf/source_normalization.py`

Responsibilities:

- normalize node records
- normalize edge records
- normalize sources
- normalize qualifiers
- normalize attributes

This code should be extracted from existing `gandalf.loader` logic, not copied
from `qlever_trapi`.

### Expected functions

- `normalize_node_record(raw_node: dict) -> dict`
- `normalize_edge_record(raw_edge: dict, infores: str) -> dict`
- `normalize_sources(raw_edge: dict, infores: str) -> list[dict]`
- `extract_edge_qualifiers(raw_edge: dict) -> list[dict]`
- `extract_edge_attributes(raw_edge: dict) -> list[dict]`
- `extract_node_attributes(raw_node: dict) -> list[dict]`

## Phase 2: Shared canonical property cache

Add a new module, for example:

- `gandalf/property_cache.py`

Responsibilities:

- build the canonical property cache from canonical node and edge records
- read it back at runtime
- support batched node and edge fetches

This module should be independent of both CSR internals and QLever internals.

## Phase 3: Refactor CSR builder to use canonical records

Refactor [gandalf/loader.py](/Users/bizon/Projects/Dogsled/gandalf/gandalf/loader.py):

- build canonical node records from `nodes.jsonl`
- build canonical edge records from `edges.jsonl`
- continue writing current CSR-specific artifacts
- additionally write the shared canonical property cache

Important:

- hot-path CSR structures should still be written from canonical edge records
- cold-path CSR attribute LMDB should still be written from canonical edge
  records
- no CSR query behavior should change

## Phase 4: Add QLever build path

Extend [scripts/build_graph.py](/Users/bizon/Projects/Dogsled/gandalf/scripts/build_graph.py)
with `--backend {csr,qlever}`.

### CSR mode

Preserve current behavior plus shared canonical property cache output.

### QLever mode

Use the same raw inputs:

- `--nodes nodes.jsonl`
- `--edges edges.jsonl`
- `--output <dir>`

Build steps:

1. normalize raw rows into canonical records
2. write shared canonical property cache
3. export RDF from canonical records
4. run `qlever index`
5. compute and write metadata JSON files
6. write backend manifest

Do not add tarball-only build behavior in the first pass.

## QLever RDF Export

Port only the RDF export logic needed for `gandalf` semantics from
`qlever_trapi`.

Suggested module:

- `gandalf/backends/qlever/rdf_export.py`

Responsibilities:

- write node triples
- write edge reification triples
- write class and predicate hierarchy triples needed by the transpiler
- write attribute/value triples for normalized records

Explicitly leave out:

- reverse traversal alias experiments
- pathfinder-specific query optimizations
- benchmarking hooks

## Metadata Build

Do not make the QLever runtime derive metadata live from QLever by default.

Instead, build these offline during artifact generation:

- `meta_kg.json`
- `sri_testing_data.json`
- `graph_metadata.json`

These files must follow current `gandalf` semantics.

Where possible, reuse existing `gandalf.graph` metadata-building logic or
extract backend-neutral metadata builders from it.

## Runtime Backend Abstraction

Add a small internal backend interface, for example under:

- `gandalf/backends/base.py`
- `gandalf/backends/csr_backend.py`
- `gandalf/backends/qlever_backend.py`
- `gandalf/backends/load_backend.py`

This abstraction should be minimal and should exist only to keep
`gandalf.server` and `gandalf.search.lookup` clean.

### Required runtime capabilities

- execute a TRAPI lookup
- return metadata payload
- return meta knowledge graph payload
- return SRI testing data payload
- close resources on shutdown

## CSR Lookup Extraction

Move current CSR lookup logic out of
[gandalf/search/lookup.py](/Users/bizon/Projects/Dogsled/gandalf/gandalf/search/lookup.py)
into a CSR-specific implementation module.

Keep the public `lookup(...)` entry point stable.

The dispatcher should choose behavior based on the backend/runtime object.

## QLever Query Engine Port

Port only the internal QLever query machinery needed to satisfy current
`gandalf` behavior.

Suggested modules:

- `gandalf/backends/qlever/request_normalization.py`
- `gandalf/backends/qlever/sparql_builder.py`
- `gandalf/backends/qlever/qlever_client.py`
- `gandalf/backends/qlever/result_assembly.py`
- `gandalf/backends/qlever/backend.py`

### Keep

- TRAPI query normalization for the `gandalf` subset
- SPARQL transpilation
- predicate handling
- qualifier handling
- subclass-aware pattern generation
- streaming result consumption for large queries
- knowledge graph and result assembly

### Remove or defer

- QLever-only HTTP envelope behavior
- QLever-only request features
- live metadata API logic as primary path
- alternate async API behavior

## Query Semantics: `gandalf` Wins

Whenever there is a mismatch, the QLever backend should be adapted to the CSR
behavior, not the other way around.

### Predicate expansion

Reuse [gandalf/search/expanders.py](/Users/bizon/Projects/Dogsled/gandalf/gandalf/search/expanders.py)
where possible for:

- inverse handling
- symmetric handling
- descendant expansion
- wildcard `related_to` behavior

The goal is one semantic implementation, not two drifting ones.

### Qualifier expansion

Reuse the current `gandalf` qualifier expansion logic where possible.

### Subclass behavior

The QLever backend must match current `gandalf` behavior for:

- subclass graph rewriting
- direct-edge precedence over subclass-expanded edges
- inferred edge synthesis
- `auxiliary_graphs`
- inferred edge provenance

If `qlever_trapi` currently differs, it should be changed to match `gandalf`.

### Dehydrated mode

The QLever backend must match current `gandalf` dehydrated behavior:

- keep only lightweight node payloads
- use the same category trimming logic
- keep edge payloads minimal in the same way

## Server Integration

Extend [gandalf/config.py](/Users/bizon/Projects/Dogsled/gandalf/gandalf/config.py)
with only the minimal QLever runtime settings:

- backend type or format
- QLever host
- QLever port
- optional access token

Extend [gandalf/server.py](/Users/bizon/Projects/Dogsled/gandalf/gandalf/server.py)
to load either:

- a CSR backend runtime
- a QLever backend runtime

The public endpoint behavior should remain unchanged:

- `POST /query`
- `POST /asyncquery`
- `GET /metadata`
- `GET /meta_knowledge_graph`
- `GET /sri_testing_data`

No alternate QLever FastAPI app should be introduced.

## Manifest and Backend Detection

For the QLever backend, add a manifest file such as `backend.json`.

Suggested fields:

- `backend`: `"qlever"`
- `dataset_name`
- `qlever_dataset_base`
- `property_cache_path`
- `meta_kg_path`
- `sri_testing_data_path`
- `graph_metadata_path`
- build flags relevant to runtime interpretation

This gives `gandalf` a single source of truth for loading backend artifacts.

## Tests

## Preserve all existing CSR tests

The first requirement is that all current `gandalf` tests still pass without
loosening any assertions.

## Add shared normalization tests

Create tests for:

- canonical node record normalization
- canonical edge record normalization
- source normalization parity with current loader behavior
- qualifier extraction parity
- attribute extraction parity

These tests should use the current `gandalf` fixture data.

## Add shared property cache tests

Create tests for:

- build/read round-trip of canonical node records
- build/read round-trip of canonical edge records
- batched fetch behavior
- provenance order and contents matching current `gandalf`

## Port selected QLever unit tests

Port and adapt the pure logic tests from `qlever_trapi` for:

- SPARQL generation
- predicate handling
- inverse handling
- qualifier constraint translation
- subclass rewrite shapes

Where `qlever_trapi` behavior differs from `gandalf`, rewrite the expected
assertions to match `gandalf`.

## Add backend parity tests

Add a common test matrix that executes supported TRAPI queries against:

- CSR backend
- QLever backend using a fake/stub QLever HTTP server

Compare:

- `message.query_graph`
- `message.knowledge_graph`
- `message.results`
- `message.auxiliary_graphs`

Root-level `logs` should also be present in both.

The parity target is current `gandalf` behavior.

## Add one real-QLever integration test

Add a small optional integration test that runs only when the `qlever` binary
is installed.

This test should verify:

- RDF export
- index creation
- QLever query execution
- shared property cache hydration
- final TRAPI response shape

Mark it clearly as an integration test.

## Python Compatibility

The QLever port must follow `gandalf`'s Python standards and compatibility
targets.

That means:

- use `gandalf`'s typing style
- avoid introducing syntax that exceeds the current supported baseline unless
  the project baseline is intentionally changed
- follow `gandalf` naming and module organization style

The first implementation should adapt the imported QLever code to `gandalf`,
not raise the repo's expectations to match `qlever_trapi`.

## Suggested File Additions

Likely new files:

- `gandalf/canonical_records.py`
- `gandalf/property_cache.py`
- `gandalf/backends/base.py`
- `gandalf/backends/load_backend.py`
- `gandalf/backends/csr_backend.py`
- `gandalf/backends/qlever/backend.py`
- `gandalf/backends/qlever/rdf_export.py`
- `gandalf/backends/qlever/request_normalization.py`
- `gandalf/backends/qlever/sparql_builder.py`
- `gandalf/backends/qlever/qlever_client.py`
- `gandalf/backends/qlever/result_assembly.py`
- `docs/qlever_backend_plan.md`

Likely modified files:

- `gandalf/loader.py`
- `gandalf/search/lookup.py`
- `gandalf/server.py`
- `gandalf/config.py`
- `gandalf/__init__.py`
- `scripts/build_graph.py`
- tests

## Implementation Order

1. Extract shared normalization code from current `gandalf.loader`.
2. Add shared canonical property cache and tests.
3. Refactor CSR build to use shared normalization and write the shared cache.
4. Extract CSR lookup into backend-specific code without behavior change.
5. Add backend runtime abstraction and backend loading.
6. Port QLever RDF export and build path.
7. Port QLever query execution internals.
8. Align QLever response assembly to `gandalf` semantics.
9. Integrate QLever backend into `gandalf.server`.
10. Add parity and integration tests.

## Success Criteria

The implementation is successful when:

- all current CSR behavior remains unchanged
- the QLever backend can be built from the same `nodes.jsonl` and `edges.jsonl`
  inputs
- the QLever backend serves the same `gandalf` API surface
- supported queries produce `gandalf`-compatible responses
- source/provenance behavior matches current `gandalf`
- runtime loads are read-only for both backends
