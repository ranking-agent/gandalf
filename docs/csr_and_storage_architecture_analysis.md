# CSR Implementation and Property Storage Architecture Analysis

This document evaluates the trade-offs between scipy's CSR sparse matrix and Gandalf's hand-rolled CSR implementation for graph topology, and between LMDB and Elasticsearch for node/edge property storage. It captures the reasoning behind Gandalf's current architecture and the considerations for any future changes.

## Background

Gandalf represents a biomedical knowledge graph using a Compressed Sparse Row (CSR) data structure for topology and a separate storage layer for node and edge properties. The system is deployed as multiple read-only pods sharing a persistent volume claim (PVC), optimized for low memory overhead via memory-mapped file access.

The question under consideration: should Gandalf adopt scipy's `csr_matrix`/`csr_array` for graph topology, and should it use Elasticsearch instead of (or in addition to) LMDB for property storage?

---

## Part 1: Graph Topology — Hand-Rolled CSR vs. Scipy CSR

### Current Implementation

Gandalf's `CSRGraph` maintains two parallel CSR structures built from plain NumPy arrays:

**Forward CSR (outgoing edges):**
- `fwd_offsets` (int64): row pointers into the targets/predicates arrays for each node
- `fwd_targets` (int32): destination node indices
- `fwd_predicates` (int32): predicate IDs for each edge

**Reverse CSR (incoming edges):**
- `rev_offsets` (int64): row pointers for target nodes
- `rev_sources` (int32): source node indices
- `rev_predicates` (int32): predicate IDs
- `rev_to_fwd` (int32): maps reverse-CSR positions to forward-CSR positions for O(1) property lookup

These arrays are persisted as `.npy` files and loaded with `np.load(mmap_mode='r')`, meaning the OS pages them in from disk on demand. Edge properties are stored separately in an `EdgePropertyStore` with deduplicated qualifier and source pools, and cold-path attributes are stored in LMDB.

### Gandalf's Traversal Access Pattern

The core path-finding algorithm (`path_finder.py`) performs a vectorized 3-hop search:

1. **Forward hop 1**: `graph.neighbors(start_idx)` — row slice to get all outgoing neighbors
2. **Forward hop 2**: For each neighbor, another row slice to get second-hop candidates
3. **Backward hop 1**: `graph.incoming_neighbors(end_idx)` — row slice from the reverse CSR
4. **Intersection**: `np.isin()` to find nodes reachable from both directions
5. **Path assembly**: `np.column_stack()` to build complete paths

Additional operations include:
- Sequential iteration within a node's edge range (`for pos in range(start, end)`)
- Binary search within sorted rows via `np.searchsorted` for specific edge lookups
- Conditional property lookups when the query specifies property filters

These are all **row-based access patterns** — the fundamental strength of CSR. However, no sparse linear algebra operations (matrix-vector multiplication, eigenvalue decomposition, etc.) are performed.

### Scipy CSR: Pros

- **Less custom code to maintain.** Scipy's CSR is a mature, well-tested implementation. Adopting it would eliminate the need to maintain Gandalf's own CSR logic.
- **Access to sparse linear algebra.** If Gandalf were to add graph-level analytics (PageRank, reachability via matrix powers, spectral clustering, connected components), scipy provides these out of the box.
- **Ecosystem compatibility.** Scipy sparse matrices interoperate with scikit-learn, NetworkX, and other libraries, which could simplify future integrations.

### Scipy CSR: Cons

- **No memory-mapped loading.** Scipy sparse matrices are in-memory Python/NumPy objects. Loading via `scipy.sparse.load_npz` or pickle allocates the underlying arrays (`indptr`, `indices`, `data`) into RAM. There is no `mmap_mode` equivalent. This directly conflicts with Gandalf's deployment model of low-memory pods reading from shared PVCs, where the OS page cache manages the working set and cheap disk substitutes for expensive RAM.
- **Single value per entry.** A `csr_matrix` stores one scalar per non-zero entry. Gandalf requires multiple parallel arrays per edge (targets, predicates, reverse-to-forward mappings). Using scipy would require maintaining side arrays indexed by the same CSR positions, negating the "less code" benefit and introducing synchronization risk.
- **No guarantee of sorted indices within rows.** Gandalf relies on `np.searchsorted` over sorted target arrays for O(log(degree)) edge lookups on the hot path. Scipy does not guarantee sorted indices within rows; `.sort_indices()` must be called explicitly, and the caller must then reach directly into `.indices` — bypassing scipy's API.
- **No linear algebra payoff.** The primary value proposition of scipy's CSR — fast sparse matrix operations — is unused. Gandalf's traversal is fundamentally neighbor iteration, not linear algebra. The dependency cost is paid without using scipy's core strengths.
- **Additional dependency.** Scipy is a substantial library. Adding it to the dependency tree increases image size and build complexity for capabilities that are not currently needed.

### Recommendation

**Retain the hand-rolled CSR implementation.** The current design is purpose-built for Gandalf's access patterns, supports memory-mapped loading critical to the deployment model, and carries per-edge metadata that scipy cannot natively represent. Scipy CSR should be reconsidered only if graph-level linear algebra operations become a concrete requirement.

---

## Part 2: Property Storage — LMDB vs. Elasticsearch

### Current Implementation

Gandalf uses LMDB for cold-path property storage (node/edge attributes, publications). The LMDB database is opened in read-only mode and accessed via memory-mapped reads. Multiple pods share the same underlying PVC, and horizontal scaling is achieved by adding pods (or duplicating the PVC for additional pod clusters).

Property lookups occur during query traversal when the query specifies property filters. Otherwise, traversal operates purely on the CSR topology.

### LMDB: Pros

- **Extremely fast reads.** Memory-mapped, zero-copy access with no network hop. Point lookups by key are the fastest possible operation. This is critical for the conditional property lookups that occur mid-traversal on the hot path.
- **Operationally simple.** Embedded library, no separate process or cluster to manage. No JVM tuning, no shard rebalancing, no cluster health monitoring.
- **Predictable latency.** No garbage collection pauses, no cluster coordination, no query parsing overhead. Latency is determined by whether the page is in the OS cache.
- **ACID transactions with low overhead.** Even though Gandalf uses read-only mode, LMDB's transactional guarantees ensure consistent reads.
- **Read-only scaling model works well.** In read-only mode, the main operational pain points of LMDB (single-writer contention, copy-on-write B-tree growth) do not apply. Multiple pods read from the same PVC with zero coordination. Scaling further means duplicating the PVC — operationally trivial and far cheaper than managing a distributed system.
- **Disk-over-RAM economics.** Because LMDB is mmap-based, the OS page cache naturally keeps hot data in memory while cold data stays on disk. This allows Gandalf to trade cheap disk space for expensive RAM, keeping per-pod memory requirements low.

### LMDB: Cons

- **Single-machine storage.** Each LMDB instance lives on one volume. Scaling beyond a single machine's disk capacity requires manual sharding or PVC duplication.
- **No query language.** LMDB provides key-value lookups and ordered range scans only. Any filtering by property values must be implemented in application code.
- **No full-text search.** Searching within property values (e.g., "find all nodes where description contains 'clinical trial'") is not supported natively.
- **Write-heavy workloads can be challenging.** Not relevant to Gandalf's read-only use case, but worth noting for completeness — LMDB has a single-writer model and copy-on-write pages can cause database growth.

### Elasticsearch: Pros

- **Rich query capabilities.** Full-text search, fuzzy matching, range queries, aggregations, and faceted filtering over property values. If users ever need to express complex property-level search conditions within graph traversal (e.g., "find edges where source publication contains 'phase III' and year > 2020"), Elasticsearch handles this natively.
- **Horizontally scalable.** Adding nodes to the cluster increases both storage capacity and query throughput. No manual sharding required.
- **Good observability ecosystem.** Kibana provides built-in dashboards for debugging, query profiling, and monitoring.
- **Schema flexibility.** Dynamic mappings adapt to evolving property schemas without explicit migration steps.

### Elasticsearch: Cons

- **Significant per-query latency overhead.** Even a simple point lookup involves a network round-trip, query parsing, and Lucene segment access. For property lookups that occur mid-traversal, this latency multiplied across thousands of candidate paths would be a substantial performance regression compared to LMDB's zero-copy mmap reads.
- **Operational complexity.** Elasticsearch is a distributed system requiring cluster management: shard allocation, rebalancing, rolling upgrades, JVM heap tuning, and monitoring. This is a qualitative step up in operational burden from an embedded library.
- **Resource-heavy.** Substantial RAM and disk overhead for inverted indices, doc values, and segment metadata. The JVM heap alone typically requires multiple gigabytes per node.
- **Eventual consistency by default.** Writes are not immediately visible unless a refresh is forced, which hurts write throughput. Not a concern for read-only workloads, but relevant during graph rebuild/loading.
- **Undermines the disk-over-RAM model.** Elasticsearch requires data to be indexed and largely resident in memory for performant queries, conflicting with Gandalf's architecture of relying on the OS page cache and cheap disk.

### Recommendation

**Retain LMDB as the primary property store.** It is well-matched to Gandalf's access pattern (conditional point lookups during traversal), deployment model (read-only, disk-backed, shared PVC), and scaling approach (add pods/duplicate PVCs). Elasticsearch should only be introduced if there is a concrete requirement for full-text search or complex property-level filtering that cannot be reasonably implemented in application code. If such a requirement arises, a hybrid approach — LMDB for traversal-time lookups, Elasticsearch for higher-level property search — would be preferable to replacing LMDB entirely.

---

## Part 3: Architectural Coherence

The decisions above are not independent — they reinforce the same architectural principle:

**Gandalf is optimized for disk-backed, memory-mapped, read-only serving with minimal per-pod resource requirements.**

| Concern | Current Approach | Alternative | Why Current Wins |
|---|---|---|---|
| Graph topology | Hand-rolled NumPy CSR with `.npy` mmap | Scipy `csr_matrix` | Supports mmap, parallel edge arrays, sorted-row binary search |
| Property storage | LMDB (read-only, mmap) | Elasticsearch | Zero-copy reads, no network hop, no cluster ops |
| Horizontal scaling | Multiple pods on shared PVC; duplicate PVC for more capacity | ES cluster auto-sharding | Operationally trivial, cost-effective (disk vs. RAM) |
| Memory management | OS page cache manages working set | Application-level caching / JVM heap | Kernel is better at this; no tuning required |

Both scipy CSR and Elasticsearch would shift Gandalf toward a model that requires more RAM per pod, introduces external system dependencies, and adds operational complexity — all in exchange for capabilities (sparse linear algebra, full-text search) that the system does not currently use and has no concrete plans to use.

The right time to revisit these decisions is when a specific requirement demands capabilities that the current architecture cannot provide. Until then, the existing approach provides the best balance of performance, simplicity, and cost.
