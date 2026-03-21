# Plan: Move Node Properties, ID Mappings, and Edge IDs to LMDB

## Goal
Reduce RAM from ~25GB to ~5GB by moving the three largest Python-object consumers into LMDB (memory-mapped, shared across processes, paged by OS on demand).

## RAM savings estimate (136M edges, 10M nodes)

| Component | Current RAM | After LMDB | Savings |
|---|---|---|---|
| edge_ids (136M strings) | ~10-14 GB | ~0 GB | ~12 GB |
| node_properties (10M dicts) | ~3-6 GB | ~0 GB | ~4 GB |
| node_id_to_idx (10M entries) | ~1.5 GB | ~0 GB | ~1.5 GB |
| idx_to_node_id (10M entries) | ~1.5 GB | ~0 GB | ~1.5 GB |
| **Total savings** | | | **~19 GB** |

Remaining in RAM: `num_nodes` (int), `predicate_to_idx` / `id_to_predicate` (tiny, ~1000 entries), edge_property_pools (tiny pools + mmap'd indices), meta_kg JSON (~50MB).

## Performance justification

All access to these data structures at query time is **point lookups** on small result sets (tens to hundreds of items). LMDB point lookups are ~1μs warm. The full-iteration patterns (`build_meta_kg`, `_build_node_categories`, etc.) only happen at build time and the results are cached as `meta_kg.json`.

## Implementation

### Step 1: Create `NodeStore` class in `gandalf/node_store.py`

A single LMDB environment with 3 named sub-databases:
- **`id_to_idx`**: `node_id_string` → `4-byte uint32` (big-endian)
- **`idx_to_id`**: `4-byte uint32` → `node_id_string` (UTF-8)
- **`properties`**: `4-byte uint32` → `msgpack({name, categories, attributes})`

Public API:
```python
class NodeStore:
    def __init__(self, path, readonly=True)
    def get_node_idx(self, node_id: str) -> Optional[int]
    def get_node_id(self, node_idx: int) -> Optional[str]
    def get_properties(self, node_idx: int) -> dict
    def get_property(self, node_idx: int, key: str, default=None)
    def iter_id_to_idx(self) -> Iterator[Tuple[str, int]]  # for build_meta_kg
    def iter_properties(self) -> Iterator[Tuple[int, dict]]  # for build_meta_kg
    @staticmethod
    def build(path, node_id_to_idx, node_properties, commit_every=50_000)
    def close()
```

### Step 2: Add edge_ids to LMDB

New file `edge_ids.lmdb` with same pattern as existing LMDB store:
- Key: `4-byte uint32` (edge_idx, big-endian)
- Value: edge_id string (UTF-8 bytes, raw — no msgpack needed)

Thin `EdgeIdStore` class or extend `LMDBPropertyStore`.

### Step 3: Modify `CSRGraph.save_mmap()`

- Build `node_store.lmdb` via `NodeStore.build()` instead of including `node_id_to_idx` and `node_properties` in `metadata.pkl`
- Build `edge_ids.lmdb` instead of `edge_ids.pkl`
- `metadata.pkl` shrinks to just `{num_nodes, predicate_to_idx}`

### Step 4: Modify `CSRGraph.load_mmap()`

- Open `NodeStore(directory / "node_store.lmdb", readonly=True)`
- Store as `graph.node_store`
- Open edge_ids LMDB instead of loading pickle

### Step 5: Update accessor methods in `CSRGraph`

- `get_node_idx()` → delegates to `self.node_store.get_node_idx()`
- `get_node_id()` → delegates to `self.node_store.get_node_id()`
- `get_node_property()` → delegates to `self.node_store.get_property()`
- `get_all_node_properties()` → delegates to `self.node_store.get_properties()`
- `get_edge_id()` → delegates to edge_ids LMDB lookup
- Remove `self.node_id_to_idx`, `self.idx_to_node_id`, `self.node_properties` from load path
- Keep direct dict access in `__init__` constructor (build-time only)

### Step 6: Update `build_metadata()` / `build_meta_kg()`

- `_build_node_categories()`: iterate `node_store.iter_properties()` or range(num_nodes) with node_store lookups
- `_build_category_prefixes()`: iterate `node_store.iter_id_to_idx()`
- `_scan_edge_triples()`: use `node_store.get_node_id()` for the examples

### Step 7: Update direct dict access in `enrichment.py`

- `graph.node_id_to_idx.get(node_id)` → `graph.get_node_idx(node_id)`
- Already uses `graph.get_all_node_properties()` so that's fine

### Step 8: Backward compatibility in `load_mmap()`

- Check for `node_store.lmdb` first; fall back to `metadata.pkl` with old behavior
- Check for `edge_ids.lmdb` first; fall back to `edge_ids.pkl`
- This lets existing serialized graphs load without re-building

### Step 9: Update tests

- Ensure existing tests pass with the new LMDB-backed stores
- Test backward compatibility with pickle format

## Files to create/modify
- `gandalf/node_store.py` (NEW)
- `gandalf/graph.py` (modify save/load/accessors/metadata builders)
- `gandalf/enrichment.py` (replace direct dict access with method calls)
- `gandalf/loader.py` (minor: ensure constructor still works with dicts)
- Tests

## Files NOT modified
- `gandalf/search/lookup.py` — already uses `graph.get_node_idx()` / `graph.get_node_id()`
- `gandalf/search/path_finder.py` — same
- `gandalf/search/reconstruct.py` — same
- `gandalf/edge_properties.py` — unchanged
- `gandalf/config.py` — unchanged
- `gandalf/lmdb_store.py` — unchanged (edge attributes stay separate)
