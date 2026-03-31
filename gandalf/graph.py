"""Main Gandalf CSR Graph class."""

import json
import logging
import pickle
import shutil
import struct
import time
from pathlib import Path
from typing import Literal, Optional, Union

import msgpack
import numpy as np

from gandalf.config import settings
from gandalf.lmdb_store import LMDBPropertyStore
from gandalf.node_store import NodeStore

logger = logging.getLogger(__name__)


def _load_npy(path: Path, mmap_mode: Literal["r+", "r", "w+", "c"] = "r") -> np.ndarray:
    """Load a .npy file, optionally copying into RAM instead of memory-mapping."""
    arr: np.ndarray = np.load(path, mmap_mode=mmap_mode)
    if settings.load_mmaps_into_memory:
        arr = np.array(arr)
    return arr


class EdgePropertyStore:
    """Memory-efficient storage for qualifier and source dedup via interning.

    Qualifiers and sources are the "hot path" data — accessed during
    traversal for every predicate-matching edge. They have very high
    deduplication ratios (~10K unique qualifier combos, ~50 unique source
    configs) so the pools are tiny and shared across workers via fork COW.

    The wrapper dicts returned by _get_props() contain only references to
    long-lived pool objects, so they are freed by refcount (no GC cycles).

    Other "cold path" data (attributes, which include publications) lives
    in LMDBPropertyStore (disk-backed, accessed only during response enrichment).
    """

    __slots__ = (
        "_sources_pool",
        "_quals_pool",
        "_sources_idx",
        "_quals_idx",
    )

    def __init__(self):
        self._sources_pool = []
        self._quals_pool = []
        self._sources_idx = None
        self._quals_idx = None

    @staticmethod
    def _make_hashable(obj):
        """Convert a JSON-compatible value to a hashable key for interning."""
        if isinstance(obj, dict):
            return tuple(
                sorted((k, EdgePropertyStore._make_hashable(v)) for k, v in obj.items())
            )
        elif isinstance(obj, (list, tuple)):
            return tuple(EdgePropertyStore._make_hashable(item) for item in obj)
        return obj

    @classmethod
    def from_property_list(cls, props_list):
        """Build an EdgePropertyStore from a list of property dicts.

        Args:
            props_list: List of dicts, each with keys like 'sources',
                        'qualifiers'. Only these two fields are stored;
                        publications/attributes belong in LMDBPropertyStore.
        """
        store = cls()
        n = len(props_list)

        sources_intern = {}
        quals_intern = {}

        sources_indices = np.empty(n, dtype=np.int32)
        quals_indices = np.empty(n, dtype=np.int32)

        for i, props in enumerate(props_list):
            # Intern sources
            sources = props.get("sources", [])
            sources_key = cls._make_hashable(sources)
            if sources_key not in sources_intern:
                sources_intern[sources_key] = len(store._sources_pool)
                store._sources_pool.append(sources)
            sources_indices[i] = sources_intern[sources_key]

            # Intern qualifiers
            quals = props.get("qualifiers", [])
            quals_key = cls._make_hashable(quals)
            if quals_key not in quals_intern:
                quals_intern[quals_key] = len(store._quals_pool)
                store._quals_pool.append(quals)
            quals_indices[i] = quals_intern[quals_key]

        store._sources_idx = sources_indices
        store._quals_idx = quals_indices

        return store

    def __len__(self):
        if self._quals_idx is None:
            return 0
        return len(self._quals_idx)

    def __getitem__(self, key):
        if isinstance(key, slice):
            indices = range(*key.indices(len(self)))
            return [self._get_props(i) for i in indices]
        return self._get_props(key)

    def _get_props(self, idx):
        """Get qualifier and source properties for an edge at the given index.

        Returns a dict with pool references (no new allocations beyond the
        wrapper dict itself, which is cycle-free and freed by refcount).
        """
        return {
            "sources": self._sources_pool[self._sources_idx[idx]],
            "qualifiers": self._quals_pool[self._quals_idx[idx]],
        }

    def get_qualifiers(self, idx):
        """Get just the qualifiers for an edge. Zero-alloc pool reference."""
        return self._quals_pool[self._quals_idx[idx]]

    def get_sources(self, idx):
        """Get just the sources for an edge. Zero-alloc pool reference."""
        return self._sources_pool[self._sources_idx[idx]]

    def get_field(self, idx, key, default=None):
        """Get a single field value without creating a full dict."""
        if key == "sources":
            return self._sources_pool[self._sources_idx[idx]]
        elif key == "qualifiers":
            return self._quals_pool[self._quals_idx[idx]]
        return default

    def dedup_stats(self):
        """Return statistics about deduplication effectiveness."""
        n = len(self)
        return {
            "total_edges": n,
            "unique_sources": len(self._sources_pool),
            "unique_qualifiers": len(self._quals_pool),
        }

    def save_mmap(self, directory: Path):
        """Save to directory as mmap-friendly files."""
        np.save(directory / "edge_sources_idx.npy", self._sources_idx)
        np.save(directory / "edge_quals_idx.npy", self._quals_idx)

        pools = {
            "sources_pool": self._sources_pool,
            "quals_pool": self._quals_pool,
        }
        with open(directory / "edge_property_pools.pkl", "wb") as f:
            pickle.dump(pools, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load_mmap(cls, directory: Path, mmap_mode: Literal["r+", "r", "w+", "c"] = "r"):
        """Load from directory, memory-mapping the index arrays."""
        store = cls()

        store._sources_idx = _load_npy(
            directory / "edge_sources_idx.npy", mmap_mode=mmap_mode
        )
        store._quals_idx = _load_npy(
            directory / "edge_quals_idx.npy", mmap_mode=mmap_mode
        )

        with open(directory / "edge_property_pools.pkl", "rb") as f:
            pools = pickle.load(f)

        store._sources_pool = pools["sources_pool"]
        store._quals_pool = pools["quals_pool"]

        return store


class EdgePropertyStoreBuilder:
    """Incrementally builds an EdgePropertyStore one edge at a time.

    Pre-allocates numpy index arrays (edge count known from pass 1),
    fills them during pass 2 streaming, and supports reorder() for
    CSR sort alignment.
    """

    def __init__(self, num_edges):
        self._sources_pool = []
        self._quals_pool = []
        self._sources_intern = {}
        self._quals_intern = {}
        self._sources_idx = np.empty(num_edges, dtype=np.int32)
        self._quals_idx = np.empty(num_edges, dtype=np.int32)

    def add(self, pos, props):
        """Intern one edge's qualifier and source properties at position pos."""
        # Intern sources
        sources = props.get("sources", [])
        sources_key = EdgePropertyStore._make_hashable(sources)
        if sources_key not in self._sources_intern:
            self._sources_intern[sources_key] = len(self._sources_pool)
            self._sources_pool.append(sources)
        self._sources_idx[pos] = self._sources_intern[sources_key]

        # Intern qualifiers
        quals = props.get("qualifiers", [])
        quals_key = EdgePropertyStore._make_hashable(quals)
        if quals_key not in self._quals_intern:
            self._quals_intern[quals_key] = len(self._quals_pool)
            self._quals_pool.append(quals)
        self._quals_idx[pos] = self._quals_intern[quals_key]

    def reorder(self, permutation):
        """Reorder index arrays to match CSR sort order."""
        self._sources_idx = self._sources_idx[permutation]
        self._quals_idx = self._quals_idx[permutation]

    def build(self):
        """Finalize to an EdgePropertyStore."""
        store = EdgePropertyStore()
        store._sources_pool = self._sources_pool
        store._quals_pool = self._quals_pool
        store._sources_idx = self._sources_idx
        store._quals_idx = self._quals_idx
        # Free intern dicts
        self._sources_intern = None
        self._quals_intern = None
        return store


class CSRGraph:
    """
    Compressed Sparse Row graph representation for fast neighbor lookups.

    Maintains two CSR structures:
    - Forward: node -> outgoing edges (who does this node point to?)
    - Reverse: node -> incoming edges (who points to this node?)

    Edge properties are stored in two tiers:
    - Hot path (qualifiers, sources): in-memory dedup store (EdgePropertyStore)
    - Cold path (attributes, including publications): disk-backed LMDB (LMDBPropertyStore)
    """

    def __init__(
        self,
        num_nodes,
        edges,
        edge_predicates,
        node_id_to_idx,
        predicate_to_idx,
        edge_properties=None,
        node_properties=None,
    ):
        """
        Args:
            num_nodes: Total number of unique nodes
            edges: List of (src_idx, dst_idx) tuples using integer indices
            edge_predicates: List of predicate IDs (parallel to edges list)
            node_id_to_idx: Dict mapping original node IDs to integer indices
            predicate_to_id: Dict mapping predicate strings to integer IDs
            edge_properties: Dict mapping (src_idx, dst_idx, pred_idx) -> properties dict
            node_properties: Dict mapping node_idx -> properties dict
        """
        self.num_nodes = num_nodes
        self.node_id_to_idx = node_id_to_idx
        self.idx_to_node_id = {idx: nid for nid, idx in node_id_to_idx.items()}
        self.node_properties = node_properties or {}

        # Predicate vocabulary
        self.predicate_to_idx = predicate_to_idx
        self.id_to_predicate = {idx: pred for pred, idx in predicate_to_idx.items()}

        # Node store (LMDB) — None when using in-memory dicts (build time)
        self.node_store = None

        # LMDB store (cold path) — set later by loader or load_mmap
        self.lmdb_store = None

        # Edge IDs from the original data — set later by loader or load_mmap
        self.edge_ids = None
        self._edge_ids_env = None

        # Graph Metadata - set later by loader or load_mmap
        self.meta_kg = None
        self.sri_testing_data = None
        self.graph_metadata = None

        # Build both forward and reverse CSR structures
        logger.debug("Building forward CSR...")
        self._build_forward_csr(edges, edge_predicates, edge_properties)

        logger.debug("Building reverse CSR...")
        self._build_reverse_csr(edges, edge_predicates)

        self.build_metadata()

    def _build_forward_csr(self, edges, edge_predicates, edge_properties):
        """Build forward adjacency list (source -> targets)."""
        # Sort edges by source for CSR construction
        edges_with_props = [
            (
                src,
                dst,
                pred_id,
                edge_properties.get((src, dst, pred_id), {}) if edge_properties else {},
            )
            for (src, dst), pred_id in zip(edges, edge_predicates)
        ]
        edges_with_props.sort(key=lambda x: (x[0], x[1], x[2]))

        # Build forward CSR arrays
        self.fwd_targets = np.array(
            [dst for src, dst, _, _ in edges_with_props], dtype=np.int32
        )
        self.fwd_predicates = np.array(
            [pred_id for src, dst, pred_id, _ in edges_with_props], dtype=np.int32
        )

        # Build deduplicated edge property store (qualifiers + sources only)
        props_list = [props for _, _, _, props in edges_with_props]
        self.edge_properties = EdgePropertyStore.from_property_list(props_list)

        stats = self.edge_properties.dedup_stats()
        logger.debug(
            "  Edge property dedup: %s edges -> "
            "%s unique source configs, "
            "%s unique qualifier combos",
            stats["total_edges"],
            stats["unique_sources"],
            stats["unique_qualifiers"],
        )

        # Build forward offsets
        self.fwd_offsets = np.zeros(self.num_nodes + 1, dtype=np.int64)

        if len(edges_with_props) > 0:
            current_src = 0
            for i, (src, dst, _, _) in enumerate(edges_with_props):
                while current_src < src:
                    self.fwd_offsets[current_src + 1] = i
                    current_src += 1
                self.fwd_offsets[src + 1] = i + 1

            for i in range(current_src + 1, self.num_nodes + 1):
                self.fwd_offsets[i] = len(edges_with_props)

    def _build_reverse_csr(self, edges, edge_predicates):
        """Build reverse adjacency list (target -> sources).

        Also builds ``rev_to_fwd`` – an int32 array that maps each reverse-CSR
        position to its corresponding forward-CSR position.  This enables O(1)
        property lookups for reverse-direction traversals and correctly handles
        duplicate (src, dst, pred) edges with different qualifiers / sources.
        """
        num_edges = len(edges)
        # Create reverse edges with original index for rev_to_fwd mapping
        # (dst, src, pred_id, original_index)
        reverse_edges = [
            (dst, src, pred_id, orig_idx)
            for orig_idx, ((src, dst), pred_id) in enumerate(
                zip(edges, edge_predicates)
            )
        ]
        reverse_edges.sort(key=lambda x: (x[0], x[1], x[2]))

        # Build reverse CSR arrays
        self.rev_sources = np.array(
            [src for dst, src, _, _ in reverse_edges], dtype=np.int32
        )
        self.rev_predicates = np.array(
            [pred_id for dst, src, pred_id, _ in reverse_edges], dtype=np.int32
        )

        # Build rev_to_fwd: map each reverse position to its forward position.
        # The forward CSR was built from edges sorted by (src, dst, pred).
        # We need to map each original edge index to its forward position.
        # _fwd_sort_order[orig_idx] = forward CSR position of that edge.
        if hasattr(self, "_fwd_sort_order") and self._fwd_sort_order is not None:
            fwd_pos = self._fwd_sort_order  # already inverse-mapped
        else:
            # Fallback: build fwd_pos from the forward CSR structure.
            # For each forward position, determine which original edge it is.
            fwd_pos = np.empty(num_edges, dtype=np.int32)
            # In the constructor path, edges were sorted by (src,dst,pred)
            # so we need to compute the sort permutation and invert it.
            sorted_edges = sorted(
                range(num_edges),
                key=lambda i: (edges[i][0], edges[i][1], int(edge_predicates[i])),
            )
            for fwd_idx, orig_idx in enumerate(sorted_edges):
                fwd_pos[orig_idx] = fwd_idx

        self.rev_to_fwd = np.array(
            [fwd_pos[orig_idx] for _, _, _, orig_idx in reverse_edges],
            dtype=np.int32,
        )

        # Build reverse offsets
        self.rev_offsets = np.zeros(self.num_nodes + 1, dtype=np.int64)

        if len(reverse_edges) > 0:
            current_dst = 0
            for i, (dst, src, _, _) in enumerate(reverse_edges):
                while current_dst < dst:
                    self.rev_offsets[current_dst + 1] = i
                    current_dst += 1
                self.rev_offsets[dst + 1] = i + 1

            for i in range(current_dst + 1, self.num_nodes + 1):
                self.rev_offsets[i] = len(reverse_edges)

    def _rebuild_rev_to_fwd(self):
        """Rebuild ``rev_to_fwd`` from existing forward and reverse CSR arrays.

        For each reverse-CSR position, find the corresponding forward-CSR
        position by matching (src, dst, pred) and tracking ordinal within
        groups of duplicate edges.
        """
        num_edges = len(self.fwd_targets)
        self.rev_to_fwd = np.empty(num_edges, dtype=np.int32)

        for node_idx in range(self.num_nodes):
            rev_start = int(self.rev_offsets[node_idx])
            rev_end = int(self.rev_offsets[node_idx + 1])

            # For each reverse edge to this node, find the matching forward position.
            # Track how many times we've seen each (src, pred) pair to handle duplicates.
            seen_counts = {}
            for rev_pos in range(rev_start, rev_end):
                src = int(self.rev_sources[rev_pos])
                pred = int(self.rev_predicates[rev_pos])
                key = (src, pred)
                ordinal = seen_counts.get(key, 0)
                seen_counts[key] = ordinal + 1

                # Find the (ordinal)-th forward edge with this (src, dst=node_idx, pred)
                fwd_start = int(self.fwd_offsets[src])
                fwd_end = int(self.fwd_offsets[src + 1])
                count = 0
                for fwd_pos in range(fwd_start, fwd_end):
                    if (
                        int(self.fwd_targets[fwd_pos]) == node_idx
                        and int(self.fwd_predicates[fwd_pos]) == pred
                    ):
                        if count == ordinal:
                            self.rev_to_fwd[rev_pos] = fwd_pos
                            break
                        count += 1

    # ------------------------------------------------------------------
    # Binary search edge lookup (replaces edge_prop_index dict, saves ~7GB)
    # ------------------------------------------------------------------

    def _find_edge_index(self, src_idx, dst_idx, pred_id):
        """Find CSR array index for an edge (src, dst, pred).

        Since forward edges are sorted by (src, dst, pred), this uses
        binary search within the source node's edge range.

        Returns edge array index, or None if not found.
        Typical cost: O(log(degree)), ~4-5 comparisons for avg degree 19.
        """
        start = int(self.fwd_offsets[src_idx])
        end = int(self.fwd_offsets[src_idx + 1])

        if start == end:
            return None

        # Binary search for dst_idx within targets[start:end]
        targets_slice = self.fwd_targets[start:end]
        left = int(np.searchsorted(targets_slice, dst_idx, side="left"))
        right = int(np.searchsorted(targets_slice, dst_idx, side="right"))

        if left == right:
            return None  # dst_idx not found

        # Linear scan within the (typically 1-3) edges to same dst for pred_id
        for i in range(start + left, start + right):
            if self.fwd_predicates[i] == pred_id:
                return i

        return None

    # ------------------------------------------------------------------
    # Neighbor queries
    # ------------------------------------------------------------------

    def neighbors(self, node_idx, predicate_filter=None):
        """
        Get neighbor indices for a node index (nodes this node points TO)
        """
        start = self.fwd_offsets[node_idx]
        end = self.fwd_offsets[node_idx + 1]

        if predicate_filter is None:
            return self.fwd_targets[start:end]
        else:
            pred_id = self.predicate_to_idx.get(predicate_filter)
            if pred_id is None:
                return np.array([], dtype=np.int32)

            neighbors = self.fwd_targets[start:end]
            predicates = self.fwd_predicates[start:end]
            mask = predicates == pred_id
            return neighbors[mask]

    def incoming_neighbors(self, node_idx, predicate_filter=None):
        """
        Get incoming neighbors (nodes that point TO this node).
        """
        start = self.rev_offsets[node_idx]
        end = self.rev_offsets[node_idx + 1]

        if predicate_filter is None:
            return self.rev_sources[start:end]
        else:
            pred_id = self.predicate_to_idx.get(predicate_filter)
            if pred_id is None:
                return np.array([], dtype=np.int32)

            sources = self.rev_sources[start:end]
            predicates = self.rev_predicates[start:end]
            mask = predicates == pred_id
            return sources[mask]

    def neighbors_with_properties(
        self, node_idx: int, predicate_filter: Optional[list] = None
    ):
        """Get neighbors with edge properties (qualifiers + sources).

        Predicate filtering is done FIRST (in-memory from CSR arrays),
        then qualifier/source properties are fetched only for matching
        edges — avoiding unnecessary dedup store lookups.

        Returns list of (neighbor_idx, predicate_str, edge_props, fwd_edge_idx)
        tuples where edge_props = {"qualifiers": [...], "sources": [...]}.
        The dict values are pool references (zero GC pressure).
        ``fwd_edge_idx`` is the forward-CSR array position — unique per edge
        even when (src, dst, pred) repeats with different qualifiers/sources.
        """
        start = int(self.fwd_offsets[node_idx])
        end = int(self.fwd_offsets[node_idx + 1])

        result = []
        for pos in range(start, end):
            pred_id = int(self.fwd_predicates[pos])
            pred_str = self.id_to_predicate[pred_id]

            if predicate_filter is not None and pred_str not in predicate_filter:
                continue

            target = int(self.fwd_targets[pos])
            props = self.edge_properties._get_props(pos)
            result.append((target, pred_str, props, pos))

        return result

    def neighbors_filtered_by_targets(
        self, node_idx: int, target_set: set, predicate_filter: Optional[set] = None
    ):
        """Get outgoing neighbors that are in *target_set*, with properties.

        Unlike ``neighbors_with_properties``, this skips property lookups
        entirely for edges whose target is not in ``target_set``.  When only
        a small fraction of neighbors match, this avoids millions of
        unnecessary dict allocations and dramatically reduces memory pressure.

        Args:
            node_idx: Source node index.
            target_set: Set of target node indices to keep.
            predicate_filter: Optional set of allowed predicate strings.

        Returns:
            List of (target_idx, predicate_str, edge_props, fwd_edge_idx) tuples
            for edges whose target is in *target_set*.
        """
        start = int(self.fwd_offsets[node_idx])
        end = int(self.fwd_offsets[node_idx + 1])

        result = []
        for pos in range(start, end):
            target = int(self.fwd_targets[pos])
            if target not in target_set:
                continue

            pred_id = int(self.fwd_predicates[pos])
            pred_str = self.id_to_predicate[pred_id]

            if predicate_filter is not None and pred_str not in predicate_filter:
                continue

            props = self.edge_properties._get_props(pos)
            result.append((target, pred_str, props, pos))

        return result

    def incoming_neighbors_with_properties(
        self, node_idx, predicate_filter: Optional[list] = None
    ):
        """Get incoming neighbors with edge properties (qualifiers + sources).

        Uses the ``rev_to_fwd`` mapping for O(1) property lookup.  This
        correctly handles duplicate (src, dst, pred) edges that differ only
        in qualifiers / sources — each reverse-CSR position maps to its own
        unique forward-CSR position.

        Returns list of (src_idx, predicate, edge_props, fwd_edge_idx) tuples.
        """
        start = int(self.rev_offsets[node_idx])
        end = int(self.rev_offsets[node_idx + 1])

        result = []
        for pos in range(start, end):
            src_idx = int(self.rev_sources[pos])
            pred_id = int(self.rev_predicates[pos])
            predicate = self.id_to_predicate[pred_id]

            if predicate_filter is not None and predicate not in predicate_filter:
                continue

            # O(1) forward edge index lookup via rev_to_fwd mapping
            fwd_idx = int(self.rev_to_fwd[pos])
            props = self.edge_properties._get_props(fwd_idx)

            result.append((src_idx, predicate, props, fwd_idx))

        return result

    def get_edges(self, node_idx):
        """Get all edges from a node as (neighbor_idx, predicate_str) tuples."""
        start = self.fwd_offsets[node_idx]
        end = self.fwd_offsets[node_idx + 1]

        neighbors = self.fwd_targets[start:end]
        pred_ids = self.fwd_predicates[start:end]

        return [
            (int(neighbor), self.id_to_predicate[int(pred_id)])
            for neighbor, pred_id in zip(neighbors, pred_ids)
        ]

    def get_incoming_edges(self, node_idx):
        """Get all incoming edges to a node as (source_idx, predicate_str) tuples."""
        start = self.rev_offsets[node_idx]
        end = self.rev_offsets[node_idx + 1]

        sources = self.rev_sources[start:end]
        pred_ids = self.rev_predicates[start:end]

        return [
            (int(source), self.id_to_predicate[int(pred_id)])
            for source, pred_id in zip(sources, pred_ids)
        ]

    # ------------------------------------------------------------------
    # Edge property accessors
    # ------------------------------------------------------------------

    def get_edge_property(self, src_idx, dst_idx, predicate, key, default=None):
        """Get a specific property for an edge.

        For 'qualifiers' and 'sources': O(log(degree)) via dedup store.
        For 'attributes': O(log(degree)) + LMDB lookup.
        """
        if key == "predicate":
            return predicate

        pred_id = self.predicate_to_idx.get(predicate)
        if pred_id is None:
            return default

        edge_idx = self._find_edge_index(src_idx, dst_idx, pred_id)
        if edge_idx is None:
            return default

        # Hot path fields from dedup store
        result = self.edge_properties.get_field(edge_idx, key)
        if result is not None:
            return result

        # Cold path fields from LMDB
        if self.lmdb_store is not None:
            detail = self.lmdb_store.get(edge_idx)
            return detail.get(key, default)

        return default

    def get_all_edge_properties(self, src_idx, dst_idx, predicate):
        """Get all properties for an edge.

        Merges hot-path (qualifiers, sources from dedup store) with
        cold-path (attributes from LMDB).
        """
        pred_id = self.predicate_to_idx.get(predicate)
        if pred_id is None:
            return {}

        edge_idx = self._find_edge_index(src_idx, dst_idx, pred_id)
        if edge_idx is None:
            return {}

        # Start with hot-path data (pool references, no GC pressure)
        props = self.edge_properties._get_props(edge_idx)

        # Merge cold-path data from LMDB
        if self.lmdb_store is not None:
            detail = self.lmdb_store.get(edge_idx)
            props.update(detail)

        props["predicate"] = predicate
        return props

    def get_edge_properties_by_index(self, fwd_edge_idx):
        """Get all properties for an edge by its forward-CSR array position.

        This is O(1) for hot-path data (qualifiers, sources) and a single
        LMDB lookup for cold-path data (attributes).  Prefer
        this over ``get_all_edge_properties`` when you already have the
        forward edge index (e.g. from ``neighbors_with_properties``).
        """
        fwd_edge_idx = int(fwd_edge_idx)
        props = self.edge_properties._get_props(fwd_edge_idx)

        pred_id = int(self.fwd_predicates[fwd_edge_idx])
        props["predicate"] = self.id_to_predicate[pred_id]

        if self.lmdb_store is not None:
            detail = self.lmdb_store.get(fwd_edge_idx)
            props.update(detail)

        return props

    def get_edge_id(self, fwd_edge_idx):
        """Return the original edge ID for a forward-CSR position, or None."""
        if getattr(self, "_edge_ids_env", None) is not None:
            return self._load_edge_id_from_lmdb(self._edge_ids_env, int(fwd_edge_idx))
        if self.edge_ids is not None:
            return self.edge_ids[int(fwd_edge_idx)]
        return None

    def get_all_edges_between(
        self, src_idx, dst_idx, predicate_filter: Optional[list] = None
    ):
        """Get all edges (with different predicates or qualifiers) between two nodes."""
        start = int(self.fwd_offsets[src_idx])
        end = int(self.fwd_offsets[src_idx + 1])

        result = []
        for pos in range(start, end):
            if int(self.fwd_targets[pos]) == dst_idx:
                pred_id = int(self.fwd_predicates[pos])
                predicate_str = self.id_to_predicate[pred_id]
                if predicate_filter is None or predicate_str in predicate_filter:
                    props = self.edge_properties._get_props(pos)
                    result.append((predicate_str, props))

        return result

    # ------------------------------------------------------------------
    # Node property accessors
    # ------------------------------------------------------------------

    def get_node_property(self, node_idx, key, default=None):
        """Get a specific property for a node"""
        if self.node_store is not None:
            return self.node_store.get_property(node_idx, key, default)
        return self.node_properties.get(node_idx, {}).get(key, default)

    def get_all_node_properties(self, node_idx):
        """Get all properties for a node as a dict"""
        if self.node_store is not None:
            return self.node_store.get_properties(node_idx)
        return self.node_properties.get(node_idx, {})

    def degree(self, node_idx, predicate_filter=None):
        """Get degree of a node, optionally filtered by predicate."""
        if predicate_filter is None:
            return self.fwd_offsets[node_idx + 1] - self.fwd_offsets[node_idx]
        else:
            return len(self.neighbors(node_idx, predicate_filter=predicate_filter))

    def get_node_idx(self, node_id):
        """Convert original node ID to internal index"""
        if self.node_store is not None:
            return self.node_store.get_node_idx(node_id)
        return self.node_id_to_idx.get(node_id)

    def get_node_id(self, node_idx):
        """Convert internal index to original node ID"""
        if self.node_store is not None:
            return self.node_store.get_node_id(node_idx)
        return self.idx_to_node_id.get(node_idx)

    def get_predicate_stats(self):
        """Get statistics about predicate usage"""
        pred_counts = {}
        for pred_id in self.fwd_predicates:
            pred_str = self.id_to_predicate[int(pred_id)]
            pred_counts[pred_str] = pred_counts.get(pred_str, 0) + 1

        return sorted(pred_counts.items(), key=lambda x: x[1], reverse=True)

    # ------------------------------------------------------------------
    # Metadata (Plater-compatible pre-computed summaries)
    # ------------------------------------------------------------------

    def _build_node_categories(self):
        """Pre-extract the primary category for each node."""
        node_categories = {}
        for node_idx in range(self.num_nodes):
            props = self.get_all_node_properties(node_idx)
            cats = props.get("categories", ["biolink:NamedThing"])
            node_categories[node_idx] = cats[0] if cats else "biolink:NamedThing"
        return node_categories

    def _build_category_prefixes(self, node_categories):
        """Collect ID prefixes per category."""
        category_prefixes = {}
        id_iter = (
            self.node_store.iter_id_to_idx()
            if self.node_store is not None
            else self.node_id_to_idx.items()
        )
        for node_id, node_idx in id_iter:
            cat = node_categories[node_idx]
            prefix = node_id.split(":")[0] if ":" in node_id else node_id
            if cat not in category_prefixes:
                category_prefixes[cat] = set()
            category_prefixes[cat].add(prefix)
        return category_prefixes

    def _scan_edge_triples(self, node_categories):
        """Single pass over forward edges to collect triple counts and examples."""
        triple_counts = {}
        triple_examples = {}

        for src_idx in range(self.num_nodes):
            start = int(self.fwd_offsets[src_idx])
            end = int(self.fwd_offsets[src_idx + 1])
            if start == end:
                continue

            subj_cat = node_categories[src_idx]

            for pos in range(start, end):
                tgt_idx = int(self.fwd_targets[pos])
                pred_id = int(self.fwd_predicates[pos])
                pred = self.id_to_predicate[pred_id]
                obj_cat = node_categories[tgt_idx]

                key = (subj_cat, pred, obj_cat)
                triple_counts[key] = triple_counts.get(key, 0) + 1

                if key not in triple_examples:
                    triple_examples[key] = (
                        self.get_node_id(src_idx),
                        self.get_node_id(tgt_idx),
                    )

        return triple_counts, triple_examples

    def build_meta_kg(self, node_categories, category_prefixes, triple_counts):
        """Build the TRAPI MetaKnowledgeGraph with full attribute metadata.

        Scans all nodes and edges to produce:
        - nodes: keyed by category, with id_prefixes and attribute descriptors
        - edges: list of (subject, predicate, object) with attribute and qualifier
                 descriptors

        This is the expensive step (full LMDB scan for edge attributes).
        The result is stored as self.meta_kg and can be persisted as meta_kg.json.
        """
        logger.info("Building meta knowledge graph...")
        t0 = time.perf_counter()

        # -- Node attributes: collect unique (attribute_type_id, attribute_source,
        #    original_attribute_name) tuples per category.
        #    Each unique original_attribute_name gets its own descriptor entry.
        # cat -> set of (type_id, source, orig_name)
        cat_attr_set = {}
        for node_idx in range(self.num_nodes):
            props = self.get_all_node_properties(node_idx)
            cat = node_categories[node_idx]
            if cat not in cat_attr_set:
                cat_attr_set[cat] = set()
            for attr in props.get("attributes", []):
                type_id = attr.get("attribute_type_id", "biolink:Attribute")
                source = attr.get("attribute_source", None)
                orig_name = attr.get("original_attribute_name")
                if orig_name:
                    cat_attr_set[cat].add((type_id, source, orig_name))

        meta_nodes = {}
        for cat, prefixes in category_prefixes.items():
            attrs = []
            for type_id, source, orig_name in sorted(cat_attr_set.get(cat, set())):
                attrs.append(
                    {
                        "attribute_type_id": type_id,
                        "attribute_source": source,
                        "original_attribute_names": [orig_name],
                        "constraint_use": False,
                        "constraint_name": None,
                    }
                )
            meta_nodes[cat] = {
                "id_prefixes": sorted(prefixes),
                "attributes": attrs,
            }

        # -- Edge attributes: full scan of LMDB + hot-path qualifiers --
        # triple_key -> set of (type_id, source, orig_name)
        triple_attr_set = {key: set() for key in triple_counts}
        # triple_key -> set of qualifier_type_ids
        triple_qual_map = {key: set() for key in triple_counts}

        # Build a mapping from edge position -> triple key for the full scan
        edge_to_triple = {}
        for src_idx in range(self.num_nodes):
            start = int(self.fwd_offsets[src_idx])
            end = int(self.fwd_offsets[src_idx + 1])
            if start == end:
                continue
            subj_cat = node_categories[src_idx]
            for pos in range(start, end):
                tgt_idx = int(self.fwd_targets[pos])
                pred_id = int(self.fwd_predicates[pos])
                pred = self.id_to_predicate[pred_id]
                obj_cat = node_categories[tgt_idx]
                edge_to_triple[pos] = (subj_cat, pred, obj_cat)

                # Collect qualifiers from hot-path store
                if isinstance(self.edge_properties, EdgePropertyStore):
                    quals = self.edge_properties.get_qualifiers(pos)
                    for q in quals:
                        qtype = q.get("qualifier_type_id")
                        if qtype:
                            triple_qual_map[(subj_cat, pred, obj_cat)].add(qtype)

        # Full scan of LMDB for edge attribute types
        if self.lmdb_store is not None:
            logger.debug("  Scanning LMDB for edge attribute types...")
            with self.lmdb_store._env.begin(buffers=True) as txn:
                cursor = txn.cursor()
                for key_buf, val_buf in cursor:
                    edge_idx = int.from_bytes(bytes(key_buf), "big")
                    triple_key = edge_to_triple.get(edge_idx)
                    if triple_key is None:
                        continue
                    detail = msgpack.unpackb(val_buf, raw=False)
                    for attr in detail.get("attributes", []):
                        type_id = attr.get("attribute_type_id", "biolink:Attribute")
                        source = attr.get("attribute_source", None)
                        orig_name = attr.get("original_attribute_name")
                        if orig_name:
                            triple_attr_set[triple_key].add(
                                (type_id, source, orig_name)
                            )

        del edge_to_triple

        meta_edges = []
        for subj_cat, pred, obj_cat in sorted(triple_counts):
            triple_key = (subj_cat, pred, obj_cat)
            attrs = []
            for type_id, source, orig_name in sorted(
                triple_attr_set.get(triple_key, set())
            ):
                attrs.append(
                    {
                        "attribute_type_id": type_id,
                        "attribute_source": source,
                        "original_attribute_names": [orig_name],
                        "constraint_use": False,
                        "constraint_name": None,
                    }
                )
            # Add qualifier type IDs as attributes in the qualifiers list
            qualifiers = []
            for qtype in sorted(triple_qual_map.get(triple_key, set())):
                qualifiers.append({"qualifier_type_id": qtype})

            meta_edges.append(
                {
                    "subject": subj_cat,
                    "predicate": pred,
                    "object": obj_cat,
                    "attributes": attrs,
                    "qualifiers": qualifiers,
                }
            )

        self.meta_kg = {"nodes": meta_nodes, "edges": meta_edges}

        t1 = time.perf_counter()
        logger.debug(
            "  Meta KG built in %.2fs: %s edge triples, %s categories",
            t1 - t0,
            len(triple_counts),
            len(category_prefixes),
        )

    def build_metadata(self):
        """Pre-compute Plater-compatible metadata from the CSR graph.

        Builds (if not already present):
        - meta_kg: TRAPI MetaKnowledgeGraph (via build_meta_kg if not already set)
        - sri_testing_data: One representative edge per (subj_cat, pred, obj_cat) triple

        Ideally both are created offline during graph building and loaded
        from disk.  This method only regenerates them as a fallback.
        """
        # If both are already loaded (e.g. from disk), nothing to do.
        if (
            getattr(self, "meta_kg", None) is not None
            and getattr(self, "sri_testing_data", None) is not None
        ):
            return

        logger.warning(
            "meta_kg and/or sri_testing_data not pre-loaded; "
            "generating on the fly (consider building offline first)"
        )

        t0 = time.perf_counter()

        node_categories = self._build_node_categories()
        category_prefixes = self._build_category_prefixes(node_categories)
        triple_counts, triple_examples = self._scan_edge_triples(node_categories)

        # Build meta_kg if not already loaded from disk
        if not hasattr(self, "meta_kg") or self.meta_kg is None:
            self.build_meta_kg(node_categories, category_prefixes, triple_counts)

        # Build SRI testing data if not already loaded from disk
        if not hasattr(self, "sri_testing_data") or self.sri_testing_data is None:
            logger.info("Building sri testing data...")
            self.sri_testing_data = {
                "edges": [
                    {
                        "subject_category": subj_cat,
                        "object_category": obj_cat,
                        "predicate": pred,
                        "subject_id": example[0],
                        "object_id": example[1],
                    }
                    for (subj_cat, pred, obj_cat), example in triple_examples.items()
                ]
            }

        t1 = time.perf_counter()
        logger.info(
            "  Metadata built in %.2fs: " "%s unique triples, " "%s categories",
            t1 - t0,
            len(triple_counts),
            len(category_prefixes),
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    @staticmethod
    def _save_edge_ids_lmdb(db_path, edge_ids, commit_every=50_000):
        """Save edge IDs list to an LMDB file."""
        import lmdb as _lmdb

        db_path = Path(db_path)
        if db_path.exists():
            shutil.rmtree(db_path)
        db_path.mkdir(parents=True, exist_ok=True)

        _INITIAL = 4 * 1024 * 1024 * 1024  # 4 GB
        env = _lmdb.open(
            str(db_path),
            map_size=_INITIAL,
            readonly=False,
            max_dbs=0,
            readahead=False,
        )
        pending = []
        txn = env.begin(write=True)
        try:
            for idx, eid in enumerate(edge_ids):
                if eid is not None:
                    key = struct.pack(">I", idx)
                    val = (
                        eid.encode("utf-8")
                        if isinstance(eid, str)
                        else str(eid).encode("utf-8")
                    )
                    try:
                        txn.put(key, val)
                        pending.append((key, val))
                    except _lmdb.MapFullError:
                        txn.abort()
                        new_size = env.info()["map_size"] * 2
                        env.set_mapsize(new_size)
                        logger.warning(
                            "    Edge IDs LMDB: map full, resized to %.0f GB",
                            new_size / (1024**3),
                        )
                        txn = env.begin(write=True)
                        for pk, pv in pending:
                            txn.put(pk, pv)
                        txn.put(key, val)
                        pending.append((key, val))
                if (idx + 1) % commit_every == 0:
                    txn.commit()
                    pending.clear()
                    txn = env.begin(write=True)
            txn.commit()
        except BaseException:
            txn.abort()
            raise
        finally:
            env.close()
        logger.debug(
            "  Edge IDs LMDB: wrote %s entries to %s", f"{len(edge_ids):,}", db_path
        )

    @staticmethod
    def _load_edge_id_from_lmdb(env, edge_idx):
        """Look up a single edge ID from an LMDB environment."""
        key = struct.pack(">I", edge_idx)
        with env.begin(buffers=True) as txn:
            val = txn.get(key)
            if val is None:
                return None
            return bytes(val).decode("utf-8")

    def save_mmap(self, directory: Union[str, Path]):
        """Save graph in memory-mappable format for fast loading.

        Creates a directory with separate files:
        - NumPy arrays as .npy files (can be memory-mapped)
        - Metadata dictionaries as pickle
        - Edge qualifier/source dedup store (mmap-friendly)
        - Edge detail properties as LMDB (if present)
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        logger.info("Saving graph to %s (mmap format)...", directory)
        t0 = time.perf_counter()

        # Save NumPy arrays as .npy files (memory-mappable)
        np.save(directory / "fwd_targets.npy", self.fwd_targets)
        np.save(directory / "fwd_predicates.npy", self.fwd_predicates)
        np.save(directory / "fwd_offsets.npy", self.fwd_offsets)
        np.save(directory / "rev_sources.npy", self.rev_sources)
        np.save(directory / "rev_predicates.npy", self.rev_predicates)
        np.save(directory / "rev_offsets.npy", self.rev_offsets)
        np.save(directory / "rev_to_fwd.npy", self.rev_to_fwd)

        # Save small metadata as pickle (predicates + num_nodes only)
        metadata = {
            "num_nodes": self.num_nodes,
            "predicate_to_idx": self.predicate_to_idx,
        }
        with open(directory / "metadata.pkl", "wb") as f:
            pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Save node ID mappings and properties as LMDB
        node_id_to_idx = (
            dict(self.node_store.iter_id_to_idx())
            if isinstance(getattr(self, "node_store", None), NodeStore)
            else self.node_id_to_idx
        )
        node_properties = (
            dict(self.node_store.iter_properties())
            if isinstance(getattr(self, "node_store", None), NodeStore)
            else self.node_properties
        )
        NodeStore.build(
            directory / "node_store.lmdb",
            node_id_to_idx,
            node_properties,
        )

        # Save hot-path edge properties (qualifier + source dedup store)
        if isinstance(self.edge_properties, EdgePropertyStore):
            self.edge_properties.save_mmap(directory)
        else:
            # Legacy fallback: pickle the whole thing
            with open(directory / "edge_properties.pkl", "wb") as f:
                pickle.dump(self.edge_properties, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Save edge IDs as LMDB if present
        if self.edge_ids is not None:
            self._save_edge_ids_lmdb(directory / "edge_ids.lmdb", self.edge_ids)

        # Save meta_kg as JSON for fast loading at query time
        if hasattr(self, "meta_kg") and self.meta_kg is not None:
            with open(directory / "meta_kg.json", "w", encoding="utf-8") as f:
                json.dump(self.meta_kg, f, separators=(",", ":"))

        # Save sri_testing_data as JSON for fast loading at query time
        if hasattr(self, "sri_testing_data") and self.sri_testing_data is not None:
            with open(directory / "sri_testing_data.json", "w", encoding="utf-8") as f:
                json.dump(self.sri_testing_data, f, separators=(",", ":"))

        # Copy LMDB store if present
        if self.lmdb_store is not None:
            lmdb_src = self.lmdb_store._path
            lmdb_dst = directory / "edge_properties.lmdb"
            if lmdb_src.resolve() != lmdb_dst.resolve():
                if lmdb_dst.exists():
                    shutil.rmtree(lmdb_dst)
                shutil.copytree(lmdb_src, lmdb_dst)

        t1 = time.perf_counter()
        logger.info("Graph saved in %.2fs", t1 - t0)

        # Print file sizes
        total_size = 0
        for entry in sorted(directory.iterdir()):
            if entry.is_file():
                size = entry.stat().st_size
                total_size += size
                logger.debug("  %s: %.1f MB", entry.name, size / 1024 / 1024)
            elif entry.is_dir():
                dir_size = sum(
                    ff.stat().st_size for ff in entry.iterdir() if ff.is_file()
                )
                total_size += dir_size
                logger.debug("  %s/: %.1f MB", entry.name, dir_size / 1024 / 1024)
        logger.debug("  Total: %.1f MB", total_size / 1024 / 1024)

    @staticmethod
    def load_mmap(
        directory: Union[str, Path], mmap_mode: Literal["r+", "r", "w+", "c"] = "r"
    ):
        """Load graph from memory-mapped format.

        Supports both the new hybrid format (dedup store + LMDB) and
        the legacy format (full EdgePropertyStore with pubs/sources/quals).
        """
        directory = Path(directory)
        logger.info(
            "Loading graph from %s (mmap_mode=%s, in_memory=%s)...",
            directory,
            mmap_mode,
            settings.load_mmaps_into_memory,
        )
        t0 = time.perf_counter()

        graph = CSRGraph.__new__(CSRGraph)

        # Load NumPy arrays (memory-mapped or copied into RAM)
        graph.fwd_targets = _load_npy(
            directory / "fwd_targets.npy", mmap_mode=mmap_mode
        )
        graph.fwd_predicates = _load_npy(
            directory / "fwd_predicates.npy", mmap_mode=mmap_mode
        )
        graph.fwd_offsets = _load_npy(
            directory / "fwd_offsets.npy", mmap_mode=mmap_mode
        )
        graph.rev_sources = _load_npy(
            directory / "rev_sources.npy", mmap_mode=mmap_mode
        )
        graph.rev_predicates = _load_npy(
            directory / "rev_predicates.npy", mmap_mode=mmap_mode
        )
        graph.rev_offsets = _load_npy(
            directory / "rev_offsets.npy", mmap_mode=mmap_mode
        )
        rev_to_fwd_path = directory / "rev_to_fwd.npy"
        if rev_to_fwd_path.exists():
            graph.rev_to_fwd = _load_npy(rev_to_fwd_path, mmap_mode=mmap_mode)
        else:
            graph.rev_to_fwd = None  # rebuilt after full load

        # Load metadata
        with open(directory / "metadata.pkl", "rb") as f:
            metadata = pickle.load(f)

        graph.num_nodes = metadata["num_nodes"]
        graph.predicate_to_idx = metadata["predicate_to_idx"]
        graph.id_to_predicate = {
            idx: pred for pred, idx in graph.predicate_to_idx.items()
        }

        # Load node store (LMDB) or fall back to legacy pickle
        node_store_path = directory / "node_store.lmdb"
        if node_store_path.exists():
            graph.node_store = NodeStore(node_store_path, readonly=True)
            # No in-memory dicts needed
            graph.node_id_to_idx = None
            graph.idx_to_node_id = None
            graph.node_properties = None
        else:
            # Legacy: node data was in metadata.pkl
            graph.node_store = None
            graph.node_id_to_idx = metadata["node_id_to_idx"]
            graph.idx_to_node_id = {
                idx: nid for nid, idx in graph.node_id_to_idx.items()
            }
            graph.node_properties = metadata["node_properties"]

        # Load edge properties - detect format
        t_props_start = time.perf_counter()
        lmdb_path = directory / "edge_properties.lmdb"

        if (directory / "edge_property_pools.pkl").exists():
            # New or legacy dedup format
            graph.edge_properties = EdgePropertyStore.load_mmap(
                directory, mmap_mode=mmap_mode
            )
        elif (directory / "edge_properties.pkl").exists():
            # Legacy format: one big pickle
            with open(directory / "edge_properties.pkl", "rb") as f:
                graph.edge_properties = pickle.load(f)
            if isinstance(graph.edge_properties, list):
                logger.debug("  Converting edge properties to deduplicated format...")
                graph.edge_properties = EdgePropertyStore.from_property_list(
                    graph.edge_properties
                )
        else:
            raise FileNotFoundError(f"No edge property files found in {directory}")

        # Load LMDB store if present
        if lmdb_path.exists():
            graph.lmdb_store = LMDBPropertyStore(lmdb_path, readonly=True)
        else:
            graph.lmdb_store = None

        # Load edge IDs — prefer LMDB, fall back to pickle
        edge_ids_lmdb_path = directory / "edge_ids.lmdb"
        edge_ids_pkl_path = directory / "edge_ids.pkl"
        if edge_ids_lmdb_path.exists():
            import lmdb as _lmdb

            graph._edge_ids_env = _lmdb.open(
                str(edge_ids_lmdb_path),
                readonly=True,
                max_dbs=0,
                map_size=256 * 1024 * 1024 * 1024,
                readahead=False,
                lock=False,
            )
            graph.edge_ids = None  # signal: use LMDB
        elif edge_ids_pkl_path.exists():
            with open(edge_ids_pkl_path, "rb") as f:
                graph.edge_ids = pickle.load(f)
            graph._edge_ids_env = None
        else:
            graph.edge_ids = None
            graph._edge_ids_env = None

        t_props_end = time.perf_counter()

        if isinstance(graph.edge_properties, EdgePropertyStore):
            stats = graph.edge_properties.dedup_stats()
            logger.debug(
                "  Edge property dedup: %s edges -> "
                "%s unique source configs, "
                "%s unique qualifier combos",
                stats["total_edges"],
                stats["unique_sources"],
                stats["unique_qualifiers"],
            )

        if graph.lmdb_store is not None:
            logger.debug("  LMDB detail store: %s", lmdb_path)

        # Rebuild rev_to_fwd if not present (legacy format)
        if graph.rev_to_fwd is None:
            logger.debug("  Rebuilding rev_to_fwd mapping...")
            graph._rebuild_rev_to_fwd()

        t1 = time.perf_counter()
        logger.info(
            "Graph loaded in %.2fs " "(edge_properties: %.2fs)",
            t1 - t0,
            t_props_end - t_props_start,
        )
        logger.info(
            "  %s nodes, %s edges, " "%s predicates",
            graph.num_nodes,
            len(graph.fwd_targets),
            len(graph.predicate_to_idx),
        )

        # Load pre-computed meta_kg from disk if available (avoids
        # expensive full LMDB scan at query-time startup).
        meta_kg_path = directory / "meta_kg.json"
        if meta_kg_path.exists():
            with open(meta_kg_path, "r", encoding="utf-8") as f:
                graph.meta_kg = json.load(f)
            logger.info("  Loaded meta_kg from %s", meta_kg_path)
        else:
            graph.meta_kg = None  # will be built by build_metadata()

        # Load pre-computed sri_testing_data from disk if available.
        sri_testing_path = directory / "sri_testing_data.json"
        if sri_testing_path.exists():
            with open(sri_testing_path, "r", encoding="utf-8") as f:
                graph.sri_testing_data = json.load(f)
            logger.info("  Loaded sri_testing_data from %s", sri_testing_path)
        else:
            graph.sri_testing_data = None  # will be built by build_metadata()

        metadata_path = directory / "graph-metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                graph.graph_metadata = json.load(f)
            logger.info("  Loaded metadata from %s", metadata_path)
        else:
            graph.graph_metadata = None

        graph.build_metadata()

        return graph
