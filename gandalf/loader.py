"""Load nodes and edges into Gandalf.

Three-pass streaming loader that keeps peak memory at ~3-4GB for 38M edges:

Pass 1: Stream JSONL to collect vocabularies (node IDs, predicates, edge count).
Pass 2: Stream JSONL again, converting each edge to integer indices stored in
        pre-allocated numpy arrays. Simultaneously interns qualifier/source data
        into the EdgePropertyStoreBuilder and writes attributes to
        a temporary LMDB keyed by original line index.
Pass 3: Sort numpy arrays by (src, dst, pred) via np.lexsort. Rewrite the temp
        LMDB in CSR-sorted order to produce the final LMDB where key == CSR
        edge index (zero indirection at query time). Reorder qualifier/source
        dedup indices to match. Build CSR offset arrays.
"""

import json
import shutil
import tempfile
from pathlib import Path

import msgpack
import numpy as np

from gandalf.graph import CSRGraph, EdgePropertyStoreBuilder
from gandalf.lmdb_store import (
    LMDBPropertyStore,
    _INITIAL_WRITE_MAP_SIZE,
    _encode_key,
    _put_with_resize,
)

import logging
import lmdb

logger = logging.getLogger(__name__)


# Fields that are structural (not stored as properties)
_CORE_FIELDS = {
    "id",
    "category",
    "subject",
    "object",
    "predicate",
    "sources",
}

# Node fields that become top-level TRAPI Node properties (not attributes)
_CORE_NODE_FIELDS = {"id", "name", "category"}

# Known qualifier fields that appear as top-level JSONL keys
_QUALIFIER_FIELDS = {
    "qualified_predicate",
    "object_aspect_qualifier",
    "object_direction_qualifier",
    "subject_aspect_qualifier",
    "subject_direction_qualifier",
    "causal_mechanism_qualifier",
    "species_context_qualifier",
}


def _extract_sources(data):
    """Extract normalized source list from edge data.

    Ensures every source has an ``upstream_resource_ids`` list (defaults to
    ``[]``) and prepends an ``infores:gandalf`` aggregator_knowledge_source
    whose upstream points to the top of the existing source chain (i.e. the
    source(s) not referenced in any other source's upstream_resource_ids).
    """
    raw = data.get("sources", [])

    # Normalize: guarantee upstream_resource_ids on every source
    sources = [
        {
            "resource_id": s["resource_id"],
            "resource_role": s["resource_role"],
            "upstream_resource_ids": s.get("upstream_resource_ids", []),
        }
        for s in raw
    ]

    # Find the top of the source chain: sources whose resource_id is NOT
    # referenced in any other source's upstream_resource_ids.  These are the
    # "leaf" providers that no one else aggregates from yet.
    all_upstream = {uid for s in sources for uid in s["upstream_resource_ids"]}
    top_ids = [
        s["resource_id"] for s in sources if s["resource_id"] not in all_upstream
    ]

    # Prepend gandalf as aggregator_knowledge_source
    gandalf_source = {
        "resource_id": "infores:gandalf",
        "resource_role": "aggregator_knowledge_source",
        "upstream_resource_ids": top_ids,
    }

    return [gandalf_source] + sources


def _extract_qualifiers(data):
    """Extract qualifiers.

    Format: Top-level fields (object_aspect_qualifier, etc.)
    """
    qualifiers = []
    for field in _QUALIFIER_FIELDS:
        if field in data:
            qualifiers.append(
                {
                    "qualifier_type_id": f"biolink:{field}",
                    "qualifier_value": data[field],
                }
            )

    return qualifiers


def _extract_attributes(data):
    """Extract attributes (everything not in core/qualifier/source fields).

    Publications are included as a TRAPI Attribute with
    ``attribute_type_id`` of ``biolink:publications``.
    """
    attributes = []
    for field, value in data.items():
        if field in _CORE_FIELDS or field in _QUALIFIER_FIELDS or field == "qualifiers":
            continue
        attributes.append(
            {
                "attribute_type_id": f"biolink:{field}",
                "value": value,
                "original_attribute_name": field,
            }
        )
    return attributes


def _extract_node_attributes(node_data):
    """Extract node attributes as TRAPI Attribute objects.

    Any field not in ``_CORE_NODE_FIELDS`` (id, name, category) is converted
    to a TRAPI-compliant Attribute dict with ``attribute_type_id``, ``value``,
    and ``original_attribute_name``.
    """
    attributes = []
    for field, value in node_data.items():
        if field in _CORE_NODE_FIELDS:
            continue
        attributes.append(
            {
                "attribute_type_id": "biolink:Attribute",
                "value": value,
                "original_attribute_name": field,
            }
        )
    return attributes


def build_graph_from_jsonl(edge_jsonl_path, node_jsonl_path):
    """Build a CSR graph from JSONL files using three-pass streaming.

    Pass 1: Collect vocabularies (node IDs, predicates, edge count).
    Pass 2: Build numpy arrays + dedup store + temp LMDB.
    Pass 3: Sort, rewrite LMDB in CSR order, build offsets.

    Peak memory: ~3-4GB for 38M edges (down from 100GB+).
    """
    edge_jsonl_path = str(edge_jsonl_path)
    node_jsonl_path = str(node_jsonl_path) if node_jsonl_path else None

    # =================================================================
    # Pass 1: Vocabulary collection
    # =================================================================
    logger.info("Pass 1: Collecting vocabularies from %s...", edge_jsonl_path)

    node_ids = set()
    predicates = set()
    edge_count = 0

    with open(edge_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            node_ids.add(data["subject"])
            node_ids.add(data["object"])
            predicates.add(data["predicate"])
            edge_count += 1

            if edge_count % 1_000_000 == 0:
                logger.debug("  %s edges scanned...", f"{edge_count:,}")

    logger.info(
        "  Found %s unique nodes, %s predicates, %s edges",
        f"{len(node_ids):,}",
        f"{len(predicates):,}",
        f"{edge_count:,}",
    )

    # Build vocabulary mappings
    node_id_to_idx = {nid: idx for idx, nid in enumerate(sorted(node_ids))}
    predicate_to_idx = {pred: idx for idx, pred in enumerate(sorted(predicates))}
    num_nodes = len(node_ids)
    del node_ids  # Free ~2GB immediately

    # Load node properties
    node_properties = {}
    if node_jsonl_path:
        logger.debug("Reading node properties from %s...", node_jsonl_path)
        with open(node_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                node_data = json.loads(line)
                node_id = node_data.get("id")
                if node_id:
                    idx = node_id_to_idx.get(node_id)
                    if idx is not None:
                        node_properties[idx] = {
                            "name": node_data.get("name", None),
                            "categories": node_data.get("category", []),
                            "attributes": _extract_node_attributes(node_data),
                        }
        logger.debug("  Loaded properties for %s nodes", f"{len(node_properties):,}")

    # =================================================================
    # Pass 2: Build arrays + dedup store + temp LMDB
    # =================================================================
    logger.info(
        "Pass 2: Building arrays and property stores (%s edges)...", f"{edge_count:,}"
    )

    # Pre-allocate numpy arrays
    src_indices = np.empty(edge_count, dtype=np.int32)
    dst_indices = np.empty(edge_count, dtype=np.int32)
    pred_indices = np.empty(edge_count, dtype=np.int32)

    # Edge IDs from the JSONL "id" field (indexed by original line order)
    edge_ids = [None] * edge_count

    # Incremental dedup builder for qualifiers + sources (hot path)
    prop_builder = EdgePropertyStoreBuilder(edge_count)

    # Temp LMDB for cold-path properties (publications, attributes)
    temp_dir = tempfile.mkdtemp(prefix="gandalf_build_")
    temp_lmdb_path = Path(temp_dir) / "temp_props.lmdb"
    temp_lmdb_path.mkdir(parents=True, exist_ok=True)

    temp_env = lmdb.open(
        str(temp_lmdb_path),
        map_size=_INITIAL_WRITE_MAP_SIZE,
        readonly=False,
        max_dbs=0,
        readahead=False,
    )

    txn = temp_env.begin(write=True)
    pending = []
    try:
        with open(edge_jsonl_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                data = json.loads(line)

                # Fill numpy arrays
                src_indices[i] = node_id_to_idx[data["subject"]]
                dst_indices[i] = node_id_to_idx[data["object"]]
                pred_indices[i] = predicate_to_idx[data["predicate"]]

                # Capture edge ID from JSONL (if present)
                edge_ids[i] = data.get("id")

                # Extract properties
                sources = _extract_sources(data)
                qualifiers = _extract_qualifiers(data)
                attributes = _extract_attributes(data)

                # Hot path: intern qualifiers + sources
                prop_builder.add(i, {"sources": sources, "qualifiers": qualifiers})

                # Cold path: write attributes to temp LMDB
                # (publications are now included in the attributes list)
                detail = {
                    "attributes": attributes,
                }
                key = _encode_key(i)
                val = msgpack.packb(detail, use_bin_type=True)
                txn = _put_with_resize(temp_env, txn, key, val, pending)

                if (i + 1) % 50_000 == 0:
                    txn.commit()
                    pending.clear()
                    txn = temp_env.begin(write=True)

                if (i + 1) % 1_000_000 == 0:
                    logger.debug(
                        "  %s/%s edges processed...", f"{i + 1:,}", f"{edge_count:,}"
                    )

        txn.commit()
    except BaseException:
        txn.abort()
        raise
    finally:
        temp_env.close()

    logger.debug("  Arrays and temp LMDB built")

    # =================================================================
    # Pass 3: Sort, rewrite LMDB, build CSR
    # =================================================================
    logger.info("Pass 3: Sorting and building CSR structure...")

    # Sort by (src, dst, pred) using lexsort (last key is primary)
    sort_order = np.lexsort((pred_indices, dst_indices, src_indices))

    # Reorder numpy arrays
    src_sorted = src_indices[sort_order]
    dst_sorted = dst_indices[sort_order]
    pred_sorted = pred_indices[sort_order]

    # Free unsorted arrays
    del src_indices, dst_indices, pred_indices

    # Reorder edge IDs to match CSR sort order
    edge_ids_sorted = [edge_ids[sort_order[j]] for j in range(edge_count)]
    del edge_ids

    # Reorder dedup store indices to match CSR order
    prop_builder.reorder(sort_order)
    edge_properties = prop_builder.build()
    del prop_builder

    stats = edge_properties.dedup_stats()
    logger.debug(
        "  Edge property dedup: %s edges -> %s unique source configs, %s unique qualifier combos",
        f"{stats['total_edges']:,}",
        f"{stats['unique_sources']:,}",
        f"{stats['unique_qualifiers']:,}",
    )

    # Rewrite temp LMDB in CSR-sorted order → final LMDB
    # This is the expensive build-time step, but ensures query-time
    # LMDB key == CSR edge index with zero indirection.
    final_lmdb_path = Path(temp_dir) / "edge_properties.lmdb"
    lmdb_store = LMDBPropertyStore.build_sorted(
        db_path=final_lmdb_path,
        temp_db_path=temp_lmdb_path,
        sort_permutation=sort_order,
        num_edges=edge_count,
    )

    # Clean up temp LMDB
    shutil.rmtree(temp_lmdb_path)
    # NOTE: sort_order is kept alive — needed for rev_to_fwd mapping below.

    # Build CSR offset arrays using searchsorted
    logger.debug("  Building CSR offsets...")
    fwd_offsets = np.zeros(num_nodes + 1, dtype=np.int64)
    if edge_count > 0:
        boundaries = np.searchsorted(src_sorted, np.arange(num_nodes + 1))
        fwd_offsets = boundaries.astype(np.int64)
    del src_sorted

    # Build reverse CSR: sort edges by (dst, src, pred)
    logger.debug("  Building reverse CSR...")
    # Reconstruct per-edge source node IDs from forward CSR offsets
    edge_src = np.empty(edge_count, dtype=np.int32)
    for node_idx in range(num_nodes):
        start = int(fwd_offsets[node_idx])
        end = int(fwd_offsets[node_idx + 1])
        edge_src[start:end] = node_idx

    rev_order = np.lexsort((pred_sorted, edge_src, dst_sorted))

    rev_dst_sorted = dst_sorted[rev_order]
    rev_sources = edge_src[rev_order]
    rev_predicates = pred_sorted[rev_order]

    rev_offsets = np.zeros(num_nodes + 1, dtype=np.int64)
    if edge_count > 0:
        boundaries = np.searchsorted(rev_dst_sorted, np.arange(num_nodes + 1))
        rev_offsets = boundaries.astype(np.int64)

    # Build rev_to_fwd mapping: for each reverse-CSR position, store the
    # corresponding forward-CSR position.  Forward positions are simply
    # 0..E-1 (the arrays are already in forward-sorted order).  The
    # inverse of sort_order maps original-edge-index → forward position.
    logger.debug("  Building rev_to_fwd mapping...")
    fwd_pos = np.empty(edge_count, dtype=np.int32)
    fwd_pos[sort_order] = np.arange(edge_count, dtype=np.int32)
    rev_to_fwd = rev_order.astype(np.int32)
    del fwd_pos, sort_order

    del edge_src, rev_dst_sorted, rev_order

    # Assemble the graph
    logger.debug("  Assembling graph...")
    graph = CSRGraph.__new__(CSRGraph)
    graph.num_nodes = num_nodes
    graph.node_id_to_idx = node_id_to_idx
    graph.idx_to_node_id = {idx: nid for nid, idx in node_id_to_idx.items()}
    graph.predicate_to_idx = predicate_to_idx
    graph.id_to_predicate = {idx: pred for pred, idx in predicate_to_idx.items()}
    graph.node_properties = node_properties

    graph.fwd_targets = dst_sorted
    graph.fwd_predicates = pred_sorted
    graph.fwd_offsets = fwd_offsets

    graph.rev_sources = rev_sources
    graph.rev_predicates = rev_predicates
    graph.rev_offsets = rev_offsets
    graph.rev_to_fwd = rev_to_fwd

    graph.edge_properties = edge_properties
    graph.edge_ids = edge_ids_sorted
    graph.lmdb_store = lmdb_store

    # Print statistics
    degrees = [graph.degree(i) for i in range(min(1000, graph.num_nodes))]
    if degrees:
        logger.info("Graph statistics:")
        logger.info("  Nodes: %s", f"{graph.num_nodes:,}")
        logger.info("  Edges: %s", f"{len(graph.fwd_targets):,}")
        logger.info("  Unique predicates: %s", f"{len(predicate_to_idx):,}")
        logger.info("  Avg degree (sampled): %.1f", np.mean(degrees))
        logger.info("  Max degree (sampled): %s", np.max(degrees))
        memory_mb = (
            (
                graph.fwd_targets.nbytes
                + graph.fwd_offsets.nbytes
                + graph.fwd_predicates.nbytes
            )
            / 1024
            / 1024
        )
        logger.info("  CSR memory usage: ~%.1f MB", memory_mb)

    graph.build_metadata()

    return graph
