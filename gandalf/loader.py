"""Load nodes and edges into Gandalf.

Three-pass streaming loader that keeps peak memory at ~3-4GB for 38M edges:

Pass 1: Stream edge triples to collect vocabularies (node IDs, predicates,
        edge count).
Pass 2: Stream edges again, converting each to integer indices stored in
        pre-allocated numpy arrays. Simultaneously interns qualifier/source data
        into the EdgePropertyStoreBuilder and writes attributes to
        a temporary LMDB keyed by original line index.
Pass 3: Sort numpy arrays by (src, dst, pred) via np.lexsort. Rewrite the temp
        LMDB in CSR-sorted order to produce the final LMDB where key == CSR
        edge index (zero indirection at query time). Reorder qualifier/source
        dedup indices to match. Build CSR offset arrays.

The build core (``_build_graph_from_source``) is agnostic to where records come
from: it consumes an abstract ``GraphSource`` of *already-normalized* nodes and
edges. Two thin wrappers select a source:

  * ``build_graph_from_jsonl`` reads KGX jsonl and normalizes (``KGXJsonlSource``).
  * ``build_graph_from_mongo`` reads pre-normalized documents (``MongoSource``).

Normalization itself lives in ``gandalf.normalize``; record validation lives in
``gandalf.sources.base``.
"""

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
from gandalf.sources import GraphSource, KGXJsonlSource

import logging
import lmdb

logger = logging.getLogger(__name__)


def _build_graph_from_source(source: GraphSource) -> CSRGraph:
    """Build a CSR graph from a :class:`GraphSource` using three-pass streaming.

    The source yields already-normalized, validated node and edge records (see
    ``gandalf.sources.base`` for the contract). Peak memory: ~3-4GB for 38M
    edges.
    """
    # =================================================================
    # Pass 1: Vocabulary collection
    # =================================================================
    logger.info("Pass 1: Collecting vocabularies...")

    node_ids = set()
    predicates = set()
    edge_count = 0

    for subject, object_, predicate in source.iter_edge_triples():
        node_ids.add(subject)
        node_ids.add(object_)
        predicates.add(predicate)
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

    # Load node properties (records are already normalized + validated)
    node_properties = {}
    logger.debug("Reading node properties...")
    for node_data in source.iter_nodes():
        idx = node_id_to_idx.get(node_data["id"])
        if idx is not None:
            node_properties[idx] = {
                "name": node_data.get("name", None),
                "categories": node_data.get("categories", []),
                "attributes": node_data.get("attributes", []),
            }
    if node_properties:
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

    # Edge IDs from the normalized "id" field (indexed by original iteration order)
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
        for i, edge in enumerate(source.iter_edges()):
            # Fill numpy arrays
            src_indices[i] = node_id_to_idx[edge["subject"]]
            dst_indices[i] = node_id_to_idx[edge["object"]]
            pred_indices[i] = predicate_to_idx[edge["predicate"]]

            # Capture edge ID from the normalized record (if present)
            edge_ids[i] = edge.get("id")

            # Hot path: intern qualifiers + sources (already normalized)
            prop_builder.add(
                i, {"sources": edge["sources"], "qualifiers": edge["qualifiers"]}
            )

            # Cold path: write attributes to temp LMDB
            # (publications are included in the attributes list)
            detail = {
                "attributes": edge["attributes"],
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
    graph.node_store = None

    graph.fwd_targets = dst_sorted
    graph.fwd_predicates = pred_sorted
    graph.fwd_offsets = fwd_offsets

    graph.rev_sources = rev_sources
    graph.rev_predicates = rev_predicates
    graph.rev_offsets = rev_offsets
    graph.rev_to_fwd = rev_to_fwd

    graph.edge_properties = edge_properties
    graph.edge_ids = edge_ids_sorted
    graph._edge_ids_env = None
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

    # Initialize the plugin-owned traversal metadata store. Bypassing
    # ``__init__`` via ``__new__`` above means we have to set this
    # explicitly before ``run_enrichers`` reads it.
    from gandalf.plugins.traversal_metadata_store import TraversalMetadataStore

    graph.traversal_metadata = TraversalMetadataStore.open_writable()

    # Run plugin enrichers so traversal_metadata is populated before any
    # query is executed against this graph.
    from gandalf.plugins.enrichers import run_enrichers

    run_enrichers(graph)

    return graph


def build_graph_from_jsonl(edge_jsonl_path, node_jsonl_path) -> CSRGraph:
    """Build a CSR graph from KGX jsonl files.

    Reads ``edge_jsonl_path`` / ``node_jsonl_path`` and applies gandalf's
    normalization, then builds the graph. ``node_jsonl_path`` may be falsy to
    build from edge endpoints only.
    """
    source = KGXJsonlSource(edge_jsonl_path, node_jsonl_path)
    return _build_graph_from_source(source)


def build_graph_from_mongo(
    *, mongo_uri: str, db: str, nodes_collection: str, edges_collection: str
) -> CSRGraph:
    """Build a CSR graph from already-normalized documents in MongoDB.

    The documents must already be in gandalf's normalized form (see
    ``gandalf.sources.base``); no normalization is applied. Requires ``pymongo``
    (install the ``mongo`` extra).
    """
    from gandalf.sources.mongo import MongoSource

    source = MongoSource(
        uri=mongo_uri,
        db=db,
        nodes_collection=nodes_collection,
        edges_collection=edges_collection,
    )
    try:
        return _build_graph_from_source(source)
    finally:
        source.close()
