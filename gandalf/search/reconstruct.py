"""Path reconstruction from edge results using join operations."""

import csv
import logging
import time
from datetime import datetime
from typing import Any, NamedTuple, Optional, cast

import numpy as np

from translator_tom.model_dicts import NodeDict

from gandalf.config import settings
from gandalf.profiler import current_profiler
from gandalf.search.path_arrays import PathArrays
from gandalf.trapi import ensure_node_category

logger = logging.getLogger(__name__)

# When set, write a debug TSV of all reconstructed paths.
# Set to a file path to write there, or "1"/"true" for an auto-named file.
DEBUG_PATHS_TSV = settings.debug_paths_tsv


# When path count exceeds this threshold, skip edge attribute enrichment
# (sources, qualifiers, attributes from LMDB) and only include
# predicates. This avoids expensive per-edge property lookups on large result sets.
LARGE_RESULT_PATH_THRESHOLD = settings.large_result_threshold

# Maximum number of intermediate paths allowed during join operations.
# When exceeded, paths are truncated to this limit and a warning is printed.
# Set to 0 to disable the limit.
MAX_PATH_LIMIT = settings.max_path_limit


def _get_most_specific_category(categories, bmt):
    """Return a single-element list with the most specific category.

    Uses BMT's ``get_descendants`` to find the leaf category — the one
    whose descendant set contains none of the other categories in the list.
    """
    if len(categories) <= 1:
        return list(categories)

    cat_set = set(categories)
    for cat in categories:
        try:
            descendants = set(bmt.get_descendants(cat, formatted=True))
        except Exception:
            descendants = set()
        if descendants & (cat_set - {cat}):
            # This category has a more-specific sibling in the list
            continue
        return [cat]

    # Fallback: return first element
    return [categories[0]]


def reconstruct_paths(
    graph,
    query_graph,
    edge_results,
    edge_order,
    edge_inverse_preds=None,
    dehydrated=None,
    bmt=None,
    logger: Optional[logging.Logger] = None,
):
    """Reconstruct complete paths by iteratively joining edge results.

    Uses two-pass joins (count then fill) to avoid temporary Python lists
    and returns a compact PathArrays object instead of per-path dicts.

    Args:
        graph: CSRGraph instance
        query_graph: Original query graph
        edge_results: Dict of edge_id -> [(subj_idx, pred, obj_idx, via_inverse, fwd_edge_idx), ...]
        edge_order: List of edge IDs in original query order
        edge_inverse_preds: (Deprecated, kept for compatibility) Dict of edge_id -> set of inverse predicates
        logger: Logger to emit this query's records to.  ``lookup`` passes the
            query's own logger so that entries reach that query's TRAPI logs
            and no other's.  Defaults to the module logger.

    Returns:
        PathArrays object, or None if no paths found
    """
    logger = logger if logger is not None else logging.getLogger(__name__)
    if len(edge_order) == 0:
        return None

    t0 = time.perf_counter()

    # Build join order based on query graph structure
    join_order = compute_join_order(query_graph, edge_results, edge_order)

    logger.debug("  Join order: %s", join_order)

    # Build mappings for query structure
    # qnode_id -> column index in node array
    qnode_to_col = {}
    # qedge_id -> column index in predicate array
    qedge_to_col = {eid: i for i, eid in enumerate(join_order)}

    # Build predicate vocabulary: predicate_string -> int
    predicate_to_idx: dict[str, int] = {}
    idx_to_predicate: list[str] = []

    def get_pred_idx(pred):
        if pred not in predicate_to_idx:
            predicate_to_idx[pred] = len(idx_to_predicate)
            idx_to_predicate.append(pred)
        return predicate_to_idx[pred]

    # Start with the first edge results
    first_edge_id = join_order[0]
    first_edge = query_graph["edges"][first_edge_id]
    subj_qnode = first_edge["subject"]
    obj_qnode = first_edge["object"]

    # Assign column indices for first edge's nodes
    qnode_to_col[subj_qnode] = 0
    qnode_to_col[obj_qnode] = 1
    num_node_cols = 2

    # Convert first edge results to numpy arrays
    first_results = edge_results[first_edge_id]
    num_paths = len(first_results)

    if num_paths == 0:
        return None

    # Pre-allocate arrays for nodes, predicates, and via_inverse flags
    max_nodes = len(query_graph["nodes"])
    num_edges = len(join_order)

    paths_nodes = np.zeros((num_paths, max_nodes), dtype=np.int32)
    paths_preds = np.zeros((num_paths, num_edges), dtype=np.int32)
    paths_via_inverse = np.zeros((num_paths, num_edges), dtype=np.bool_)
    paths_fwd_edge_idx = np.zeros((num_paths, num_edges), dtype=np.int32)

    # Fill in first edge data
    first = _edge_arrays(first_results, get_pred_idx)
    paths_nodes[:, 0] = first.subj
    paths_nodes[:, 1] = first.obj
    paths_preds[:, 0] = first.pred
    paths_via_inverse[:, 0] = first.via_inverse
    paths_fwd_edge_idx[:, 0] = first.fwd_edge_idx

    logger.debug(
        "  Starting with %s paths from edge '%s'", f"{num_paths:,}", first_edge_id
    )

    prof = current_profiler()

    # Iteratively join with remaining edges using two-pass approach:
    # Pass 1: count output rows, Pass 2: fill pre-allocated arrays.
    # This avoids creating millions of tiny numpy row copies in Python lists.
    for join_idx, edge_id in enumerate(join_order[1:], 1):
        edge = query_graph["edges"][edge_id]
        subj_qnode = edge["subject"]
        obj_qnode = edge["object"]

        logger.debug(
            "  Join %s/%s: Adding edge '%s' (%s paths)...",
            join_idx,
            len(join_order) - 1,
            edge_id,
            f"{len(paths_nodes):,}",
        )

        join_cm = prof.stage(
            "join", level=join_idx, edge_id=edge_id, paths_in=len(paths_nodes)
        )
        join_cm.__enter__()
        t_join_start = time.perf_counter()

        subj_in_paths = subj_qnode in qnode_to_col
        obj_in_paths = obj_qnode in qnode_to_col

        # The edge's matches in query-aligned direction, as arrays
        edge = _edge_arrays(edge_results[edge_id], get_pred_idx)

        if not subj_in_paths and not obj_in_paths:
            logger.debug("    Warning: Cartesian product needed for edge '%s'", edge_id)
        for qnode, in_paths in ((subj_qnode, subj_in_paths), (obj_qnode, obj_in_paths)):
            if not in_paths:
                qnode_to_col[qnode] = num_node_cols
                num_node_cols += 1
        subj_col = qnode_to_col[subj_qnode]
        obj_col = qnode_to_col[obj_qnode]

        # Join on whichever ends are already bound; the ends that are not
        # get filled in from the edge.  With neither bound, every key is
        # equal and the join is a cartesian product.
        if subj_in_paths and obj_in_paths:
            path_keys = _pair_keys(paths_nodes[:, subj_col], paths_nodes[:, obj_col])
            edge_keys = _pair_keys(edge.subj, edge.obj)
            new_node_cols = {}
        elif subj_in_paths:
            path_keys, edge_keys = paths_nodes[:, subj_col], edge.subj
            new_node_cols = {obj_col: edge.obj}
        elif obj_in_paths:
            path_keys, edge_keys = paths_nodes[:, obj_col], edge.obj
            new_node_cols = {subj_col: edge.subj}
        else:
            path_keys = np.zeros(len(paths_nodes), dtype=np.int8)
            edge_keys = np.zeros(len(edge.subj), dtype=np.int8)
            new_node_cols = {subj_col: edge.subj, obj_col: edge.obj}

        paths_nodes, paths_preds, paths_via_inverse, paths_fwd_edge_idx = _join(
            paths_nodes,
            paths_preds,
            paths_via_inverse,
            paths_fwd_edge_idx,
            path_keys,
            edge_keys,
            edge,
            new_node_cols,
            join_idx,
            logger=logger,
        )

        t_join_end = time.perf_counter()
        prof.add_metric("paths_out", len(paths_nodes))
        join_cm.__exit__(None, None, None)
        logger.debug(
            " -> %s paths (%.2fs)", f"{len(paths_nodes):,}", t_join_end - t_join_start
        )

        if len(paths_nodes) == 0:
            logger.debug("  No valid paths found after joining edge '%s'", edge_id)
            break

    t1 = time.perf_counter()
    logger.debug("  Path reconstruction took %.2fs", t1 - t0)

    num_paths = len(paths_nodes)
    if num_paths == 0:
        return None

    # Determine lightweight (dehydrated) mode: explicit request takes
    # precedence, otherwise fall back to the automatic path-count threshold.
    if dehydrated is not None:
        lightweight = dehydrated
    else:
        lightweight = num_paths > LARGE_RESULT_PATH_THRESHOLD

    if lightweight:
        if dehydrated:
            logger.debug(
                "  %s paths (lightweight mode: dehydrated response requested, "
                "skipping edge attributes)...",
                f"{num_paths:,}",
            )
        else:
            logger.debug(
                "  %s paths (lightweight mode: >%s paths, skipping edge attributes)...",
                f"{num_paths:,}",
                f"{LARGE_RESULT_PATH_THRESHOLD:,}",
            )
    else:
        logger.debug("  %s paths", f"{num_paths:,}")

    # Build node property cache
    t_cache_start = time.perf_counter()
    with prof.stage("node_cache_build"):
        unique_node_indices = np.unique(paths_nodes[:, :num_node_cols])

        # Fetch all node properties and IDs in two single-transaction batch
        # reads (sorted for B-tree locality) instead of one LMDB lookup per
        # node.  For result sets with hundreds of thousands of unique nodes
        # this collapses ~N transactions into 2.
        props_batch = graph.get_all_node_properties_batch(unique_node_indices)
        id_batch = graph.get_node_ids_batch(unique_node_indices)

        node_cache = {}
        node_id_cache = {}
        # Nodes share a few dozen distinct category lists, and resolving the
        # most specific one costs a BMT call per category, so resolve each
        # distinct list once.
        specific_categories: dict[tuple, list] = {}
        for node_idx in unique_node_indices:
            # pop (not get) so each property dict is freed as it is consumed,
            # avoiding a transient 2x peak alongside node_cache.
            all_props = props_batch.pop(int(node_idx), {})
            # TRAPI 2.0 admits no nulls and requires at least one category on
            # every Node, so a missing name yields no "name" key and a node
            # with no stored category falls back to NamedThing.  (Node
            # attributes may legitimately be an empty list.)
            node_props: NodeDict
            if lightweight and bmt is not None:
                categories = tuple(all_props.get("categories", []))
                specific = specific_categories.get(categories)
                if specific is None:
                    specific = _get_most_specific_category(list(categories), bmt)
                    specific_categories[categories] = specific
                # A list of its own per node, so no response code that
                # edits one node's categories can reach another's.
                node_props = {"categories": list(specific)}
                name = all_props.get("name")
                if name is not None:
                    node_props["name"] = name
            else:
                node_props = cast("NodeDict", all_props.copy())
                if node_props.get("name") is None:
                    node_props.pop("name", None)
                if "attributes" not in node_props:
                    node_props["attributes"] = []
            node_cache[node_idx] = ensure_node_category(node_props)
            node_id_cache[node_idx] = id_batch.get(int(node_idx))
        prof.add_metric("unique_nodes", int(len(unique_node_indices)))

    t_cache_end = time.perf_counter()
    logger.debug(
        "  Cached properties for %s unique nodes (%.2fs)",
        f"{len(unique_node_indices):,}",
        t_cache_end - t_cache_start,
    )

    # Build reverse mappings
    col_to_qnode = {v: k for k, v in qnode_to_col.items()}
    col_to_qedge = {v: k for k, v in qedge_to_col.items()}

    path_arrays = PathArrays(
        paths_nodes=paths_nodes,
        paths_preds=paths_preds,
        paths_via_inverse=paths_via_inverse,
        paths_fwd_edge_idx=paths_fwd_edge_idx,
        node_cache=node_cache,
        node_id_cache=node_id_cache,
        idx_to_predicate=idx_to_predicate,
        qnode_to_col=qnode_to_col,
        qedge_to_col=qedge_to_col,
        col_to_qnode=col_to_qnode,
        col_to_qedge=col_to_qedge,
        num_node_cols=num_node_cols,
        num_edges=num_edges,
        lightweight=lightweight,
    )

    if DEBUG_PATHS_TSV:
        _dump_debug_tsv(path_arrays, query_graph, join_order, graph, logger=logger)

    return path_arrays


def _dump_debug_tsv(
    path_arrays,
    query_graph,
    join_order,
    graph,
    logger: Optional[logging.Logger] = None,
):
    """Write all reconstructed paths to a TSV file for debugging.

    Each row is one path.  Columns are dynamically generated based on the
    number of hops so this works for arbitrary-length queries.

    Column layout (interleaved nodes and edges in path order):
        path_index,
        n0_qnode, n0_curie, n0_name, n0_category,
        e0_qedge, e0_predicate, e0_via_inverse, e0_fwd_edge_idx, e0_sources, e0_qualifiers,
        n1_qnode, n1_curie, n1_name, n1_category,
        ...
        nN_qnode, nN_curie, nN_name, nN_category
    """
    logger = logger if logger is not None else logging.getLogger(__name__)
    # Determine output path
    tsv_path = DEBUG_PATHS_TSV
    if tsv_path.lower() in ("1", "true", "yes"):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        tsv_path = f"debug_paths_{ts}.tsv"

    pa = path_arrays
    num_paths = len(pa)

    if num_paths == 0:
        logger.debug("  Debug TSV: no paths to write")
        return

    if num_paths > 1_000_000:
        logger.warning(
            "  Debug TSV: writing %s paths — file may be very large",
            f"{num_paths:,}",
        )

    # Build ordered sequence of (node_qid, edge_qid) pairs along the path.
    # Walk edges in join_order, tracking which nodes we've visited, to produce
    # a linear node-edge-node-edge-...-node sequence.
    ordered_nodes: list[str] = []  # qnode ids in path order
    ordered_edges = []  # qedge ids in path order (between consecutive nodes)

    visited_nodes = set()
    for edge_id in join_order:
        edge_def = query_graph["edges"][edge_id]
        subj_qnode = edge_def["subject"]
        obj_qnode = edge_def["object"]

        if not ordered_nodes:
            # First edge — add both nodes
            ordered_nodes.append(subj_qnode)
            ordered_nodes.append(obj_qnode)
            visited_nodes.add(subj_qnode)
            visited_nodes.add(obj_qnode)
            ordered_edges.append(edge_id)
        else:
            subj_in = subj_qnode in visited_nodes
            obj_in = obj_qnode in visited_nodes

            if subj_in and not obj_in:
                # Subject already in path; find where it is and insert edge+obj after it
                insert_pos = ordered_nodes.index(subj_qnode)
                ordered_nodes.insert(insert_pos + 1, obj_qnode)
                ordered_edges.insert(insert_pos, edge_id)
                visited_nodes.add(obj_qnode)
            elif obj_in and not subj_in:
                # Object already in path; insert subj+edge before it
                insert_pos = ordered_nodes.index(obj_qnode)
                ordered_nodes.insert(insert_pos, subj_qnode)
                ordered_edges.insert(insert_pos, edge_id)
                visited_nodes.add(subj_qnode)
            elif subj_in and obj_in:
                # Both already in path — insert edge between them
                si = ordered_nodes.index(subj_qnode)
                oi = ordered_nodes.index(obj_qnode)
                edge_insert = min(si, oi)
                ordered_edges.insert(edge_insert, edge_id)
            else:
                # Neither in path (cartesian) — append both
                ordered_nodes.append(subj_qnode)
                ordered_nodes.append(obj_qnode)
                ordered_edges.append(edge_id)
                visited_nodes.add(subj_qnode)
                visited_nodes.add(obj_qnode)

    # Check if we can look up edge properties (sources, qualifiers)
    has_edge_props = (
        hasattr(graph, "edge_properties") and graph.edge_properties is not None
    )

    # Build header
    header = ["path_index"]
    for i, qnode_id in enumerate(ordered_nodes):
        prefix = f"n{i}"
        header.extend(
            [
                f"{prefix}_qnode",
                f"{prefix}_curie",
                f"{prefix}_name",
                f"{prefix}_category",
            ]
        )
        if i < len(ordered_edges):
            eprefix = f"e{i}"
            header.extend(
                [
                    f"{eprefix}_qedge",
                    f"{eprefix}_predicate",
                    f"{eprefix}_via_inverse",
                    f"{eprefix}_fwd_edge_idx",
                    f"{eprefix}_sources",
                    f"{eprefix}_qualifiers",
                ]
            )

    try:
        with open(tsv_path, "w", newline="") as fh:
            writer = csv.writer(fh, delimiter="\t")
            writer.writerow(header)

            for path_idx in range(num_paths):
                row: list[Any] = [path_idx]
                for i, qnode_id in enumerate(ordered_nodes):
                    col = pa.qnode_to_col.get(qnode_id)
                    if col is not None:
                        node_idx = int(pa.paths_nodes[path_idx, col])
                        curie = pa.node_id_cache.get(node_idx, "")
                        props = pa.node_cache.get(node_idx, {})
                        name = props.get("name", "")
                        categories = props.get("categories", [])
                        category = categories[0] if categories else ""
                    else:
                        curie = ""
                        name = ""
                        category = ""
                    row.extend([qnode_id, curie, name, category])

                    if i < len(ordered_edges):
                        edge_id = ordered_edges[i]
                        ecol = pa.qedge_to_col.get(edge_id)
                        if ecol is not None:
                            pred_idx = int(pa.paths_preds[path_idx, ecol])
                            predicate = pa.idx_to_predicate[pred_idx]
                            via_inv: Any = bool(pa.paths_via_inverse[path_idx, ecol])
                            fwd_eidx = int(pa.paths_fwd_edge_idx[path_idx, ecol])
                        else:
                            predicate = ""
                            via_inv = ""
                            fwd_eidx = -1

                        # Look up sources and qualifiers from edge properties
                        sources_str = ""
                        quals_str = ""
                        if has_edge_props and fwd_eidx >= 0:
                            try:
                                sources = graph.edge_properties.get_sources(fwd_eidx)
                                if sources:
                                    sources_str = "|".join(
                                        f"{s.get('resource_id', '')}:{s.get('resource_role', '')}"
                                        for s in sources
                                    )
                            except Exception:
                                sources_str = "<error>"
                            try:
                                quals = graph.edge_properties.get_qualifiers(fwd_eidx)
                                if quals:
                                    quals_str = "|".join(
                                        f"{q.get('qualifier_type_id', '')}={q.get('qualifier_value', '')}"
                                        for q in quals
                                    )
                            except Exception:
                                quals_str = "<error>"

                        row.extend(
                            [
                                edge_id,
                                predicate,
                                via_inv,
                                fwd_eidx,
                                sources_str,
                                quals_str,
                            ]
                        )

                writer.writerow(row)

        logger.info("  Debug TSV: wrote %s paths to %s", f"{num_paths:,}", tsv_path)
    except OSError as exc:
        logger.error("  Debug TSV: failed to write %s: %s", tsv_path, exc)


class _EdgeArrays(NamedTuple):
    """One query edge's matches as parallel arrays, in query direction."""

    subj: np.ndarray
    obj: np.ndarray
    pred: np.ndarray
    via_inverse: np.ndarray
    fwd_edge_idx: np.ndarray


def _edge_arrays(edge_matches, get_pred_idx) -> _EdgeArrays:
    """Convert ``query_edge`` match tuples into query-aligned arrays.

    A match found through an inverse lookup is stored object-to-subject, so
    its ends are swapped back to the query's direction here.

    >>> vocab = {}
    >>> arrays = _edge_arrays(
    ...     [(1, "biolink:treats", 2, False, 7), (3, "biolink:treated_by", 4, True, 8)],
    ...     lambda p: vocab.setdefault(p, len(vocab)),
    ... )
    >>> arrays.subj.tolist(), arrays.obj.tolist(), arrays.pred.tolist()
    ([1, 4], [2, 3], [0, 1])
    """
    n = len(edge_matches)
    if n == 0:
        empty = np.empty(0, dtype=np.int32)
        return _EdgeArrays(empty, empty, empty, np.empty(0, dtype=np.bool_), empty)
    subj, preds, obj, via_inverse, fwd = zip(*edge_matches)
    subj_arr = np.fromiter(subj, dtype=np.int32, count=n)
    obj_arr = np.fromiter(obj, dtype=np.int32, count=n)
    inv_arr = np.fromiter(via_inverse, dtype=np.bool_, count=n)
    return _EdgeArrays(
        subj=np.where(inv_arr, obj_arr, subj_arr),
        obj=np.where(inv_arr, subj_arr, obj_arr),
        pred=np.fromiter((get_pred_idx(p) for p in preds), dtype=np.int32, count=n),
        via_inverse=inv_arr,
        fwd_edge_idx=np.fromiter(fwd, dtype=np.int32, count=n),
    )


def _pair_keys(subj: np.ndarray, obj: np.ndarray) -> np.ndarray:
    """Pack (subject, object) node index pairs into single int64 join keys."""
    return (subj.astype(np.int64) << 32) | obj.astype(np.int64)


def _join(
    paths_nodes,
    paths_preds,
    paths_via_inverse,
    paths_fwd_edge_idx,
    path_keys: np.ndarray,
    edge_keys: np.ndarray,
    edge: _EdgeArrays,
    new_node_cols: dict,
    join_idx: int,
    logger: Optional[logging.Logger] = None,
):
    """Extend each path with every edge match whose key equals the path's key.

    Output rows come in path order and, within a path, in the edge matches'
    original order.  When the output would exceed ``MAX_PATH_LIMIT`` it is
    truncated to the first ``MAX_PATH_LIMIT`` rows of that order.

    Args:
        paths_nodes, paths_preds, paths_via_inverse, paths_fwd_edge_idx: The
            partial paths so far.
        path_keys: One join key per path.
        edge_keys: One join key per edge match (same dtype as ``path_keys``).
        edge: The edge's matches, query-aligned.
        new_node_cols: ``{node column: array}`` for the edge ends not yet in
            the paths, to be filled from the matching edge.
        join_idx: The edge's column in the per-edge arrays.

    Returns:
        The joined ``(nodes, preds, via_inverse, fwd_edge_idx)`` arrays.
    """
    logger = logger if logger is not None else logging.getLogger(__name__)

    # Sort the edges by key (stably, so equal keys keep their original
    # order) and find each path's run of matching edges.
    edge_order = np.argsort(edge_keys, kind="stable")
    sorted_keys = edge_keys[edge_order]
    lo = np.searchsorted(sorted_keys, path_keys, side="left")
    counts = np.searchsorted(sorted_keys, path_keys, side="right") - lo

    output_count = int(counts.sum())
    if MAX_PATH_LIMIT > 0 and output_count > MAX_PATH_LIMIT:
        logger.warning(
            "Truncating %s intermediate paths to %s",
            f"{output_count:,}",
            f"{MAX_PATH_LIMIT:,}",
        )
        # Keep whole paths up to the limit, then part of the path that
        # crosses it, then nothing.
        ends = np.cumsum(counts)
        cut = int(np.searchsorted(ends, MAX_PATH_LIMIT, side="left"))
        counts = counts.copy()
        counts[cut] = MAX_PATH_LIMIT - (int(ends[cut - 1]) if cut > 0 else 0)
        counts[cut + 1 :] = 0
        output_count = MAX_PATH_LIMIT

    # Output row -> (source path, matching edge)
    path_of_row = np.repeat(np.arange(len(path_keys)), counts)
    run_starts = np.repeat(np.cumsum(counts) - counts, counts)
    offset_in_run = np.arange(output_count) - run_starts
    edge_of_row = edge_order[np.repeat(lo, counts) + offset_in_run]

    new_nodes = paths_nodes[path_of_row]
    for col, values in new_node_cols.items():
        new_nodes[:, col] = values[edge_of_row]
    new_preds = paths_preds[path_of_row]
    new_preds[:, join_idx] = edge.pred[edge_of_row]
    new_via_inv = paths_via_inverse[path_of_row]
    new_via_inv[:, join_idx] = edge.via_inverse[edge_of_row]
    new_fwd_eidx = paths_fwd_edge_idx[path_of_row]
    new_fwd_eidx[:, join_idx] = edge.fwd_edge_idx[edge_of_row]

    return new_nodes, new_preds, new_via_inv, new_fwd_eidx


def compute_join_order(query_graph, edge_results, edge_order):
    """Compute optimal join order to minimize intermediate results.

    Strategy:
    1. Start with smallest edge
    2. Greedily add edges that share nodes with current partial path
    3. Prefer edges that will filter (both nodes already in path)
    """
    remaining_edges = set(edge_order)
    join_order = []
    nodes_in_path = set()

    # Start with the edge with fewest results
    first_edge = min(remaining_edges, key=lambda e: len(edge_results.get(e, [])))
    join_order.append(first_edge)
    remaining_edges.remove(first_edge)

    # Add nodes from first edge to path
    first_edge_info = query_graph["edges"][first_edge]
    nodes_in_path.add(first_edge_info["subject"])
    nodes_in_path.add(first_edge_info["object"])

    # Greedily add remaining edges
    while remaining_edges:
        best_edge = None
        best_score = -1

        for edge_id in remaining_edges:
            edge = query_graph["edges"][edge_id]
            subj = edge["subject"]
            obj = edge["object"]

            # Score based on:
            # - How many nodes are already in path (higher is better for joining)
            # - Size of edge results (smaller is better)
            nodes_shared = (subj in nodes_in_path) + (obj in nodes_in_path)
            result_size = len(edge_results.get(edge_id, []))

            # Prefer edges with shared nodes, then smaller result sets
            score = nodes_shared * 1000000000 - result_size

            if score > best_score:
                best_score = score
                best_edge = edge_id

        join_order.append(best_edge)
        remaining_edges.remove(best_edge)

        # Add new nodes to path
        edge_info = query_graph["edges"][best_edge]
        nodes_in_path.add(edge_info["subject"])
        nodes_in_path.add(edge_info["object"])

    return join_order
