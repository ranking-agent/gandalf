"""Path reconstruction from edge results using join operations."""

import os
import time
from collections import defaultdict

import numpy as np

from gandalf.search.path_arrays import PathArrays


# When path count exceeds this threshold, skip edge attribute enrichment
# (sources, qualifiers, publications, attributes from LMDB) and only include
# predicates. This avoids expensive per-edge property lookups on large result sets.
LARGE_RESULT_PATH_THRESHOLD = int(
    os.environ.get("GANDALF_LARGE_RESULT_THRESHOLD", "50000")
)

# Maximum number of intermediate paths allowed during join operations.
# When exceeded, paths are truncated to this limit and a warning is printed.
# Set to 0 to disable the limit.
MAX_PATH_LIMIT = int(
    os.environ.get("GANDALF_MAX_PATH_LIMIT", "0")
)


def reconstruct_paths(graph, query_graph, edge_results, edge_order, verbose,
                      edge_inverse_preds=None):
    """Reconstruct complete paths by iteratively joining edge results.

    Uses two-pass joins (count then fill) to avoid temporary Python lists
    and returns a compact PathArrays object instead of per-path dicts.

    Args:
        graph: CSRGraph instance
        query_graph: Original query graph
        edge_results: Dict of edge_id -> [(subj_idx, pred, obj_idx, via_inverse, fwd_edge_idx), ...]
        edge_order: List of edge IDs in original query order
        verbose: Print progress
        edge_inverse_preds: (Deprecated, kept for compatibility) Dict of edge_id -> set of inverse predicates

    Returns:
        PathArrays object, or None if no paths found
    """
    if len(edge_order) == 0:
        return None

    t0 = time.perf_counter()

    # Build join order based on query graph structure
    join_order = compute_join_order(query_graph, edge_results, edge_order, verbose)

    if verbose:
        print(f"  Join order: {join_order}")

    # Build mappings for query structure
    # qnode_id -> column index in node array
    qnode_to_col = {}
    # qedge_id -> column index in predicate array
    qedge_to_col = {eid: i for i, eid in enumerate(join_order)}

    # Build predicate vocabulary: predicate_string -> int
    predicate_to_idx = {}
    idx_to_predicate = []

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
    for i, (subj_idx, predicate, obj_idx, via_inverse, fwd_edge_idx) in enumerate(first_results):
        if via_inverse:
            paths_nodes[i, 0] = obj_idx
            paths_nodes[i, 1] = subj_idx
        else:
            paths_nodes[i, 0] = subj_idx
            paths_nodes[i, 1] = obj_idx
        paths_preds[i, 0] = get_pred_idx(predicate)
        paths_via_inverse[i, 0] = via_inverse
        paths_fwd_edge_idx[i, 0] = fwd_edge_idx

    if verbose:
        print(f"  Starting with {num_paths:,} paths from edge '{first_edge_id}'")

    # Iteratively join with remaining edges using two-pass approach:
    # Pass 1: count output rows, Pass 2: fill pre-allocated arrays.
    # This avoids creating millions of tiny numpy row copies in Python lists.
    for join_idx, edge_id in enumerate(join_order[1:], 1):
        edge = query_graph["edges"][edge_id]
        subj_qnode = edge["subject"]
        obj_qnode = edge["object"]

        if verbose:
            print(
                f"  Join {join_idx}/{len(join_order) - 1}: Adding edge '{edge_id}' ({len(paths_nodes):,} paths)...",
                end="",
            )

        t_join_start = time.perf_counter()

        subj_in_paths = subj_qnode in qnode_to_col
        obj_in_paths = obj_qnode in qnode_to_col

        edge_data = edge_results[edge_id]

        # Normalize edge data to query-aligned direction
        normalized_edge_data = []
        for subj_idx, predicate, obj_idx, via_inverse, fwd_edge_idx in edge_data:
            if via_inverse:
                normalized_edge_data.append((obj_idx, predicate, subj_idx, via_inverse, fwd_edge_idx))
            else:
                normalized_edge_data.append((subj_idx, predicate, obj_idx, via_inverse, fwd_edge_idx))
        edge_data = normalized_edge_data

        if subj_in_paths and obj_in_paths:
            paths_nodes, paths_preds, paths_via_inverse, paths_fwd_edge_idx = _join_both_in_paths(
                paths_nodes, paths_preds, paths_via_inverse, paths_fwd_edge_idx,
                edge_data, qnode_to_col[subj_qnode], qnode_to_col[obj_qnode],
                join_idx, max_nodes, num_edges, get_pred_idx,
            )

        elif subj_in_paths:
            if obj_qnode not in qnode_to_col:
                qnode_to_col[obj_qnode] = num_node_cols
                num_node_cols += 1
            paths_nodes, paths_preds, paths_via_inverse, paths_fwd_edge_idx = _join_on_subject(
                paths_nodes, paths_preds, paths_via_inverse, paths_fwd_edge_idx,
                edge_data, qnode_to_col[subj_qnode], qnode_to_col[obj_qnode],
                join_idx, max_nodes, num_edges, get_pred_idx,
            )

        elif obj_in_paths:
            if subj_qnode not in qnode_to_col:
                qnode_to_col[subj_qnode] = num_node_cols
                num_node_cols += 1
            paths_nodes, paths_preds, paths_via_inverse, paths_fwd_edge_idx = _join_on_object(
                paths_nodes, paths_preds, paths_via_inverse, paths_fwd_edge_idx,
                edge_data, qnode_to_col[subj_qnode], qnode_to_col[obj_qnode],
                join_idx, max_nodes, num_edges, get_pred_idx,
            )

        else:
            # Neither node in paths - cartesian product
            if verbose:
                print(f"\n    Warning: Cartesian product needed for edge '{edge_id}'")

            if subj_qnode not in qnode_to_col:
                qnode_to_col[subj_qnode] = num_node_cols
                num_node_cols += 1
            if obj_qnode not in qnode_to_col:
                qnode_to_col[obj_qnode] = num_node_cols
                num_node_cols += 1
            paths_nodes, paths_preds, paths_via_inverse, paths_fwd_edge_idx = _join_cartesian(
                paths_nodes, paths_preds, paths_via_inverse, paths_fwd_edge_idx,
                edge_data, qnode_to_col[subj_qnode], qnode_to_col[obj_qnode],
                join_idx, max_nodes, num_edges, get_pred_idx,
            )

        t_join_end = time.perf_counter()
        if verbose:
            print(f" -> {len(paths_nodes):,} paths ({t_join_end - t_join_start:.2f}s)")

        if len(paths_nodes) == 0:
            if verbose:
                print(f"  No valid paths found after joining edge '{edge_id}'")
            break

    t1 = time.perf_counter()
    if verbose:
        print(f"  Path reconstruction took {t1 - t0:.2f}s")

    num_paths = len(paths_nodes)
    if num_paths == 0:
        return None

    # Check if we exceed the large result threshold
    lightweight = num_paths > LARGE_RESULT_PATH_THRESHOLD

    if verbose:
        if lightweight:
            print(
                f"  {num_paths:,} paths (lightweight mode: "
                f">{LARGE_RESULT_PATH_THRESHOLD:,} paths, skipping edge attributes)..."
            )
        else:
            print(f"  {num_paths:,} paths")

    # Build node property cache
    t_cache_start = time.perf_counter()
    unique_node_indices = np.unique(paths_nodes[:, :num_node_cols])

    node_cache = {}
    for node_idx in unique_node_indices:
        node_props = graph.get_all_node_properties(node_idx).copy()
        node_props["id"] = graph.get_node_id(node_idx)
        if "categories" not in node_props:
            node_props["categories"] = []
        if "attributes" not in node_props:
            node_props["attributes"] = []
        node_cache[node_idx] = node_props

    t_cache_end = time.perf_counter()
    if verbose:
        print(
            f"  Cached properties for {len(unique_node_indices):,} unique nodes "
            f"({t_cache_end - t_cache_start:.2f}s)"
        )

    # Build reverse mappings
    col_to_qnode = {v: k for k, v in qnode_to_col.items()}
    col_to_qedge = {v: k for k, v in qedge_to_col.items()}

    return PathArrays(
        paths_nodes=paths_nodes,
        paths_preds=paths_preds,
        paths_via_inverse=paths_via_inverse,
        paths_fwd_edge_idx=paths_fwd_edge_idx,
        node_cache=node_cache,
        idx_to_predicate=idx_to_predicate,
        qnode_to_col=qnode_to_col,
        qedge_to_col=qedge_to_col,
        col_to_qnode=col_to_qnode,
        col_to_qedge=col_to_qedge,
        num_node_cols=num_node_cols,
        num_edges=num_edges,
        lightweight=lightweight,
    )


def _join_both_in_paths(
    paths_nodes, paths_preds, paths_via_inverse, paths_fwd_edge_idx,
    edge_data, subj_col, obj_col, join_idx, max_nodes, num_edges, get_pred_idx,
):
    """Join when both nodes already in path - validate consistency."""
    edge_index = defaultdict(list)
    for subj_idx, predicate, obj_idx, via_inverse, fwd_edge_idx in edge_data:
        edge_index[(subj_idx, obj_idx)].append((get_pred_idx(predicate), via_inverse, fwd_edge_idx))

    # Pass 1: Count output rows
    output_count = 0
    for path_idx in range(len(paths_nodes)):
        key = (int(paths_nodes[path_idx, subj_col]), int(paths_nodes[path_idx, obj_col]))
        if key in edge_index:
            output_count += len(edge_index[key])
    if MAX_PATH_LIMIT > 0 and output_count > MAX_PATH_LIMIT:
        print(f"\n    Warning: Truncating {output_count:,} intermediate paths to {MAX_PATH_LIMIT:,}")
        output_count = MAX_PATH_LIMIT

    # Pass 2: Fill pre-allocated arrays
    new_nodes = np.empty((output_count, max_nodes), dtype=np.int32)
    new_preds = np.empty((output_count, num_edges), dtype=np.int32)
    new_via_inv = np.empty((output_count, num_edges), dtype=np.bool_)
    new_fwd_eidx = np.empty((output_count, num_edges), dtype=np.int32)
    w = 0
    for path_idx in range(len(paths_nodes)):
        if w >= output_count:
            break
        key = (int(paths_nodes[path_idx, subj_col]), int(paths_nodes[path_idx, obj_col]))
        if key in edge_index:
            for pred_idx, via_inverse, fwd_edge_idx in edge_index[key]:
                if w >= output_count:
                    break
                new_nodes[w] = paths_nodes[path_idx]
                new_preds[w] = paths_preds[path_idx]
                new_preds[w, join_idx] = pred_idx
                new_via_inv[w] = paths_via_inverse[path_idx]
                new_via_inv[w, join_idx] = via_inverse
                new_fwd_eidx[w] = paths_fwd_edge_idx[path_idx]
                new_fwd_eidx[w, join_idx] = fwd_edge_idx
                w += 1

    return new_nodes, new_preds, new_via_inv, new_fwd_eidx


def _join_on_subject(
    paths_nodes, paths_preds, paths_via_inverse, paths_fwd_edge_idx,
    edge_data, subj_col, obj_col, join_idx, max_nodes, num_edges, get_pred_idx,
):
    """Join on subject node, add object node."""
    edge_index = defaultdict(list)
    for subj_idx, predicate, obj_idx, via_inverse, fwd_edge_idx in edge_data:
        edge_index[subj_idx].append((get_pred_idx(predicate), obj_idx, via_inverse, fwd_edge_idx))

    # Pass 1: Count
    output_count = 0
    for path_idx in range(len(paths_nodes)):
        sidx = int(paths_nodes[path_idx, subj_col])
        if sidx in edge_index:
            output_count += len(edge_index[sidx])
    if MAX_PATH_LIMIT > 0 and output_count > MAX_PATH_LIMIT:
        print(f"\n    Warning: Truncating {output_count:,} intermediate paths to {MAX_PATH_LIMIT:,}")
        output_count = MAX_PATH_LIMIT

    # Pass 2: Fill
    new_nodes = np.empty((output_count, max_nodes), dtype=np.int32)
    new_preds = np.empty((output_count, num_edges), dtype=np.int32)
    new_via_inv = np.empty((output_count, num_edges), dtype=np.bool_)
    new_fwd_eidx = np.empty((output_count, num_edges), dtype=np.int32)
    w = 0
    for path_idx in range(len(paths_nodes)):
        if w >= output_count:
            break
        sidx = int(paths_nodes[path_idx, subj_col])
        if sidx in edge_index:
            for pred_idx, obj_idx, via_inverse, fwd_edge_idx in edge_index[sidx]:
                if w >= output_count:
                    break
                new_nodes[w] = paths_nodes[path_idx]
                new_nodes[w, obj_col] = obj_idx
                new_preds[w] = paths_preds[path_idx]
                new_preds[w, join_idx] = pred_idx
                new_via_inv[w] = paths_via_inverse[path_idx]
                new_via_inv[w, join_idx] = via_inverse
                new_fwd_eidx[w] = paths_fwd_edge_idx[path_idx]
                new_fwd_eidx[w, join_idx] = fwd_edge_idx
                w += 1

    return new_nodes, new_preds, new_via_inv, new_fwd_eidx


def _join_on_object(
    paths_nodes, paths_preds, paths_via_inverse, paths_fwd_edge_idx,
    edge_data, subj_col, obj_col, join_idx, max_nodes, num_edges, get_pred_idx,
):
    """Join on object node, add subject node."""
    edge_index = defaultdict(list)
    for subj_idx, predicate, obj_idx, via_inverse, fwd_edge_idx in edge_data:
        edge_index[obj_idx].append((subj_idx, get_pred_idx(predicate), via_inverse, fwd_edge_idx))

    # Pass 1: Count
    output_count = 0
    for path_idx in range(len(paths_nodes)):
        oidx = int(paths_nodes[path_idx, obj_col])
        if oidx in edge_index:
            output_count += len(edge_index[oidx])
    if MAX_PATH_LIMIT > 0 and output_count > MAX_PATH_LIMIT:
        print(f"\n    Warning: Truncating {output_count:,} intermediate paths to {MAX_PATH_LIMIT:,}")
        output_count = MAX_PATH_LIMIT

    # Pass 2: Fill
    new_nodes = np.empty((output_count, max_nodes), dtype=np.int32)
    new_preds = np.empty((output_count, num_edges), dtype=np.int32)
    new_via_inv = np.empty((output_count, num_edges), dtype=np.bool_)
    new_fwd_eidx = np.empty((output_count, num_edges), dtype=np.int32)
    w = 0
    for path_idx in range(len(paths_nodes)):
        if w >= output_count:
            break
        oidx = int(paths_nodes[path_idx, obj_col])
        if oidx in edge_index:
            for sidx, pred_idx, via_inverse, fwd_edge_idx in edge_index[oidx]:
                if w >= output_count:
                    break
                new_nodes[w] = paths_nodes[path_idx]
                new_nodes[w, subj_col] = sidx
                new_preds[w] = paths_preds[path_idx]
                new_preds[w, join_idx] = pred_idx
                new_via_inv[w] = paths_via_inverse[path_idx]
                new_via_inv[w, join_idx] = via_inverse
                new_fwd_eidx[w] = paths_fwd_edge_idx[path_idx]
                new_fwd_eidx[w, join_idx] = fwd_edge_idx
                w += 1

    return new_nodes, new_preds, new_via_inv, new_fwd_eidx


def _join_cartesian(
    paths_nodes, paths_preds, paths_via_inverse, paths_fwd_edge_idx,
    edge_data, subj_col, obj_col, join_idx, max_nodes, num_edges, get_pred_idx,
):
    """Neither node in paths - cartesian product."""
    output_count = len(paths_nodes) * len(edge_data)
    if MAX_PATH_LIMIT > 0 and output_count > MAX_PATH_LIMIT:
        print(f"\n    Warning: Truncating {output_count:,} intermediate paths to {MAX_PATH_LIMIT:,}")
        output_count = MAX_PATH_LIMIT

    new_nodes = np.empty((output_count, max_nodes), dtype=np.int32)
    new_preds = np.empty((output_count, num_edges), dtype=np.int32)
    new_via_inv = np.empty((output_count, num_edges), dtype=np.bool_)
    new_fwd_eidx = np.empty((output_count, num_edges), dtype=np.int32)
    w = 0
    for path_idx in range(len(paths_nodes)):
        if w >= output_count:
            break
        for subj_idx, predicate, obj_idx, via_inverse, fwd_edge_idx in edge_data:
            if w >= output_count:
                break
            new_nodes[w] = paths_nodes[path_idx]
            new_nodes[w, subj_col] = subj_idx
            new_nodes[w, obj_col] = obj_idx
            new_preds[w] = paths_preds[path_idx]
            new_preds[w, join_idx] = get_pred_idx(predicate)
            new_via_inv[w] = paths_via_inverse[path_idx]
            new_via_inv[w, join_idx] = via_inverse
            new_fwd_eidx[w] = paths_fwd_edge_idx[path_idx]
            new_fwd_eidx[w, join_idx] = fwd_edge_idx
            w += 1

    return new_nodes, new_preds, new_via_inv, new_fwd_eidx


def compute_join_order(query_graph, edge_results, edge_order, verbose):
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
