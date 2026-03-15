"""3-hop path finding functions for direct graph traversal."""

import logging
import time
from collections import defaultdict

import numpy as np

from gandalf.graph import CSRGraph

logger = logging.getLogger(__name__)


def _return_with_properties(
    graph: CSRGraph, paths: list[list[int]]
) -> list[dict[str, dict[str, str]]]:
    """Given paths, return them with useful properties attached."""
    start_time = time.time()
    logger.debug("Assembling enriched paths...")
    hydrated_paths = []
    for path_idx in paths:
        [start_idx, n1_idx, n2_idx, end_idx] = path_idx
        # Get predicates - take the first one if multiple exist
        edges_01 = graph.get_all_edges_between(start_idx, n1_idx)
        edges_12 = graph.get_all_edges_between(n1_idx, n2_idx)
        edges_23 = graph.get_all_edges_between(n2_idx, end_idx)

        pred_01 = edges_01[0][0] if edges_01 else None
        pred_12 = edges_12[0][0] if edges_12 else None
        pred_23 = edges_23[0][0] if edges_23 else None
        n0 = {
            "id": graph.get_node_id(start_idx),
            "category": graph.get_node_property(start_idx, "category", []),
            "name": graph.get_node_property(start_idx, "name"),
        }
        e0 = {
            "predicate": pred_01,
        }
        n1 = {
            "id": graph.get_node_id(n1_idx),
            "category": graph.get_node_property(n1_idx, "category", []),
            "name": graph.get_node_property(n1_idx, "name"),
        }
        e1 = {
            "predicate": pred_12,
        }
        n2 = {
            "id": graph.get_node_id(n2_idx),
            "category": graph.get_node_property(n2_idx, "category", []),
            "name": graph.get_node_property(n2_idx, "name"),
        }
        e2 = {
            "predicate": pred_23,
        }
        n3 = {
            "id": graph.get_node_id(end_idx),
            "category": graph.get_node_property(end_idx, "category", []),
            "name": graph.get_node_property(end_idx, "name"),
        }
        hydrated_paths.append(
            {
                "n0": n0,
                "e0": e0,
                "n1": n1,
                "e1": e1,
                "n2": n2,
                "e2": e2,
                "n3": n3,
            }
        )
    logger.debug(
        "Done! Hydrating %s paths took %s",
        f"{len(hydrated_paths):,}",
        time.time() - start_time,
    )
    return hydrated_paths


def find_3hop_paths_with_properties(graph: CSRGraph, start_id, end_id, max_paths=None):
    """Find all 3-hop paths between two nodes with edge and node properties.

    Returns list of dicts with path information.

    Args:
        graph: CSRGraph instance
        start_id: Starting node ID
        end_id: Ending node ID
        max_paths: If specified, only enrich the first N paths (for performance)
    """
    # Convert IDs to indices
    start_idx = graph.get_node_idx(start_id)
    end_idx = graph.get_node_idx(end_id)

    if start_idx is None or end_idx is None:
        return []

    if start_idx == end_idx:
        return []

    logger.debug(
        "Start node '%s' has degree: %s", start_id, f"{graph.degree(start_idx):,}"
    )
    logger.debug("End node '%s' has degree: %s", end_id, f"{graph.degree(end_idx):,}")

    # Get raw paths (as indices)
    paths_idx = _find_3hop_paths_directed_idx(
        graph,
        start_idx,
        end_idx,
        start_from_end=(graph.degree(start_idx) > graph.degree(end_idx)),
    )

    logger.debug("Found %s paths", f"{len(paths_idx):,}")

    # Limit paths if requested
    if max_paths and len(paths_idx) > max_paths:
        logger.debug(
            "Limiting to first %s paths for property enrichment", f"{max_paths:,}"
        )
        paths_idx = paths_idx[:max_paths]

    return _return_with_properties(graph, paths_idx)


def _find_3hop_paths_directed_idx(graph, start_idx, end_idx, start_from_end=False):
    """Helper: find paths using indices, returns paths as index lists."""
    if start_from_end:
        # Search from end to start, then reverse
        paths = _do_unfiltered_search(graph, end_idx, start_idx)
        return [[p[3], p[2], p[1], p[0]] for p in paths]
    else:
        return _do_unfiltered_search(graph, start_idx, end_idx)


def _do_unfiltered_search(graph: CSRGraph, start_idx, end_idx):
    """Actual bidirectional search implementation returning full 3-hop paths."""

    # Forward: start -> n1
    t0 = time.perf_counter()
    forward_1 = graph.neighbors(start_idx)
    t1 = time.perf_counter()
    logger.debug("Imatinib 1-hop: %s %s", len(forward_1), t1 - t0)

    if len(forward_1) == 0:
        return []

    # Forward: start -> n1 -> n2  (vectorized parent tracking)
    t0 = time.perf_counter()

    src_bufs = []  # n1 indices
    dst_bufs = []  # n2 indices

    for n1_idx in forward_1:
        if n1_idx == end_idx:  # skip direct edge
            continue

        n2s = graph.neighbors(n1_idx)
        if len(n2s) == 0:
            continue

        src_bufs.append(np.full(len(n2s), n1_idx, dtype=np.int32))
        dst_bufs.append(n2s)

    if not src_bufs:
        return []

    src_n1 = np.concatenate(src_bufs)
    dst_n2 = np.concatenate(dst_bufs)

    t1 = time.perf_counter()
    logger.debug("Imatinib 2-hop edges: %s %s", len(dst_n2), t1 - t0)

    # Backward: end -> n2
    t0 = time.perf_counter()
    backward_1 = graph.incoming_neighbors(end_idx)
    t1 = time.perf_counter()
    logger.debug("Asthma 1-hop: %s %s", len(backward_1), t1 - t0)

    if len(backward_1) == 0:
        return []

    backward_unique = np.unique(backward_1)

    # Intersection: keep only valid n2
    t0 = time.perf_counter()
    mask = np.isin(dst_n2, backward_unique, assume_unique=False)
    src_n1 = src_n1[mask]
    dst_n2 = dst_n2[mask]
    t1 = time.perf_counter()
    logger.debug("Intersection edges: %s %s", len(dst_n2), t1 - t0)

    if len(dst_n2) == 0:
        return []

    # Assemble full paths (vectorized)
    paths = np.column_stack(
        [
            np.full(len(src_n1), start_idx, dtype=np.int32),
            src_n1,
            dst_n2,
            np.full(len(src_n1), end_idx, dtype=np.int32),
        ]
    )

    return paths.tolist()


def find_3hop_paths_filtered(
    graph: CSRGraph,
    start_id,
    end_id,
    allowed_predicates=None,
    excluded_predicates=None,
):
    """Find 3-hop paths with predicate filtering.

    Args:
        graph: CSRGraph instance
        start_id: Starting node ID
        end_id: Ending node ID
        allowed_predicates: If provided, only use edges with these predicates
        excluded_predicates: If provided, skip edges with these predicates
    """
    start_idx = graph.get_node_idx(start_id)
    end_idx = graph.get_node_idx(end_id)

    if start_idx is None or end_idx is None:
        return []

    if start_idx == end_idx:
        return []

    # Helper to check if predicate is allowed
    def is_predicate_allowed(predicate):
        if excluded_predicates and predicate in excluded_predicates:
            return False
        if allowed_predicates and predicate not in allowed_predicates:
            return False
        return True

    logger.debug(
        "Start node '%s' has degree: %s", start_id, f"{graph.degree(start_idx):,}"
    )
    logger.debug("End node '%s' has degree: %s", end_id, f"{graph.degree(end_idx):,}")

    # Forward: start -> n1 (with filtering)
    # Get all edges from start, filter by predicate
    forward_1_filtered = []  # List of (n1_idx, predicate) tuples

    for n1_idx, predicate, _, _ in graph.neighbors_with_properties(start_idx):
        if n1_idx == end_idx:
            continue
        if is_predicate_allowed(predicate):
            forward_1_filtered.append((n1_idx, predicate))

    logger.debug("After filtering edge 1: %s neighbors", f"{len(forward_1_filtered):,}")

    if len(forward_1_filtered) == 0:
        return []

    # Build mapping: n2_idx -> list of (n1_idx, pred_01, pred_12) tuples
    # This tracks both intermediate nodes and the predicates used
    forward_paths = defaultdict(list)

    for n1_idx, pred_01 in forward_1_filtered:
        for n2_idx, pred_12, _, _ in graph.neighbors_with_properties(n1_idx):
            if n2_idx == start_idx:
                continue
            if is_predicate_allowed(pred_12):
                forward_paths[n2_idx].append((n1_idx, pred_01, pred_12))

    logger.debug("Nodes reachable in 2 hops: %s", f"{len(forward_paths):,}")

    # Check which n2 nodes connect to end
    backward_connections: dict[int, list] = {}  # n2_idx -> predicate (n2 -> end)

    for n2_idx in forward_paths.keys():
        # Get all edges from n2 to end
        edges_to_end = graph.get_all_edges_between(n2_idx, end_idx)
        for predicate, _ in edges_to_end:
            if is_predicate_allowed(predicate):
                # Store first valid predicate (or could store all)
                if n2_idx not in backward_connections:
                    backward_connections[n2_idx] = []
                backward_connections[n2_idx].append(predicate)

    logger.debug(
        "After filtering edge 3: %s nodes connect to end",
        f"{len(backward_connections):,}",
    )

    # Find intersection and build paths
    paths = []
    for n2_idx in forward_paths:
        if n2_idx in backward_connections:
            # For each path through this n2
            for n1_idx, pred_01, pred_12 in forward_paths[n2_idx]:
                # For each valid predicate from n2 to end
                for pred_23 in backward_connections[n2_idx]:
                    n0 = {
                        "id": graph.get_node_id(start_idx),
                        "category": graph.get_node_property(start_idx, "category", []),
                        "name": graph.get_node_property(start_idx, "name"),
                    }
                    e0 = {
                        "predicate": pred_01,
                    }
                    n1 = {
                        "id": graph.get_node_id(n1_idx),
                        "category": graph.get_node_property(n1_idx, "category", []),
                        "name": graph.get_node_property(n1_idx, "name"),
                    }
                    e1 = {
                        "predicate": pred_12,
                    }
                    n2 = {
                        "id": graph.get_node_id(n2_idx),
                        "category": graph.get_node_property(n2_idx, "category", []),
                        "name": graph.get_node_property(n2_idx, "name"),
                    }
                    e2 = {
                        "predicate": pred_23,
                    }
                    n3 = {
                        "id": graph.get_node_id(end_idx),
                        "category": graph.get_node_property(end_idx, "category", []),
                        "name": graph.get_node_property(end_idx, "name"),
                    }
                    paths.append(
                        {
                            "n0": n0,
                            "e0": e0,
                            "n1": n1,
                            "e1": e1,
                            "n2": n2,
                            "e2": e2,
                            "n3": n3,
                        }
                    )

    logger.debug("Found %s filtered paths", f"{len(paths):,}")

    return paths


def find_mechanistic_paths(graph: CSRGraph, start_id, end_id):
    """Find paths using only mechanistic/causal predicates.

    Good for finding actual biological mechanisms rather than associations.
    """
    mechanistic_predicates = {
        "biolink:treats",
        "biolink:affects",
        "biolink:regulates",
        "biolink:increases_expression_of",
        "biolink:decreases_expression_of",
        "biolink:gene_associated_with_condition",
        "biolink:has_metabolite",
        "biolink:metabolized_by",
        "biolink:applied_to_treat",
        "biolink:contraindicated_for",
        "biolink:directly_physically_interacts_with",
        "biolink:has_contraindication",
        "biolink:subject_of_treatment_application_or_study_for_treatment_by",
        "biolink:contribution_from",
    }

    return find_3hop_paths_filtered(
        graph,
        start_id,
        end_id,
        allowed_predicates=mechanistic_predicates,
    )


def do_one_hop(graph: CSRGraph, start_id: str):
    """Get all neighbors from a single node."""
    start_idx = graph.get_node_idx(start_id)

    neighbors = graph.neighbors(start_idx)

    return neighbors
