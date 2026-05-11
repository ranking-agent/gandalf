"""Edge query functions for graph traversal with predicate/qualifier filtering."""

import logging
import time
from typing import Optional

from gandalf.profiler import current_profiler
from gandalf.search.attribute_constraints import matches_attribute_constraints
from gandalf.search.node_filters import NodeFilter, apply_node_filters
from gandalf.search.qualifiers import edge_matches_qualifier_constraints

logger = logging.getLogger(__name__)


def query_subclass_edge(graph, start_idxes, end_idxes, depth):
    """Traverse ``subclass_of`` edges to find subclass relationships.

    The synthetic subclass edge connects a child node (subject) to a
    superclass node (object).  The superclass node holds the original
    pinned IDs, so ``end_idxes`` will be pinned.

    We perform a BFS starting from each pinned end (superclass) node,
    following **incoming** ``subclass_of`` edges up to *depth* hops.
    Depth 0 means the node itself (identity -- no hop needed).

    Args:
        graph: CSRGraph instance
        start_idxes: Indices for the child (subject) side, or None
        end_idxes: Indices for the superclass (object) side
        depth: Maximum subclass_of hops

    Returns:
        List of (child_idx, "biolink:subclass_of", parent_idx, False, fwd_edge_idx) tuples.
        The depth-0 self-match is included with fwd_edge_idx=-1 (no real edge).
    """
    matches = []

    # Resolve the subclass_of predicate index once
    subclass_pred = "biolink:subclass_of"
    subclass_pred_filter = {subclass_pred}

    if end_idxes is None:
        return matches

    for superclass_idx in end_idxes:
        # BFS: current frontier -> next frontier, up to `depth` levels
        # Depth 0 = identity match (the node itself)
        frontier = {superclass_idx}
        visited = {superclass_idx}

        # Always include the depth-0 self-match (no real edge, sentinel -1)
        matches.append((superclass_idx, subclass_pred, superclass_idx, False, -1))

        for _hop in range(depth):
            next_frontier = set()
            for node_idx in frontier:
                # Walk incoming subclass_of edges: child --subclass_of--> node_idx
                for (
                    child_idx,
                    _predicate,
                    _props,
                    fwd_eidx,
                ) in graph.incoming_neighbors_with_properties(
                    node_idx, predicate_filter=subclass_pred_filter
                ):
                    if child_idx in visited:
                        continue
                    visited.add(child_idx)
                    next_frontier.add(child_idx)
                    matches.append(
                        (child_idx, subclass_pred, superclass_idx, False, fwd_eidx)
                    )
            frontier = next_frontier
            if not frontier:
                break

    logger.debug(
        "  Subclass traversal: found %s matches (depth=%s)", len(matches), depth
    )

    return matches


def query_edge(
    graph,
    start_idxes,
    end_idxes,
    start_categories,
    end_categories,
    allowed_predicates,
    qualifier_constraints,
    inverse_predicates: Optional[list[str]] = None,
    node_filters: Optional[list[NodeFilter]] = None,
    attribute_constraints: Optional[list] = None,
    start_node_constraints: Optional[list] = None,
    end_node_constraints: Optional[list] = None,
):
    """Query for a single edge with given constraints.

    Handles symmetric and inverse predicates at query time by checking both
    edge directions when appropriate. For example, if searching for predicate P
    and P has inverse Q, edges stored as B--Q-->A will be returned as A--P-->B.

    Args:
        graph: CSRGraph instance
        start_idxes: List of pinned start node indices, or None if unpinned
        end_idxes: List of pinned end node indices, or None if unpinned
        start_categories: List of allowed categories for start node
        end_categories: List of allowed categories for end node
        allowed_predicates: List of forward predicate strings (canonical/symmetric descendants)
        qualifier_constraints: List of qualifier constraint dicts from query
        inverse_predicates: List of inverse predicate strings for reverse direction
            matching. None means don't check inverse direction. Empty list means
            match all predicates in inverse direction (wildcard).
        node_filters: Pre-built list of NodeFilter closures (from
            ``build_node_filters``). Empty list / None means no filtering.
        attribute_constraints: List of TRAPI AttributeConstraint dicts for
            filtering edges by their attributes. All must be satisfied (AND).
        start_node_constraints: List of TRAPI AttributeConstraint dicts for
            filtering the start (subject) node by its attributes.
        end_node_constraints: List of TRAPI AttributeConstraint dicts for
            filtering the end (object) node by its attributes.

    Returns:
        List of (subject_idx, predicate, object_idx, via_inverse, fwd_edge_idx) tuples where
        via_inverse indicates if the edge was found through inverse/symmetric lookup and
        fwd_edge_idx is the forward-CSR array position (unique per physical edge).
    """
    if node_filters is None:
        node_filters = []
    matches = []
    seen_edges = set()  # Track (subj, pred, obj, fwd_edge_idx) to avoid duplicates

    # Build set of inverse predicates for quick lookup.
    # None  -> don't check inverse direction at all (default)
    # []    -> match ALL predicates in inverse direction (wildcard, e.g. related_to)
    # [pred]-> match only the listed predicates in inverse direction
    check_inverse = inverse_predicates is not None
    inverse_pred_set = set(inverse_predicates) if inverse_predicates else set()

    def add_match(subj_idx, predicate, obj_idx, fwd_edge_idx, via_inverse=False):
        """Add a match, avoiding duplicates. Includes via_inverse flag.

        Dedup key includes ``fwd_edge_idx`` so that edges with the same
        (subj, pred, obj) but different qualifiers / sources are kept as
        separate matches.  It also includes ``via_inverse`` because the
        same physical edge found in both forward and inverse directions
        represents two distinct query bindings (e.g. SN=A,h=B vs SN=B,h=A
        for symmetric predicates).
        """
        key = (subj_idx, predicate, obj_idx, fwd_edge_idx, via_inverse)
        if key not in seen_edges:
            seen_edges.add(key)
            matches.append((subj_idx, predicate, obj_idx, via_inverse, fwd_edge_idx))

    prof = current_profiler()
    n_allowed_preds = len(allowed_predicates) if allowed_predicates else 0
    n_inverse_preds = len(inverse_pred_set) if check_inverse else 0

    # Case 1: Start pinned, end unpinned
    if start_idxes is not None and end_idxes is None:
        with prof.stage(
            "query_forward",
            n_pinned=len(start_idxes),
            n_predicates=n_allowed_preds,
            n_inverse_preds=n_inverse_preds,
            check_inverse=check_inverse,
            n_end_categories=len(end_categories) if end_categories else 0,
            has_attribute_constraints=bool(attribute_constraints),
        ):
            _query_forward(
                graph,
                start_idxes,
                allowed_predicates,
                end_categories,
                qualifier_constraints,
                check_inverse,
                inverse_pred_set,
                add_match,
                node_filters=node_filters,
                attribute_constraints=attribute_constraints,
                start_node_constraints=start_node_constraints,
                end_node_constraints=end_node_constraints,
            )

    # Case 2: Start unpinned, end pinned
    elif start_idxes is None and end_idxes is not None:
        with prof.stage(
            "query_backward",
            n_pinned=len(end_idxes),
            n_predicates=n_allowed_preds,
            n_inverse_preds=n_inverse_preds,
            check_inverse=check_inverse,
            n_start_categories=len(start_categories) if start_categories else 0,
            has_attribute_constraints=bool(attribute_constraints),
        ):
            _query_backward(
                graph,
                end_idxes,
                allowed_predicates,
                start_categories,
                qualifier_constraints,
                check_inverse,
                inverse_pred_set,
                add_match,
                node_filters=node_filters,
                attribute_constraints=attribute_constraints,
                start_node_constraints=start_node_constraints,
                end_node_constraints=end_node_constraints,
            )

    # Case 3: Both pinned
    elif start_idxes is not None and end_idxes is not None:
        with prof.stage(
            "query_both_pinned",
            n_start=len(start_idxes),
            n_end=len(end_idxes),
            n_predicates=n_allowed_preds,
            n_inverse_preds=n_inverse_preds,
            check_inverse=check_inverse,
            has_attribute_constraints=bool(attribute_constraints),
        ):
            _query_both_pinned(
                graph,
                start_idxes,
                end_idxes,
                allowed_predicates,
                qualifier_constraints,
                check_inverse,
                inverse_pred_set,
                add_match,
                node_filters=node_filters,
                attribute_constraints=attribute_constraints,
                start_node_constraints=start_node_constraints,
                end_node_constraints=end_node_constraints,
            )

    else:
        raise Exception("Both nodes unpinned - bad query planning")

    prof.add_metric("matches", len(matches))
    return matches


def _query_forward(
    graph,
    start_idxes,
    allowed_predicates,
    end_categories,
    qualifier_constraints,
    check_inverse,
    inverse_pred_set,
    add_match,
    node_filters=None,
    attribute_constraints=None,
    start_node_constraints=None,
    end_node_constraints=None,
):
    """Case 1: Start pinned, end unpinned - forward search from pinned nodes."""
    logger.debug("  Forward search from %s pinned nodes", len(start_idxes))

    t0 = time.perf_counter()

    fwd_pred_filter = set(allowed_predicates) if allowed_predicates else None
    inv_pred_filter = inverse_pred_set if (check_inverse and inverse_pred_set) else None

    neighbors_scanned = 0
    neighbors_after_pred = 0
    fwd_offsets = graph.fwd_offsets
    rev_offsets = graph.rev_offsets
    slow_nodes = []  # Track nodes that take > 0.1s

    for start_idx in start_idxes:
        # Check start node attribute constraints once per start node
        if start_node_constraints:
            start_attrs = graph.get_node_property(start_idx, "attributes", [])
            if not matches_attribute_constraints(start_attrs, start_node_constraints):
                continue

        t_node_start = time.perf_counter()
        node_neighbors = 0

        # Count edges scanned at the CSR level (before predicate filter)
        neighbors_scanned += int(fwd_offsets[start_idx + 1] - fwd_offsets[start_idx])
        if check_inverse:
            neighbors_scanned += int(
                rev_offsets[start_idx + 1] - rev_offsets[start_idx]
            )

        # Check outgoing edges (direct matches) - predicate filtered at CSR level
        for obj_idx, predicate, props, fwd_edge_idx in graph.neighbors_with_properties(
            start_idx, predicate_filter=fwd_pred_filter
        ):
            node_neighbors += 1

            # Check object categories
            if end_categories:
                obj_cats = graph.get_node_property(obj_idx, "categories", [])
                if not any(cat in obj_cats for cat in end_categories):
                    continue

            # Check node filters (plugin-defined: degree, IC, etc.)
            if not apply_node_filters(node_filters, graph, obj_idx):
                continue

            # Check end node attribute constraints
            if end_node_constraints:
                obj_attrs = graph.get_node_property(obj_idx, "attributes", [])
                if not matches_attribute_constraints(obj_attrs, end_node_constraints):
                    continue

            # Check qualifier constraints
            if qualifier_constraints:
                edge_qualifiers = props.get("qualifiers", [])
                if not edge_matches_qualifier_constraints(
                    edge_qualifiers, qualifier_constraints
                ):
                    continue

            # Check edge attribute constraints (cold path: LMDB lookup)
            if attribute_constraints:
                if not _edge_passes_attribute_constraints(
                    graph, fwd_edge_idx, attribute_constraints
                ):
                    continue

            add_match(start_idx, predicate, obj_idx, fwd_edge_idx)

        # Check incoming edges for symmetric/inverse predicates
        # An incoming edge with inverse(P) represents an outgoing edge with P
        if check_inverse:
            for (
                other_idx,
                stored_pred,
                props,
                fwd_edge_idx,
            ) in graph.incoming_neighbors_with_properties(
                start_idx, predicate_filter=inv_pred_filter
            ):
                node_neighbors += 1

                # Check object categories (the "other" node becomes our object)
                if end_categories:
                    obj_cats = graph.get_node_property(other_idx, "categories", [])
                    if not any(cat in obj_cats for cat in end_categories):
                        continue

                # Check node filters (plugin-defined: degree, IC, etc.)
                if not apply_node_filters(node_filters, graph, other_idx):
                    continue

                # Check end node attribute constraints (other_idx is the "object" via inverse)
                if end_node_constraints:
                    obj_attrs = graph.get_node_property(other_idx, "attributes", [])
                    if not matches_attribute_constraints(
                        obj_attrs, end_node_constraints
                    ):
                        continue

                # Check qualifier constraints
                if qualifier_constraints:
                    edge_qualifiers = props.get("qualifiers", [])
                    if not edge_matches_qualifier_constraints(
                        edge_qualifiers, qualifier_constraints
                    ):
                        continue

                # Check edge attribute constraints (cold path: LMDB lookup)
                if attribute_constraints:
                    if not _edge_passes_attribute_constraints(
                        graph, fwd_edge_idx, attribute_constraints
                    ):
                        continue

                # Report the actual edge as stored in the graph
                # The edge is: other_idx --[stored_pred]--> start_idx
                # Mark as via_inverse since found through inverse lookup
                add_match(
                    other_idx, stored_pred, start_idx, fwd_edge_idx, via_inverse=True
                )

        t_node_end = time.perf_counter()
        node_time = t_node_end - t_node_start
        neighbors_after_pred += node_neighbors

        if node_time > 0.1:  # Track slow nodes
            slow_nodes.append((start_idx, node_neighbors, node_time))

    t1 = time.perf_counter()
    _record_traversal_metrics(
        graph, neighbors_scanned, neighbors_after_pred, slow_nodes
    )
    logger.debug(
        "  Scanned %s edges, %s passed predicate filter",
        neighbors_scanned,
        neighbors_after_pred,
    )
    if slow_nodes:
        logger.debug("  Slow nodes (>0.1s): %s", len(slow_nodes))
        for node_idx, neighbors, node_time in slow_nodes[:5]:  # Show top 5
            logger.debug(
                "    Node %s: %s neighbors, %.2fs", node_idx, neighbors, node_time
            )
    logger.debug("  Forward search completed in %.3fs", t1 - t0)


def _query_backward(
    graph,
    end_idxes,
    allowed_predicates,
    start_categories,
    qualifier_constraints,
    check_inverse,
    inverse_pred_set,
    add_match,
    node_filters=None,
    attribute_constraints=None,
    start_node_constraints=None,
    end_node_constraints=None,
):
    """Case 2: Start unpinned, end pinned - backward search from pinned nodes."""
    logger.debug("  Backward search from %s pinned nodes", len(end_idxes))

    t0 = time.perf_counter()

    fwd_pred_filter = set(allowed_predicates) if allowed_predicates else None
    inv_pred_filter = inverse_pred_set if (check_inverse and inverse_pred_set) else None

    neighbors_scanned = 0
    neighbors_after_pred = 0
    fwd_offsets = graph.fwd_offsets
    rev_offsets = graph.rev_offsets
    slow_nodes = []  # Track nodes that take > 0.1s

    for i, end_idx in enumerate(end_idxes):
        # Check end node attribute constraints once per end node
        if end_node_constraints:
            end_attrs = graph.get_node_property(end_idx, "attributes", [])
            if not matches_attribute_constraints(end_attrs, end_node_constraints):
                continue

        t_node_start = time.perf_counter()
        node_neighbors = 0

        # Count edges scanned at the CSR level (before predicate filter)
        neighbors_scanned += int(rev_offsets[end_idx + 1] - rev_offsets[end_idx])
        if check_inverse:
            neighbors_scanned += int(fwd_offsets[end_idx + 1] - fwd_offsets[end_idx])

        # Check incoming edges (direct matches) - predicate filtered at CSR level
        for (
            subj_idx,
            predicate,
            props,
            fwd_edge_idx,
        ) in graph.incoming_neighbors_with_properties(
            end_idx, predicate_filter=fwd_pred_filter
        ):
            node_neighbors += 1

            # Check subject categories
            if start_categories:
                subj_cats = graph.get_node_property(subj_idx, "categories", [])
                if not any(cat in subj_cats for cat in start_categories):
                    continue

            # Check node filters (plugin-defined: degree, IC, etc.)
            if not apply_node_filters(node_filters, graph, subj_idx):
                continue

            # Check start node attribute constraints
            if start_node_constraints:
                subj_attrs = graph.get_node_property(subj_idx, "attributes", [])
                if not matches_attribute_constraints(
                    subj_attrs, start_node_constraints
                ):
                    continue

            # Check qualifier constraints
            if qualifier_constraints:
                edge_qualifiers = props.get("qualifiers", [])
                if not edge_matches_qualifier_constraints(
                    edge_qualifiers, qualifier_constraints
                ):
                    continue

            # Check edge attribute constraints (cold path: LMDB lookup)
            if attribute_constraints:
                if not _edge_passes_attribute_constraints(
                    graph, fwd_edge_idx, attribute_constraints
                ):
                    continue

            add_match(subj_idx, predicate, end_idx, fwd_edge_idx)

        # Check outgoing edges for symmetric/inverse predicates
        # An outgoing edge with inverse(P) represents an incoming edge with P
        if check_inverse:
            for (
                other_idx,
                stored_pred,
                props,
                fwd_edge_idx,
            ) in graph.neighbors_with_properties(
                end_idx, predicate_filter=inv_pred_filter
            ):
                node_neighbors += 1

                # Check subject categories (the "other" node becomes our subject)
                if start_categories:
                    subj_cats = graph.get_node_property(other_idx, "categories", [])
                    if not any(cat in subj_cats for cat in start_categories):
                        continue

                # Check node filters (plugin-defined: degree, IC, etc.)
                if not apply_node_filters(node_filters, graph, other_idx):
                    continue

                # Check start node attribute constraints (other_idx is the "subject" via inverse)
                if start_node_constraints:
                    subj_attrs = graph.get_node_property(other_idx, "attributes", [])
                    if not matches_attribute_constraints(
                        subj_attrs, start_node_constraints
                    ):
                        continue

                # Check qualifier constraints
                if qualifier_constraints:
                    edge_qualifiers = props.get("qualifiers", [])
                    if not edge_matches_qualifier_constraints(
                        edge_qualifiers, qualifier_constraints
                    ):
                        continue

                # Check edge attribute constraints (cold path: LMDB lookup)
                if attribute_constraints:
                    if not _edge_passes_attribute_constraints(
                        graph, fwd_edge_idx, attribute_constraints
                    ):
                        continue

                # Report the actual edge as stored in the graph
                # The edge is: end_idx --[stored_pred]--> other_idx
                # Mark as via_inverse since found through inverse lookup
                add_match(
                    end_idx, stored_pred, other_idx, fwd_edge_idx, via_inverse=True
                )

        t_node_end = time.perf_counter()
        node_time = t_node_end - t_node_start
        neighbors_after_pred += node_neighbors

        if node_time > 0.1:  # Track slow nodes
            slow_nodes.append((end_idx, node_neighbors, node_time))

    t1 = time.perf_counter()
    _record_traversal_metrics(
        graph, neighbors_scanned, neighbors_after_pred, slow_nodes
    )
    logger.debug(
        "  Scanned %s edges, %s passed predicate filter",
        neighbors_scanned,
        neighbors_after_pred,
    )
    if slow_nodes:
        logger.debug("  Slow nodes (>0.1s): %s", len(slow_nodes))
        for node_idx, neighbors, node_time in slow_nodes[:5]:  # Show top 5
            logger.debug(
                "    Node %s: %s neighbors, %.2fs", node_idx, neighbors, node_time
            )


def _query_both_pinned(
    graph,
    start_idxes,
    end_idxes,
    allowed_predicates,
    qualifier_constraints,
    check_inverse,
    inverse_pred_set,
    add_match,
    node_filters=None,
    attribute_constraints=None,
    start_node_constraints=None,
    end_node_constraints=None,
):
    """Case 3: Both ends pinned - intersection search."""
    logger.debug(
        "  Both ends pinned: %s start, %s end", len(start_idxes), len(end_idxes)
    )

    t0 = time.perf_counter()

    # Build target set up front so we can filter during traversal instead
    # of accumulating a large forward_edges dict and intersecting later.
    # This avoids property-dict allocations for edges whose target is not
    # in end_set (the vast majority in typical queries).
    end_set = set(end_idxes)
    pred_filter_set = set(allowed_predicates) if allowed_predicates else None

    t_neighbors_start = time.perf_counter()
    neighbors_scanned = 0
    neighbors_after_pred = 0
    slow_nodes = []

    for start_idx in start_idxes:
        # Check node filters on the start node
        if not apply_node_filters(node_filters, graph, start_idx):
            continue

        # Check start node attribute constraints
        if start_node_constraints:
            start_attrs = graph.get_node_property(start_idx, "attributes", [])
            if not matches_attribute_constraints(start_attrs, start_node_constraints):
                continue

        t_node_start = time.perf_counter()

        # Count total neighbors for diagnostics (cheap CSR offset math)
        node_start = int(graph.fwd_offsets[start_idx])
        node_end = int(graph.fwd_offsets[start_idx + 1])
        node_neighbors = node_end - node_start
        neighbors_scanned += node_neighbors

        # Only fetch properties for edges whose target is in end_set
        for (
            obj_idx,
            predicate,
            props,
            fwd_edge_idx,
        ) in graph.neighbors_filtered_by_targets(
            start_idx, end_set, predicate_filter=pred_filter_set
        ):
            neighbors_after_pred += 1

            # Check node filters on the end node
            if not apply_node_filters(node_filters, graph, obj_idx):
                continue

            # Check end node attribute constraints
            if end_node_constraints:
                obj_attrs = graph.get_node_property(obj_idx, "attributes", [])
                if not matches_attribute_constraints(obj_attrs, end_node_constraints):
                    continue

            # Check qualifier constraints inline
            if qualifier_constraints:
                edge_qualifiers = props.get("qualifiers", [])
                if not edge_matches_qualifier_constraints(
                    edge_qualifiers, qualifier_constraints
                ):
                    continue

            # Check edge attribute constraints (cold path: LMDB lookup)
            if attribute_constraints:
                if not _edge_passes_attribute_constraints(
                    graph, fwd_edge_idx, attribute_constraints
                ):
                    continue

            add_match(start_idx, predicate, obj_idx, fwd_edge_idx)

        t_node_end = time.perf_counter()
        node_time = t_node_end - t_node_start
        if node_time > 0.1:
            slow_nodes.append((start_idx, node_neighbors, node_time))

    # Also check reverse direction for symmetric/inverse predicates
    # Look for edges: end_node --inverse(P)--> start_node
    if check_inverse:
        start_set = set(start_idxes)
        for end_idx in end_idxes:
            # Check node filters on the end node
            if not apply_node_filters(node_filters, graph, end_idx):
                continue

            # Check end node attribute constraints
            if end_node_constraints:
                end_attrs = graph.get_node_property(end_idx, "attributes", [])
                if not matches_attribute_constraints(end_attrs, end_node_constraints):
                    continue

            # Count edges scanned at the CSR level (end_idx outgoing)
            neighbors_scanned += int(
                graph.fwd_offsets[end_idx + 1] - graph.fwd_offsets[end_idx]
            )

            for (
                obj_idx,
                stored_pred,
                props,
                fwd_edge_idx,
            ) in graph.neighbors_filtered_by_targets(
                end_idx, start_set, predicate_filter=inverse_pred_set or None
            ):
                neighbors_after_pred += 1

                # Check node filters on the target (start) node
                if not apply_node_filters(node_filters, graph, obj_idx):
                    continue

                # Check start node attribute constraints (obj_idx is a start node)
                if start_node_constraints:
                    subj_attrs = graph.get_node_property(obj_idx, "attributes", [])
                    if not matches_attribute_constraints(
                        subj_attrs, start_node_constraints
                    ):
                        continue

                # Check qualifier constraints before adding
                if qualifier_constraints:
                    edge_qualifiers = props.get("qualifiers", [])
                    if not edge_matches_qualifier_constraints(
                        edge_qualifiers, qualifier_constraints
                    ):
                        continue

                # Check edge attribute constraints (cold path: LMDB lookup)
                if attribute_constraints:
                    if not _edge_passes_attribute_constraints(
                        graph, fwd_edge_idx, attribute_constraints
                    ):
                        continue

                # Report the actual edge as stored in the graph
                # The edge is: end_idx --[stored_pred]--> obj_idx
                # (where obj_idx is a start node)
                # Mark as via_inverse since found through inverse lookup
                add_match(end_idx, stored_pred, obj_idx, fwd_edge_idx, via_inverse=True)

    t1 = time.perf_counter()
    _record_traversal_metrics(
        graph, neighbors_scanned, neighbors_after_pred, slow_nodes
    )
    logger.debug(
        "    Neighbor traversal: %.3fs (%s scanned, %s after pred filter)",
        t1 - t_neighbors_start,
        neighbors_scanned,
        neighbors_after_pred,
    )
    if slow_nodes:
        logger.debug("    Slow nodes (>0.1s): %s", len(slow_nodes))
        for node_idx, neighbors, node_time in slow_nodes[:5]:
            logger.debug(
                "      Node %s: %s neighbors, %.2fs", node_idx, neighbors, node_time
            )


_SLOW_NODE_EVENT_LIMIT = 10


def _record_traversal_metrics(
    graph, neighbors_scanned, neighbors_after_pred, slow_nodes
):
    """Surface neighborhood-size and slow-node detail to the profiler.

    Aggregates onto the currently-active query_* stage. ``slow_nodes`` is
    the list ``[(node_idx, neighbors, duration_seconds), ...]`` collected by
    the search loop for nodes that took longer than the inline threshold.

    Two complementary edge-count metrics are recorded:
    - ``neighbors_scanned``: total CSR edges considered (predicate filter
      applied at the integer level). Cheap to compute from offsets.
    - ``neighbors_after_pred``: edges that survived predicate filtering and
      entered the Python loop. The ratio of the two is the predicate hit
      rate — the headline diagnostic for query_forward / query_backward.
    """
    prof = current_profiler()
    prof.add_metric("neighbors_scanned", int(neighbors_scanned))
    prof.add_metric("neighbors_after_pred", int(neighbors_after_pred))
    prof.add_metric("slow_nodes", len(slow_nodes))
    if not slow_nodes:
        return
    # Top-N by wall time so big offenders aren't crowded out by the cap.
    top = sorted(slow_nodes, key=lambda r: r[2], reverse=True)[:_SLOW_NODE_EVENT_LIMIT]
    for node_idx, neighbors, node_time in top:
        node_id = None
        try:
            node_id = graph.get_node_id(node_idx)
        except Exception:
            pass
        prof.event(
            "slow_node",
            node_idx=int(node_idx),
            node_id=node_id,
            neighbors=int(neighbors),
            duration_ms=node_time * 1000.0,
        )


def _edge_passes_attribute_constraints(graph, fwd_edge_idx, attribute_constraints):
    """Check if an edge's attributes satisfy all attribute constraints.

    Fetches edge attributes from LMDB (cold path) and applies the constraint
    matching logic.  Only called for edges that already passed all other
    filters (predicates, categories, qualifiers), so the number of LMDB
    lookups is bounded by the surviving candidate set.
    """
    if graph.lmdb_store is None:
        # No LMDB store — no attributes to check against
        return False

    detail = graph.lmdb_store.get(fwd_edge_idx)
    edge_attrs = detail.get("attributes", [])
    return matches_attribute_constraints(edge_attrs, attribute_constraints)
