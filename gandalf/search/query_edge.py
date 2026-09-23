"""Edge query functions for graph traversal with predicate/qualifier filtering."""

import logging
import time
from typing import Optional

from gandalf.profiler import current_profiler
from gandalf.search.attribute_constraints import matches_attribute_constraints
from gandalf.search.edge_constraints import EdgeConstraints
from gandalf.search.node_filters import NodeFilter, apply_node_filters

logger = logging.getLogger(__name__)


def query_subclass_edge(
    graph,
    start_idxes,
    end_idxes,
    depth,
    logger: Optional[logging.Logger] = None,
):
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
        logger: Logger to emit this query's records to.  Defaults to the
            module logger.

    Returns:
        List of (child_idx, "biolink:subclass_of", parent_idx, False, fwd_edge_idx) tuples.
        The depth-0 self-match is included with fwd_edge_idx=-1 (no real edge).
    """
    logger = logger if logger is not None else logging.getLogger(__name__)
    matches: list[tuple] = []

    # Resolve the subclass_of predicate index once
    subclass_pred = "biolink:subclass_of"

    if end_idxes is None:
        return matches

    subclass_mask = graph.predicate_mask([subclass_pred])

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
                    node_idx, predicate_mask=subclass_mask
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
    edge_constraints: EdgeConstraints,
    inverse_predicates: Optional[list[str]] = None,
    node_filters: Optional[list[NodeFilter]] = None,
    start_node_constraints: Optional[list] = None,
    end_node_constraints: Optional[list] = None,
    logger: Optional[logging.Logger] = None,
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
        edge_constraints: The QEdge's parsed TRAPI 2.0 ``constraints`` object
            (qualifiers, attributes, knowledge_level, agent_type, sources).
            An empty ``EdgeConstraints`` filters nothing.
        inverse_predicates: List of inverse predicate strings for reverse direction
            matching. None means don't check inverse direction. Empty list means
            match all predicates in inverse direction (wildcard).
        node_filters: Pre-built list of NodeFilter closures (from
            ``build_node_filters``). Empty list / None means no filtering.
        start_node_constraints: List of TRAPI AttributeConstraint dicts for
            filtering the start (subject) node by its attributes.
        end_node_constraints: List of TRAPI AttributeConstraint dicts for
            filtering the end (object) node by its attributes.
        logger: Logger to emit this query's records to.  ``lookup`` passes the
            query's own logger so that entries reach that query's TRAPI logs
            and no other's.  Defaults to the module logger.

    Returns:
        List of (subject_idx, predicate, object_idx, via_inverse, fwd_edge_idx) tuples where
        via_inverse indicates if the edge was found through inverse/symmetric lookup and
        fwd_edge_idx is the forward-CSR array position (unique per physical edge).
    """
    logger = logger if logger is not None else logging.getLogger(__name__)
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

    # Predicate filters as lookup tables over predicate ids, built once so
    # that each node's neighbour slice is filtered in one vectorized step.
    # None means "any predicate".
    forward_mask = (
        graph.predicate_mask(allowed_predicates) if allowed_predicates else None
    )
    inverse_mask = graph.predicate_mask(inverse_pred_set) if inverse_pred_set else None

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
            has_edge_constraints=bool(edge_constraints),
        ):
            _query_forward(
                graph,
                start_idxes,
                forward_mask,
                end_categories,
                edge_constraints,
                check_inverse,
                inverse_mask,
                add_match,
                node_filters=node_filters,
                start_node_constraints=start_node_constraints,
                end_node_constraints=end_node_constraints,
                logger=logger,
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
            has_edge_constraints=bool(edge_constraints),
        ):
            _query_backward(
                graph,
                end_idxes,
                forward_mask,
                start_categories,
                edge_constraints,
                check_inverse,
                inverse_mask,
                add_match,
                node_filters=node_filters,
                start_node_constraints=start_node_constraints,
                end_node_constraints=end_node_constraints,
                logger=logger,
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
            has_edge_constraints=bool(edge_constraints),
        ):
            _query_both_pinned(
                graph,
                start_idxes,
                end_idxes,
                forward_mask,
                edge_constraints,
                check_inverse,
                inverse_mask,
                add_match,
                node_filters=node_filters,
                start_node_constraints=start_node_constraints,
                end_node_constraints=end_node_constraints,
                logger=logger,
            )

    else:
        raise Exception("Both nodes unpinned - bad query planning")

    prof.add_metric("matches", len(matches))
    return matches


def _node_checker(graph, categories, node_filters, attribute_constraints):
    """Build a memoized test of whether a node satisfies a query node.

    Checks the node's categories (any of *categories*), the plugin node
    filters, and the query node's attribute constraints.  The node record is
    read once per node, and the verdict is cached for the rest of the edge
    query: hub nodes are reached from many pinned nodes, and each read
    unpacks the node's whole stored record.

    Returns:
        A ``node_idx -> bool`` callable, or None when there is nothing to
        check (every node passes).
    """
    if not categories and not node_filters and not attribute_constraints:
        return None
    verdicts: dict[int, bool] = {}

    def passes(node_idx: int) -> bool:
        verdict = verdicts.get(node_idx)
        if verdict is None:
            verdict = _node_passes(
                graph, node_idx, categories, node_filters, attribute_constraints
            )
            verdicts[node_idx] = verdict
        return verdict

    return passes


def _node_passes(graph, node_idx, categories, node_filters, attribute_constraints):
    """Uncached check behind :func:`_node_checker`."""
    props: dict = {}
    if categories or attribute_constraints:
        props = graph.get_all_node_properties(node_idx)
    if categories:
        node_categories = props.get("categories", [])
        if not any(cat in node_categories for cat in categories):
            return False
    if not apply_node_filters(node_filters, graph, node_idx):
        return False
    if attribute_constraints and not matches_attribute_constraints(
        props.get("attributes", []), attribute_constraints
    ):
        return False
    return True


def _query_forward(
    graph,
    start_idxes,
    forward_mask,
    end_categories,
    edge_constraints,
    check_inverse,
    inverse_mask,
    add_match,
    node_filters=None,
    start_node_constraints=None,
    end_node_constraints=None,
    logger: Optional[logging.Logger] = None,
):
    """Case 1: Start pinned, end unpinned - forward search from pinned nodes.

    ``forward_mask`` / ``inverse_mask`` are :meth:`CSRGraph.predicate_mask`
    tables, or None for "any predicate".
    """
    logger = logger if logger is not None else logging.getLogger(__name__)
    logger.debug("  Forward search from %s pinned nodes", len(start_idxes))

    t0 = time.perf_counter()

    total_neighbors = 0
    slow_nodes = []  # Track nodes that take > 0.1s

    start_ok = _node_checker(graph, None, None, start_node_constraints)
    end_ok = _node_checker(graph, end_categories, node_filters, end_node_constraints)

    for start_idx in start_idxes:
        # Check start node attribute constraints once per start node
        if start_ok is not None and not start_ok(start_idx):
            continue

        t_node_start = time.perf_counter()
        node_neighbors = 0

        # Check outgoing edges (direct matches)
        for obj_idx, predicate, props, fwd_edge_idx in graph.neighbors_with_properties(
            start_idx, predicate_mask=forward_mask
        ):
            node_neighbors += 1

            # Check the object: categories, node filters, attribute constraints
            if end_ok is not None and not end_ok(obj_idx):
                continue

            # Check the QEdge's constraints (qualifiers, knowledge_level,
            # agent_type, sources, attributes)
            if edge_constraints and not edge_constraints.permits(
                graph, props, fwd_edge_idx
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
                start_idx, predicate_mask=inverse_mask
            ):
                node_neighbors += 1

                # The "other" node becomes our object
                if end_ok is not None and not end_ok(other_idx):
                    continue

                # Check the QEdge's constraints (qualifiers, knowledge_level,
                # agent_type, sources, attributes)
                if edge_constraints and not edge_constraints.permits(
                    graph, props, fwd_edge_idx
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
        total_neighbors += node_neighbors

        if node_time > 0.1:  # Track slow nodes
            slow_nodes.append((start_idx, node_neighbors, node_time))

    t1 = time.perf_counter()
    _record_traversal_metrics(graph, total_neighbors, slow_nodes)
    logger.debug("  Traversed %s total neighbors", total_neighbors)
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
    forward_mask,
    start_categories,
    edge_constraints,
    check_inverse,
    inverse_mask,
    add_match,
    node_filters=None,
    start_node_constraints=None,
    end_node_constraints=None,
    logger: Optional[logging.Logger] = None,
):
    """Case 2: Start unpinned, end pinned - backward search from pinned nodes.

    Predicate masks are as for :func:`_query_forward`.
    """
    logger = logger if logger is not None else logging.getLogger(__name__)
    logger.debug("  Backward search from %s pinned nodes", len(end_idxes))

    t0 = time.perf_counter()

    total_neighbors = 0
    slow_nodes = []  # Track nodes that take > 0.1s

    end_ok = _node_checker(graph, None, None, end_node_constraints)
    start_ok = _node_checker(
        graph, start_categories, node_filters, start_node_constraints
    )

    for end_idx in end_idxes:
        # Check end node attribute constraints once per end node
        if end_ok is not None and not end_ok(end_idx):
            continue

        t_node_start = time.perf_counter()
        node_neighbors = 0

        # Check incoming edges (direct matches)
        for (
            subj_idx,
            predicate,
            props,
            fwd_edge_idx,
        ) in graph.incoming_neighbors_with_properties(
            end_idx, predicate_mask=forward_mask
        ):
            node_neighbors += 1

            # Check the subject: categories, node filters, attribute constraints
            if start_ok is not None and not start_ok(subj_idx):
                continue

            # Check the QEdge's constraints (qualifiers, knowledge_level,
            # agent_type, sources, attributes)
            if edge_constraints and not edge_constraints.permits(
                graph, props, fwd_edge_idx
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
            ) in graph.neighbors_with_properties(end_idx, predicate_mask=inverse_mask):
                node_neighbors += 1

                # The "other" node becomes our subject
                if start_ok is not None and not start_ok(other_idx):
                    continue

                # Check the QEdge's constraints (qualifiers, knowledge_level,
                # agent_type, sources, attributes)
                if edge_constraints and not edge_constraints.permits(
                    graph, props, fwd_edge_idx
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
        total_neighbors += node_neighbors

        if node_time > 0.1:  # Track slow nodes
            slow_nodes.append((end_idx, node_neighbors, node_time))

    t1 = time.perf_counter()
    _record_traversal_metrics(graph, total_neighbors, slow_nodes)
    logger.debug("  Traversed %s total incoming neighbors", total_neighbors)
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
    forward_mask,
    edge_constraints,
    check_inverse,
    inverse_mask,
    add_match,
    node_filters=None,
    start_node_constraints=None,
    end_node_constraints=None,
    logger: Optional[logging.Logger] = None,
):
    """Case 3: Both ends pinned - intersection search.

    Predicate masks are as for :func:`_query_forward`.
    """
    logger = logger if logger is not None else logging.getLogger(__name__)
    logger.debug(
        "  Both ends pinned: %s start, %s end", len(start_idxes), len(end_idxes)
    )

    t0 = time.perf_counter()

    # Build target set up front so we can filter during traversal instead
    # of accumulating a large forward_edges dict and intersecting later.
    # This avoids property-dict allocations for edges whose target is not
    # in end_set (the vast majority in typical queries).
    end_set = set(end_idxes)

    # Both ends are pinned, so categories are not rechecked: only the node
    # filters and each end's attribute constraints apply.
    start_ok = _node_checker(graph, None, node_filters, start_node_constraints)
    end_ok = _node_checker(graph, None, node_filters, end_node_constraints)

    t_neighbors_start = time.perf_counter()
    total_neighbors = 0
    slow_nodes = []

    for start_idx in start_idxes:
        # Check node filters and attribute constraints on the start node
        if start_ok is not None and not start_ok(start_idx):
            continue

        t_node_start = time.perf_counter()

        # Count total neighbors for diagnostics (cheap CSR offset math)
        node_start = int(graph.fwd_offsets[start_idx])
        node_end = int(graph.fwd_offsets[start_idx + 1])
        node_neighbors = node_end - node_start
        total_neighbors += node_neighbors

        # Only fetch properties for edges whose target is in end_set
        for (
            obj_idx,
            predicate,
            props,
            fwd_edge_idx,
        ) in graph.neighbors_filtered_by_targets(
            start_idx, end_set, predicate_mask=forward_mask
        ):
            # Check node filters and attribute constraints on the end node
            if end_ok is not None and not end_ok(obj_idx):
                continue

            # Check the QEdge's constraints (qualifiers, knowledge_level,
            # agent_type, sources, attributes)
            if edge_constraints and not edge_constraints.permits(
                graph, props, fwd_edge_idx
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
            # Check node filters and attribute constraints on the end node
            if end_ok is not None and not end_ok(end_idx):
                continue

            for (
                obj_idx,
                stored_pred,
                props,
                fwd_edge_idx,
            ) in graph.neighbors_filtered_by_targets(
                end_idx, start_set, predicate_mask=inverse_mask
            ):
                total_neighbors += 1

                # obj_idx is a start node: check its filters and constraints
                if start_ok is not None and not start_ok(obj_idx):
                    continue

                # Check the QEdge's constraints (qualifiers, knowledge_level,
                # agent_type, sources, attributes)
                if edge_constraints and not edge_constraints.permits(
                    graph, props, fwd_edge_idx
                ):
                    continue

                # Report the actual edge as stored in the graph
                # The edge is: end_idx --[stored_pred]--> obj_idx
                # (where obj_idx is a start node)
                # Mark as via_inverse since found through inverse lookup
                add_match(end_idx, stored_pred, obj_idx, fwd_edge_idx, via_inverse=True)

    t1 = time.perf_counter()
    _record_traversal_metrics(graph, total_neighbors, slow_nodes)
    logger.debug(
        "    Neighbor traversal: %.3fs (%s neighbors)",
        t1 - t_neighbors_start,
        total_neighbors,
    )
    if slow_nodes:
        logger.debug("    Slow nodes (>0.1s): %s", len(slow_nodes))
        for node_idx, neighbors, node_time in slow_nodes[:5]:
            logger.debug(
                "      Node %s: %s neighbors, %.2fs", node_idx, neighbors, node_time
            )


_SLOW_NODE_EVENT_LIMIT = 10


def _record_traversal_metrics(graph, total_neighbors, slow_nodes):
    """Surface neighborhood-size and slow-node detail to the profiler.

    Aggregates onto the currently-active query_* stage. ``slow_nodes`` is
    the list ``[(node_idx, neighbors, duration_seconds), ...]`` collected by
    the search loop for nodes that took longer than the inline threshold.
    """
    prof = current_profiler()
    prof.add_metric("total_neighbors", int(total_neighbors))
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
