"""Edge query functions for graph traversal with predicate/qualifier filtering."""

import logging
import time
from typing import Optional

from gandalf.search.attribute_constraints import matches_attribute_constraints
from gandalf.search.qualifiers import edge_matches_qualifier_constraints

logger = logging.getLogger(__name__)


def _get_information_content(graph, node_idx):
    """Extract the information_content value from a node's attributes."""
    attrs = graph.get_node_property(node_idx, "attributes", [])
    for attr in attrs:
        if attr.get("original_attribute_name") == "information_content":
            return attr.get("value")
    return None


def _node_total_degree(graph, node_idx):
    """Compute total degree (outgoing + incoming) for a node."""
    out_deg = int(graph.fwd_offsets[node_idx + 1] - graph.fwd_offsets[node_idx])
    in_deg = int(graph.rev_offsets[node_idx + 1] - graph.rev_offsets[node_idx])
    return out_deg + in_deg


def _passes_node_filters(graph, node_idx, max_node_degree, min_information_content):
    """Check if a node passes degree and information content filters."""
    if max_node_degree is not None:
        if _node_total_degree(graph, node_idx) > max_node_degree:
            return False
    if min_information_content is not None:
        ic = _get_information_content(graph, node_idx)
        if ic is None or ic < min_information_content:
            return False
    return True


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
                    predicate,
                    _props,
                    fwd_eidx,
                ) in graph.incoming_neighbors_with_properties(node_idx):
                    if predicate != subclass_pred:
                        continue
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
    max_node_degree: Optional[int] = None,
    min_information_content: Optional[float] = None,
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
        max_node_degree: If set, filter out nodes with total degree (in + out)
            exceeding this value during traversal.
        min_information_content: If set, filter out nodes whose
            information_content attribute is below this value.
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

    # Case 1: Start pinned, end unpinned
    if start_idxes is not None and end_idxes is None:
        _query_forward(
            graph,
            start_idxes,
            allowed_predicates,
            end_categories,
            qualifier_constraints,
            check_inverse,
            inverse_pred_set,
            add_match,
            max_node_degree=max_node_degree,
            min_information_content=min_information_content,
            attribute_constraints=attribute_constraints,
            start_node_constraints=start_node_constraints,
            end_node_constraints=end_node_constraints,
        )

    # Case 2: Start unpinned, end pinned
    elif start_idxes is None and end_idxes is not None:
        _query_backward(
            graph,
            end_idxes,
            allowed_predicates,
            start_categories,
            qualifier_constraints,
            check_inverse,
            inverse_pred_set,
            add_match,
            max_node_degree=max_node_degree,
            min_information_content=min_information_content,
            attribute_constraints=attribute_constraints,
            start_node_constraints=start_node_constraints,
            end_node_constraints=end_node_constraints,
        )

    # Case 3: Both pinned
    elif start_idxes is not None and end_idxes is not None:
        _query_both_pinned(
            graph,
            start_idxes,
            end_idxes,
            allowed_predicates,
            qualifier_constraints,
            check_inverse,
            inverse_pred_set,
            add_match,
            max_node_degree=max_node_degree,
            min_information_content=min_information_content,
            attribute_constraints=attribute_constraints,
            start_node_constraints=start_node_constraints,
            end_node_constraints=end_node_constraints,
        )

    else:
        raise Exception("Both nodes unpinned - bad query planning")

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
    max_node_degree=None,
    min_information_content=None,
    attribute_constraints=None,
    start_node_constraints=None,
    end_node_constraints=None,
):
    """Case 1: Start pinned, end unpinned - forward search from pinned nodes."""
    logger.debug("  Forward search from %s pinned nodes", len(start_idxes))

    t0 = time.perf_counter()

    total_neighbors = 0
    slow_nodes = []  # Track nodes that take > 0.1s

    for start_idx in start_idxes:
        # Check start node attribute constraints once per start node
        if start_node_constraints:
            start_attrs = graph.get_node_property(start_idx, "attributes", [])
            if not matches_attribute_constraints(start_attrs, start_node_constraints):
                continue

        t_node_start = time.perf_counter()
        node_neighbors = 0

        # Check outgoing edges (direct matches)
        for obj_idx, predicate, props, fwd_edge_idx in graph.neighbors_with_properties(
            start_idx
        ):
            node_neighbors += 1
            # Check predicate
            if allowed_predicates and predicate not in allowed_predicates:
                continue

            # Check object categories
            if end_categories:
                obj_cats = graph.get_node_property(obj_idx, "categories", [])
                if not any(cat in obj_cats for cat in end_categories):
                    continue

            # Check node filters (degree and information content)
            if not _passes_node_filters(
                graph, obj_idx, max_node_degree, min_information_content
            ):
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
            ) in graph.incoming_neighbors_with_properties(start_idx):
                node_neighbors += 1

                # Check if stored predicate is one of our inverse predicates
                if inverse_pred_set and stored_pred not in inverse_pred_set:
                    continue

                # Check object categories (the "other" node becomes our object)
                if end_categories:
                    obj_cats = graph.get_node_property(other_idx, "categories", [])
                    if not any(cat in obj_cats for cat in end_categories):
                        continue

                # Check node filters (degree and information content)
                if not _passes_node_filters(
                    graph, other_idx, max_node_degree, min_information_content
                ):
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
        total_neighbors += node_neighbors

        if node_time > 0.1:  # Track slow nodes
            slow_nodes.append((start_idx, node_neighbors, node_time))

    t1 = time.perf_counter()
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
    allowed_predicates,
    start_categories,
    qualifier_constraints,
    check_inverse,
    inverse_pred_set,
    add_match,
    max_node_degree=None,
    min_information_content=None,
    attribute_constraints=None,
    start_node_constraints=None,
    end_node_constraints=None,
):
    """Case 2: Start unpinned, end pinned - backward search from pinned nodes."""
    logger.debug("  Backward search from %s pinned nodes", len(end_idxes))

    t0 = time.perf_counter()

    total_neighbors = 0
    slow_nodes = []  # Track nodes that take > 0.1s

    for i, end_idx in enumerate(end_idxes):
        # Check end node attribute constraints once per end node
        if end_node_constraints:
            end_attrs = graph.get_node_property(end_idx, "attributes", [])
            if not matches_attribute_constraints(end_attrs, end_node_constraints):
                continue

        t_node_start = time.perf_counter()
        node_neighbors = 0

        # Check incoming edges (direct matches)
        for (
            subj_idx,
            predicate,
            props,
            fwd_edge_idx,
        ) in graph.incoming_neighbors_with_properties(end_idx):
            node_neighbors += 1
            # Check predicate
            if allowed_predicates and predicate not in allowed_predicates:
                continue

            # Check subject categories
            if start_categories:
                subj_cats = graph.get_node_property(subj_idx, "categories", [])
                if not any(cat in subj_cats for cat in start_categories):
                    continue

            # Check node filters (degree and information content)
            if not _passes_node_filters(
                graph, subj_idx, max_node_degree, min_information_content
            ):
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
            ) in graph.neighbors_with_properties(end_idx):
                node_neighbors += 1

                # Check if stored predicate is one of our inverse predicates
                if inverse_pred_set and stored_pred not in inverse_pred_set:
                    continue

                # Check subject categories (the "other" node becomes our subject)
                if start_categories:
                    subj_cats = graph.get_node_property(other_idx, "categories", [])
                    if not any(cat in subj_cats for cat in start_categories):
                        continue

                # Check node filters (degree and information content)
                if not _passes_node_filters(
                    graph, other_idx, max_node_degree, min_information_content
                ):
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
        total_neighbors += node_neighbors

        if node_time > 0.1:  # Track slow nodes
            slow_nodes.append((end_idx, node_neighbors, node_time))

    t1 = time.perf_counter()
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
    allowed_predicates,
    qualifier_constraints,
    check_inverse,
    inverse_pred_set,
    add_match,
    max_node_degree=None,
    min_information_content=None,
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
    total_neighbors = 0
    slow_nodes = []

    for start_idx in start_idxes:
        # Check node filters on the start node
        if not _passes_node_filters(
            graph, start_idx, max_node_degree, min_information_content
        ):
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
        total_neighbors += node_neighbors

        # Only fetch properties for edges whose target is in end_set
        for (
            obj_idx,
            predicate,
            props,
            fwd_edge_idx,
        ) in graph.neighbors_filtered_by_targets(
            start_idx, end_set, predicate_filter=pred_filter_set
        ):
            # Check node filters on the end node
            if not _passes_node_filters(
                graph, obj_idx, max_node_degree, min_information_content
            ):
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
            if not _passes_node_filters(
                graph, end_idx, max_node_degree, min_information_content
            ):
                continue

            # Check end node attribute constraints
            if end_node_constraints:
                end_attrs = graph.get_node_property(end_idx, "attributes", [])
                if not matches_attribute_constraints(end_attrs, end_node_constraints):
                    continue

            for (
                obj_idx,
                stored_pred,
                props,
                fwd_edge_idx,
            ) in graph.neighbors_filtered_by_targets(
                end_idx, start_set, predicate_filter=inverse_pred_set or None
            ):
                total_neighbors += 1

                # Check node filters on the target (start) node
                if not _passes_node_filters(
                    graph, obj_idx, max_node_degree, min_information_content
                ):
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
