"""Edge query functions for graph traversal with predicate/qualifier filtering.

The traversal works on arrays: it reads the edges of many pinned nodes at
once, applies each filter to all of them as a mask, and returns the matches
as an :class:`~gandalf.search.edge_matches.EdgeMatches`.  See "Traversal
works on arrays" in CLAUDE.MD.
"""

import logging
import time
from typing import Iterator, Optional

import numpy as np

from gandalf.profiler import current_profiler
from gandalf.attribute_columns import prefilter
from gandalf.search.attribute_constraints import matches_attribute_constraints
from gandalf.search.edge_constraints import EdgeConstraints
from gandalf.search.edge_matches import EdgeMatches
from gandalf.search.node_filters import NodeFilter, node_filter_mask

#: Candidate edges read per traversal step.  Pinned nodes are taken in runs
#: whose degrees add up to about this many, so the arrays a step works on
#: stay a few megabytes however many nodes are pinned.  A single node with
#: more edges than this is still read in one step.
TRAVERSAL_CHUNK_EDGES = 262_144

#: A traversal step slower than this (seconds) is reported to the profiler.
_SLOW_STEP_SECONDS = 0.1

_SUBCLASS_OF = "biolink:subclass_of"

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
        The matches as an :class:`EdgeMatches` of child (subject),
        ``biolink:subclass_of``, superclass (object).  The depth-0
        self-match is included with fwd_edge_idx=-1 (no real edge).
    """
    logger = logger if logger is not None else logging.getLogger(__name__)
    matches: list[tuple] = []

    if end_idxes is None:
        return EdgeMatches.empty([_SUBCLASS_OF])

    subclass_mask = graph.predicate_mask([_SUBCLASS_OF])

    for superclass_idx in _as_index_list(end_idxes):
        # BFS: current frontier -> next frontier, up to `depth` levels
        # Depth 0 = identity match (the node itself)
        frontier = {superclass_idx}
        visited = {superclass_idx}

        # Always include the depth-0 self-match (no real edge, sentinel -1)
        matches.append((superclass_idx, superclass_idx, -1))

        for _hop in range(depth):
            next_frontier = set()
            for node_idx in frontier:
                # Walk incoming subclass_of edges: child --subclass_of--> node_idx
                _owner, children, _pred_ids, fwd_eidxes = graph.edges_of(
                    np.array([node_idx]), incoming=True, predicate_mask=subclass_mask
                )
                for child_idx, fwd_eidx in zip(children.tolist(), fwd_eidxes.tolist()):
                    if child_idx in visited:
                        continue
                    visited.add(child_idx)
                    next_frontier.add(child_idx)
                    matches.append((child_idx, superclass_idx, fwd_eidx))
            frontier = next_frontier
            if not frontier:
                break

    logger.debug(
        "  Subclass traversal: found %s matches (depth=%s)", len(matches), depth
    )

    columns = np.array(matches, dtype=np.int32).reshape(-1, 3)
    return EdgeMatches(
        subj=columns[:, 0].copy(),
        obj=columns[:, 1].copy(),
        pred=np.zeros(len(columns), dtype=np.int32),
        via_inverse=np.zeros(len(columns), dtype=np.bool_),
        fwd_edge_idx=columns[:, 2].copy(),
        predicates=[_SUBCLASS_OF],
    )


def _as_index_list(node_idxes) -> list[int]:
    """Node indices (a list or an array) as a list of Python ints."""
    if isinstance(node_idxes, np.ndarray):
        indices: list[int] = node_idxes.tolist()
        return indices
    return list(node_idxes)


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
    pair_targets: Optional[dict[int, set[int]]] = None,
    logger: Optional[logging.Logger] = None,
) -> EdgeMatches:
    """Query for a single edge with given constraints.

    Handles symmetric and inverse predicates at query time by checking both
    edge directions when appropriate. For example, if searching for predicate P
    and P has inverse Q, edges stored as B--Q-->A will be returned as A--P-->B.

    The traversal works on arrays throughout (:meth:`CSRGraph.edges_of`): no
    Python object is made per candidate edge, and each filter is applied to a
    whole step's edges at once.

    Args:
        graph: CSRGraph instance
        start_idxes: Pinned start node indices (a list or an array), or None
            if unpinned
        end_idxes: Pinned end node indices (a list or an array), or None if
            unpinned
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
            ``build_node_filters``), applied to an unpinned end only. Empty
            list / None means no filtering.
        start_node_constraints: List of TRAPI AttributeConstraint dicts for
            filtering the start (subject) node by its attributes.
        end_node_constraints: List of TRAPI AttributeConstraint dicts for
            filtering the end (object) node by its attributes.
        pair_targets: Each start node's allowed end nodes, when both ends are
            pinned: an edge then matches only between a listed pair, not
            between any start and any end.  None allows every pair.
        logger: Logger to emit this query's records to.  ``lookup`` passes the
            query's own logger so that entries reach that query's TRAPI logs
            and no other's.  Defaults to the module logger.

    Returns:
        The matching edges as an :class:`EdgeMatches`, each edge as stored in
        the graph, once per direction it matched in: ``via_inverse`` marks an
        edge found through the inverse/symmetric lookup, and ``fwd_edge_idx``
        is its forward-CSR array position (unique per physical edge).
    """
    logger = logger if logger is not None else logging.getLogger(__name__)
    if node_filters is None:
        node_filters = []

    # None  -> don't check inverse direction at all (default)
    # []    -> match ALL predicates in inverse direction (wildcard, e.g. related_to)
    # [pred]-> match only the listed predicates in inverse direction
    check_inverse = inverse_predicates is not None
    inverse_pred_set = set(inverse_predicates) if inverse_predicates else set()

    # Predicate filters as lookup tables over predicate ids, built once so
    # that each step's edges are filtered in one vectorized operation.
    # None means "any predicate".
    forward_mask = (
        graph.predicate_mask(allowed_predicates) if allowed_predicates else None
    )
    inverse_mask = graph.predicate_mask(inverse_pred_set) if inverse_pred_set else None

    prof = current_profiler()
    n_allowed_preds = len(allowed_predicates) if allowed_predicates else 0
    n_inverse_preds = len(inverse_pred_set) if check_inverse else 0
    t0 = time.perf_counter()

    # Case 1: Start pinned, end unpinned
    if start_idxes is not None and end_idxes is None:
        logger.debug("  Forward search from %s pinned nodes", len(start_idxes))
        with prof.stage(
            "query_forward",
            n_pinned=len(start_idxes),
            n_predicates=n_allowed_preds,
            n_inverse_preds=n_inverse_preds,
            check_inverse=check_inverse,
            n_end_categories=len(end_categories) if end_categories else 0,
            has_edge_constraints=bool(edge_constraints),
        ):
            parts, total_neighbors, slow_nodes = _query_from_pinned(
                graph,
                start_idxes,
                pinned_is_subject=True,
                forward_mask=forward_mask,
                check_inverse=check_inverse,
                inverse_mask=inverse_mask,
                edge_constraints=edge_constraints,
                pinned_ok=_NodeCheck(graph, None, None, start_node_constraints),
                other_ok=_NodeCheck(
                    graph, end_categories, node_filters, end_node_constraints
                ),
            )
            _record_traversal_metrics(graph, total_neighbors, slow_nodes)

    # Case 2: Start unpinned, end pinned
    elif start_idxes is None and end_idxes is not None:
        logger.debug("  Backward search from %s pinned nodes", len(end_idxes))
        with prof.stage(
            "query_backward",
            n_pinned=len(end_idxes),
            n_predicates=n_allowed_preds,
            n_inverse_preds=n_inverse_preds,
            check_inverse=check_inverse,
            n_start_categories=len(start_categories) if start_categories else 0,
            has_edge_constraints=bool(edge_constraints),
        ):
            parts, total_neighbors, slow_nodes = _query_from_pinned(
                graph,
                end_idxes,
                pinned_is_subject=False,
                forward_mask=forward_mask,
                check_inverse=check_inverse,
                inverse_mask=inverse_mask,
                edge_constraints=edge_constraints,
                pinned_ok=_NodeCheck(graph, None, None, end_node_constraints),
                other_ok=_NodeCheck(
                    graph, start_categories, node_filters, start_node_constraints
                ),
            )
            _record_traversal_metrics(graph, total_neighbors, slow_nodes)

    # Case 3: Both pinned
    elif start_idxes is not None and end_idxes is not None:
        logger.debug(
            "  Both ends pinned: %s start, %s end", len(start_idxes), len(end_idxes)
        )
        with prof.stage(
            "query_both_pinned",
            n_start=len(start_idxes),
            n_end=len(end_idxes),
            n_predicates=n_allowed_preds,
            n_inverse_preds=n_inverse_preds,
            check_inverse=check_inverse,
            has_edge_constraints=bool(edge_constraints),
        ):
            parts, total_neighbors, slow_nodes = _query_both_pinned(
                graph,
                start_idxes,
                end_idxes,
                forward_mask=forward_mask,
                check_inverse=check_inverse,
                inverse_mask=inverse_mask,
                edge_constraints=edge_constraints,
                # Both ends are pinned, so categories and node filters are
                # not rechecked: only each end's attribute constraints apply.
                # An end is pinned either by the query, whose pinned nodes
                # are never filtered, or by an earlier edge that discovered
                # it as an unpinned end, where the node filters ran.
                start_ok=_NodeCheck(graph, None, None, start_node_constraints),
                end_ok=_NodeCheck(graph, None, None, end_node_constraints),
                pair_targets=pair_targets,
            )
            _record_traversal_metrics(graph, total_neighbors, slow_nodes)

    else:
        raise Exception("Both nodes unpinned - bad query planning")

    matches = EdgeMatches.concatenate(parts, graph.id_to_predicate)
    logger.debug(
        "  Traversed %s neighbors in %.3fs", total_neighbors, time.perf_counter() - t0
    )
    prof.add_metric("matches", len(matches))
    return matches


class _NodeCheck:
    """A memoized test of whether nodes satisfy a query node.

    Checks a node's categories (any of *categories*), the plugin node
    filters, and the query node's attribute constraints.

    Categories are tested for a whole step's nodes at once against the
    graph's category index (``graph.node_categories``), where it has one.
    Everything else is judged once per node for the whole edge query, since
    hub nodes are reached from many pinned nodes: attribute constraints from
    the graph's attribute columns where they can answer, plugin filters a
    step's nodes at a time (``node_filter_mask``), and whatever still needs
    the node's record -- including the categories of a graph without the
    index -- with the step's nodes read from the node store in one batch.

    Falsy when there is nothing to check (every node passes).
    """

    def __init__(self, graph, categories, node_filters, attribute_constraints):
        self._graph = graph
        self._node_filters = node_filters or []
        self._attribute_constraints = attribute_constraints
        self._category_index = (
            getattr(graph, "node_categories", None) if categories else None
        )
        indexed = self._category_index is not None
        self._indexed_categories = categories if indexed else None
        # Categories left to check on each node's own record
        self._categories = None if indexed else categories
        self._verdicts: dict[int, bool] = {}

    def __bool__(self) -> bool:
        return bool(
            self._indexed_categories
            or self._categories
            or self._node_filters
            or self._attribute_constraints
        )

    def mask(self, node_idxes: np.ndarray) -> np.ndarray:
        """Whether each of *node_idxes* passes, as a boolean array."""
        mask = np.ones(len(node_idxes), dtype=np.bool_)
        if len(node_idxes) == 0:
            return mask
        if self._category_index is not None:
            mask = self._category_index.any_of(node_idxes, self._indexed_categories)
        if not (self._categories or self._node_filters or self._attribute_constraints):
            return mask

        # Only the nodes still standing are looked at one by one.
        candidates = np.flatnonzero(mask)
        unique, inverse = np.unique(node_idxes[candidates], return_inverse=True)
        nodes = unique.tolist()
        verdicts = self._verdicts
        self._judge([node for node in nodes if node not in verdicts])
        passed = np.fromiter(
            (verdicts[node] for node in nodes), dtype=np.bool_, count=len(nodes)
        )
        mask[candidates] = passed[inverse]
        return mask

    def _judge(self, nodes: list[int]) -> None:
        """Record a verdict for each of *nodes* (ascending, not yet judged)."""
        if not nodes:
            return
        graph = self._graph
        categories = self._categories
        constraints = self._attribute_constraints
        node_arr = np.asarray(nodes, dtype=np.int64)
        passed = np.ones(len(nodes), dtype=np.bool_)
        need_record = np.zeros(len(nodes), dtype=np.bool_)
        # Attribute constraints from the graph's columns where they can
        # answer; the nodes they cannot judge are read below.
        if constraints:
            columns = getattr(graph, "attribute_columns", None)
            passed, need_record = prefilter(
                constraints,
                columns.node_constraint if columns is not None else None,
                node_arr,
            )
        # Plugin filters judge the nodes still standing, in one batch each
        for node_filter in self._node_filters:
            standing = np.flatnonzero(passed)
            if len(standing) == 0:
                break
            passed[standing] = node_filter_mask(node_filter, graph, node_arr[standing])
        need_record &= passed
        read = passed if categories else need_record
        props_by_node: dict = {}
        if read.any():
            props_by_node = graph.get_all_node_properties_batch(node_arr[read].tolist())
        verdicts = self._verdicts
        for node, ok, need in zip(nodes, passed.tolist(), need_record.tolist()):
            if ok and (categories or need):
                props = props_by_node.get(node, {})
                if categories:
                    node_categories = props.get("categories", [])
                    ok = any(cat in node_categories for cat in categories)
                if ok and need:
                    ok = bool(
                        matches_attribute_constraints(
                            props.get("attributes", []), constraints
                        )
                    )
            verdicts[node] = ok


def _unique_in_order(node_idxes) -> np.ndarray:
    """*node_idxes* as an index array, each node once, at its first position.

    A node pinned twice yields the same edges twice, so traversing it once is
    what keeps a match from being reported more than once.
    """
    nodes = np.asarray(node_idxes, dtype=np.int64).reshape(-1)
    _, first_at = np.unique(nodes, return_index=True)
    if len(first_at) == len(nodes):
        return nodes
    unique: np.ndarray = nodes[np.sort(first_at)]
    return unique


def _degrees(graph, nodes: np.ndarray, incoming: bool) -> np.ndarray:
    """Each node's out-degree (or in-degree), from the CSR offsets alone."""
    offsets = graph.rev_offsets if incoming else graph.fwd_offsets
    degrees: np.ndarray = offsets[nodes + 1] - offsets[nodes]
    return degrees


def _steps(weights: np.ndarray, limit: int) -> Iterator[tuple[int, int]]:
    """Split ``range(len(weights))`` into runs weighing about *limit* each.

    A run always holds at least one item, so one heavier than *limit* is a
    run of its own.

    >>> list(_steps(np.array([3, 3, 3, 10, 1, 1]), 6))
    [(0, 2), (2, 3), (3, 4), (4, 6)]
    >>> list(_steps(np.array([], dtype=np.int64), 6))
    []
    """
    cumulative = np.cumsum(weights)
    start, n = 0, len(weights)
    while start < n:
        done = int(cumulative[start - 1]) if start else 0
        end = int(np.searchsorted(cumulative, done + limit, side="right"))
        end = max(end, start + 1)
        yield start, end
        start = end


def _read_edges(
    graph,
    nodes: np.ndarray,
    incoming: bool,
    predicate_mask,
    other_ok: Optional["_NodeCheck"],
    edge_constraints,
    other_allowed=None,
) -> tuple[int, tuple]:
    """Read one traversal step's edges and apply every filter to them.

    Args:
        nodes: The pinned nodes of this step.
        incoming: Read their incoming edges rather than their outgoing ones.
        other_ok: Check on the node at each edge's other end, or None.
        other_allowed: ``(pinned, other) -> mask`` restricting the other end
            to given nodes, applied before anything is read about it.

    Returns:
        ``(scanned, (owner, pinned, other, pred_ids, fwd_edge_idx))``:
        how many edges passed the predicate filter (and *other_allowed*), and
        the edges that passed everything, *owner* being each one's position
        in *nodes*.
    """
    owner, other, pred_ids, fwd_edge_idx = graph.edges_of(
        nodes, incoming=incoming, predicate_mask=predicate_mask
    )
    pinned = nodes[owner]
    keep = None
    if other_allowed is not None:
        keep = other_allowed(pinned, other)
        owner, pinned, other, pred_ids, fwd_edge_idx = (
            owner[keep],
            pinned[keep],
            other[keep],
            pred_ids[keep],
            fwd_edge_idx[keep],
        )
    scanned = len(owner)

    # The node at the other end first (categories, node filters, attribute
    # constraints), then the QEdge's constraints (qualifiers,
    # knowledge_level, agent_type, sources, attributes) on what is left.
    for check in (other_ok, edge_constraints):
        if not check or len(owner) == 0:
            continue
        if check is other_ok:
            keep = other_ok.mask(other)
        else:
            keep = edge_constraints.permits_mask(graph, fwd_edge_idx)
        if not keep.all():
            owner, pinned, other, pred_ids, fwd_edge_idx = (
                owner[keep],
                pinned[keep],
                other[keep],
                pred_ids[keep],
                fwd_edge_idx[keep],
            )
    return scanned, (owner, pinned, other, pred_ids, fwd_edge_idx)


def _stored(pinned, other, pred_ids, fwd_edge_idx, incoming: bool, via_inverse: bool):
    """One step's edges as an ``EdgeMatches.concatenate`` part.

    An incoming edge is stored other -> pinned, an outgoing one pinned -> other.
    """
    subj, obj = (other, pinned) if incoming else (pinned, other)
    return subj, obj, pred_ids, via_inverse, fwd_edge_idx


def _query_from_pinned(
    graph,
    pinned_idxes,
    pinned_is_subject: bool,
    forward_mask,
    check_inverse: bool,
    inverse_mask,
    edge_constraints,
    pinned_ok: "_NodeCheck",
    other_ok: "_NodeCheck",
) -> tuple[list, int, list]:
    """Cases 1 and 2: search from the pinned end of an edge to its open end.

    From a pinned subject the direct matches are its outgoing edges; from a
    pinned object, its incoming ones.  With ``check_inverse`` the edges in
    the other direction are read too, under ``inverse_mask``: an edge stored
    B --inverse(P)--> A stands for A --P--> B.  Those are reported as stored
    and marked ``via_inverse``.

    ``forward_mask`` / ``inverse_mask`` are :meth:`CSRGraph.predicate_mask`
    tables, or None for "any predicate".

    Matches come pinned node by pinned node, each node's direct matches
    before its inverse ones.

    Returns:
        ``(parts, total_neighbors, slow_nodes)``: the matches as
        ``EdgeMatches.concatenate`` parts, how many edges passed the
        predicate filters, and ``(node_idx, neighbors, seconds)`` for each
        slow traversal step (see :func:`_record_traversal_metrics`).
    """
    nodes = _unique_in_order(pinned_idxes)
    # Attribute constraints on the pinned node, once per node
    if pinned_ok:
        nodes = nodes[pinned_ok.mask(nodes)]

    direct_incoming = not pinned_is_subject
    weights = _degrees(graph, nodes, direct_incoming)
    if check_inverse:
        weights = weights + _degrees(graph, nodes, not direct_incoming)

    parts: list[tuple] = []
    total_neighbors = 0
    slow_nodes: list[tuple] = []
    for start, end in _steps(weights, TRAVERSAL_CHUNK_EDGES):
        t_step = time.perf_counter()
        step_nodes = nodes[start:end]
        scanned, (owner, pinned, other, pred_ids, fwd_edge_idx) = _read_edges(
            graph, step_nodes, direct_incoming, forward_mask, other_ok, edge_constraints
        )
        direct = _stored(pinned, other, pred_ids, fwd_edge_idx, direct_incoming, False)

        if not check_inverse:
            parts.append(direct)
        else:
            inv_scanned, (inv_owner, pinned, other, pred_ids, fwd_edge_idx) = (
                _read_edges(
                    graph,
                    step_nodes,
                    not direct_incoming,
                    inverse_mask,
                    other_ok,
                    edge_constraints,
                )
            )
            scanned += inv_scanned
            inverse = _stored(
                pinned, other, pred_ids, fwd_edge_idx, not direct_incoming, True
            )
            if end - start == 1:
                parts.extend((direct, inverse))
            else:
                # Each node's direct matches, then its inverse ones: a stable
                # sort by node of the direct block followed by the inverse.
                order = np.argsort(np.concatenate([owner, inv_owner]), kind="stable")
                merged = EdgeMatches.concatenate([direct, inverse], ())
                parts.append(
                    (
                        merged.subj[order],
                        merged.obj[order],
                        merged.pred[order],
                        merged.via_inverse[order],
                        merged.fwd_edge_idx[order],
                    )
                )

        total_neighbors += scanned
        seconds = time.perf_counter() - t_step
        if seconds > _SLOW_STEP_SECONDS:
            heaviest = start + int(np.argmax(weights[start:end]))
            slow_nodes.append((int(nodes[heaviest]), scanned, seconds))

    return parts, total_neighbors, slow_nodes


def _pair_keys(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Pack two node index arrays into one int64 key per pair."""
    keys: np.ndarray = (first.astype(np.int64) << 32) | second.astype(np.int64)
    return keys


def _query_both_pinned(
    graph,
    start_idxes,
    end_idxes,
    forward_mask,
    check_inverse: bool,
    inverse_mask,
    edge_constraints,
    start_ok: "_NodeCheck",
    end_ok: "_NodeCheck",
    pair_targets=None,
) -> tuple[list, int, list]:
    """Case 3: Both ends pinned - intersection search.

    Reads each start node's outgoing edges and keeps those ending on a pinned
    end node; then, with ``check_inverse``, each end node's outgoing edges
    ending on a pinned start node, reported as stored and marked
    ``via_inverse``.  An edge whose other end is not pinned is dropped before
    anything else is looked up about it.

    Predicate masks are as for :func:`_query_from_pinned`.  With
    ``pair_targets``, each start node is searched for its own allowed ends
    only, and each end node, in the inverse pass, for the starts that list it.

    Returns:
        ``(parts, total_neighbors, slow_nodes)`` as :func:`_query_from_pinned`
        does; direct matches all come before inverse ones.
    """
    starts = _unique_in_order(start_idxes)
    ends = _unique_in_order(end_idxes)

    if pair_targets is None:
        start_has_targets = np.full(len(starts), len(ends) > 0)
        end_has_sources = np.full(len(ends), len(starts) > 0)

        def end_allowed(pinned, other):
            return np.isin(other, ends)

        def start_allowed(pinned, other):
            return np.isin(other, starts)

    else:
        # The listed (start, end) pairs between pinned nodes, as int64 keys
        end_set = set(ends.tolist())
        pairs = [
            (start, target)
            for start in starts.tolist()
            for target in pair_targets.get(start, ())
            if target in end_set
        ]
        pair_array = np.array(pairs, dtype=np.int64).reshape(-1, 2)
        pair_keys = _pair_keys(pair_array[:, 0], pair_array[:, 1])
        start_has_targets = np.isin(starts, pair_array[:, 0])
        end_has_sources = np.isin(ends, pair_array[:, 1])

        def end_allowed(pinned, other):
            return np.isin(_pair_keys(pinned, other), pair_keys)

        def start_allowed(pinned, other):
            return np.isin(_pair_keys(other, pinned), pair_keys)

    parts: list[tuple] = []
    total_neighbors = 0
    slow_nodes: list[tuple] = []

    def search(nodes, has_partner, nodes_ok, other_ok, other_allowed, mask, inverse):
        """Edges out of *nodes* that end on an allowed pinned node."""
        nonlocal total_neighbors
        nodes = nodes[has_partner]
        if nodes_ok:
            nodes = nodes[nodes_ok.mask(nodes)]
        weights = _degrees(graph, nodes, incoming=False)
        for start, end in _steps(weights, TRAVERSAL_CHUNK_EDGES):
            t_step = time.perf_counter()
            scanned, (_owner, pinned, other, pred_ids, fwd_edge_idx) = _read_edges(
                graph,
                nodes[start:end],
                False,
                mask,
                other_ok,
                edge_constraints,
                other_allowed=other_allowed,
            )
            parts.append(_stored(pinned, other, pred_ids, fwd_edge_idx, False, inverse))
            # For diagnostics: every neighbour of a start node, but only the
            # pinned ones of an end node.
            neighbors = scanned if inverse else int(weights[start:end].sum())
            total_neighbors += neighbors
            seconds = time.perf_counter() - t_step
            if seconds > _SLOW_STEP_SECONDS:
                heaviest = start + int(np.argmax(weights[start:end]))
                slow_nodes.append((int(nodes[heaviest]), neighbors, seconds))

    search(
        starts, start_has_targets, start_ok, end_ok, end_allowed, forward_mask, False
    )
    # Also check reverse direction for symmetric/inverse predicates:
    # edges end_node --inverse(P)--> start_node
    if check_inverse:
        search(
            ends, end_has_sources, end_ok, start_ok, start_allowed, inverse_mask, True
        )

    return parts, total_neighbors, slow_nodes


_SLOW_NODE_EVENT_LIMIT = 10


def _record_traversal_metrics(graph, total_neighbors, slow_nodes):
    """Surface neighborhood-size and slow-node detail to the profiler.

    Aggregates onto the currently-active query_* stage. ``slow_nodes`` is
    the list ``[(node_idx, neighbors, duration_seconds), ...]`` collected by
    the search for traversal steps that took longer than the inline
    threshold: the step's highest-degree node, the neighbours the whole step
    read, and how long it took.
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
