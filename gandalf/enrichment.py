"""Enrich a TRAPI knowledge graph with full node and edge properties.

After ``lookup()`` builds a TRAPI response, the knowledge-graph nodes and
edges may be missing some properties — especially in *lightweight* mode
(>50 000 paths) where edge attribute lookups are skipped entirely.

``enrich_knowledge_graph()`` takes a TRAPI message **and** the backing
``CSRGraph`` and walks every node / edge in ``message.knowledge_graph``,
attaching all available properties from the graph's in-memory stores and
LMDB cold-path store.
"""

from __future__ import annotations

from gandalf.graph import CSRGraph


def enrich_knowledge_graph(message: dict, graph: CSRGraph) -> dict:
    """Attach all available properties to knowledge-graph nodes and edges.

    The function mutates *message* in place **and** returns it for
    convenience.

    Node properties added (when present in the graph):
        * ``name``
        * ``categories``
        * ``attributes`` — TRAPI Attribute objects (defaults to ``[]``)

    Edge properties added (when present in the graph):
        * ``sources``       — from the in-memory dedup store (hot path)
        * ``qualifiers``    — from the in-memory dedup store (hot path)
        * ``attributes``    — from LMDB (cold path; includes publications)

    Args:
        message: A TRAPI ``message`` dict that contains at least
            ``message["knowledge_graph"]["nodes"]`` and
            ``message["knowledge_graph"]["edges"]``.
        graph: The :class:`CSRGraph` instance that was used to produce the
            message (needed to look up properties).

    Returns:
        The same *message* dict, now enriched.
    """
    kg = message.get("message", {}).get("knowledge_graph", {})

    _enrich_nodes(kg.get("nodes", {}), graph)
    _enrich_edges(kg.get("edges", {}), graph)

    return message


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _enrich_nodes(nodes: dict, graph: CSRGraph) -> None:
    """Fill in missing properties for every node in the knowledge graph."""
    for node_id, node in nodes.items():
        node_idx = graph.node_id_to_idx.get(node_id)
        if node_idx is None:
            # Node not in graph (e.g. synthetic inferred node) — skip
            continue

        stored = graph.get_all_node_properties(node_idx)
        if not stored:
            continue

        # Populate fields that are absent from the KG node
        if "name" not in node and "name" in stored:
            node["name"] = stored["name"]

        if "categories" not in node and "categories" in stored:
            node["categories"] = stored["categories"]

        # Attributes should always be a list
        if "attributes" not in node:
            node["attributes"] = stored.get("attributes", [])


def _enrich_edges(edges: dict, graph: CSRGraph) -> None:
    """Fill in missing properties for every edge in the knowledge graph.

    To resolve edge properties we need the forward-CSR edge index.  We
    locate it by looking up the (subject → object) adjacency list in the
    CSR and matching on predicate (and qualifiers when already present).
    """
    # Batch-collect the forward edge indices we need to fetch from LMDB
    # so we can do a single batch read instead of one-at-a-time.
    edge_idx_map: dict[str, int] = {}  # edge_uuid -> fwd_edge_idx

    for edge_uuid, edge in edges.items():
        subj_id = edge.get("subject")
        obj_id = edge.get("object")
        predicate = edge.get("predicate")

        if subj_id is None or obj_id is None or predicate is None:
            continue

        subj_idx = graph.node_id_to_idx.get(subj_id)
        obj_idx = graph.node_id_to_idx.get(obj_id)
        if subj_idx is None or obj_idx is None:
            continue

        fwd_idx = _find_fwd_edge_idx(graph, subj_idx, obj_idx, predicate, edge)
        if fwd_idx is not None:
            edge_idx_map[edge_uuid] = fwd_idx

    # Batch LMDB read for cold-path properties (attributes)
    lmdb_results: dict[int, dict] = {}
    if graph.lmdb_store is not None and edge_idx_map:
        unique_indices = list(set(edge_idx_map.values()))
        lmdb_results = graph.lmdb_store.get_batch(unique_indices)

    # Now enrich each edge
    for edge_uuid, edge in edges.items():
        fwd_idx = edge_idx_map.get(edge_uuid)
        if fwd_idx is None:
            # Could not resolve — ensure defaults are present
            edge.setdefault("sources", [])
            edge.setdefault("qualifiers", [])
            edge.setdefault("attributes", [])
            continue

        # Hot-path properties (in-memory dedup store)
        if "sources" not in edge:
            edge["sources"] = graph.edge_properties.get_sources(fwd_idx)

        if "qualifiers" not in edge:
            edge["qualifiers"] = graph.edge_properties.get_qualifiers(fwd_idx)

        # Cold-path properties (LMDB)
        detail = lmdb_results.get(fwd_idx, {})

        if "attributes" not in edge:
            edge["attributes"] = detail.get("attributes", [])


def _find_fwd_edge_idx(
    graph: CSRGraph,
    subj_idx: int,
    obj_idx: int,
    predicate: str,
    edge: dict,
) -> int | None:
    """Return the forward-CSR position for an edge, or *None*.

    Scans the forward adjacency list of *subj_idx* for entries that point
    to *obj_idx* with the matching predicate.  When multiple parallel edges
    exist (same subject/predicate/object but different qualifiers or
    sources), we disambiguate using the qualifier and source data already
    present on *edge*.
    """
    pred_idx = graph.predicate_to_idx.get(predicate)
    if pred_idx is None:
        return None

    start = int(graph.fwd_offsets[subj_idx])
    end = int(graph.fwd_offsets[subj_idx + 1])

    candidates: list[int] = []
    for pos in range(start, end):
        if (
            int(graph.fwd_targets[pos]) == obj_idx
            and int(graph.fwd_predicates[pos]) == pred_idx
        ):
            candidates.append(pos)

    if not candidates:
        return None

    if len(candidates) == 1:
        return candidates[0]

    # Multiple parallel edges — try to disambiguate using qualifiers/sources
    # already present on the edge dict.
    edge_quals = edge.get("qualifiers")
    edge_sources = edge.get("sources")

    if edge_quals is not None or edge_sources is not None:
        for pos in candidates:
            match = True
            if edge_quals is not None:
                stored_quals = graph.edge_properties.get_qualifiers(pos)
                if _normalize_quals(stored_quals) != _normalize_quals(edge_quals):
                    match = False
            if match and edge_sources is not None:
                stored_sources = graph.edge_properties.get_sources(pos)
                if _normalize_sources(stored_sources) != _normalize_sources(
                    edge_sources
                ):
                    match = False
            if match:
                return pos

    # Fall back to the first candidate
    return candidates[0]


def _normalize_quals(quals: list) -> frozenset:
    """Create a hashable representation of a qualifier list for comparison."""
    return frozenset(
        (q.get("qualifier_type_id", ""), q.get("qualifier_value", ""))
        for q in (quals or [])
    )


def _normalize_sources(sources: list) -> frozenset:
    """Create a hashable representation of a source list for comparison."""
    return frozenset(
        (s.get("resource_id", ""), s.get("resource_role", "")) for s in (sources or [])
    )
