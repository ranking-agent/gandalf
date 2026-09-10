"""Validation for TRAPI request semantics.

Lightweight validation functions for TRAPI request payloads.  These are
kept separate from ``gandalf.server`` so they can be imported without
triggering module-level graph loading.
"""

from fastapi import HTTPException


def normalize_query_graph(query_graph: dict) -> None:
    """Drop optional fields whose value is ``None`` from a query graph in place.

    On the default (non-validating) request path the raw client body is used
    as-is (see ``server._request_dict``), skipping the
    ``model_dump(exclude_none=True)`` re-walk that the validating path applies.
    A client that sends an optional field explicitly as ``null`` -- e.g.
    ``"ids": null`` on an unpinned node -- therefore leaves the key present
    with a ``None`` value.

    Downstream code reads these fields with the ``node.get("ids", [])`` idiom,
    which only substitutes the default when the key is *absent*; a present
    ``None`` slips through and breaks operations like ``len(...)`` / ``set(...)``
    / ``x in ...`` with an opaque 500 (e.g. ``search/lookup.py`` reading a
    node's ``ids``).

    Restore the invariant the pipeline relies on -- optional fields are absent
    rather than ``None`` -- by stripping ``None`` values from each node and
    edge. This mirrors ``model_dump(exclude_none=True)`` but only walks the
    (small) query graph rather than the full request body, so it does not
    reintroduce the per-request cost the fast path was added to avoid.
    """
    for container in (query_graph.get("nodes"), query_graph.get("edges")):
        if not isinstance(container, dict):
            continue
        for element in container.values():
            if not isinstance(element, dict):
                continue
            for key in [k for k, v in element.items() if v is None]:
                del element[key]
            # A QEdge's TRAPI 2.0 constraints object needs the same treatment:
            # ``{"constraints": {"qualifiers": null}}`` must read as "no
            # qualifier constraint", not as a null the parser has to defend
            # against.
            constraints = element.get("constraints")
            if isinstance(constraints, dict):
                for key in [k for k, v in constraints.items() if v is None]:
                    del constraints[key]
                if not constraints:
                    del element["constraints"]


def validate_set_interpretation(query_graph: dict) -> None:
    """Validate node-level ``set_interpretation`` values.

    Raises ``HTTPException(422)`` for unsupported or invalid configurations:
    - ``MANY`` is not supported.
    - ``ALL`` requires ``ids`` on the node.
    - ``COLLATE`` is only valid for unpinned nodes (without ``ids``).
    """
    for qnode_id, qnode in query_graph.get("nodes", {}).items():
        interp = qnode.get("set_interpretation")
        if interp is None:
            continue
        if interp == "MANY":
            raise HTTPException(
                422,
                f"set_interpretation MANY is not supported (node '{qnode_id}')",
            )
        if interp == "ALL" and not qnode.get("ids"):
            raise HTTPException(
                422,
                f"set_interpretation ALL requires ids (node '{qnode_id}')",
            )
        if interp == "COLLATE" and qnode.get("ids"):
            raise HTTPException(
                422,
                f"set_interpretation COLLATE is only valid for unpinned nodes "
                f"without ids (node '{qnode_id}')",
            )


# TRAPI 1.x request fields that 2.0 replaced.  Each maps to the 2.0 spelling
# so the error can name the fix.
_RETIRED_QEDGE_FIELDS = {
    "qualifier_constraints": "constraints.qualifiers",
    "attribute_constraints": "constraints.attributes",
}
_RETIRED_QPATH_CONSTRAINT_FIELDS = {
    "intermediate_categories": "required_intermediate_categories",
}


def reject_retired_trapi_fields(query_graph: dict) -> None:
    """Reject TRAPI 1.x query-graph fields that 2.0 renamed.

    The 2.0 schema sets ``additionalProperties: true`` on QEdge, so a stray
    ``qualifier_constraints`` would simply be ignored -- and a client that
    asked for filtering would silently get unfiltered results back, which is
    worse than an error.  Name the 2.0 replacement instead.

    Raises ``HTTPException(400)`` for the first retired field found.
    """
    for qedge_id, qedge in (query_graph.get("edges") or {}).items():
        if not isinstance(qedge, dict):
            continue
        for retired, replacement in _RETIRED_QEDGE_FIELDS.items():
            if retired in qedge:
                raise HTTPException(
                    400,
                    f"edge '{qedge_id}' uses '{retired}', which TRAPI 2.0 "
                    f"replaced with '{replacement}'",
                )

    for qpath_id, qpath in (query_graph.get("paths") or {}).items():
        if not isinstance(qpath, dict):
            continue
        for constraint in qpath.get("constraints") or []:
            if not isinstance(constraint, dict):
                continue
            for retired, replacement in _RETIRED_QPATH_CONSTRAINT_FIELDS.items():
                if retired in constraint:
                    raise HTTPException(
                        400,
                        f"path '{qpath_id}' uses '{retired}', which TRAPI 2.0 "
                        f"replaced with '{replacement}'",
                    )


def validate_edge_node_references(query_graph: dict) -> None:
    """Validate that every qedge references nodes that exist in the graph.

    The query planner indexes into ``query_graph["nodes"]`` using each
    qedge's ``subject`` and ``object``. If an edge references a node key
    that is not present, that lookup raises ``KeyError`` deep in the planner
    and surfaces to the client as an opaque 500. Catch it up front and return
    a 400 instead, since this is a malformed request rather than a server
    fault.

    Raises ``HTTPException(400)`` for the first edge found referencing a
    missing node.
    """
    nodes = query_graph.get("nodes") or {}
    edges = query_graph.get("edges") or {}
    for qedge_id, qedge in edges.items():
        for endpoint in ("subject", "object"):
            node_ref = qedge.get(endpoint)
            if node_ref not in nodes:
                raise HTTPException(
                    400,
                    f"edge '{qedge_id}' references missing {endpoint} node "
                    f"'{node_ref}'",
                )
