"""Validation for TRAPI request semantics.

Lightweight validation functions for TRAPI request payloads.  These are
kept separate from ``gandalf.server`` so they can be imported without
triggering module-level graph loading.
"""

from fastapi import HTTPException


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
