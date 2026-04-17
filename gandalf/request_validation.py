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
