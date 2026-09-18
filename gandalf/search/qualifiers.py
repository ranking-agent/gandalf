"""Qualifier constraint matching for edge filtering.

TRAPI 2.0 expresses a ``QualifierSetConstraint`` as a plain mapping from
``qualifier_type_id`` to the required ``qualifier_value``::

    {"biolink:object_aspect_qualifier": "activity"}

replacing the 1.x ``{"qualifier_set": [{"qualifier_type_id": ...,
"qualifier_value": ...}]}`` wrapper.  A list of such mappings lives at
``QEdge.constraints.qualifiers``.
"""

from typing import Iterable, Mapping, Optional, Sequence, Union

# One constraint maps a qualifier type to either the single value TRAPI sends
# or, after ``QualifierExpander`` widens it, the list of acceptable descendant
# values.
QualifierSetConstraint = Mapping[str, Union[str, Sequence[str]]]


def edge_matches_qualifier_constraints(
    edge_qualifiers: Optional[Iterable[dict]],
    qualifier_constraints: Optional[Sequence[QualifierSetConstraint]],
) -> bool:
    """Check whether an edge's qualifiers satisfy a QEdge's qualifier constraints.

    Semantics follow TRAPI: OR between the constraints in the list (the edge
    matches if it satisfies at least one), AND within a single constraint (the
    edge must carry every type-value pair it names).

    A constraint value may be either the single string TRAPI sends or a list of
    acceptable values, which is what ``QualifierExpander`` produces when it
    widens a value to its Biolink descendants.  A list matches if the edge
    carries *any* of its members.

    Args:
        edge_qualifiers: The edge's TRAPI Qualifier dicts, each with
            ``qualifier_type_id`` and ``qualifier_value``.
        qualifier_constraints: The QEdge's ``constraints.qualifiers`` list.

    Returns:
        True if the edge satisfies at least one constraint.  An empty or
        missing constraint list matches every edge.

    Examples:
        >>> quals = [
        ...     {"qualifier_type_id": "biolink:object_aspect_qualifier",
        ...      "qualifier_value": "activity"},
        ... ]
        >>> edge_matches_qualifier_constraints(quals, None)
        True
        >>> edge_matches_qualifier_constraints(
        ...     quals, [{"biolink:object_aspect_qualifier": "activity"}]
        ... )
        True
        >>> edge_matches_qualifier_constraints(
        ...     quals, [{"biolink:object_aspect_qualifier": "abundance"}]
        ... )
        False
        >>> edge_matches_qualifier_constraints(
        ...     quals,
        ...     [{"biolink:object_aspect_qualifier": ["abundance", "activity"]}],
        ... )
        True
    """
    # No constraints means all edges match
    if not qualifier_constraints:
        return True

    # type_id -> set of values carried by the edge
    edge_qualifiers_by_type: dict[str, set[str]] = {}
    for qualifier in edge_qualifiers or ():
        type_id = qualifier.get("qualifier_type_id")
        value = qualifier.get("qualifier_value")
        if type_id and value:
            edge_qualifiers_by_type.setdefault(type_id, set()).add(value)

    # OR between constraints
    for constraint in qualifier_constraints:
        if not constraint:
            # An empty constraint names no requirement, so nothing can fail it.
            return True

        # AND within a constraint
        if all(
            _qualifier_matches(edge_qualifiers_by_type.get(type_id), required)
            for type_id, required in constraint.items()
        ):
            return True

    return False


def _qualifier_matches(
    edge_values: Optional[set], required: Union[str, Sequence[str]]
) -> bool:
    """Check one type-value pair of a constraint against the edge's values."""
    if not edge_values:
        return False
    if isinstance(required, str):
        return required in edge_values
    return not edge_values.isdisjoint(required)
