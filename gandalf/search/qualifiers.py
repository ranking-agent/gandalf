"""Qualifier constraint matching for edge filtering."""


def edge_matches_qualifier_constraints(edge_qualifiers, qualifier_constraints):
    """Check if an edge's qualifiers match the query's qualifier constraints.

    Qualifier constraints use OR semantics between qualifier_sets and AND semantics
    within each qualifier_set. An edge matches if it satisfies at least one
    qualifier_set (i.e., has ALL qualifiers in that set).

    Supports two formats for constraint qualifiers:
    - Original: {"qualifier_type_id": "...", "qualifier_value": "..."} - exact match
    - Expanded: {"qualifier_type_id": "...", "qualifier_values": [...]} - match any value

    Args:
        edge_qualifiers: List of qualifier dicts from the edge, each with
                        'qualifier_type_id' and 'qualifier_value'
        qualifier_constraints: List of constraint dicts from the query, each with
                              a 'qualifier_set' containing qualifiers to match

    Returns:
        True if the edge matches at least one qualifier_set, False otherwise.
        Returns True if qualifier_constraints is None or empty.
    """
    # No constraints means all edges match
    if not qualifier_constraints:
        return True

    # Build a set of (type_id, value) tuples from edge qualifiers for fast lookup
    edge_qualifier_set = set()
    # Also build a dict mapping type_id -> set of values for expanded matching
    edge_qualifiers_by_type: dict[str, set[str]] = {}
    if edge_qualifiers:
        for q in edge_qualifiers:
            type_id = q.get("qualifier_type_id")
            value = q.get("qualifier_value")
            if type_id and value:
                edge_qualifier_set.add((type_id, value))
                if type_id not in edge_qualifiers_by_type:
                    edge_qualifiers_by_type[type_id] = set()
                edge_qualifiers_by_type[type_id].add(value)

    # Check if edge satisfies at least one qualifier_set (OR semantics)
    for constraint in qualifier_constraints:
        qualifier_set = constraint.get("qualifier_set", [])
        if not qualifier_set:
            # Empty qualifier_set matches any edge
            return True

        # Check if edge has ALL qualifiers in this set (AND semantics)
        all_match = True
        for required_qualifier in qualifier_set:
            req_type = required_qualifier.get("qualifier_type_id")

            # Check for expanded format (qualifier_values - plural)
            req_values = required_qualifier.get("qualifier_values")
            if req_values is not None:
                # Expanded format: edge must have this type with ANY of the values
                edge_values = edge_qualifiers_by_type.get(req_type, set())
                if not edge_values.intersection(req_values):
                    all_match = False
                    break
            else:
                # Original format: exact match required
                req_value = required_qualifier.get("qualifier_value")
                if (req_type, req_value) not in edge_qualifier_set:
                    all_match = False
                    break

        if all_match:
            return True

    return False
