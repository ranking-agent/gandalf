"""TRAPI attribute constraint matching for nodes and edges.

Attribute constraints allow callers to filter results based on attribute
values.  Each constraint specifies an attribute_type_id (matched via the
``id`` field), an operator, and a value.  Multiple constraints use AND
logic — all must be satisfied.

See the TRAPI spec ``AttributeConstraint`` schema for full details.
"""

import re


def matches_attribute_constraints(attributes, constraints):
    """Check if a list of TRAPI attributes satisfies all constraints.

    Args:
        attributes: List of TRAPI Attribute dicts, each with at least
            ``attribute_type_id`` and ``value``.
        constraints: List of TRAPI AttributeConstraint dicts, each with
            ``id``, ``operator``, ``value``, and optionally ``not``.

    Returns:
        True if ALL constraints are satisfied (AND logic).
        Returns True if constraints is None or empty.
    """
    if not constraints:
        return True
    if not attributes:
        # No attributes but there are constraints — check if all are negated
        for c in constraints:
            if not c.get("not", False):
                return False
        return True

    for constraint in constraints:
        result = _evaluate_constraint(attributes, constraint)
        if constraint.get("not", False):
            result = not result
        if not result:
            return False

    return True


def _evaluate_constraint(attributes, constraint):
    """Evaluate a single constraint against a list of attributes.

    Finds attributes matching the constraint's ``id`` (matched against
    ``attribute_type_id`` or ``original_attribute_name``), then applies
    the operator.  If no matching attribute is found, returns False.

    For operators ``>`` and ``<`` with list constraint values, at least
    one comparison must be true (OR logic per TRAPI spec).
    """
    constraint_id = constraint["id"]
    operator = constraint["operator"]
    constraint_value = constraint["value"]

    # Find matching attributes by attribute_type_id or original_attribute_name
    matching_values = []
    for attr in attributes:
        if (
            attr.get("attribute_type_id") == constraint_id
            or attr.get("original_attribute_name") == constraint_id
        ):
            matching_values.append(attr.get("value"))

    if not matching_values:
        return False

    # At least one matching attribute must satisfy the operator
    for attr_value in matching_values:
        if _apply_operator(operator, attr_value, constraint_value):
            return True

    return False


def _apply_operator(operator, attr_value, constraint_value):
    """Apply an operator to compare an attribute value against a constraint value.

    Args:
        operator: One of "==", ">", "<", "matches", "==="
        attr_value: The value from the attribute
        constraint_value: The value from the constraint

    Returns:
        True if the comparison holds.
    """
    if operator == "==":
        return attr_value == constraint_value

    elif operator == "===":
        # Strict equality: type, value, and for lists also order
        return (
            type(attr_value) is type(constraint_value)
            and attr_value == constraint_value
        )

    elif operator == ">":
        return _compare_numeric(attr_value, constraint_value, ">")

    elif operator == "<":
        return _compare_numeric(attr_value, constraint_value, "<")

    elif operator == "matches":
        return _matches_regex(attr_value, constraint_value)

    return False


def _compare_numeric(attr_value, constraint_value, direction):
    """Numeric comparison supporting list constraint values (OR logic).

    Per TRAPI spec: with lists and '>' or '<', at least one comparison
    must be true (OR logic).
    """
    if isinstance(constraint_value, list):
        for cv in constraint_value:
            try:
                if direction == ">" and float(attr_value) > float(cv):
                    return True
                if direction == "<" and float(attr_value) < float(cv):
                    return True
            except (TypeError, ValueError):
                continue
        return False

    try:
        if direction == ">":
            return float(attr_value) > float(constraint_value)
        return float(attr_value) < float(constraint_value)
    except (TypeError, ValueError):
        return False


def _matches_regex(attr_value, pattern):
    """Regex match using re.search on string values."""
    try:
        return re.search(str(pattern), str(attr_value)) is not None
    except re.error:
        return False
