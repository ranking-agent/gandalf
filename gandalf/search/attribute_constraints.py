"""TRAPI attribute constraint matching for nodes and edges.

Attribute constraints allow callers to filter results based on attribute
values.  Each constraint specifies an attribute_type_id (matched via the
``id`` field), an operator, and a value.  Multiple constraints use AND
logic — all must be satisfied.

Attribute values in a knowledge graph are frequently lists — ``publications``
is the canonical example.  Every operator except ``===`` is evaluated against
each member of such a list, so ``==`` reads as "contains".  That makes
filtering an edge down to specific PubMed IDs a plain equality constraint::

    {
        "id": "biolink:publications",
        "operator": "==",
        "value": ["PMID:23456789", "PMID:11111111"],
    }

keeps only edges whose publications include at least one of those PMIDs.
Publication identifiers are compared canonically, so ``PMID:23456789``,
``pubmed:23456789``, ``https://pubmed.ncbi.nlm.nih.gov/23456789`` and the bare
``23456789`` all select the same article.

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
    # Per TRAPI: a list constraint value means OR — the constraint is satisfied
    # if the attribute matches ANY member of the list.  `===` is excluded: it
    # intentionally compares lists by exact type/value/order.
    if isinstance(constraint_value, list) and operator != "===":
        return any(_apply_operator(operator, attr_value, cv) for cv in constraint_value)

    # A list-valued attribute (publications, for example) is satisfied when ANY
    # of its members satisfies the operator, so `==` reads as "contains".
    # `===` is again excluded so it can compare the list as a whole.
    if isinstance(attr_value, (list, tuple)) and operator != "===":
        return any(_apply_operator(operator, av, constraint_value) for av in attr_value)

    if operator == "==":
        return _equals(attr_value, constraint_value)

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


def _equals(attr_value, constraint_value):
    """Equality comparison that understands publication identifiers.

    Falls back to canonical publication comparison when plain equality fails,
    so the different spellings of a PubMed ID found across knowledge sources
    (``PMID:123``, ``pubmed:123``, a pubmed.ncbi.nlm.nih.gov URL) all match one
    another and the bare accession ``123``.
    """
    if attr_value == constraint_value:
        return True
    return _publications_equal(attr_value, constraint_value)


# The spellings of a PubMed reference seen in KGX edge attributes.
_PUBMED_CURIE = re.compile(r"^\s*(?:pmid|pubmed)\s*:\s*(\d+)\s*$", re.IGNORECASE)
_PUBMED_URL = re.compile(
    r"^\s*https?://(?:www\.)?(?:pubmed\.ncbi\.nlm\.nih\.gov|ncbi\.nlm\.nih\.gov/pubmed)"
    r"/(\d+)/?\s*$",
    re.IGNORECASE,
)
_BARE_ACCESSION = re.compile(r"^\s*(\d+)\s*$")


def _publications_equal(attr_value, constraint_value):
    """Compare two values as PubMed identifiers.

    At least one side must be an *explicit* PubMed identifier — a ``PMID:``/
    ``pubmed:`` CURIE or a PubMed URL.  A bare accession only counts as a
    PubMed ID opposite such a side, which keeps this out of the way of
    ordinary numeric attributes.

    Returns:
        True if both values denote the same PubMed article.
    """
    attr_explicit = _explicit_pubmed_accession(attr_value)
    constraint_explicit = _explicit_pubmed_accession(constraint_value)
    if attr_explicit is None and constraint_explicit is None:
        return False

    attr_accession = (
        attr_explicit if attr_explicit is not None else _bare_accession(attr_value)
    )
    constraint_accession = (
        constraint_explicit
        if constraint_explicit is not None
        else _bare_accession(constraint_value)
    )
    return attr_accession is not None and attr_accession == constraint_accession


def _explicit_pubmed_accession(value):
    """Return the accession digits of an explicit PubMed identifier, else None."""
    if not isinstance(value, str):
        return None
    match = _PUBMED_CURIE.match(value) or _PUBMED_URL.match(value)
    return match.group(1) if match else None


def _bare_accession(value):
    """Return the digits of a bare accession (``"123"`` or ``123``), else None."""
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        return None
    match = _BARE_ACCESSION.match(str(value))
    return match.group(1) if match else None


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
