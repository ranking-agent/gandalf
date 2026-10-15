"""TRAPI 1.6 at the edges of a server that is TRAPI 2.0 inside.

This module exists only on the TRAPI 1.6 maintenance branch.  Everything
between the request and the response -- the graph format, the search, the
response builder, ``gandalf.search.edge_constraints`` -- is shared with the
TRAPI 2.0 line untouched, so a graph built by either line is served by both
and changes merge across with little friction.  The 1.6 differences are
applied at the two boundaries instead:

Requests (1.6 -> internal)
    * A QEdge's ``qualifier_constraints`` / ``attribute_constraints`` lists are
      read into the internal ``constraints`` object by
      :func:`qedge_constraints`; ``EdgeConstraints.parse`` calls it.
    * ``log_level`` and ``bypass_cache`` sit at the top level of a 1.6 Query
      rather than in ``parameters``; :func:`gandalf.trapi.query_parameters`
      folds them in, so ``parameters`` stays gandalf's own settings object.

Responses (internal -> 1.6), by :func:`to_trapi_1_6`
    * A NodeBinding / EdgeBinding is a list of ``{"id", "attributes"}``
      objects rather than one ``{"ids": [...]}`` object.  (Results written
      straight to JSON are already 1.6-shaped: see ``_single_path_json``.)
    * ``Result.analyses`` is required, so a result that binds no QEdge
      carries ``[]`` rather than no ``analyses`` at all.
    * ``AuxiliaryGraph.attributes`` and ``Node.attributes`` are required.
    * ``knowledge_level`` and ``agent_type`` are Edge *attributes*
      (``biolink:knowledge_level`` / ``biolink:agent_type``), not top-level
      Edge properties.

Every conversion is idempotent, so it is safe to run again over a response
that a response annotator plugin has added to.
"""

from typing import Any

import orjson

#: The attribute types 1.6 carries knowledge_level and agent_type as.
_KNOWLEDGE_LEVEL = "biolink:knowledge_level"
_AGENT_TYPE = "biolink:agent_type"


class TRAPI16ConstraintError(ValueError):
    """Raised when a 1.6 ``qualifier_constraints`` value is malformed."""


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------


def qedge_constraints(qedge: Any) -> Any:
    """Read a TRAPI 1.6 QEdge's constraints as the internal ``constraints`` object.

    1.6 keeps qualifier and attribute constraints in two lists on the QEdge,
    and spells a QualifierSetConstraint as ``{"qualifier_set": [{
    "qualifier_type_id": ..., "qualifier_value": ...}, ...]}``; internally
    (and in TRAPI 2.0) it is a ``{qualifier_type_id: qualifier_value}``
    mapping.

    Args:
        qedge: A TRAPI 1.6 QEdge dict.

    Returns:
        The internal constraints object, or None when the QEdge has none.

    Raises:
        TRAPI16ConstraintError: for a malformed ``qualifier_constraints``.

    Examples:
        >>> qedge_constraints({"subject": "n0", "object": "n1"}) is None
        True
        >>> qedge_constraints({"qualifier_constraints": [{"qualifier_set": [
        ...     {"qualifier_type_id": "biolink:object_aspect_qualifier",
        ...      "qualifier_value": "activity"}]}]})
        {'qualifiers': [{'biolink:object_aspect_qualifier': 'activity'}]}
        >>> qedge_constraints({"attribute_constraints": [
        ...     {"id": "biolink:p_value", "operator": "<", "value": 0.05}]})
        {'attributes': [{'id': 'biolink:p_value', 'operator': '<', 'value': 0.05}]}
    """
    constraints: dict[str, Any] = {}

    qualifier_constraints = qedge.get("qualifier_constraints")
    if qualifier_constraints:
        if not isinstance(qualifier_constraints, list):
            raise TRAPI16ConstraintError(
                f"qualifier_constraints must be a list, got {qualifier_constraints!r}"
            )
        constraints["qualifiers"] = [
            _qualifier_set(constraint) for constraint in qualifier_constraints
        ]

    attribute_constraints = qedge.get("attribute_constraints")
    if attribute_constraints:
        # Same AttributeConstraint shape in both versions; the internal
        # parser validates it.
        constraints["attributes"] = attribute_constraints

    return constraints or None


def _qualifier_set(constraint: Any) -> dict[str, str]:
    """One 1.6 QualifierConstraint as a ``{type: value}`` mapping."""
    qualifier_set = (
        constraint.get("qualifier_set") if isinstance(constraint, dict) else None
    )
    if not isinstance(qualifier_set, list):
        raise TRAPI16ConstraintError(
            "each qualifier_constraints entry must be an object with a "
            f"'qualifier_set' list, got {constraint!r}"
        )
    mapping: dict[str, str] = {}
    for qualifier in qualifier_set:
        type_id = (
            qualifier.get("qualifier_type_id") if isinstance(qualifier, dict) else None
        )
        value = (
            qualifier.get("qualifier_value") if isinstance(qualifier, dict) else None
        )
        if not isinstance(type_id, str) or not isinstance(value, str):
            raise TRAPI16ConstraintError(
                "each qualifier in a qualifier_set needs a string "
                f"'qualifier_type_id' and 'qualifier_value', got {qualifier!r}"
            )
        if type_id in mapping and mapping[type_id] != value:
            # AND of two values of one type: the internal form keys on the
            # type, and no edge in this graph carries one type twice.
            raise TRAPI16ConstraintError(
                f"qualifier_set names {type_id} more than once; this server "
                "matches one value per qualifier type"
            )
        mapping[type_id] = value
    return mapping


def fold_top_level_parameters(query: Any, parameters: dict) -> dict:
    """``parameters`` with a 1.6 Query's top-level ``log_level`` / ``bypass_cache``.

    Returns *parameters* itself when there is nothing to fold in, and a new
    dict otherwise: the request's own ``parameters`` is never changed.

    Examples:
        >>> fold_top_level_parameters({"log_level": "DEBUG"}, {"subclass": False})
        {'subclass': False, 'log_level': 'DEBUG'}
        >>> p = {"subclass": False}
        >>> fold_top_level_parameters({}, p) is p
        True
    """
    folded = None
    for key in ("log_level", "bypass_cache"):
        value = query.get(key)
        if value is not None:
            if folded is None:
                folded = dict(parameters)
            folded[key] = value
    return folded if folded is not None else parameters


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------


def bound_ids(binding: Any) -> list:
    """The IDs a NodeBinding / EdgeBinding binds, in either version's shape.

    For code that reads results after they have been converted (response
    annotators, ``gandalf.validation``) and may also be handed internal ones.

    Examples:
        >>> bound_ids([{"id": "A", "attributes": []}, {"id": "B", "attributes": []}])
        ['A', 'B']
        >>> bound_ids({"ids": ["A"]})
        ['A']
        >>> bound_ids(None)
        []
    """
    if isinstance(binding, dict):
        return list(binding.get("ids") or ())
    if isinstance(binding, list):
        return [b["id"] for b in binding if isinstance(b, dict) and "id" in b]
    return []


def to_trapi_1_6(response: Any) -> Any:
    """Turn a response built in the internal (2.0) shape into TRAPI 1.6, in place.

    Costs one pass over the distinct knowledge-graph edges and auxiliary
    graphs, and one over the results still built as dicts -- not over the
    results written straight to JSON, which ``_single_path_json`` already
    writes in the 1.6 shape.

    Args:
        response: A TRAPI Response dict (mutated in place).

    Returns:
        The same dict.

    Examples:
        >>> r = to_trapi_1_6({"message": {
        ...     "knowledge_graph": {"nodes": {}, "edges": {"e": {
        ...         "predicate": "biolink:treats", "knowledge_level": "knowledge_assertion",
        ...         "agent_type": "manual_agent", "attributes": []}}},
        ...     "results": [{"node_bindings": {"n0": {"ids": ["A"]}},
        ...                  "analyses": [{"resource_id": "infores:x",
        ...                                "edge_bindings": {"e0": {"ids": ["e"]}}}]}],
        ...     "auxiliary_graphs": {"a": {"edges": ["e"]}}}})
        >>> r["message"]["results"][0]["node_bindings"]
        {'n0': [{'id': 'A', 'attributes': []}]}
        >>> r["message"]["results"][0]["analyses"][0]["edge_bindings"]
        {'e0': [{'id': 'e', 'attributes': []}]}
        >>> r["message"]["auxiliary_graphs"]
        {'a': {'edges': ['e'], 'attributes': []}}
        >>> [a["attribute_type_id"] for a in r["message"]["knowledge_graph"]["edges"]["e"]["attributes"]]
        ['biolink:knowledge_level', 'biolink:agent_type']
        >>> to_trapi_1_6(r) == r  # idempotent
        True
    """
    message = response.get("message")
    if not isinstance(message, dict):
        return response

    knowledge_graph = message.get("knowledge_graph") or {}
    for node in (knowledge_graph.get("nodes") or {}).values():
        node_to_1_6(node)
    for edge in (knowledge_graph.get("edges") or {}).values():
        edge_to_1_6(edge)

    for aux_graph in (message.get("auxiliary_graphs") or {}).values():
        if isinstance(aux_graph, dict):
            aux_graph.setdefault("attributes", [])

    results = message.get("results")
    if results:
        # Bindings are shared between results (see "Bindings are shared
        # objects" in CLAUDE.MD); convert each one once and share the result.
        converted: dict[int, Any] = {}
        for result in results:
            if isinstance(result, dict):
                _result_to_1_6(result, converted)

    return response


def _result_to_1_6(result: dict, converted: dict) -> None:
    """Convert one Result's bindings, in place."""
    node_bindings = result.get("node_bindings")
    if node_bindings:
        for qnode_id, binding in node_bindings.items():
            node_bindings[qnode_id] = _bindings_to_1_6(binding, converted)

    analyses = result.get("analyses")
    if analyses is None:
        result["analyses"] = []
        return
    for analysis in analyses:
        edge_bindings = analysis.get("edge_bindings")
        if edge_bindings:
            for qedge_id, binding in edge_bindings.items():
                edge_bindings[qedge_id] = _bindings_to_1_6(binding, converted)
        elif edge_bindings is None:
            analysis["edge_bindings"] = {}


def _bindings_to_1_6(binding: Any, converted: dict) -> Any:
    """``{"ids": [a, b]}`` as ``[{"id": a, "attributes": []}, ...]``.

    Anything else (already a list, as on the rehydrate path) is left alone.
    """
    if not isinstance(binding, dict):
        return binding
    key = id(binding)
    done = converted.get(key)
    if done is None:
        extra = {k: v for k, v in binding.items() if k != "ids"}
        done = converted[key] = [
            {"id": bound_id, "attributes": [], **extra}
            for bound_id in binding.get("ids") or ()
        ]
    return done


def lacks_stored_attributes(attributes: Any) -> bool:
    """Whether an Edge's attributes are absent but for knowledge_level/agent_type.

    What a dehydrated 1.6 edge carries, so rehydrating it must still fill in
    the stored attributes.

    Examples:
        >>> lacks_stored_attributes(None), lacks_stored_attributes([])
        (True, True)
        >>> lacks_stored_attributes([{"attribute_type_id": "biolink:agent_type",
        ...                           "value": "manual_agent"}])
        True
        >>> lacks_stored_attributes([{"attribute_type_id": "biolink:publications",
        ...                           "value": []}])
        False
    """
    if attributes is None:
        return True
    return isinstance(attributes, list) and all(
        isinstance(attribute, dict)
        and attribute.get("attribute_type_id") in (_KNOWLEDGE_LEVEL, _AGENT_TYPE)
        for attribute in attributes
    )


def node_to_1_6(node: Any) -> Any:
    """Give a Node the ``attributes`` list TRAPI 1.6 requires.

    A dehydrated response, and a node the graph does not know, carry none.

    Examples:
        >>> node_to_1_6({"categories": ["biolink:Gene"]})
        {'categories': ['biolink:Gene'], 'attributes': []}
    """
    if node.get("attributes") is None:
        node["attributes"] = []
    return node


def edge_to_1_6(edge: Any) -> Any:
    """Move an Edge's ``knowledge_level`` / ``agent_type`` into its attributes.

    Attributes still held as the graph's stored JSON (``bytes``, see
    ``gandalf.trapi.edge_attributes``) are extended without being decoded,
    and come out byte-identical to appending to the decoded list.  The graph
    never stores these two as attributes, so there is nothing to dedupe
    against there; a list may come from a client (the rehydrate path), and an
    attribute already present is kept rather than doubled.

    Examples:
        >>> edge_to_1_6({"knowledge_level": "not_provided", "agent_type": "not_provided",
        ...              "attributes": b'[{"attribute_type_id":"biolink:x","value":1}]'})["attributes"]
        b'[{"attribute_type_id":"biolink:x","value":1},{"attribute_type_id":"biolink:knowledge_level","value":"not_provided"},{"attribute_type_id":"biolink:agent_type","value":"not_provided"}]'
        >>> edge_to_1_6({"knowledge_level": "knowledge_assertion", "agent_type": "manual_agent",
        ...              "attributes": [{"attribute_type_id": "biolink:knowledge_level",
        ...                              "value": "prediction"}]})["attributes"]
        [{'attribute_type_id': 'biolink:knowledge_level', 'value': 'prediction'}, {'attribute_type_id': 'biolink:agent_type', 'value': 'manual_agent'}]
        >>> edge_to_1_6({"predicate": "biolink:treats"})
        {'predicate': 'biolink:treats'}
    """
    knowledge_level = edge.pop("knowledge_level", None)
    agent_type = edge.pop("agent_type", None)
    if knowledge_level is None and agent_type is None:
        return edge

    added = []
    if knowledge_level is not None:
        added.append({"attribute_type_id": _KNOWLEDGE_LEVEL, "value": knowledge_level})
    if agent_type is not None:
        added.append({"attribute_type_id": _AGENT_TYPE, "value": agent_type})

    attributes = edge.get("attributes")
    if type(attributes) is bytes:
        encoded = orjson.dumps(added)[1:-1]
        body = attributes[1:-1]
        edge["attributes"] = b"[" + (body + b"," if body else b"") + encoded + b"]"
        return edge

    if attributes is None:
        attributes = edge["attributes"] = []
    present = {
        attribute.get("attribute_type_id")
        for attribute in attributes
        if isinstance(attribute, dict)
    }
    attributes.extend(a for a in added if a["attribute_type_id"] not in present)
    return edge
