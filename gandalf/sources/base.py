"""Abstract graph source + validation of gandalf's normalized record contract.

A :class:`GraphSource` yields *already-normalized* node and edge records that the
loader's build core (``gandalf.loader._build_graph_from_source``) consumes. Two
concrete sources exist: ``KGXJsonlSource`` (reads KGX jsonl and normalizes via
``gandalf.normalize``) and ``MongoSource`` (reads documents that an upstream
pipeline already normalized).

The normalized contract enforced by :func:`validate_normalized_edge` /
:func:`validate_normalized_node`:

NormalizedEdge::

    subject:    str   (required, non-empty)
    object:     str   (required, non-empty)
    predicate:  str   (required, non-empty)
    id:         str | None
    sources:    list[ {resource_id:str, resource_role:str,
                        upstream_resource_ids:list[str]} ]
    qualifiers: list[ {qualifier_type_id:str, qualifier_value:str} ]
    attributes: list[ {attribute_type_id:str, value:Any,
                        original_attribute_name:str} ]

NormalizedNode::

    id:         str   (required, non-empty)
    name:       str | None
    categories: list[str]   # NOTE plural; raw KGX uses singular ``category``
    attributes: list[ {attribute_type_id:str, value:Any,
                        original_attribute_name:str} ]
"""

import abc
from typing import Iterator, Tuple


class SourceValidationError(ValueError):
    """Raised when a source yields a record not in gandalf's normalized form."""


class GraphSource(abc.ABC):
    """A re-iterable source of normalized node and edge records.

    Implementations must return a *fresh* generator from each method on every
    call (the loader iterates edges twice: once for vocabulary, once for the
    build).

    Ordering invariant: ``iter_edge_triples()`` and ``iter_edges()`` MUST yield
    edges in the same order, and that order MUST be stable across calls. The
    build maps edge ``i`` from ``iter_edges()`` to the position ``i`` counted in
    ``iter_edge_triples()``; a mismatch silently corrupts the graph.
    """

    @abc.abstractmethod
    def iter_edge_triples(self) -> Iterator[Tuple[str, str, str]]:
        """Pass 1: yield ``(subject, object, predicate)`` only.

        Cheap path used for vocabulary collection; no normalization is performed.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def iter_edges(self) -> Iterator[dict]:
        """Pass 2: yield fully normalized + validated edge dicts."""
        raise NotImplementedError

    @abc.abstractmethod
    def iter_nodes(self) -> Iterator[dict]:
        """Yield fully normalized + validated node dicts."""
        raise NotImplementedError


def _require_str(record_kind: str, record_id, field: str, value) -> None:
    if not isinstance(value, str) or not value:
        raise SourceValidationError(
            f"{record_kind} {record_id!r}: field {field!r} must be a non-empty "
            f"string, got {value!r}"
        )


def validate_normalized_edge(edge: dict) -> None:
    """Validate a normalized edge against the contract; raise on the first issue."""
    if not isinstance(edge, dict):
        raise SourceValidationError(f"edge must be a dict, got {type(edge).__name__}")

    edge_id = edge.get("id")
    if edge_id is not None and not isinstance(edge_id, str):
        raise SourceValidationError(
            f"edge {edge_id!r}: field 'id' must be a string or None, got {edge_id!r}"
        )

    for field in ("subject", "object", "predicate"):
        _require_str("edge", edge_id, field, edge.get(field))

    for field in ("sources", "qualifiers", "attributes"):
        if not isinstance(edge.get(field), list):
            raise SourceValidationError(
                f"edge {edge_id!r}: field {field!r} must be a list, "
                f"got {edge.get(field)!r}"
            )

    for source in edge["sources"]:
        if not isinstance(source, dict):
            raise SourceValidationError(
                f"edge {edge_id!r}: each source must be a dict, got {source!r}"
            )
        _require_str(
            "edge", edge_id, "sources[].resource_id", source.get("resource_id")
        )
        _require_str(
            "edge", edge_id, "sources[].resource_role", source.get("resource_role")
        )
        if not isinstance(source.get("upstream_resource_ids"), list):
            raise SourceValidationError(
                f"edge {edge_id!r}: source {source.get('resource_id')!r} must have an "
                f"'upstream_resource_ids' list, got {source.get('upstream_resource_ids')!r}"
            )

    for qualifier in edge["qualifiers"]:
        if not isinstance(qualifier, dict):
            raise SourceValidationError(
                f"edge {edge_id!r}: each qualifier must be a dict, got {qualifier!r}"
            )
        _require_str(
            "edge",
            edge_id,
            "qualifiers[].qualifier_type_id",
            qualifier.get("qualifier_type_id"),
        )
        # A TRAPI qualifier_value must be a scalar string; a list/dict here later
        # raises "unhashable type" deep in meta-KG / dedup. Catch it up front.
        if not isinstance(qualifier.get("qualifier_value"), str):
            raise SourceValidationError(
                f"edge {edge_id!r}: qualifier "
                f"{qualifier.get('qualifier_type_id')!r} 'qualifier_value' must be a "
                f"string, got {qualifier.get('qualifier_value')!r}"
            )

    for attribute in edge["attributes"]:
        if not isinstance(attribute, dict):
            raise SourceValidationError(
                f"edge {edge_id!r}: each attribute must be a dict, got {attribute!r}"
            )
        _require_str(
            "edge",
            edge_id,
            "attributes[].attribute_type_id",
            attribute.get("attribute_type_id"),
        )
        if "value" not in attribute:
            raise SourceValidationError(
                f"edge {edge_id!r}: attribute "
                f"{attribute.get('attribute_type_id')!r} is missing 'value'"
            )


def validate_normalized_node(node: dict) -> None:
    """Validate a normalized node against the contract; raise on the first issue."""
    if not isinstance(node, dict):
        raise SourceValidationError(f"node must be a dict, got {type(node).__name__}")

    node_id = node.get("id")
    _require_str("node", node_id, "id", node_id)

    name = node.get("name")
    if name is not None and not isinstance(name, str):
        raise SourceValidationError(
            f"node {node_id!r}: field 'name' must be a string or None, got {name!r}"
        )

    if not isinstance(node.get("categories"), list):
        raise SourceValidationError(
            f"node {node_id!r}: field 'categories' must be a list, "
            f"got {node.get('categories')!r}"
        )

    if not isinstance(node.get("attributes"), list):
        raise SourceValidationError(
            f"node {node_id!r}: field 'attributes' must be a list, "
            f"got {node.get('attributes')!r}"
        )
    for attribute in node["attributes"]:
        if not isinstance(attribute, dict):
            raise SourceValidationError(
                f"node {node_id!r}: each attribute must be a dict, got {attribute!r}"
            )
        _require_str(
            "node",
            node_id,
            "attributes[].attribute_type_id",
            attribute.get("attribute_type_id"),
        )
