"""TRAPI 2.0 ``QEdge.constraints`` parsing and matching.

TRAPI 2.0 gathers every constraint a client can place on a query edge into one
``constraints`` object::

    "e0": {
        "subject": "n0",
        "object": "n1",
        "constraints": {
            "knowledge_level": {"behavior": "ALLOW",
                                "values": ["knowledge_assertion"]},
            "agent_type": {"behavior": "DENY", "values": ["text_mining_agent"]},
            "sources": {"behavior": "ALLOW", "values": ["infores:ctd"],
                        "primary_only": true},
            "qualifiers": [{"biolink:object_aspect_qualifier": "activity"}],
            "attributes": [{"id": "biolink:publications", "operator": "==",
                            "value": ["PMID:23456789"]}]
        }
    }

This replaces the 1.x ``qualifier_constraints`` / ``attribute_constraints``
lists and adds the ``knowledge_level``, ``agent_type`` and ``sources``
allow/deny constraints.

:class:`EdgeConstraints` parses that object once per QEdge and then answers a
single question per candidate edge, so the traversal loops in
``gandalf.search.query_edge`` stay a flat sequence of checks no matter how many
constraint kinds the spec grows.
"""

from dataclasses import dataclass
from typing import Any, Optional

from gandalf.search.attribute_constraints import matches_attribute_constraints
from gandalf.search.qualifiers import edge_matches_qualifier_constraints

ALLOW = "ALLOW"
DENY = "DENY"

# The QEdge.constraints keys this server understands.  A client that sends
# something else gets told rather than silently served unfiltered results.
KNOWN_CONSTRAINTS = frozenset(
    {"knowledge_level", "agent_type", "sources", "qualifiers", "attributes"}
)

PRIMARY_KNOWLEDGE_SOURCE = "primary_knowledge_source"


class ConstraintError(ValueError):
    """Raised when a QEdge's ``constraints`` object is malformed."""


@dataclass(frozen=True)
class AllowDeny:
    """A TRAPI ``AllowDenyConstraint``: a set of values, allowed or denied.

    ALLOW is satisfied when at least one of ``values`` is present (OR); DENY is
    satisfied when none of them is (NOT (x OR y)).

    Examples:
        >>> allow = AllowDeny.parse({"behavior": "ALLOW", "values": ["a", "b"]},
        ...                         "knowledge_level")
        >>> allow.permits({"a"}), allow.permits({"c"}), allow.permits(set())
        (True, False, False)
        >>> deny = AllowDeny.parse({"behavior": "DENY", "values": ["a"]},
        ...                        "agent_type")
        >>> deny.permits({"a"}), deny.permits({"c"}), deny.permits(set())
        (False, True, True)
    """

    behavior: str
    values: frozenset

    @classmethod
    def parse(cls, raw: Any, field: str) -> "AllowDeny":
        """Build an AllowDeny from a raw constraint dict, or raise."""
        if not isinstance(raw, dict):
            raise ConstraintError(
                f"constraints.{field} must be an object with 'behavior' and "
                f"'values', got {raw!r}"
            )
        behavior = raw.get("behavior")
        if behavior not in (ALLOW, DENY):
            raise ConstraintError(
                f"constraints.{field}.behavior must be '{ALLOW}' or '{DENY}', "
                f"got {behavior!r}"
            )
        values = raw.get("values")
        if not isinstance(values, list) or not values:
            raise ConstraintError(
                f"constraints.{field}.values must be a non-empty list, "
                f"got {values!r}"
            )
        return cls(behavior=behavior, values=frozenset(values))

    def permits(self, present: set) -> bool:
        """Check the values an edge actually carries against this constraint."""
        overlaps = not self.values.isdisjoint(present)
        return overlaps if self.behavior == ALLOW else not overlaps


@dataclass(frozen=True)
class SourcesConstraint:
    """A TRAPI ``constraints.sources``: an AllowDeny over source infores CURIEs.

    ``primary_only`` narrows the check from every RetrievalSource on the edge
    to just the one with the ``primary_knowledge_source`` role.
    """

    allow_deny: AllowDeny
    primary_only: bool = False

    @classmethod
    def parse(cls, raw: Any) -> "SourcesConstraint":
        """Build a SourcesConstraint from a raw constraint dict, or raise."""
        allow_deny = AllowDeny.parse(raw, "sources")
        primary_only = raw.get("primary_only", False)
        if not isinstance(primary_only, bool):
            raise ConstraintError(
                f"constraints.sources.primary_only must be a boolean, "
                f"got {primary_only!r}"
            )
        return cls(allow_deny=allow_deny, primary_only=primary_only)

    def permits(self, sources: Optional[list]) -> bool:
        """Check an edge's RetrievalSource list against this constraint."""
        present = {
            source.get("resource_id")
            for source in sources or ()
            if not self.primary_only
            or source.get("resource_role") == PRIMARY_KNOWLEDGE_SOURCE
        }
        present.discard(None)
        return self.allow_deny.permits(present)


@dataclass(frozen=True)
class EdgeConstraints:
    """Every constraint one QEdge places on the edges bound to it.

    Parsed once per QEdge by :meth:`parse`, then applied per candidate edge by
    :meth:`permits`.  An instance with no constraints is falsy, so callers can
    skip the check entirely on the (common) unconstrained path.
    """

    qualifiers: tuple = ()
    attributes: tuple = ()
    knowledge_level: Optional[AllowDeny] = None
    agent_type: Optional[AllowDeny] = None
    sources: Optional[SourcesConstraint] = None

    @classmethod
    def parse(cls, qedge: dict) -> "EdgeConstraints":
        """Build the constraints for one QEdge from its ``constraints`` object.

        Args:
            qedge: A TRAPI QEdge dict.

        Returns:
            The parsed constraints; empty when the QEdge has none.

        Raises:
            ConstraintError: if ``constraints`` is not an object, names a key
                this server does not implement, or holds a malformed value.

        Examples:
            >>> EdgeConstraints.parse({"subject": "n0", "object": "n1"})
            EdgeConstraints(qualifiers=(), attributes=(), knowledge_level=None, agent_type=None, sources=None)
            >>> bool(EdgeConstraints.parse({"constraints": {
            ...     "agent_type": {"behavior": "DENY",
            ...                    "values": ["text_mining_agent"]}}}))
            True
        """
        raw = qedge.get("constraints")
        if raw is None:
            return cls()
        if not isinstance(raw, dict):
            raise ConstraintError(f"QEdge 'constraints' must be an object, got {raw!r}")

        unknown = sorted(set(raw) - KNOWN_CONSTRAINTS)
        if unknown:
            raise ConstraintError(
                f"unsupported QEdge constraint(s) {', '.join(unknown)}; "
                f"this server implements {', '.join(sorted(KNOWN_CONSTRAINTS))}"
            )

        return cls(
            qualifiers=tuple(_parse_list(raw, "qualifiers")),
            attributes=tuple(_parse_list(raw, "attributes")),
            knowledge_level=_parse_allow_deny(raw, "knowledge_level"),
            agent_type=_parse_allow_deny(raw, "agent_type"),
            sources=(
                SourcesConstraint.parse(raw["sources"])
                if raw.get("sources") is not None
                else None
            ),
        )

    def with_qualifiers(self, qualifiers) -> "EdgeConstraints":
        """Return a copy carrying *qualifiers*, used to install expanded values."""
        return EdgeConstraints(
            qualifiers=tuple(qualifiers or ()),
            attributes=self.attributes,
            knowledge_level=self.knowledge_level,
            agent_type=self.agent_type,
            sources=self.sources,
        )

    def __bool__(self) -> bool:
        return bool(
            self.qualifiers
            or self.attributes
            or self.knowledge_level
            or self.agent_type
            or self.sources
        )

    @property
    def needs_attributes(self) -> bool:
        """Whether checking an edge requires the cold-path LMDB attribute read."""
        return bool(self.attributes)

    def permits(self, graph, props: dict, fwd_edge_idx: int) -> bool:
        """Check one candidate edge against every constraint (all must hold).

        Args:
            graph: The CSRGraph being traversed.
            props: The edge's traversal properties, as returned by
                ``EdgePropertyStore._get_props`` (qualifiers, sources).
            fwd_edge_idx: The edge's forward-CSR position, used to read
                knowledge_level / agent_type and the cold-path attributes.

        Returns:
            True if the edge satisfies every constraint.
        """
        if self.qualifiers and not edge_matches_qualifier_constraints(
            props.get("qualifiers"), self.qualifiers
        ):
            return False

        # knowledge_level and agent_type are read on demand rather than
        # assembled into every traversal dict, since most queries never
        # constrain on them.
        if self.knowledge_level is not None or self.agent_type is not None:
            knowledge_level, agent_type = graph.edge_properties.get_kl_at(fwd_edge_idx)
            if self.knowledge_level is not None and not self.knowledge_level.permits(
                {knowledge_level}
            ):
                return False
            if self.agent_type is not None and not self.agent_type.permits(
                {agent_type}
            ):
                return False

        if self.sources is not None and not self.sources.permits(props.get("sources")):
            return False

        # Cold path last: it costs an LMDB read, so only edges that already
        # passed every in-memory check pay for it.
        if self.attributes and not _edge_attributes_match(
            graph, fwd_edge_idx, self.attributes
        ):
            return False

        return True


def _parse_list(raw: dict, field: str) -> list:
    """Read a list-valued constraint, rejecting anything that is not a list."""
    value = raw.get(field)
    if value is None:
        return []
    if not isinstance(value, list):
        raise ConstraintError(f"constraints.{field} must be a list, got {value!r}")
    return value


def _parse_allow_deny(raw: dict, field: str) -> Optional[AllowDeny]:
    """Read an allow/deny constraint, or None when the QEdge omits it."""
    value = raw.get(field)
    return None if value is None else AllowDeny.parse(value, field)


def _edge_attributes_match(graph, fwd_edge_idx: int, attribute_constraints) -> bool:
    """Check an edge's attributes, fetched from LMDB, against the constraints."""
    if graph.lmdb_store is None:
        # No LMDB store — no attributes to check against
        return False

    detail = graph.lmdb_store.get(fwd_edge_idx)
    return bool(
        matches_attribute_constraints(
            detail.get("attributes", []), attribute_constraints
        )
    )
