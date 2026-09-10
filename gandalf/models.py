"""Pydantic models for GANDALF API request validation and response documentation.

Provides TRAPI-compatible request/response models with OpenAPI examples
for the Swagger UI documentation.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ---------------------------------------------------------------------------
# TRAPI log components
# ---------------------------------------------------------------------------


class LogLevel(str, Enum):
    """TRAPI log severity levels."""

    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"
    DEBUG = "DEBUG"


class SetInterpretation(str, Enum):
    """TRAPI set interpretation modes for QNode IDs."""

    BATCH = "BATCH"
    ALL = "ALL"
    MANY = "MANY"
    COLLATE = "COLLATE"


class LogEntry(BaseModel):
    """A single TRAPI log entry conforming to the Translator Reasoner API spec."""

    timestamp: str = Field(
        ...,
        description="ISO 8601 timestamp, UTC with millisecond precision and an "
        "explicit offset (e.g. 2026-08-24T19:42:44.661+00:00)",
    )
    level: Optional[str] = Field(None, description="Log severity level")
    code: Optional[str] = Field(None, description="Standardized short code")
    message: str = Field(..., description="Human-readable log message")

    model_config = ConfigDict(extra="allow")


# ---------------------------------------------------------------------------
# Query graph components (request validation)
# ---------------------------------------------------------------------------


class QNode(BaseModel):
    """A node in the TRAPI query graph.

    At least one of ``ids`` or ``categories`` should typically be provided.
    A node with ``ids`` is "pinned" to specific entities; one with only
    ``categories`` matches any entity of that type.
    """

    ids: Optional[List[str]] = Field(
        None, description="CURIE identifiers to pin this node to specific entities"
    )
    categories: Optional[List[str]] = Field(
        None,
        description="Biolink categories to restrict the node type "
        "(e.g. 'biolink:Gene')",
    )
    constraints: Optional[List[Dict[str, Any]]] = Field(
        None, description="Attribute constraints for filtering nodes"
    )
    is_set: Optional[bool] = Field(
        None,
        description="Deprecated: use set_interpretation instead. "
        "Whether this node represents a set of entities",
    )
    set_interpretation: Optional[SetInterpretation] = Field(
        None,
        description="Indicates how multiple CURIEs in the ids property are "
        "interpreted. BATCH (default): each CURIE is treated independently. "
        "ALL: all CURIEs must appear in each Result. "
        "MANY: member CURIEs form sets in Results (not supported). "
        "COLLATE: multiple matching nodes are combined into a single Result "
        "(only valid for unpinned nodes without ids).",
    )
    member_ids: Optional[List[str]] = Field(
        None,
        description="CURIE identifiers for set members "
        "(used with MANY/ALL set_interpretation)",
    )

    model_config = ConfigDict(extra="allow")


class AllowDenyBehavior(str, Enum):
    """Whether an AllowDenyConstraint's values are required or forbidden."""

    ALLOW = "ALLOW"
    DENY = "DENY"


class AllowDenyConstraint(BaseModel):
    """A TRAPI list of values that bound edges must, or must not, carry.

    ALLOW is satisfied when at least one listed value is present (OR); DENY
    when none of them is.
    """

    behavior: AllowDenyBehavior = Field(
        ..., description="Whether the values are required (ALLOW) or forbidden (DENY)"
    )
    values: List[str] = Field(
        ..., min_length=1, description="The values to allow or deny"
    )

    model_config = ConfigDict(extra="allow")


class SourcesConstraint(AllowDenyConstraint):
    """An ``AllowDenyConstraint`` over the infores CURIEs in an edge's sources."""

    primary_only: Optional[bool] = Field(
        False,
        description="When true, the constraint applies only to the source with "
        "the primary_knowledge_source role rather than to every source on the "
        "edge.",
    )


class QEdgeConstraints(BaseModel):
    """Constraints a QEdge places on the edges bound to it (TRAPI 2.0).

    Replaces TRAPI 1.x's separate ``qualifier_constraints`` and
    ``attribute_constraints`` lists, and adds allow/deny constraints on an
    edge's ``knowledge_level``, ``agent_type`` and sources.  Every constraint
    given must hold (AND).
    """

    knowledge_level: Optional[AllowDenyConstraint] = Field(
        None,
        description="Allow or deny Biolink knowledge_level values on bound "
        "edges (e.g. 'knowledge_assertion', 'prediction')",
    )
    agent_type: Optional[AllowDenyConstraint] = Field(
        None,
        description="Allow or deny Biolink agent_type values on bound edges "
        "(e.g. 'manual_agent', 'text_mining_agent')",
    )
    sources: Optional[SourcesConstraint] = Field(
        None,
        description="Allow or deny infores CURIEs in the sources of bound edges",
    )
    qualifiers: Optional[List[Dict[str, str]]] = Field(
        None,
        description="QualifierSetConstraints, each a mapping of "
        "qualifier_type_id to the required qualifier_value. AND within a "
        "mapping, OR between mappings.",
    )
    attributes: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="AttributeConstraints applied to bound edges; all must "
        "be satisfied (AND)",
    )

    model_config = ConfigDict(extra="allow")


class QEdge(BaseModel):
    """An edge in the TRAPI query graph.

    Connects two nodes (``subject`` → ``object``) with optional predicate
    filters and a ``constraints`` object.
    """

    subject: str = Field(..., description="Key of the subject node in the query graph")
    object: str = Field(..., description="Key of the object node in the query graph")
    predicates: Optional[List[str]] = Field(
        None,
        description="Biolink predicates to filter edges " "(e.g. 'biolink:treats')",
    )
    knowledge_type: Optional[str] = Field(
        None,
        description="'lookup' (the default when absent) or 'inferred'. Only "
        "'lookup' is supported.",
    )
    constraints: Optional[QEdgeConstraints] = Field(
        None,
        description="Constraints on the edges bound to this QEdge: "
        "knowledge_level, agent_type, sources, qualifiers, attributes.",
    )

    model_config = ConfigDict(extra="allow")


class QPathConstraint(BaseModel):
    """A constraint applied to a Pathfinder query path.

    Constrains intermediate nodes on the path between the path's
    subject and object endpoints (e.g. by biolink category).
    """

    required_intermediate_categories: Optional[List[str]] = Field(
        None,
        description="Biolink categories that must appear at intermediate "
        "nodes along the path (e.g. 'biolink:Gene'). Each path returned must "
        "contain at least one node of each category listed.",
    )

    model_config = ConfigDict(extra="allow")


class QPath(BaseModel):
    """A path in the TRAPI Pathfinder query graph.

    Represents an arbitrary-length connection between two pinned nodes
    (``subject`` → ``object``). Unlike a QEdge, which describes a single
    direct edge, a QPath asks the knowledge graph to discover intermediate
    nodes and edges that link the endpoints. Optional ``predicates`` and
    ``constraints`` restrict which paths qualify.
    """

    subject: str = Field(..., description="Key of the subject node in the query graph")
    object: str = Field(..., description="Key of the object node in the query graph")
    predicates: Optional[List[str]] = Field(
        None,
        description="Biolink predicates that may appear on edges along "
        "the path (e.g. 'biolink:related_to')",
    )
    constraints: Optional[List[QPathConstraint]] = Field(
        None,
        description="Constraints applied to intermediate nodes along the path",
    )

    model_config = ConfigDict(extra="allow")


class QueryGraph(BaseModel):
    """TRAPI query graph containing nodes plus edges and/or paths to match.

    Supports both the standard one/multi-hop query form (``nodes`` +
    ``edges``) and the Pathfinder query form (``nodes`` + ``paths``), as
    well as hybrid graphs that mix both. ``nodes`` is always required;
    at least one of ``edges`` or ``paths`` must be supplied.
    """

    nodes: Dict[str, QNode] = Field(
        ..., description="Named query nodes keyed by identifier (e.g. 'n0', 'n1')"
    )
    edges: Optional[Dict[str, QEdge]] = Field(
        None,
        description="Named query edges keyed by identifier (e.g. 'e0', 'e1'). "
        "Required for standard TRAPI queries; may be omitted in Pathfinder "
        "queries that supply ``paths`` instead.",
    )
    paths: Optional[Dict[str, QPath]] = Field(
        None,
        description="Named query paths keyed by identifier (e.g. 'p0', 'p1') "
        "for TRAPI Pathfinder queries. Each path connects two pinned nodes "
        "with optional predicate filters and intermediate-node constraints.",
    )

    @model_validator(mode="after")
    def _require_edges_or_paths(self) -> "QueryGraph":
        if not self.edges and not self.paths:
            raise ValueError(
                "query_graph must include 'edges' (standard query) or "
                "'paths' (Pathfinder query); at least one is required."
            )
        return self


class Message(BaseModel):
    """TRAPI message containing the query graph and optional results."""

    query_graph: QueryGraph = Field(
        ..., description="The query graph specifying the pattern to match"
    )
    results: Optional[List[Dict[str, Any]]] = Field(
        None, description="Result bindings (populated in responses)"
    )
    knowledge_graph: Optional[Dict[str, Any]] = Field(
        None, description="Knowledge graph subgraph (populated in responses)"
    )

    model_config = ConfigDict(extra="allow")


# ---------------------------------------------------------------------------
# POST /query — request model
# ---------------------------------------------------------------------------

_QUERY_EXAMPLE_ONEHOP: dict = {
    "message": {
        "query_graph": {
            "nodes": {
                "n0": {"ids": ["CHEBI:6801"]},
                "n1": {"categories": ["biolink:Gene"]},
            },
            "edges": {
                "e0": {
                    "subject": "n0",
                    "object": "n1",
                    "predicates": ["biolink:affects"],
                }
            },
        }
    }
}

_QUERY_EXAMPLE_TWOHOP: dict = {
    "message": {
        "query_graph": {
            "nodes": {
                "n0": {"ids": ["CHEBI:6801"]},
                "n1": {"categories": ["biolink:Gene"]},
                "n2": {"categories": ["biolink:Disease"]},
            },
            "edges": {
                "e0": {
                    "subject": "n0",
                    "object": "n1",
                    "predicates": ["biolink:affects"],
                },
                "e1": {
                    "subject": "n1",
                    "object": "n2",
                    "predicates": ["biolink:gene_associated_with_condition"],
                },
            },
        }
    }
}

_QUERY_EXAMPLE_QUALIFIERS: dict = {
    "message": {
        "query_graph": {
            "nodes": {
                "n0": {"ids": ["CHEBI:6801"]},
                "n1": {"categories": ["biolink:Gene"]},
            },
            "edges": {
                "e0": {
                    "subject": "n0",
                    "object": "n1",
                    "predicates": ["biolink:affects"],
                    "constraints": {
                        "qualifiers": [{"biolink:object_aspect_qualifier": "activity"}],
                        "knowledge_level": {
                            "behavior": "ALLOW",
                            "values": ["knowledge_assertion"],
                        },
                        "agent_type": {
                            "behavior": "DENY",
                            "values": ["text_mining_agent"],
                        },
                    },
                }
            },
        }
    }
}

_QUERY_EXAMPLE_PATHFINDER: dict = {
    "message": {
        "query_graph": {
            "nodes": {
                "n0": {"ids": ["CHEBI:6801"]},
                "n1": {"ids": ["MONDO:0005148"]},
            },
            "paths": {
                "p0": {
                    "subject": "n0",
                    "object": "n1",
                    "predicates": ["biolink:related_to"],
                    "constraints": [
                        {"required_intermediate_categories": ["biolink:Gene"]}
                    ],
                }
            },
        }
    }
}

_QUERY_EXAMPLE_WITH_PARAMS: dict = {
    "message": {
        "query_graph": {
            "nodes": {
                "n0": {"ids": ["CHEBI:6801"]},
                "n1": {"categories": ["biolink:Gene"]},
            },
            "edges": {
                "e0": {
                    "subject": "n0",
                    "object": "n1",
                    "predicates": ["biolink:affects"],
                }
            },
        }
    },
    "parameters": {
        "log_level": "DEBUG",
        "timeout": 60,
        "subclass": True,
        "subclass_depth": 1,
        "dehydrated": False,
        "filter_config": {"max_node_degree": 50},
        "annotator_config": {},
    },
}


class QueryParameters(BaseModel):
    """TRAPI ``parameters`` for ``/query`` and ``/asyncquery``.

    TRAPI 2.0 defines this object for "query-time parameters that don't affect
    the semantics of the query or intended workflow, but may affect overall
    behavior of the server", and requires the server to repeat it back in the
    Response.  ``timeout``, ``log_level`` and ``bypass_cache`` are the standard
    TRAPI members (``log_level`` and ``bypass_cache`` moved here from the
    request's top level in 2.0); the rest are gandalf's own, carried under the
    schema's ``additionalProperties``.

    All fields are optional; absent fields fall back to server defaults.
    """

    timeout: Optional[float] = Field(
        None,
        description="Seconds the client is willing to wait. When exceeded the "
        "query stops and the Response reports a 'Timeout' status. A negative "
        "value disables the server's default timeout. A value below what this "
        "server can answer within is refused with HTTP 409.",
    )
    log_level: Optional[LogLevel] = Field(
        None, description="The least critical level of logs to return"
    )
    bypass_cache: Optional[bool] = Field(
        None,
        description="Request fresh information from sources rather than cached "
        "information. Gandalf answers from its own graph and holds no query "
        "cache, so this is accepted and has no effect.",
    )
    subclass: Optional[bool] = Field(
        None, description="Enable biolink subclass inference (default True)"
    )
    subclass_depth: Optional[int] = Field(
        None, description="Maximum subclass_of hops to traverse (default 1)"
    )
    dehydrated: Optional[bool] = Field(
        None,
        description="Return a dehydrated response (skip edge attribute "
        "enrichment). Automatically enabled when path count exceeds the large "
        "result threshold.",
    )
    rehydrate: Optional[bool] = Field(
        None,
        description="When true, skip lookup entirely and only enrich the "
        "knowledge_graph already supplied in message (presence-based).",
    )
    filter_config: Optional[Dict[str, Any]] = Field(
        None,
        description="Plugin-defined node filter settings passed to lookup(). "
        "Each registered NodeFilter plugin reads its own key; unknown keys are "
        "ignored.",
    )
    annotator_config: Optional[Dict[str, Any]] = Field(
        None,
        description="Per-request opt-in dict of response-annotator settings. "
        "Each key activates one registered annotator plugin and the value (a "
        "dict) is the plugin's per-request settings. Unknown keys are ignored.",
    )

    # ``use_enum_values`` keeps log_level a plain string after validation.  The
    # server passes it to ``logging.setLevel``, which rejects anything that is
    # not exactly ``str``, and echoes the whole object back in the Response,
    # where an enum member would not survive JSON serialization.
    model_config = ConfigDict(extra="allow", use_enum_values=True)


class TRAPIQuery(BaseModel):
    """Request body for ``POST /query``.

    Contains a TRAPI message with a query graph specifying the pattern
    to match against the knowledge graph.

    Examples:
        One-hop query (drug → gene)::

            {
                "message": {
                    "query_graph": {
                        "nodes": {
                            "n0": {"ids": ["CHEBI:6801"]},
                            "n1": {"categories": ["biolink:Gene"]}
                        },
                        "edges": {
                            "e0": {
                                "subject": "n0",
                                "object": "n1",
                                "predicates": ["biolink:affects"]
                            }
                        }
                    }
                }
            }
    """

    message: Message = Field(
        ..., description="TRAPI message containing the query graph"
    )
    submitter: Optional[str] = Field(
        None,
        description="Any string self-identifying the submitter of this query, "
        "to aid in tracking the source of queries.",
    )
    parameters: Optional[QueryParameters] = Field(
        None,
        description="TRAPI query parameters: timeout, log_level, bypass_cache, "
        "plus gandalf's subclass, subclass_depth, dehydrated, rehydrate, "
        "filter_config and annotator_config.",
    )

    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "examples": [
                _QUERY_EXAMPLE_ONEHOP,
                _QUERY_EXAMPLE_TWOHOP,
                _QUERY_EXAMPLE_QUALIFIERS,
                _QUERY_EXAMPLE_PATHFINDER,
                _QUERY_EXAMPLE_WITH_PARAMS,
            ]
        },
    )


# ---------------------------------------------------------------------------
# POST /asyncquery — request model
# ---------------------------------------------------------------------------


class WorkflowStep(BaseModel):
    """A single TRAPI workflow operation."""

    id: str = Field(..., description="Workflow operation identifier (e.g. 'lookup')")
    parameters: Optional[Dict[str, Any]] = Field(
        None, description="Operation-specific parameters"
    )


class AsyncTRAPIQuery(BaseModel):
    """Request body for ``POST /asyncquery``.

    Contains a callback URL, a TRAPI message, and an optional workflow
    specification.
    """

    callback: str = Field(
        ..., description="URL to POST results to when the query completes"
    )
    message: Message = Field(
        ..., description="TRAPI message containing the query graph"
    )
    workflow: Optional[List[WorkflowStep]] = Field(
        None,
        description="Workflow operations (defaults to [{'id': 'lookup'}])",
    )
    submitter: Optional[str] = Field(
        None,
        description="Any string self-identifying the submitter of this query, "
        "to aid in tracking the source of queries.",
    )
    set_interpretation: Optional[str] = Field(
        None, description="Set interpretation mode (only 'BATCH' is supported)"
    )
    parameters: Optional[QueryParameters] = Field(
        None,
        description="TRAPI query parameters: timeout, log_level, bypass_cache, "
        "plus gandalf's subclass, subclass_depth, dehydrated, rehydrate, "
        "filter_config and annotator_config.",
    )

    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "examples": [
                {
                    "callback": "https://example.com/callback",
                    "message": {
                        "query_graph": {
                            "nodes": {
                                "n0": {"ids": ["CHEBI:6801"]},
                                "n1": {"categories": ["biolink:Gene"]},
                            },
                            "edges": {
                                "e0": {
                                    "subject": "n0",
                                    "object": "n1",
                                    "predicates": ["biolink:affects"],
                                }
                            },
                        }
                    },
                    "workflow": [{"id": "lookup"}],
                    "parameters": {"subclass": True},
                }
            ]
        },
    )


# ---------------------------------------------------------------------------
# Response models (for Swagger documentation)
# ---------------------------------------------------------------------------


class TRAPIResponse(BaseModel):
    """Response from ``POST /query``.

    Contains the original query graph, a knowledge graph subgraph with
    matching nodes and edges, result bindings, and the TRAPI 2.0 Response
    metadata (including an echo of the request's ``parameters``, which the
    spec requires the server to repeat).
    """

    message: Dict[str, Any] = Field(
        ...,
        description="TRAPI message with query_graph, knowledge_graph, and results",
    )
    status: Optional[str] = Field(
        None,
        description="Short status code for the outcome, e.g. 'Success' or " "'Timeout'",
    )
    description: Optional[str] = Field(
        None, description="Brief human-readable description of the outcome"
    )
    logs: Optional[List[LogEntry]] = Field(
        None, description="Log entries produced while answering the query"
    )
    parameters: Optional[QueryParameters] = Field(
        None,
        description="The query parameters this service received, repeated as "
        "TRAPI 2.0 requires",
    )
    schema_version: Optional[str] = Field(
        None, description="Version of the TRAPI schema used in this document"
    )
    biolink_version: Optional[str] = Field(
        None, description="Version of the Biolink Model used in this document"
    )
    data_release_versions: Optional[Dict[str, str]] = Field(
        None, description="Versions of the data sources used in this document"
    )

    model_config = ConfigDict(extra="allow")


class NodeResponse(BaseModel):
    """Response from ``GET /node/{curie}``."""

    id: str = Field(..., description="Node CURIE identifier")
    name: Optional[str] = Field(None, description="Human-readable node name")
    categories: Optional[List[str]] = Field(
        None, description="Biolink categories for this node"
    )

    model_config = ConfigDict(extra="allow")


class NodeDegreeResponse(BaseModel):
    """Response from ``GET /node_degree/{curie}``."""

    id: str = Field(..., description="Node CURIE identifier")
    degree: int = Field(
        ..., description="Total node degree (incoming + outgoing edges)"
    )

    model_config = ConfigDict(extra="allow")


class EdgeItem(BaseModel):
    """A single edge returned by the edges endpoint."""

    subject: str = Field(..., description="Subject node CURIE")
    object: str = Field(..., description="Object node CURIE")
    predicate: str = Field(..., description="Biolink predicate")
    edge_id: str = Field(..., description="Unique edge identifier")
    sources: List[Dict[str, Any]] = Field(
        default_factory=list, description="Provenance sources"
    )
    qualifiers: List[Dict[str, Any]] = Field(
        default_factory=list, description="Edge qualifiers"
    )


class EdgesResponse(BaseModel):
    """Response from ``GET /edges/{curie}``."""

    query_curie: str = Field(..., description="The queried CURIE")
    edges: List[EdgeItem] = Field(..., description="Matching edges")


class EdgesCountResponse(BaseModel):
    """Response from ``GET /edges/{curie}?count_only=true``."""

    query_curie: str = Field(..., description="The queried CURIE")
    count: int = Field(..., description="Number of matching edges")


class EdgeSummaryItem(BaseModel):
    """A single entry in the edge summary: [predicate, category, count]."""


class EdgeSummaryResponse(BaseModel):
    """Response from ``GET /edge_summary/{curie}``."""

    query_curie: str = Field(..., description="The queried CURIE")
    edge_summary: List[List[Any]] = Field(
        ...,
        description="List of [predicate, category, count] triples",
    )


class MetadataResponse(BaseModel):
    """Response from ``GET /metadata``."""

    node_count: int = Field(..., description="Total number of nodes")
    edge_count: int = Field(..., description="Total number of edges")
    predicate_count: int = Field(..., description="Number of unique predicates")
    category_count: int = Field(..., description="Number of unique categories")
    predicates: Dict[str, int] = Field(
        ..., description="Predicate → edge count mapping"
    )
    categories: Dict[str, int] = Field(..., description="Category → node count mapping")

    model_config = ConfigDict(extra="allow")
