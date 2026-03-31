"""Pydantic models for GANDALF API request validation and response documentation.

Provides TRAPI-compatible request/response models with OpenAPI examples
for the Swagger UI documentation.
"""

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# TRAPI log components
# ---------------------------------------------------------------------------


class LogLevel(str, Enum):
    """TRAPI log severity levels."""

    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"
    DEBUG = "DEBUG"


class LogEntry(BaseModel):
    """A single TRAPI log entry conforming to the Translator Reasoner API spec."""

    timestamp: str = Field(..., description="ISO 8601 timestamp (UTC with 'Z' tag)")
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
        None, description="Whether this node represents a set of entities"
    )

    model_config = ConfigDict(extra="allow")


class QEdge(BaseModel):
    """An edge in the TRAPI query graph.

    Connects two nodes (``subject`` → ``object``) with optional predicate
    and qualifier filters.
    """

    subject: str = Field(..., description="Key of the subject node in the query graph")
    object: str = Field(..., description="Key of the object node in the query graph")
    predicates: Optional[List[str]] = Field(
        None,
        description="Biolink predicates to filter edges " "(e.g. 'biolink:treats')",
    )
    qualifier_constraints: Optional[List[Dict[str, Any]]] = Field(
        None, description="Qualifier constraints for edge filtering"
    )
    attribute_constraints: Optional[List[Dict[str, Any]]] = Field(
        None, description="Attribute constraints for edge filtering"
    )

    model_config = ConfigDict(extra="allow")


class QueryGraph(BaseModel):
    """TRAPI query graph containing nodes and edges to match."""

    nodes: Dict[str, QNode] = Field(
        ..., description="Named query nodes keyed by identifier (e.g. 'n0', 'n1')"
    )
    edges: Dict[str, QEdge] = Field(
        ..., description="Named query edges keyed by identifier (e.g. 'e0', 'e1')"
    )


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
                    "qualifier_constraints": [
                        {
                            "qualifier_set": [
                                {
                                    "qualifier_type_id": "biolink:object_aspect_qualifier",
                                    "qualifier_value": "activity",
                                }
                            ]
                        }
                    ],
                }
            },
        }
    }
}


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
    subclass: Optional[bool] = Field(
        None,
        description="Enable biolink subclass inference "
        "(overridden by query parameter if provided)",
    )
    subclass_depth: Optional[int] = Field(
        None, description="Maximum subclass_of hops to traverse (default 1)"
    )
    log_level: Optional[Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]] = (
        Field(
            None,
            description="Set logging level for this request "
            "(e.g. 'DEBUG' to see serialization timings)",
        )
    )
    dehydrated: Optional[bool] = Field(
        None,
        description="Return a dehydrated response (skip edge attribute enrichment). "
        "Automatically enabled when path count exceeds the large result threshold "
        "(overridden by query parameter if provided)",
    )

    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "examples": [
                _QUERY_EXAMPLE_ONEHOP,
                _QUERY_EXAMPLE_TWOHOP,
                _QUERY_EXAMPLE_QUALIFIERS,
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
    set_interpretation: Optional[str] = Field(
        None, description="Set interpretation mode (only 'BATCH' is supported)"
    )
    log_level: Optional[Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]] = (
        Field(
            None,
            description="Set logging level for this request "
            "(e.g. 'DEBUG' to see detailed query processing)",
        )
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
    matching nodes and edges, and result bindings.
    """

    message: Dict[str, Any] = Field(
        ...,
        description="TRAPI message with query_graph, knowledge_graph, and results",
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
