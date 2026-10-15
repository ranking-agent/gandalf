"""Pydantic request models and gandalf's own (non-TRAPI) response models.

Everything TRAPI-shaped comes from ``translator_tom``, the Translator-wide
TRAPI object model: it is the same model the rest of Translator validates
against, its docstrings are the spec's own descriptions, and it tracks the
spec so gandalf does not have to re-derive it.  What lives here is only the
two things TOM cannot know about:

* ``QueryParameters`` -- gandalf's own query-time settings (``subclass``,
  ``dehydrated``, ``timeout``, ...).  TRAPI 1.6 has no ``parameters`` object;
  it is a gandalf extension, carried under the additional properties a 1.6
  ``Query`` admits.  ``log_level`` and ``bypass_cache`` are the Query's own
  top-level properties in 1.6.
* The Plater-compatible endpoints (``/node_degree``, ``/edges``, ``/metadata``,
  ...), which are gandalf extensions and not TRAPI at all.

The OpenAPI examples for ``/query`` and ``/asyncquery`` also live here, since
they are gandalf's documentation rather than the spec's.

The TRAPI models themselves are not re-exported: import them from
``translator_tom`` (``from translator_tom import Edge, QNode, Response``) so
there is one name for each concept across Translator rather than a gandalf
alias for it.

Response construction deliberately does *not* go through TOM's models: gandalf
assembles plain dicts and serializes them with orjson, which measures ~4x
faster and ~2x lighter than materializing a TOM model per result.  The
TypedDicts in ``translator_tom.model_dicts`` give those dicts static types at
no runtime cost instead -- see ``gandalf.trapi``.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field
from translator_tom import AsyncQuery, LogLevel, Query, Response

__all__ = [
    "QueryParameters",
    "TRAPIQuery",
    "AsyncTRAPIQuery",
    "TRAPIResponse",
    "NodeResponse",
    "NodeDegreeResponse",
    "EdgeItem",
    "EdgesResponse",
    "EdgesCountResponse",
    "EdgeSummaryItem",
    "EdgeSummaryResponse",
    "MetadataResponse",
]


# ---------------------------------------------------------------------------
# OpenAPI examples for the TRAPI endpoints
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
                    "constraints": [{"intermediate_categories": ["biolink:Gene"]}],
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
    "log_level": "DEBUG",
    "parameters": {
        "timeout": 60,
        "subclass": True,
        "subclass_depth": 1,
        "dehydrated": False,
        "filter_config": {"max_node_degree": 50},
        "annotator_config": {},
    },
}


class QueryParameters(BaseModel):
    """Gandalf's ``parameters`` for ``/query`` and ``/asyncquery``.

    TRAPI 1.6 has no ``parameters`` object; gandalf carries its own settings
    in one under the additional properties a 1.6 ``Query`` admits.  All are
    optional; absent fields fall back to server defaults.  ``log_level`` and
    ``bypass_cache`` are standard top-level Query properties in 1.6 and are
    read from there; they are accepted here too, as on the TRAPI 2.0 line.
    """

    model_config = ConfigDict(extra="allow")

    timeout: Optional[float] = Field(
        None,
        description="Seconds the client is willing to wait. Absent: the "
        "server default; negative: no timeout.",
    )
    log_level: Optional[LogLevel] = Field(
        None, description="Use the Query's top-level log_level under TRAPI 1.6."
    )
    bypass_cache: Optional[bool] = Field(
        None, description="Use the Query's top-level bypass_cache under TRAPI 1.6."
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


class TRAPIQuery(Query):
    """Request body for ``POST /query``.

    TOM's ``Query`` with gandalf's ``parameters`` and the examples that
    document this server's endpoint.
    """

    parameters: Optional[QueryParameters] = None

    model_config = ConfigDict(
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


class AsyncTRAPIQuery(AsyncQuery):
    """Request body for ``POST /asyncquery``.

    TOM's ``AsyncQuery`` (a ``Query`` plus the required ``callback``) with
    gandalf's ``parameters``.
    """

    parameters: Optional[QueryParameters] = None

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "callback": "https://example.com/callback",
                    **_QUERY_EXAMPLE_ONEHOP,
                    "workflow": [{"id": "lookup"}],
                    "parameters": {"subclass": True},
                }
            ]
        },
    )


class TRAPIResponse(Response):
    """Response from ``POST /query``: TOM's TRAPI 1.6 ``Response``.

    Declared as the route's ``response_model`` so the OpenAPI document shows
    the real TRAPI response shape.  It is only *constructed* when
    ``validate_responses`` is enabled, which is a dev/testing switch -- see
    ``gandalf.server._trapi_response``.
    """


# ---------------------------------------------------------------------------
# Gandalf's own (Plater-compatible) response models
# ---------------------------------------------------------------------------


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
