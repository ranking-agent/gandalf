"""Tests for GANDALF Pydantic request/response models."""

import pytest
from pydantic import ValidationError

from gandalf.models import (
    AsyncTRAPIQuery,
    EdgeItem,
    EdgesCountResponse,
    EdgesResponse,
    EdgeSummaryResponse,
    Message,
    MetadataResponse,
    NodeResponse,
    QEdge,
    QNode,
    QPath,
    QPathConstraint,
    QueryGraph,
    SetInterpretation,
    TRAPIQuery,
    TRAPIResponse,
    WorkflowStep,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ONEHOP_QUERY = {
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

TWOHOP_QUERY = {
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


# ---------------------------------------------------------------------------
# TRAPIQuery validation
# ---------------------------------------------------------------------------


class TestTRAPIQuery:
    """Tests for the TRAPIQuery request model."""

    def test_valid_onehop_query(self):
        q = TRAPIQuery(**ONEHOP_QUERY)
        assert q.message.query_graph.nodes["n0"].ids == ["CHEBI:6801"]
        assert q.message.query_graph.edges["e0"].subject == "n0"

    def test_valid_twohop_query(self):
        q = TRAPIQuery(**TWOHOP_QUERY)
        assert len(q.message.query_graph.nodes) == 3
        assert len(q.message.query_graph.edges) == 2

    def test_missing_message_raises(self):
        with pytest.raises(ValidationError, match="message"):
            TRAPIQuery()

    def test_missing_query_graph_raises(self):
        with pytest.raises(ValidationError, match="query_graph"):
            TRAPIQuery(message={})

    def test_missing_nodes_raises(self):
        with pytest.raises(ValidationError, match="nodes"):
            TRAPIQuery(message={"query_graph": {"edges": {}}})

    def test_missing_edges_and_paths_raises(self):
        with pytest.raises(ValidationError, match="edges"):
            TRAPIQuery(message={"query_graph": {"nodes": {}}})

    def test_edge_missing_subject_raises(self):
        with pytest.raises(ValidationError, match="subject"):
            TRAPIQuery(
                message={
                    "query_graph": {
                        "nodes": {"n0": {}},
                        "edges": {"e0": {"object": "n0"}},
                    }
                }
            )

    def test_edge_missing_object_raises(self):
        with pytest.raises(ValidationError, match="object"):
            TRAPIQuery(
                message={
                    "query_graph": {
                        "nodes": {"n0": {}},
                        "edges": {"e0": {"subject": "n0"}},
                    }
                }
            )

    def test_optional_subclass_fields(self):
        q = TRAPIQuery(
            **ONEHOP_QUERY, parameters={"subclass": True, "subclass_depth": 2}
        )
        assert q.parameters.subclass is True
        assert q.parameters.subclass_depth == 2

    def test_optional_subclass_defaults_none(self):
        q = TRAPIQuery(**ONEHOP_QUERY)
        assert q.parameters is None

    def test_log_level_valid_values(self):
        for level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            q = TRAPIQuery(**ONEHOP_QUERY, log_level=level)
            assert q.log_level == level

    def test_log_level_invalid_value_raises(self):
        with pytest.raises(ValidationError, match="log_level"):
            TRAPIQuery(**ONEHOP_QUERY, log_level="TRACE")

    def test_log_level_defaults_none(self):
        q = TRAPIQuery(**ONEHOP_QUERY)
        assert q.log_level is None

    def test_extra_fields_allowed(self):
        data = {**ONEHOP_QUERY, "some_custom_field": "value"}
        q = TRAPIQuery(**data)
        assert q.some_custom_field == "value"

    def test_qualifier_constraints_accepted(self):
        data = {
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
        q = TRAPIQuery(**data)
        qc = q.message.query_graph.edges["e0"].qualifier_constraints
        assert len(qc) == 1


# ---------------------------------------------------------------------------
# Pathfinder query graphs (nodes + paths)
# ---------------------------------------------------------------------------


PATHFINDER_QUERY = {
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


class TestPathfinderQuery:
    """Tests for Pathfinder-style query graphs (nodes + paths)."""

    def test_valid_pathfinder_query(self):
        q = TRAPIQuery(**PATHFINDER_QUERY)
        assert q.message.query_graph.nodes["n0"].ids == ["CHEBI:6801"]
        assert q.message.query_graph.edges is None
        path = q.message.query_graph.paths["p0"]
        assert path.subject == "n0"
        assert path.object == "n1"
        assert path.predicates == ["biolink:related_to"]
        assert path.constraints[0].intermediate_categories == ["biolink:Gene"]

    def test_path_missing_subject_raises(self):
        with pytest.raises(ValidationError, match="subject"):
            TRAPIQuery(
                message={
                    "query_graph": {
                        "nodes": {"n0": {}, "n1": {}},
                        "paths": {"p0": {"object": "n1"}},
                    }
                }
            )

    def test_path_missing_object_raises(self):
        with pytest.raises(ValidationError, match="object"):
            TRAPIQuery(
                message={
                    "query_graph": {
                        "nodes": {"n0": {}, "n1": {}},
                        "paths": {"p0": {"subject": "n0"}},
                    }
                }
            )

    def test_hybrid_edges_and_paths_allowed(self):
        data = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["CHEBI:6801"]},
                        "n1": {"ids": ["MONDO:0005148"]},
                    },
                    "edges": {
                        "e0": {"subject": "n0", "object": "n1"},
                    },
                    "paths": {
                        "p0": {"subject": "n0", "object": "n1"},
                    },
                }
            }
        }
        q = TRAPIQuery(**data)
        assert q.message.query_graph.edges["e0"].subject == "n0"
        assert q.message.query_graph.paths["p0"].subject == "n0"

    def test_pathfinder_roundtrip_excludes_none(self):
        q = TRAPIQuery(**PATHFINDER_QUERY)
        raw = q.model_dump(exclude_none=True)
        qg = raw["message"]["query_graph"]
        assert "edges" not in qg
        assert qg["paths"]["p0"]["subject"] == "n0"
        assert qg["paths"]["p0"]["constraints"][0]["intermediate_categories"] == [
            "biolink:Gene"
        ]

    def test_qpath_direct_construction(self):
        p = QPath(
            subject="n0",
            object="n1",
            predicates=["biolink:treats"],
            constraints=[QPathConstraint(intermediate_categories=["biolink:Gene"])],
        )
        assert p.subject == "n0"
        assert p.constraints[0].intermediate_categories == ["biolink:Gene"]

    def test_query_graph_requires_edges_or_paths(self):
        with pytest.raises(ValidationError, match="edges"):
            QueryGraph(nodes={"n0": QNode()})


# ---------------------------------------------------------------------------
# TRAPIQuery round-trip (model_dump compatibility with lookup())
# ---------------------------------------------------------------------------


class TestTRAPIQueryRoundTrip:
    """Verify model_dump(exclude_none=True) produces lookup()-compatible dicts."""

    def test_onehop_roundtrip(self):
        q = TRAPIQuery(**ONEHOP_QUERY)
        raw = q.model_dump(exclude_none=True)
        assert raw["message"]["query_graph"]["nodes"]["n0"]["ids"] == ["CHEBI:6801"]
        assert raw["message"]["query_graph"]["edges"]["e0"]["subject"] == "n0"
        # No None values should be present
        assert "subclass" not in raw
        assert "subclass_depth" not in raw
        assert "parameters" not in raw

    def test_twohop_roundtrip(self):
        q = TRAPIQuery(**TWOHOP_QUERY)
        raw = q.model_dump(exclude_none=True)
        assert len(raw["message"]["query_graph"]["nodes"]) == 3
        assert len(raw["message"]["query_graph"]["edges"]) == 2

    def test_node_without_ids_roundtrip(self):
        """Unpinned nodes (categories only) should not have 'ids' key."""
        q = TRAPIQuery(**ONEHOP_QUERY)
        raw = q.model_dump(exclude_none=True)
        n1 = raw["message"]["query_graph"]["nodes"]["n1"]
        assert "ids" not in n1
        assert n1["categories"] == ["biolink:Gene"]


# ---------------------------------------------------------------------------
# AsyncTRAPIQuery validation
# ---------------------------------------------------------------------------


class TestAsyncTRAPIQuery:
    """Tests for the AsyncTRAPIQuery request model."""

    def test_valid_async_query(self):
        data = {
            "callback": "https://example.com/callback",
            **ONEHOP_QUERY,
        }
        q = AsyncTRAPIQuery(**data)
        assert q.callback == "https://example.com/callback"
        assert q.message.query_graph.nodes["n0"].ids == ["CHEBI:6801"]

    def test_missing_callback_raises(self):
        with pytest.raises(ValidationError, match="callback"):
            AsyncTRAPIQuery(**ONEHOP_QUERY)

    def test_missing_message_raises(self):
        with pytest.raises(ValidationError, match="message"):
            AsyncTRAPIQuery(callback="https://example.com/callback")

    def test_workflow_validation(self):
        data = {
            "callback": "https://example.com/callback",
            **ONEHOP_QUERY,
            "workflow": [{"id": "lookup"}],
        }
        q = AsyncTRAPIQuery(**data)
        assert len(q.workflow) == 1
        assert q.workflow[0].id == "lookup"

    def test_set_interpretation_field(self):
        data = {
            "callback": "https://example.com/callback",
            **ONEHOP_QUERY,
            "set_interpretation": "BATCH",
        }
        q = AsyncTRAPIQuery(**data)
        assert q.set_interpretation == "BATCH"


# ---------------------------------------------------------------------------
# QNode set_interpretation validation
# ---------------------------------------------------------------------------


class TestQNodeSetInterpretation:
    """Tests for QNode set_interpretation and member_ids fields."""

    def test_valid_batch(self):
        node = QNode(ids=["A"], set_interpretation="BATCH")
        assert node.set_interpretation == SetInterpretation.BATCH

    def test_valid_all(self):
        node = QNode(ids=["A", "B"], set_interpretation="ALL")
        assert node.set_interpretation == SetInterpretation.ALL

    def test_valid_many(self):
        node = QNode(set_interpretation="MANY", member_ids=["A", "B"])
        assert node.set_interpretation == SetInterpretation.MANY

    def test_valid_collate(self):
        node = QNode(categories=["biolink:Gene"], set_interpretation="COLLATE")
        assert node.set_interpretation == SetInterpretation.COLLATE

    def test_invalid_value_raises(self):
        with pytest.raises(ValidationError):
            QNode(set_interpretation="INVALID")

    def test_default_none(self):
        node = QNode(ids=["A"])
        assert node.set_interpretation is None

    def test_member_ids_field(self):
        node = QNode(member_ids=["A", "B"], set_interpretation="ALL")
        assert node.member_ids == ["A", "B"]

    def test_roundtrip_excludes_none(self):
        q = TRAPIQuery(**ONEHOP_QUERY)
        raw = q.model_dump(exclude_none=True)
        n0 = raw["message"]["query_graph"]["nodes"]["n0"]
        assert "set_interpretation" not in n0
        assert "member_ids" not in n0

    def test_roundtrip_includes_when_set(self):
        data = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {
                            "ids": ["CHEBI:6801", "CHEBI:1234"],
                            "set_interpretation": "ALL",
                        },
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
        q = TRAPIQuery(**data)
        raw = q.model_dump(exclude_none=True)
        assert (
            raw["message"]["query_graph"]["nodes"]["n0"]["set_interpretation"] == "ALL"
        )


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class TestResponseModels:
    """Tests for response documentation models."""

    def test_node_response(self):
        r = NodeResponse(
            id="CHEBI:6801", name="Metformin", categories=["biolink:SmallMolecule"]
        )
        assert r.id == "CHEBI:6801"

    def test_node_response_extra_fields(self):
        r = NodeResponse(
            id="CHEBI:6801",
            name="Metformin",
            categories=["biolink:SmallMolecule"],
            description="A biguanide drug",
        )
        assert r.description == "A biguanide drug"

    def test_edges_response(self):
        r = EdgesResponse(
            query_curie="CHEBI:6801",
            edges=[
                EdgeItem(
                    subject="CHEBI:6801",
                    object="NCBIGene:5468",
                    predicate="biolink:affects",
                    edge_id="e1",
                )
            ],
        )
        assert len(r.edges) == 1

    def test_edges_count_response(self):
        r = EdgesCountResponse(query_curie="CHEBI:6801", count=42)
        assert r.count == 42

    def test_edge_summary_response(self):
        r = EdgeSummaryResponse(
            query_curie="CHEBI:6801",
            edge_summary=[["biolink:affects", "biolink:Gene", 15]],
        )
        assert len(r.edge_summary) == 1

    def test_metadata_response(self):
        r = MetadataResponse(
            node_count=1000,
            edge_count=5000,
            predicate_count=50,
            category_count=10,
            predicates={"biolink:affects": 100},
            categories={"biolink:Gene": 500},
        )
        assert r.node_count == 1000

    def test_trapi_response(self):
        r = TRAPIResponse(message={"query_graph": {}, "results": []})
        assert "query_graph" in r.message


# ---------------------------------------------------------------------------
# OpenAPI schema extras (examples present)
# ---------------------------------------------------------------------------


class TestOpenAPIExamples:
    """Verify that models include examples in their JSON schema."""

    def test_trapi_query_has_examples(self):
        schema = TRAPIQuery.model_json_schema()
        assert "examples" in schema

    def test_trapi_query_examples_are_valid(self):
        """Each example in the schema should pass validation."""
        schema = TRAPIQuery.model_json_schema()
        for example in schema["examples"]:
            q = TRAPIQuery(**example)
            assert q.message.query_graph.nodes is not None

    def test_async_query_has_examples(self):
        schema = AsyncTRAPIQuery.model_json_schema()
        assert "examples" in schema

    def test_async_query_examples_are_valid(self):
        schema = AsyncTRAPIQuery.model_json_schema()
        for example in schema["examples"]:
            q = AsyncTRAPIQuery(**example)
            assert q.callback is not None
