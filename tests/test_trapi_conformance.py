"""Conformance: every response gandalf serves is valid TRAPI 2.0.

``translator_tom`` is the Translator-wide TRAPI object model, so validating
against it is the closest in-repo equivalent of the external validators: a
response that ``Response.from_dict`` accepts is one the model admits, and one
that round-trips byte-identically through ``to_json()`` agrees with the model
on which properties should be present at all.

That second check is the one that matters here.  TOM serializes with
``exclude_none`` and ``exclude_defaults``, so a response carrying a null or an
empty value where the model's default is "absent" comes back *different* --
which is exactly the class of bug (``qualifiers: []``,
``upstream_resource_ids: []``, ``name: null``) that a shape-only check misses.
"""

import orjson
import pytest
from translator_tom import MetaKnowledgeGraph, Response

from gandalf.enrichment import enrich_knowledge_graph
from gandalf.search import lookup
from gandalf.trapi import Deadline
from tests.search_fixtures import graph  # noqa: F401

METFORMIN = "CHEBI:6801"
DIABETES = "MONDO:0005148"
DIABETES_MELLITUS = "MONDO:0005015"


def one_hop(predicate="biolink:affects", object_node=None, constraints=None) -> dict:
    """A one-hop query, optionally pinned at the object end or constrained."""
    qedge: dict = {"subject": "n0", "object": "n1", "predicates": [predicate]}
    if constraints is not None:
        qedge["constraints"] = constraints
    return {
        "message": {
            "query_graph": {
                "nodes": {
                    "n0": {"ids": [METFORMIN]},
                    "n1": (
                        {"ids": [object_node]}
                        if object_node
                        else {"categories": ["biolink:Gene"]}
                    ),
                },
                "edges": {"e0": qedge},
            }
        }
    }


def assert_valid_trapi(response: dict) -> None:
    """Assert a response is valid TRAPI 2.0 and agrees with the model exactly."""
    model = Response.from_dict(response)
    assert orjson.loads(model.to_json()) == orjson.loads(orjson.dumps(response)), (
        "response differs from TOM's serialization of it, which means it "
        "carries a property TRAPI 2.0 would leave absent"
    )


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------


class TestResponsesAreValidTRAPI:
    @pytest.mark.parametrize("dehydrated", [False, True])
    @pytest.mark.parametrize(
        "query_kwargs",
        [
            {},
            {"predicate": "biolink:treats", "object_node": DIABETES},
            {
                "constraints": {
                    "qualifiers": [{"biolink:object_aspect_qualifier": "activity"}]
                }
            },
            {
                "constraints": {
                    "knowledge_level": {
                        "behavior": "ALLOW",
                        "values": ["knowledge_assertion"],
                    },
                    "sources": {"behavior": "ALLOW", "values": ["infores:ctd"]},
                }
            },
        ],
        ids=["unpinned", "pinned-both-ends", "qualified", "allow-deny"],
    )
    def test_lookup_response(self, graph, bmt, query_kwargs, dehydrated):  # noqa: F811
        response = lookup(
            graph, one_hop(**query_kwargs), bmt=bmt, dehydrated=dehydrated
        )
        assert response["message"]["results"]
        assert_valid_trapi(response)

    def test_subclass_inference_response(self, graph, bmt):  # noqa: F811
        """Carries auxiliary graphs and a composite inferred edge."""
        response = lookup(
            graph,
            one_hop(predicate="biolink:treats", object_node=DIABETES_MELLITUS),
            bmt=bmt,
            subclass=True,
            subclass_depth=1,
        )
        assert response["message"]["auxiliary_graphs"]
        assert_valid_trapi(response)

    def test_empty_response(self, graph, bmt):  # noqa: F811
        """No matches: results stays an empty list, which 2.0 asks for."""
        response = lookup(
            graph,
            {
                "message": {
                    "query_graph": {
                        "nodes": {
                            "n0": {"ids": ["NOSUCH:0000"]},
                            "n1": {"categories": ["biolink:Gene"]},
                        },
                        "edges": {"e0": {"subject": "n0", "object": "n1"}},
                    }
                }
            },
            bmt=bmt,
        )
        assert response["message"]["results"] == []
        assert_valid_trapi(response)

    def test_timeout_response(self, graph, bmt):  # noqa: F811
        response = lookup(
            graph, one_hop(), bmt=bmt, deadline=Deadline(-1.0), log_level="DEBUG"
        )
        assert response["status"] == "Timeout"
        assert_valid_trapi(response)

    def test_response_with_logs(self, graph, bmt):  # noqa: F811
        response = lookup(graph, one_hop(), bmt=bmt, log_level="DEBUG")
        assert response["logs"]
        assert_valid_trapi(response)

    def test_rehydrated_response(self, graph, bmt):  # noqa: F811
        """A dehydrated response handed back for enrichment."""
        dehydrated = lookup(graph, one_hop(), bmt=bmt, dehydrated=True)
        enrich_knowledge_graph(dehydrated, graph)
        assert_valid_trapi(dehydrated)

    def test_rehydrated_response_with_client_nulls(self, graph, bmt):  # noqa: F811
        """Nulls the client put in must not survive into the served response."""
        message = {
            "message": {
                "knowledge_graph": {
                    "nodes": {
                        METFORMIN: {"name": None, "categories": []},
                        "SYNTHETIC:1": {"name": None},
                    },
                    "edges": {
                        "x": {
                            "subject": METFORMIN,
                            "object": DIABETES,
                            "predicate": "biolink:treats",
                            "qualifiers": [],
                            "attributes": None,
                        }
                    },
                },
                "results": [],
            }
        }
        enrich_knowledge_graph(message, graph)
        assert_valid_trapi(message)


class TestMetaKnowledgeGraphIsValidTRAPI:
    def test_meta_knowledge_graph(self, graph, bmt):  # noqa: F811
        graph.build_metadata()
        model = MetaKnowledgeGraph.from_dict(graph.meta_kg)
        assert orjson.loads(model.to_json()) == orjson.loads(
            orjson.dumps(graph.meta_kg)
        )


class TestRequestsValidateAgainstTOM:
    """The documented OpenAPI examples must be valid TRAPI requests."""

    def test_query_examples(self):
        from gandalf.models import TRAPIQuery

        examples = (TRAPIQuery.model_config.get("json_schema_extra") or {})["examples"]
        assert examples
        for example in examples:
            TRAPIQuery.from_dict(example)

    def test_asyncquery_examples(self):
        from gandalf.models import AsyncTRAPIQuery

        examples = (AsyncTRAPIQuery.model_config.get("json_schema_extra") or {})[
            "examples"
        ]
        assert examples
        for example in examples:
            AsyncTRAPIQuery.from_dict(example)
