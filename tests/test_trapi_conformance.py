"""Conformance: every response gandalf serves is valid TRAPI 1.6.

(On the TRAPI 1.6 branch ``translator_tom`` is the 1.x line, which models
TRAPI 1.6, so the checks below are against 1.6.)

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

Dehydrated responses are deliberately excluded.  That mode exists to make the
payload as small as possible and omits ``Edge.sources``, which TRAPI
requires, so such a response is knowingly not schema-valid: validating it
would only assert a thing gandalf has chosen not to do.  What *is* pinned
below is the shape of that choice, so the omission stays deliberate rather
than drifting -- see :class:`gandalf.trapi.InFlightEdge`.
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


def one_hop(
    predicate="biolink:affects", object_node=None, constraints=None, **qedge_fields
) -> dict:
    """A one-hop query, optionally pinned at the object end or constrained."""
    qedge: dict = {"subject": "n0", "object": "n1", "predicates": [predicate]}
    qedge.update(qedge_fields)
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
    """Assert a response is valid TRAPI 1.6 and agrees with the model exactly."""
    model = Response.from_dict(response)
    assert orjson.loads(model.to_json()) == orjson.loads(orjson.dumps(response)), (
        "response differs from TOM's serialization of it, which means it "
        "carries a property TRAPI would leave absent"
    )


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------


class TestResponsesAreValidTRAPI:
    @pytest.mark.parametrize(
        "query_kwargs",
        [
            {},
            {"predicate": "biolink:treats", "object_node": DIABETES},
            {
                "qualifier_constraints": [
                    {
                        "qualifier_set": [
                            {
                                "qualifier_type_id": "biolink:object_aspect_qualifier",
                                "qualifier_value": "activity",
                            }
                        ]
                    }
                ]
            },
            {
                "attribute_constraints": [
                    {
                        "id": "biolink:publications",
                        "name": "publications",
                        "operator": "matches",
                        "value": ".",
                    }
                ]
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
        # allow-deny is the internal (2.0) constraints object, which lookup()
        # still reads; 1.6 clients cannot send it.
        ids=["unpinned", "pinned-both-ends", "qualified", "attribute", "allow-deny"],
    )
    def test_lookup_response(self, graph, bmt, query_kwargs):  # noqa: F811
        response = lookup(graph, one_hop(**query_kwargs), bmt=bmt, dehydrated=False)
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
        """No matches: results stays an empty list."""
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


class TestDehydratedResponsesTradeConformanceForSize:
    """Dehydrated mode is smaller than TRAPI allows, on purpose.

    It is the one response shape gandalf does not hold to the schema, so the
    exact shape of the exception is asserted here: anything more than this
    omitted, or anything omitted that should not be, is a change of contract
    rather than a bug fix.
    """

    def test_edges_omit_sources_and_attributes(self, graph, bmt):  # noqa: F811
        """Everything else stays, qualifiers included: they are part of what
        the edge states, not a detail of it.  Metformin's `affects` edges are
        qualified but for one, which has no qualifiers property at all.  Under
        1.6 the only attributes left are knowledge_level and agent_type, which
        1.6 carries as attributes."""
        edges = lookup(graph, one_hop(), bmt=bmt, dehydrated=True)["message"][
            "knowledge_graph"
        ]["edges"]
        full = lookup(graph, one_hop(), bmt=bmt, dehydrated=False)["message"][
            "knowledge_graph"
        ]["edges"]
        assert edges.keys() == full.keys()
        qualified = {edge_id for edge_id, edge in full.items() if "qualifiers" in edge}
        assert qualified and qualified != edges.keys()
        for edge_id, edge in edges.items():
            assert set(edge) == {
                "subject",
                "object",
                "predicate",
                "attributes",
                *(["qualifiers"] if edge_id in qualified else []),
            }, edge_id
            assert [a["attribute_type_id"] for a in edge["attributes"]] == [
                "biolink:knowledge_level",
                "biolink:agent_type",
            ], edge_id
            assert edge.get("qualifiers") == full[edge_id].get("qualifiers"), edge_id

    def test_that_is_the_only_reason_they_are_invalid(self, graph, bmt):  # noqa: F811
        """Nothing else about a dehydrated response is non-conformant.

        Put the missing sources back and it validates, which keeps the
        exception to the one property rather than a general licence.
        """
        response = lookup(graph, one_hop(), bmt=bmt, dehydrated=True)
        for edge in response["message"]["knowledge_graph"]["edges"].values():
            edge["sources"] = [
                {
                    "resource_id": "infores:ctd",
                    "resource_role": "primary_knowledge_source",
                }
            ]
        assert_valid_trapi(response)

    def test_dehydrated_is_materially_smaller(self, graph, bmt):  # noqa: F811
        """Even here, where every edge is qualified: the qualifiers are the
        most a dehydrated edge keeps."""
        dehydrated = lookup(graph, one_hop(), bmt=bmt, dehydrated=True)
        full = lookup(graph, one_hop(), bmt=bmt, dehydrated=False)
        # 0.6 on the 2.0 line.  Under 1.6 knowledge_level and agent_type are
        # attribute objects rather than two plain properties, ~130 bytes more
        # per dehydrated edge, which puts this fixture at ~0.6.
        assert len(orjson.dumps(dehydrated)) < len(orjson.dumps(full)) * 0.65

    def test_no_nulls_even_when_dehydrated(self, graph, bmt):  # noqa: F811
        """Smaller is not licence to emit a null: those are never valid."""
        response = lookup(graph, one_hop(), bmt=bmt, dehydrated=True)

        def nulls(value, path=""):
            if value is None:
                yield path or "<root>"
            elif isinstance(value, dict):
                for key, sub in value.items():
                    yield from nulls(sub, f"{path}/{key}")
            elif isinstance(value, list):
                for i, sub in enumerate(value):
                    yield from nulls(sub, f"{path}/{i}")

        assert list(nulls(response)) == []
