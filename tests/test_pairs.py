"""Tests for ``parameters.pairs`` and ``parameters.kg_only``.

``pairs`` restricts a one-hop lookup between pinned ends to listed
``[subject_id, object_id]`` pairs, so a caller filling in the edges of many
specific pairs gets those edges rather than the edges of every combination of
pinned subject and object.  ``kg_only`` leaves the results out.
"""

import pytest
from fastapi.testclient import TestClient

from gandalf.search import lookup
from tests.search_fixtures import graph  # noqa: F401


def _query(subjects, objects, **parameters):
    return {
        "message": {
            "query_graph": {
                "nodes": {"n0": {"ids": subjects}, "n1": {"ids": objects}},
                "edges": {
                    "e0": {
                        "subject": "n0",
                        "object": "n1",
                        "predicates": ["biolink:related_to"],
                    }
                },
            }
        },
        "parameters": {"subclass": False, **parameters},
    }


def _node_pairs(response):
    """Each knowledge-graph edge's endpoints, unordered."""
    return {
        frozenset((edge["subject"], edge["object"]))
        for edge in response["message"]["knowledge_graph"]["edges"].values()
    }


SUBJECTS = ["CHEBI:6801", "NCBIGene:5468"]
OBJECTS = ["MONDO:0005148", "NCBIGene:3643"]


class TestPairsLookup:
    def test_without_pairs_every_combination_is_returned(self, graph, bmt):
        response = lookup(graph, _query(SUBJECTS, OBJECTS), bmt=bmt, subclass=False)
        assert _node_pairs(response) == {
            frozenset(("CHEBI:6801", "MONDO:0005148")),
            frozenset(("CHEBI:6801", "NCBIGene:3643")),
            frozenset(("NCBIGene:5468", "MONDO:0005148")),
            frozenset(("NCBIGene:5468", "NCBIGene:3643")),
        }

    def test_pairs_restrict_to_listed_pairs(self, graph, bmt):
        pairs = [["CHEBI:6801", "MONDO:0005148"], ["NCBIGene:5468", "NCBIGene:3643"]]
        response = lookup(
            graph, _query(SUBJECTS, OBJECTS), bmt=bmt, subclass=False, pairs=pairs
        )
        assert _node_pairs(response) == {
            frozenset(("CHEBI:6801", "MONDO:0005148")),
            frozenset(("NCBIGene:5468", "NCBIGene:3643")),
        }
        for result in response["message"]["results"]:
            bound = (
                result["node_bindings"]["n0"]["ids"][0],
                result["node_bindings"]["n1"]["ids"][0],
            )
            assert list(bound) in pairs

    def test_pair_matches_an_edge_stored_the_other_way(self, graph, bmt):
        """NCBIGene:5468 --interacts_with--> NCBIGene:3643 answers the pair
        (3643, 5468) through its inverse."""
        response = lookup(
            graph,
            _query(["NCBIGene:3643"], ["NCBIGene:5468"]),
            bmt=bmt,
            subclass=False,
            pairs=[["NCBIGene:3643", "NCBIGene:5468"]],
        )
        assert _node_pairs(response) == {frozenset(("NCBIGene:5468", "NCBIGene:3643"))}

    def test_pair_outside_the_pinned_ids_matches_nothing(self, graph, bmt):
        response = lookup(
            graph,
            _query(["CHEBI:6801"], ["MONDO:0005148"]),
            bmt=bmt,
            subclass=False,
            pairs=[["NCBIGene:5468", "NCBIGene:3643"]],
        )
        assert response["message"]["results"] == []
        assert response["message"]["knowledge_graph"]["edges"] == {}

    def test_unknown_ids_and_empty_pairs_match_nothing(self, graph, bmt):
        for pairs in ([], [["CHEBI:6801", "MONDO:9999999"]]):
            response = lookup(
                graph,
                _query(["CHEBI:6801"], ["MONDO:0005148", "MONDO:9999999"]),
                bmt=bmt,
                subclass=False,
                pairs=pairs,
            )
            assert response["message"]["knowledge_graph"]["edges"] == {}


class TestKgOnly:
    def test_kg_only_keeps_the_edges_and_drops_the_results(self, graph, bmt):
        full = lookup(graph, _query(SUBJECTS, OBJECTS), bmt=bmt, subclass=False)
        kg_only = lookup(
            graph, _query(SUBJECTS, OBJECTS), bmt=bmt, subclass=False, kg_only=True
        )
        assert full["message"]["results"]
        assert kg_only["message"]["results"] == []
        assert (
            kg_only["message"]["knowledge_graph"] == full["message"]["knowledge_graph"]
        )

    def test_kg_only_keeps_subclass_support_graphs(self, graph, bmt):
        query = _query(["CHEBI:6801"], ["MONDO:0005015"], subclass=True)
        full = lookup(graph, query, bmt=bmt)
        kg_only = lookup(graph, query, bmt=bmt, kg_only=True)
        assert kg_only["message"]["results"] == []
        assert kg_only["message"]["auxiliary_graphs"]
        assert (
            kg_only["message"]["auxiliary_graphs"]
            == full["message"]["auxiliary_graphs"]
        )


@pytest.fixture
def client(graph, bmt, monkeypatch):  # noqa: F811
    monkeypatch.setenv("GANDALF_SKIP_PRELOAD", "true")
    monkeypatch.setenv("GANDALF_OTEL_ENABLED", "false")
    from gandalf import server as gandalf_server

    monkeypatch.setattr(gandalf_server, "GRAPH", graph)
    monkeypatch.setattr(gandalf_server, "BMT", bmt)
    return TestClient(gandalf_server.APP)


class TestPairsOverHttp:
    def test_pairs_and_kg_only_reach_lookup(self, client):
        response = client.post(
            "/query",
            json=_query(
                SUBJECTS,
                OBJECTS,
                pairs=[["CHEBI:6801", "MONDO:0005148"]],
                kg_only=True,
            ),
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["message"]["results"] == []
        assert _node_pairs(body) == {frozenset(("CHEBI:6801", "MONDO:0005148"))}

    @pytest.mark.parametrize(
        "query",
        [
            _query(SUBJECTS, OBJECTS, pairs=[["CHEBI:6801"]]),
            _query(SUBJECTS, OBJECTS, pairs="CHEBI:6801"),
            _query(SUBJECTS, OBJECTS, pairs=[], subclass=True),
            {
                "message": {
                    "query_graph": {
                        "nodes": {"n0": {"ids": SUBJECTS}, "n1": {}},
                        "edges": {"e0": {"subject": "n0", "object": "n1"}},
                    }
                },
                "parameters": {"subclass": False, "pairs": []},
            },
            {
                "message": {
                    "query_graph": {
                        "nodes": {
                            "n0": {"ids": SUBJECTS},
                            "n1": {"ids": OBJECTS},
                            "n2": {},
                        },
                        "edges": {
                            "e0": {"subject": "n0", "object": "n1"},
                            "e1": {"subject": "n1", "object": "n2"},
                        },
                    }
                },
                "parameters": {"subclass": False, "pairs": []},
            },
        ],
        ids=["short-pair", "not-a-list", "subclass", "unpinned", "two-edges"],
    )
    def test_pairs_it_cannot_apply_are_a_400(self, client, query):
        response = client.post("/query", json=query)
        assert response.status_code == 400, response.text
        assert "pairs" in response.text
