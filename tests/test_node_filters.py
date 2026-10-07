"""Tests for the built-in node filter plugins (max_node_degree, min_information_content).

These exercise the plugins through the public ``lookup(..., filter_config=...)``
API, proving the plugin registry behaves identically to the previous hardcoded
filter chain.
"""

import json

import pytest

from tests.search_fixtures import graph  # noqa: F401

from gandalf.loader import build_graph_from_jsonl
from gandalf.search import lookup


class TestMaxNodeDegree:
    """Tests for the max_node_degree parameter."""

    def test_max_node_degree_filters_high_degree_nodes(self, graph, bmt):
        """Nodes with total degree > max_node_degree should be excluded.

        Metformin affects 4 genes: PPARG(deg=3), INSR(deg=3), GCK(deg=3), TNF(deg=1).
        Setting max_node_degree=2 should filter out PPARG, INSR, GCK and keep only TNF.
        """
        query = {
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
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, bmt=bmt, filter_config={"max_node_degree": 2})
        results = response["message"]["results"]

        # Only TNF (degree=1) should pass the filter
        assert len(results) == 1
        gene_ids = {r["node_bindings"]["n1"]["ids"][0] for r in results}
        assert gene_ids == {"NCBIGene:7124"}

    def test_max_node_degree_allows_nodes_at_threshold(self, graph, bmt):
        """Nodes with degree exactly equal to max_node_degree should be kept."""
        query = {
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
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, bmt=bmt, filter_config={"max_node_degree": 3})
        results = response["message"]["results"]

        # PPARG, GCK (degree=3) and TNF (degree=1) should all pass
        assert len(results) == 3

    def test_max_node_degree_absent_means_no_filtering(self, graph, bmt):
        """When max_node_degree key is absent, no filtering should occur."""
        query = {
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
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, bmt=bmt, filter_config={})
        results = response["message"]["results"]

        assert len(results) == 4

    def test_max_node_degree_zero_filters_all(self, graph, bmt):
        """Setting max_node_degree=0 should filter all nodes (all have degree > 0)."""
        query = {
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
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, bmt=bmt, filter_config={"max_node_degree": 0})
        results = response["message"]["results"]

        assert len(results) == 0


class TestMinInformationContent:
    """Tests for the min_information_content parameter."""

    def test_min_ic_filters_low_ic_nodes(self, graph, bmt):
        """Nodes with IC below min_information_content should be excluded.

        Metformin affects 4 genes: PPARG(IC=92.3), INSR(IC=88.7), GCK(IC=81.2), TNF(IC=94.5).
        Setting min_information_content=90 should keep only PPARG and TNF.
        """
        query = {
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
                        },
                    },
                },
            },
        }

        response = lookup(
            graph, query, bmt=bmt, filter_config={"min_information_content": 90}
        )
        results = response["message"]["results"]

        assert len(results) == 2
        gene_ids = {r["node_bindings"]["n1"]["ids"][0] for r in results}
        assert gene_ids == {"NCBIGene:5468", "NCBIGene:7124"}

    def test_min_ic_absent_means_no_filtering(self, graph, bmt):
        """When min_information_content key is absent, no filtering should occur."""
        query = {
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
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, bmt=bmt, filter_config={})
        results = response["message"]["results"]

        assert len(results) == 4

    def test_min_ic_very_high_filters_all(self, graph, bmt):
        """Setting min_information_content higher than all IC values filters everything."""
        query = {
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
                        },
                    },
                },
            },
        }

        response = lookup(
            graph, query, bmt=bmt, filter_config={"min_information_content": 100}
        )
        results = response["message"]["results"]

        assert len(results) == 0

    def test_min_ic_filters_in_two_hop_query(self, graph, bmt):
        """min_information_content should filter intermediate nodes in multi-hop queries.

        Two-hop: Metformin --affects--> Gene --gene_associated--> T2D
        Without filtering, 3 genes bridge the path: PPARG(92.3), INSR(88.7), GCK(81.2).
        T2D has IC=78.2, below 85, but it is pinned, and pinned nodes are never
        filtered, so only GCK drops out.
        """
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["CHEBI:6801"]},
                        "n1": {"categories": ["biolink:Gene"]},
                        "n2": {"ids": ["MONDO:0005148"]},
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
                },
            },
        }

        # Without filtering: 3 results (PPARG, INSR, GCK as intermediates)
        response_unfiltered = lookup(graph, query, bmt=bmt)
        assert len(response_unfiltered["message"]["results"]) == 3

        # With min_information_content=85: GCK (81.2) filtered in first hop;
        # T2D (78.2) is pinned, so it stays → PPARG and INSR
        response_filtered = lookup(
            graph, query, bmt=bmt, filter_config={"min_information_content": 85}
        )
        gene_ids = {
            r["node_bindings"]["n1"]["ids"][0]
            for r in response_filtered["message"]["results"]
        }
        assert gene_ids == {"NCBIGene:5468", "NCBIGene:3643"}

    def test_min_ic_filters_backward_discovered_nodes(self, graph, bmt):
        """min_information_content should filter discovered nodes in backward search.

        Query: Gene --gene_associated_with_condition--> T2D
        Discovered genes: PPARG(IC=92.3), INSR(IC=88.7), GCK(IC=81.2)
        With min_information_content=90, only PPARG passes.
        """
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"categories": ["biolink:Gene"]},
                        "n1": {"ids": ["MONDO:0005148"]},
                    },
                    "edges": {
                        "e0": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:gene_associated_with_condition"],
                        },
                    },
                },
            },
        }

        # Without filtering: 3 genes
        response_unfiltered = lookup(graph, query, bmt=bmt)
        assert len(response_unfiltered["message"]["results"]) == 3

        # With min_information_content=90: only PPARG (92.3) passes
        response_filtered = lookup(
            graph, query, bmt=bmt, filter_config={"min_information_content": 90}
        )
        results = response_filtered["message"]["results"]

        assert len(results) == 1
        assert results[0]["node_bindings"]["n0"]["ids"][0] == "NCBIGene:5468"


class TestCombinedFilters:
    """Tests for combining max_node_degree and min_information_content."""

    def test_both_filters_applied(self, graph, bmt):
        """Both filters should be applied together.

        Metformin affects 4 genes:
        - PPARG: degree=3, IC=92.3
        - INSR:  degree=3, IC=88.7
        - GCK:   degree=3, IC=81.2
        - TNF:   degree=1, IC=94.5

        max_node_degree=2 keeps: TNF
        min_information_content=90 keeps: PPARG, TNF
        Both together keeps: TNF (only node passing both)
        """
        query = {
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
                        },
                    },
                },
            },
        }

        response = lookup(
            graph,
            query,
            bmt=bmt,
            filter_config={"max_node_degree": 2, "min_information_content": 90},
        )
        results = response["message"]["results"]

        assert len(results) == 1
        gene_ids = {r["node_bindings"]["n1"]["ids"][0] for r in results}
        assert gene_ids == {"NCBIGene:7124"}


def _graph_of(tmp_path, pairs):
    """A graph of ``related_to`` edges between TEST nodes, one per pair."""
    names = sorted({name for pair in pairs for name in pair})
    nodes_file = tmp_path / "nodes.jsonl"
    edges_file = tmp_path / "edges.jsonl"
    nodes_file.write_text(
        "".join(
            json.dumps(
                {"id": f"TEST:{n}", "name": n, "category": ["biolink:NamedThing"]}
            )
            + "\n"
            for n in names
        )
    )
    edges_file.write_text(
        "".join(
            json.dumps(
                {
                    "id": f"{s}-{o}",
                    "subject": f"TEST:{s}",
                    "predicate": "biolink:related_to",
                    "object": f"TEST:{o}",
                    "primary_knowledge_source": "infores:test",
                }
            )
            + "\n"
            for s, o in pairs
        )
    )
    return build_graph_from_jsonl(str(edges_file), str(nodes_file))


def _hub_graph(tmp_path, brca1_is_hub: bool):
    """cancer (degree 6) -- BRCA1 -- drug_a (degree 1) / drug_b (degree 5).

    With ``max_node_degree`` 3, cancer and drug_b are over the limit, and so
    is BRCA1 when it is given three more neighbours.
    """
    pairs = [("brca1", "cancer"), ("drug_a", "brca1"), ("drug_b", "brca1")]
    pairs += [(f"gene_{i}", "cancer") for i in range(5)]
    pairs += [("drug_b", f"target_{i}") for i in range(4)]
    if brca1_is_hub:
        pairs += [(f"partner_{i}", "brca1") for i in range(3)]
    return _graph_of(tmp_path, pairs)


def _one_hop(subject: str, obj: str) -> dict:
    return {
        "subject": subject,
        "object": obj,
        "predicates": ["biolink:related_to"],
    }


@pytest.mark.parametrize("subclass", [False, True])
@pytest.mark.parametrize("cancer_first", [True, False])
@pytest.mark.parametrize("brca1_is_hub", [False, True])
def test_pinned_nodes_are_never_filtered(
    tmp_path, bmt, subclass: bool, cancer_first: bool, brca1_is_hub: bool
) -> None:
    """cancer -> BRCA1 -> drug, with cancer and BRCA1 pinned: the degree limit
    keeps the hub drug out, but not the pinned hubs, whichever way round the
    cancer edge is written and whether or not the pinned nodes are expanded
    to their subclasses."""
    graph = _hub_graph(tmp_path, brca1_is_hub)
    cancer_edge = (
        _one_hop("cancer", "brca1") if cancer_first else _one_hop("brca1", "cancer")
    )
    query = {
        "message": {
            "query_graph": {
                "nodes": {
                    "cancer": {"ids": ["TEST:cancer"]},
                    "brca1": {"ids": ["TEST:brca1"]},
                    "drug": {},
                },
                "edges": {"e0": cancer_edge, "e1": _one_hop("brca1", "drug")},
            },
        },
    }

    response = lookup(
        graph,
        query,
        bmt=bmt,
        subclass=subclass,
        filter_config={"max_node_degree": 3},
    )
    drugs = {
        result["node_bindings"]["drug"]["ids"][0]
        for result in response["message"]["results"]
    }

    assert "TEST:drug_a" in drugs
    # Still filtered where they are candidates for the unpinned drug node.
    assert "TEST:drug_b" not in drugs
    assert "TEST:cancer" not in drugs


@pytest.mark.parametrize("subclass", [False, True])
@pytest.mark.parametrize("cancer_first", [True, False])
def test_an_edge_between_two_pinned_nodes_survives_the_filters(
    tmp_path, bmt, subclass: bool, cancer_first: bool
) -> None:
    graph = _hub_graph(tmp_path, brca1_is_hub=True)
    ends = ("cancer", "brca1") if cancer_first else ("brca1", "cancer")
    query = {
        "message": {
            "query_graph": {
                "nodes": {end: {"ids": [f"TEST:{end}"]} for end in ends},
                "edges": {"e0": _one_hop(*ends)},
            },
        },
    }

    response = lookup(
        graph,
        query,
        bmt=bmt,
        subclass=subclass,
        filter_config={"max_node_degree": 3},
    )
    edges = response["message"]["knowledge_graph"]["edges"]

    assert ("TEST:brca1", "TEST:cancer") in {
        (e["subject"], e["object"])
        for e in edges.values()
        if e["predicate"] == "biolink:related_to"
    }


@pytest.mark.parametrize("subclass", [False, True])
def test_an_unpinned_hub_between_two_pinned_nodes_is_filtered(
    tmp_path, bmt, subclass: bool
) -> None:
    """The second edge of start -> x -> end runs with both ends pinned; x
    was filtered when the first edge discovered it, so the hub stays out."""
    pairs = [("start", "hub"), ("hub", "end"), ("start", "plain"), ("plain", "end")]
    pairs += [("hub", f"other_{i}") for i in range(3)]
    graph = _graph_of(tmp_path, pairs)
    query = {
        "message": {
            "query_graph": {
                "nodes": {
                    "start": {"ids": ["TEST:start"]},
                    "x": {},
                    "end": {"ids": ["TEST:end"]},
                },
                "edges": {"e0": _one_hop("start", "x"), "e1": _one_hop("x", "end")},
            },
        },
    }

    response = lookup(
        graph,
        query,
        bmt=bmt,
        subclass=subclass,
        filter_config={"max_node_degree": 3},
    )

    assert [
        result["node_bindings"]["x"]["ids"][0]
        for result in response["message"]["results"]
    ] == ["TEST:plain"]
