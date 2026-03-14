"""Tests for max_node_degree and min_information_content filtering in lookup."""

from tests.search_fixtures import graph  # noqa: F401

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

        response = lookup(graph, query, bmt=bmt, max_node_degree=2)
        results = response["message"]["results"]

        # Only TNF (degree=1) should pass the filter
        assert len(results) == 1
        gene_ids = {r["node_bindings"]["n1"][0]["id"] for r in results}
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

        response = lookup(graph, query, bmt=bmt, max_node_degree=3)
        results = response["message"]["results"]

        # PPARG, GCK (degree=3) and TNF (degree=1) should all pass
        assert len(results) == 3

    def test_max_node_degree_none_means_no_filtering(self, graph, bmt):
        """When max_node_degree is None (default), no filtering should occur."""
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

        response = lookup(graph, query, bmt=bmt, max_node_degree=None)
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

        response = lookup(graph, query, bmt=bmt, max_node_degree=0)
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

        response = lookup(graph, query, bmt=bmt, min_information_content=90)
        results = response["message"]["results"]

        assert len(results) == 2
        gene_ids = {r["node_bindings"]["n1"][0]["id"] for r in results}
        assert gene_ids == {"NCBIGene:5468", "NCBIGene:7124"}

    def test_min_ic_none_means_no_filtering(self, graph, bmt):
        """When min_information_content is None (default), no filtering should occur."""
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

        response = lookup(graph, query, bmt=bmt, min_information_content=None)
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

        response = lookup(graph, query, bmt=bmt, min_information_content=100)
        results = response["message"]["results"]

        assert len(results) == 0

    def test_min_ic_filters_in_two_hop_query(self, graph, bmt):
        """min_information_content should filter intermediate nodes in multi-hop queries.

        Two-hop: Metformin --affects--> Gene --gene_associated--> T2D
        Without filtering, 3 genes bridge the path: PPARG(92.3), INSR(88.7), GCK(81.2).
        T2D has IC=78.2, which is also below 85, so it gets filtered in the second
        hop as well, yielding 0 total results.
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

        # With min_information_content=85: GCK (81.2) filtered in first hop,
        # AND T2D (78.2) filtered in second hop → 0 results
        response_filtered = lookup(graph, query, bmt=bmt, min_information_content=85)
        assert len(response_filtered["message"]["results"]) == 0


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
        response_filtered = lookup(graph, query, bmt=bmt, min_information_content=90)
        results = response_filtered["message"]["results"]

        assert len(results) == 1
        assert results[0]["node_bindings"]["n0"][0]["id"] == "NCBIGene:5468"


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
            graph, query, bmt=bmt,
            max_node_degree=2, min_information_content=90,
        )
        results = response["message"]["results"]

        assert len(results) == 1
        gene_ids = {r["node_bindings"]["n1"][0]["id"] for r in results}
        assert gene_ids == {"NCBIGene:7124"}
