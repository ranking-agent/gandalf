"""Tests for basic lookup queries: one-hop, two-hop, and edge cases."""

from tests.search_fixtures import graph  # noqa: F401

from gandalf.search import lookup


class TestLookupOneHop:
    """Tests for single-hop queries using the lookup function."""

    def test_one_hop_pinned_both_ends_single_result(self, graph, bmt):
        """Query with pinned start and end should return 1 result with multiple edge bindings.

        Note: biolink:treats has descendants ameliorates_condition and
        preventative_for_condition, so all 3 Metformin-T2D edges match.
        Results are aggregated by unique node paths, so we get 1 result.
        """
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["CHEBI:6801"]},
                        "n1": {"ids": ["MONDO:0005148"]},
                    },
                    "edges": {
                        "e0": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:treats"],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, bmt=bmt)
        results = response["message"]["results"]

        # Results are aggregated by unique node paths
        # Same node pair (Metformin -> T2D) = 1 result with 4 edge bindings
        assert len(results) == 1
        result = results[0]
        assert "n0" in result["node_bindings"]
        assert "n1" in result["node_bindings"]
        assert result["node_bindings"]["n0"][0]["id"] == "CHEBI:6801"
        assert result["node_bindings"]["n1"][0]["id"] == "MONDO:0005148"

        # Edge bindings should contain all 4 matching edges
        edge_bindings = result["analyses"][0]["edge_bindings"]["e0"]
        assert len(edge_bindings) == 4

    def test_one_hop_pinned_start_unpinned_end(self, graph, bmt):
        """Query with pinned start should return all matching edges."""
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

        response = lookup(graph, query, bmt=bmt)
        results = response["message"]["results"]

        # CHEBI:6801 (Metformin) affects 4 genes:
        # NCBIGene:5468 (PPARG), NCBIGene:3643 (INSR), NCBIGene:2645 (GCK), NCBIGene:7124 (TNF)
        assert len(results) == 4
        gene_ids = {r["node_bindings"]["n1"][0]["id"] for r in results}
        assert gene_ids == {
            "NCBIGene:5468",
            "NCBIGene:3643",
            "NCBIGene:2645",
            "NCBIGene:7124",
        }

    def test_one_hop_no_matching_predicate(self, graph, bmt):
        """Query with non-matching predicate should return 0 paths."""
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["CHEBI:6801"]},
                        "n1": {"ids": ["MONDO:0005148"]},
                    },
                    "edges": {
                        "e0": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:causes"],  # Not in our data
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, bmt=bmt)
        results = response["message"]["results"]

        assert len(results) == 0

    def test_one_hop_multiple_results_same_predicate(self, graph, bmt):
        """Query should return all edges matching the predicate."""
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

        response = lookup(graph, query, bmt=bmt)
        results = response["message"]["results"]

        # Three genes associated with MONDO:0005148:
        # NCBIGene:5468 (PPARG), NCBIGene:3643 (INSR), NCBIGene:2645 (GCK)
        assert len(results) == 3

        # Collect all gene IDs from results
        gene_ids = {r["node_bindings"]["n0"][0]["id"] for r in results}
        assert gene_ids == {"NCBIGene:5468", "NCBIGene:3643", "NCBIGene:2645"}


class TestLookupTwoHop:
    """Tests for two-hop queries using the lookup function."""

    def test_two_hop_linear_path(self, graph, bmt):
        """Two-hop query should return paths through intermediate nodes."""
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

        response = lookup(graph, query, bmt=bmt)
        results = response["message"]["results"]

        # Three paths from Metformin through genes to Type 2 Diabetes:
        # CHEBI:6801 --affects--> NCBIGene:5468 (PPARG) --gene_associated--> MONDO:0005148
        # CHEBI:6801 --affects--> NCBIGene:3643 (INSR) --gene_associated--> MONDO:0005148
        # CHEBI:6801 --affects--> NCBIGene:2645 (GCK) --gene_associated--> MONDO:0005148
        assert len(results) == 3
        gene_ids = {r["node_bindings"]["n1"][0]["id"] for r in results}
        assert gene_ids == {"NCBIGene:5468", "NCBIGene:3643", "NCBIGene:2645"}
        # All paths should start with Metformin and end with Type 2 Diabetes
        for result in results:
            assert result["node_bindings"]["n0"][0]["id"] == "CHEBI:6801"
            assert result["node_bindings"]["n2"][0]["id"] == "MONDO:0005148"

    def test_two_hop_multiple_intermediate_nodes(self, graph, bmt):
        """Two-hop query with multiple valid intermediate nodes."""
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"categories": ["biolink:Gene"]},
                        "n1": {"ids": ["GO:0006006"]},
                        "n2": {"categories": ["biolink:Disease"]},
                    },
                    "edges": {
                        "e0": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:participates_in"],
                        },
                        "e1": {
                            "subject": "n0",
                            "object": "n2",
                            "predicates": ["biolink:gene_associated_with_condition"],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, bmt=bmt)
        results = response["message"]["results"]

        # NCBIGene:2645 (GCK) participates_in GO:0006006 AND
        # NCBIGene:2645 (GCK) gene_associated_with_condition MONDO:0005148
        # This is a query where n0 appears in both edges
        assert len(results) == 1
        assert results[0]["node_bindings"]["n0"][0]["id"] == "NCBIGene:2645"


class TestLookupEdgeCases:
    """Tests for edge cases in the lookup function."""

    def test_nonexistent_start_node(self, graph, bmt):
        """Query with non-existent start node should return 0 paths."""
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["NONEXISTENT:12345"]},
                        "n1": {"ids": ["MONDO:0005148"]},
                    },
                    "edges": {
                        "e0": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:treats"],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, bmt=bmt)
        results = response["message"]["results"]

        assert len(results) == 0

    def test_nonexistent_end_node(self, graph, bmt):
        """Query with non-existent end node should return 0 paths."""
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["CHEBI:6801"]},
                        "n1": {"ids": ["NONEXISTENT:99999"]},
                    },
                    "edges": {
                        "e0": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:treats"],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, bmt=bmt)
        results = response["message"]["results"]

        assert len(results) == 0

    def test_category_filter_excludes_non_matching(self, graph, bmt):
        """Category filter should exclude nodes not matching the category."""
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["CHEBI:6801"]},
                        "n1": {"categories": ["biolink:Pathway"]},  # Only pathways
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

        response = lookup(graph, query, bmt=bmt)
        results = response["message"]["results"]

        # CHEBI:6801 affects NCBIGene:5468 (Gene) and CHEBI:17234 (SmallMolecule)
        # Neither is a Pathway, so should return 0
        assert len(results) == 0
