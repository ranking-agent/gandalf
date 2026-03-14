"""Tests for result grouping by node bindings."""

from tests.search_fixtures import graph  # noqa: F401

from gandalf.search import lookup


class TestNodeBindingGrouping:
    """Tests that results are correctly grouped by node, with exactly one node per binding."""

    def test_three_hop_pinned_endpoints_one_node_per_binding(self, graph, bmt):
        """Three-hop query with pinned endpoints should have one node per binding per result.

        Query: Metformin --affects--> Gene --gene_associated--> Disease --has_phenotype--> Hypoglycemia
        n0 (pinned), n1 (unpinned Gene), n2 (unpinned Disease), n3 (pinned)

        Should produce 3 results (one per gene), each with exactly one node per binding.
        """
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["CHEBI:6801"]},
                        "n1": {"categories": ["biolink:Gene"]},
                        "n2": {"categories": ["biolink:Disease"]},
                        "n3": {"ids": ["HP:0001943"]},
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
                        "e2": {
                            "subject": "n2",
                            "object": "n3",
                            "predicates": ["biolink:has_phenotype"],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, bmt=bmt, subclass=False)
        results = response["message"]["results"]

        # Three genes connect Metformin to Type 2 Diabetes: PPARG, INSR, GCK
        assert len(results) == 3

        for result in results:
            # Every node binding must have exactly one entry
            for qnode_id, bindings in result["node_bindings"].items():
                assert len(bindings) == 1, (
                    f"node_bindings[{qnode_id!r}] has {len(bindings)} entries, expected 1"
                )

            # Pinned endpoints should match the query
            assert result["node_bindings"]["n0"][0]["id"] == "CHEBI:6801"
            assert result["node_bindings"]["n3"][0]["id"] == "HP:0001943"

            # Intermediate disease should be T2D (only disease with has_phenotype to Hypoglycemia)
            assert result["node_bindings"]["n2"][0]["id"] == "MONDO:0005148"

        # Each result should have a distinct gene for n1
        gene_ids = {r["node_bindings"]["n1"][0]["id"] for r in results}
        assert gene_ids == {"NCBIGene:5468", "NCBIGene:3643", "NCBIGene:2645"}

    def test_three_hop_pinned_endpoints_with_subclass(self, graph, bmt):
        """Three-hop with subclass expansion still produces one node per binding.

        Same query but with subclass=True. Subclass expansion on pinned nodes
        should not produce duplicate results or multi-entry bindings.
        """
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["CHEBI:6801"]},
                        "n1": {"categories": ["biolink:Gene"]},
                        "n2": {"categories": ["biolink:Disease"]},
                        "n3": {"ids": ["HP:0001943"]},
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
                        "e2": {
                            "subject": "n2",
                            "object": "n3",
                            "predicates": ["biolink:has_phenotype"],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, bmt=bmt, subclass=True, subclass_depth=1)
        results = response["message"]["results"]

        assert len(results) >= 3

        for result in results:
            # Every node binding must have exactly one entry
            for qnode_id, bindings in result["node_bindings"].items():
                assert len(bindings) == 1, (
                    f"node_bindings[{qnode_id!r}] has {len(bindings)} entries, expected 1"
                )

    def test_two_hop_each_result_has_single_node_per_binding(self, graph, bmt):
        """Two-hop query: each result must have exactly one node per binding."""
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

        assert len(results) == 3

        for result in results:
            for qnode_id, bindings in result["node_bindings"].items():
                assert len(bindings) == 1, (
                    f"node_bindings[{qnode_id!r}] has {len(bindings)} entries, expected 1"
                )

    def test_subclass_grouping_no_duplicate_results(self, graph, bmt):
        """Subclass expansion should not produce duplicate results with identical node_bindings.

        When querying with a superclass ID (MONDO:0005015 = Diabetes Mellitus),
        subclass expansion matches both the superclass and its subclass
        (MONDO:0005148 = Type 2 Diabetes).  Both paths should be grouped into
        a single result because the bound ID is the queried superclass ID.
        """
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["CHEBI:6801"]},
                        "n1": {"ids": ["MONDO:0005015"]},
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

        response = lookup(graph, query, bmt=bmt, subclass=True, subclass_depth=1)
        results = response["message"]["results"]

        # Collect the node binding fingerprints (n0_id, n1_id) for each result
        fingerprints = []
        for result in results:
            fp = tuple(
                (qn, result["node_bindings"][qn][0]["id"])
                for qn in sorted(result["node_bindings"])
            )
            fingerprints.append(fp)

        # No two results should have the same fingerprint (no duplicates)
        assert len(fingerprints) == len(set(fingerprints)), (
            f"Duplicate results found: {fingerprints}"
        )

        # Every result should have exactly one node per binding
        for result in results:
            for qnode_id, bindings in result["node_bindings"].items():
                assert len(bindings) == 1, (
                    f"node_bindings[{qnode_id!r}] has {len(bindings)} entries, expected 1"
                )
