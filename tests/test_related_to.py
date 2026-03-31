"""Tests for biolink:related_to predicate expansion."""

from tests.search_fixtures import graph  # noqa: F401

from gandalf.search import lookup


class TestRelatedToPredicateExpansion:
    """Tests for biolink:related_to predicate handling.

    biolink:related_to is the root of the predicate hierarchy and should act
    as a wildcard matching any predicate in both forward AND inverse directions.
    """

    def test_related_to_both_pinned_forward_direction(self, graph, bmt):
        """related_to should match edges stored in the query direction.

        Graph has: CHEBI:6801 --treats--> MONDO:0005148
        Query:     CHEBI:6801 --related_to--> MONDO:0005148
        Should find the treats edge (and other forward edges).
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
                            "predicates": ["biolink:related_to"],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, bmt=bmt)
        results = response["message"]["results"]

        assert len(results) >= 1
        assert results[0]["node_bindings"]["n0"][0]["id"] == "CHEBI:6801"
        assert results[0]["node_bindings"]["n1"][0]["id"] == "MONDO:0005148"

    def test_related_to_both_pinned_inverse_direction(self, graph, bmt):
        """related_to should match edges stored in the REVERSE direction.

        Graph has: CHEBI:6801 --treats--> MONDO:0005148
        Query:     MONDO:0005148 --related_to--> CHEBI:6801  (reversed)
        Should still find edges via inverse lookup.
        """
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["MONDO:0005148"]},
                        "n1": {"ids": ["CHEBI:6801"]},
                    },
                    "edges": {
                        "e0": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:related_to"],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, bmt=bmt)
        results = response["message"]["results"]

        # Before the fix this returned 0 results because inverse edges
        # were not checked when related_to was used
        assert len(results) >= 1
        assert results[0]["node_bindings"]["n0"][0]["id"] == "MONDO:0005148"
        assert results[0]["node_bindings"]["n1"][0]["id"] == "CHEBI:6801"

    def test_related_to_pinned_start_unpinned_end(self, graph, bmt):
        """related_to with pinned start should find ALL neighbors (both directions).

        Graph has: CHEBI:6801 --affects--> NCBIGene:5468 (outgoing)
                   NCBIGene:5468 --gene_associated_with_condition--> MONDO:0005148 (outgoing)
                   NCBIGene:5468 --interacts_with--> NCBIGene:3643 (outgoing)
        Query:     NCBIGene:5468 --related_to--> ?
        Should find neighbors in both outgoing AND incoming directions.
        """
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["NCBIGene:5468"]},  # PPARG
                        "n1": {},
                    },
                    "edges": {
                        "e0": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:related_to"],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, bmt=bmt)
        results = response["message"]["results"]

        result_ids = {r["node_bindings"]["n1"][0]["id"] for r in results}

        # Forward (outgoing) edges from PPARG:
        #   PPARG --gene_associated_with_condition--> MONDO:0005148
        #   PPARG --interacts_with--> NCBIGene:3643
        assert "MONDO:0005148" in result_ids, "Should find outgoing edge targets"
        assert "NCBIGene:3643" in result_ids, "Should find outgoing edge targets"

        # Inverse (incoming) edges to PPARG:
        #   CHEBI:6801 --affects--> PPARG (incoming to PPARG)
        assert (
            "CHEBI:6801" in result_ids
        ), "Should find incoming edge sources via inverse direction"

    def test_related_to_unpinned_start_pinned_end(self, graph, bmt):
        """related_to with pinned end should find ALL neighbors (both directions).

        Graph has edges pointing TO MONDO:0005148:
            CHEBI:6801 --treats--> MONDO:0005148
            NCBIGene:5468 --gene_associated_with_condition--> MONDO:0005148
        And edges pointing FROM MONDO:0005148:
            MONDO:0005148 --has_phenotype--> HP:0001943
            MONDO:0005148 --subclass_of--> MONDO:0005015
        Query: ? --related_to--> MONDO:0005148
        Should find sources of incoming edges AND targets of outgoing edges
        (via inverse).
        """
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {},
                        "n1": {"ids": ["MONDO:0005148"]},
                    },
                    "edges": {
                        "e0": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:related_to"],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, bmt=bmt)
        results = response["message"]["results"]

        result_ids = {r["node_bindings"]["n0"][0]["id"] for r in results}

        # Forward (incoming edges to MONDO:0005148, i.e. subjects of edges
        # pointing at it):
        #   CHEBI:6801 --treats--> MONDO:0005148
        #   NCBIGene:5468 --gene_associated_with_condition--> MONDO:0005148
        assert "CHEBI:6801" in result_ids, "Should find subjects of incoming edges"
        assert "NCBIGene:5468" in result_ids, "Should find subjects of incoming edges"

        # Inverse (outgoing edges from MONDO:0005148, found via inverse lookup):
        #   MONDO:0005148 --has_phenotype--> HP:0001943
        assert (
            "HP:0001943" in result_ids
        ), "Should find outgoing edge targets via inverse direction"

    def test_related_to_no_predicates_same_as_related_to(self, graph, bmt):
        """Query with no predicates should behave the same as related_to."""
        # Query with related_to
        query_related = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["NCBIGene:5468"]},  # PPARG
                        "n1": {},
                    },
                    "edges": {
                        "e0": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:related_to"],
                        },
                    },
                },
            },
        }

        # Query with no predicates at all
        query_none = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["NCBIGene:5468"]},  # PPARG
                        "n1": {},
                    },
                    "edges": {
                        "e0": {
                            "subject": "n0",
                            "object": "n1",
                        },
                    },
                },
            },
        }

        response_related = lookup(graph, query_related, bmt=bmt)
        response_none = lookup(graph, query_none, bmt=bmt)

        ids_related = {
            r["node_bindings"]["n1"][0]["id"]
            for r in response_related["message"]["results"]
        }
        ids_none = {
            r["node_bindings"]["n1"][0]["id"]
            for r in response_none["message"]["results"]
        }

        assert ids_related == ids_none

    def test_related_to_two_hop_with_inverse(self, graph, bmt):
        """Two-hop query where related_to must use inverse to find the path.

        Query: MONDO:0005148 --related_to--> ? --related_to--> NCBIGene:3643
        One valid path (requiring inverse on first hop):
            MONDO:0005148 <--gene_associated_with_condition-- NCBIGene:5468
            NCBIGene:5468 --interacts_with--> NCBIGene:3643
        """
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["MONDO:0005148"]},
                        "n1": {"categories": ["biolink:Gene"]},
                        "n2": {"ids": ["NCBIGene:3643"]},
                    },
                    "edges": {
                        "e0": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:related_to"],
                        },
                        "e1": {
                            "subject": "n1",
                            "object": "n2",
                            "predicates": ["biolink:related_to"],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, bmt=bmt)
        results = response["message"]["results"]

        # Should find at least PPARG as the intermediate node
        intermediate_ids = {r["node_bindings"]["n1"][0]["id"] for r in results}
        assert (
            "NCBIGene:5468" in intermediate_ids
        ), "Should find PPARG via inverse lookup on first hop"

    def test_related_to_two_hop_symmetric_result_count(self, graph, bmt):
        """Two-hop related_to queries should return the same results regardless of direction.

        Query A: CHEBI:6801 --related_to--> Gene --related_to--> NCBIGene:3643
        Query B: NCBIGene:3643 --related_to--> Gene --related_to--> CHEBI:6801
        Should find the same intermediate Gene nodes.
        """
        query_a = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["CHEBI:6801"]},
                        "n1": {"categories": ["biolink:Gene"]},
                        "n2": {"ids": ["NCBIGene:3643"]},
                    },
                    "edges": {
                        "e0": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:related_to"],
                        },
                        "e1": {
                            "subject": "n1",
                            "object": "n2",
                            "predicates": ["biolink:related_to"],
                        },
                    },
                },
            },
        }

        query_b = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["NCBIGene:3643"]},
                        "n1": {"categories": ["biolink:Gene"]},
                        "n2": {"ids": ["CHEBI:6801"]},
                    },
                    "edges": {
                        "e0": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:related_to"],
                        },
                        "e1": {
                            "subject": "n1",
                            "object": "n2",
                            "predicates": ["biolink:related_to"],
                        },
                    },
                },
            },
        }

        response_a = lookup(graph, query_a, bmt=bmt)
        response_b = lookup(graph, query_b, bmt=bmt)

        intermediates_a = {
            r["node_bindings"]["n1"][0]["id"] for r in response_a["message"]["results"]
        }
        intermediates_b = {
            r["node_bindings"]["n1"][0]["id"] for r in response_b["message"]["results"]
        }

        assert intermediates_a == intermediates_b, (
            f"Forward and reverse two-hop queries should find the same intermediate "
            f"nodes, but got {intermediates_a} vs {intermediates_b}"
        )
        assert len(response_a["message"]["results"]) == len(
            response_b["message"]["results"]
        ), (
            f"Forward and reverse two-hop queries should return the same number of "
            f"results, but got {len(response_a['message']['results'])} vs "
            f"{len(response_b['message']['results'])}"
        )

    def test_related_to_two_hop_inverse_intermediate_pinning(self, graph, bmt):
        """Verify intermediate nodes are correctly pinned when first hop uses inverse.

        Query: MONDO:0005148 --related_to--> Gene --related_to--> CHEBI:6801
        First hop must use inverse to find Gene nodes (edges stored as Gene -> MONDO).
        Those Gene nodes must then correctly pin the second hop.
        The reversed query should produce the same results.
        """
        query_forward = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["MONDO:0005148"]},
                        "n1": {"categories": ["biolink:Gene"]},
                        "n2": {"ids": ["CHEBI:6801"]},
                    },
                    "edges": {
                        "e0": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:related_to"],
                        },
                        "e1": {
                            "subject": "n1",
                            "object": "n2",
                            "predicates": ["biolink:related_to"],
                        },
                    },
                },
            },
        }

        query_reversed = {
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
                            "predicates": ["biolink:related_to"],
                        },
                        "e1": {
                            "subject": "n1",
                            "object": "n2",
                            "predicates": ["biolink:related_to"],
                        },
                    },
                },
            },
        }

        response_fwd = lookup(graph, query_forward, bmt=bmt)
        response_rev = lookup(graph, query_reversed, bmt=bmt)

        intermediates_fwd = {
            r["node_bindings"]["n1"][0]["id"]
            for r in response_fwd["message"]["results"]
        }
        intermediates_rev = {
            r["node_bindings"]["n1"][0]["id"]
            for r in response_rev["message"]["results"]
        }

        # Both directions should find genes that connect MONDO:0005148 and CHEBI:6801
        assert len(intermediates_fwd) > 0, "Should find at least one intermediate gene"
        assert intermediates_fwd == intermediates_rev, (
            f"Intermediate genes should be the same regardless of query direction: "
            f"{intermediates_fwd} vs {intermediates_rev}"
        )
