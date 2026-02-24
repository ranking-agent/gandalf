"""Tests for symmetric predicate handling and edge validation."""

from tests.search_fixtures import graph  # noqa: F401

from gandalf.search import lookup
from gandalf.validation import validate_edge_exists


class TestSymmetricPredicates:
    """Tests for symmetric predicate handling (e.g., interacts_with).

    The test fixture contains:
    - NCBIGene:5468 (PPARG) --interacts_with--> NCBIGene:3643 (INSR)

    Symmetric predicates should be found regardless of query direction:
    - Query PPARG -> INSR should find the direct edge
    - Query INSR -> PPARG should also find this edge (via symmetric property)
    """

    def test_symmetric_predicate_forward_direction(self, graph, bmt):
        """Query in the direction the edge is stored should return the edge."""
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["NCBIGene:5468"]},  # PPARG
                        "n1": {"ids": ["NCBIGene:3643"]},  # INSR
                    },
                    "edges": {
                        "e0": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:interacts_with"],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, bmt=bmt, verbose=False)
        results = response["message"]["results"]

        # Should find the direct edge PPARG -> INSR
        assert len(results) == 1
        assert results[0]["node_bindings"]["n0"][0]["id"] == "NCBIGene:5468"
        assert results[0]["node_bindings"]["n1"][0]["id"] == "NCBIGene:3643"

        # The edge in the knowledge graph should be in the stored direction
        kg_edges = response["message"]["knowledge_graph"]["edges"]
        assert len(kg_edges) == 1
        edge = list(kg_edges.values())[0]
        assert edge["predicate"] == "biolink:interacts_with"
        # Edge should be in the actual stored direction
        assert edge["subject"] == "NCBIGene:5468"
        assert edge["object"] == "NCBIGene:3643"

    def test_symmetric_predicate_reverse_direction(self, graph, bmt):
        """Query in reverse direction should also find the edge via symmetric property.

        Graph has: PPARG --interacts_with--> INSR
        Query asks: INSR --interacts_with--> PPARG (reverse)
        Should find the edge because interacts_with is symmetric.
        """
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["NCBIGene:3643"]},  # INSR (query subject)
                        "n1": {"ids": ["NCBIGene:5468"]},  # PPARG (query object)
                    },
                    "edges": {
                        "e0": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:interacts_with"],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, bmt=bmt, verbose=False)
        results = response["message"]["results"]

        # Should find 1 result - the symmetric edge
        assert len(results) == 1
        # Node bindings should reflect the query structure
        assert results[0]["node_bindings"]["n0"][0]["id"] == "NCBIGene:3643"
        assert results[0]["node_bindings"]["n1"][0]["id"] == "NCBIGene:5468"

        # The edge in knowledge graph should be the ACTUAL stored edge
        kg_edges = response["message"]["knowledge_graph"]["edges"]
        assert len(kg_edges) == 1
        edge = list(kg_edges.values())[0]
        assert edge["predicate"] == "biolink:interacts_with"
        # Edge should be in the actual stored direction (PPARG -> INSR)
        assert edge["subject"] == "NCBIGene:5468"
        assert edge["object"] == "NCBIGene:3643"

    def test_symmetric_predicate_pinned_start_unpinned_end(self, graph, bmt):
        """Query with pinned start should find neighbors via symmetric predicate."""
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["NCBIGene:5468"]},  # PPARG
                        "n1": {"categories": ["biolink:Gene"]},
                    },
                    "edges": {
                        "e0": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:interacts_with"],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, bmt=bmt, verbose=False)
        results = response["message"]["results"]

        # Should find INSR as interacting partner
        assert len(results) == 1
        assert results[0]["node_bindings"]["n1"][0]["id"] == "NCBIGene:3643"

    def test_symmetric_predicate_unpinned_start_pinned_end(self, graph, bmt):
        """Query with pinned end should find neighbors via symmetric predicate.

        Graph has: PPARG --interacts_with--> INSR
        Query: ? --interacts_with--> PPARG
        Should find INSR via symmetric property.
        """
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"categories": ["biolink:Gene"]},
                        "n1": {"ids": ["NCBIGene:5468"]},  # PPARG (pinned as object)
                    },
                    "edges": {
                        "e0": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:interacts_with"],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, bmt=bmt, verbose=False)
        results = response["message"]["results"]

        # Should find INSR as the subject (interacts with PPARG)
        assert len(results) == 1
        assert results[0]["node_bindings"]["n0"][0]["id"] == "NCBIGene:3643"

    def test_symmetric_predicate_two_hop_forward(self, graph, bmt):
        """Two-hop query with symmetric predicate in forward direction.

        Path: Metformin --affects--> PPARG --interacts_with--> INSR
        """
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["CHEBI:6801"]},  # Metformin
                        "n1": {"categories": ["biolink:Gene"]},  # PPARG
                        "n2": {"ids": ["NCBIGene:3643"]},  # INSR
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
                            "predicates": ["biolink:interacts_with"],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, bmt=bmt, verbose=False)
        results = response["message"]["results"]

        # Should find 1 path: Metformin -> PPARG -> INSR
        assert len(results) == 1
        assert results[0]["node_bindings"]["n0"][0]["id"] == "CHEBI:6801"
        assert results[0]["node_bindings"]["n1"][0]["id"] == "NCBIGene:5468"
        assert results[0]["node_bindings"]["n2"][0]["id"] == "NCBIGene:3643"

    def test_symmetric_predicate_two_hop_reverse(self, graph, bmt):
        """Two-hop query with symmetric predicate in reverse direction.

        Query path: Metformin --affects--> PPARG <--interacts_with-- INSR
        Stored as: PPARG --interacts_with--> INSR
        The symmetric predicate should allow INSR to be found as connecting to PPARG.
        """
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["CHEBI:6801"]},  # Metformin
                        "n1": {"categories": ["biolink:Gene"]},  # Will be PPARG
                        "n2": {"ids": ["NCBIGene:3643"]},  # INSR
                    },
                    "edges": {
                        "e0": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:affects"],
                        },
                        "e1": {
                            "subject": "n2",  # INSR as subject
                            "object": "n1",  # PPARG as object
                            "predicates": ["biolink:interacts_with"],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, bmt=bmt, verbose=False)
        results = response["message"]["results"]

        # Should find 1 path via symmetric interacts_with
        assert len(results) == 1
        assert results[0]["node_bindings"]["n0"][0]["id"] == "CHEBI:6801"
        assert results[0]["node_bindings"]["n1"][0]["id"] == "NCBIGene:5468"
        assert results[0]["node_bindings"]["n2"][0]["id"] == "NCBIGene:3643"


class TestSymmetricPredicateValidation:
    """Tests that validate returned edges actually exist in the graph.

    These tests ensure that the edges returned in the knowledge_graph
    are real edges that exist in the graph, not "phantom edges".
    """

    def test_symmetric_edge_validation_forward(self, graph, bmt):
        """Edges returned from forward symmetric query should exist in graph."""
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["NCBIGene:5468"]},  # PPARG
                        "n1": {"ids": ["NCBIGene:3643"]},  # INSR
                    },
                    "edges": {
                        "e0": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:interacts_with"],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, bmt=bmt, verbose=False)
        kg_edges = response["message"]["knowledge_graph"]["edges"]

        # Every edge in the knowledge graph should exist in the actual graph
        for edge_id, edge in kg_edges.items():
            error = validate_edge_exists(
                graph,
                edge["subject"],
                edge["predicate"],
                edge["object"],
            )
            assert error is None, f"Edge {edge_id} not found: {edge}"

    def test_symmetric_edge_validation_reverse(self, graph, bmt):
        """Edges returned from reverse symmetric query should exist in graph.

        This is the key test for the phantom edge bug. When querying in
        the reverse direction of a symmetric predicate, the returned
        edge should still be the actual stored edge.
        """
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["NCBIGene:3643"]},  # INSR (query subject)
                        "n1": {"ids": ["NCBIGene:5468"]},  # PPARG (query object)
                    },
                    "edges": {
                        "e0": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:interacts_with"],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, bmt=bmt, verbose=False)
        kg_edges = response["message"]["knowledge_graph"]["edges"]

        # Should have found at least one edge
        assert len(kg_edges) >= 1, "Expected at least one edge in response"

        # Every edge in the knowledge graph should exist in the actual graph
        for edge_id, edge in kg_edges.items():
            error = validate_edge_exists(
                graph,
                edge["subject"],
                edge["predicate"],
                edge["object"],
            )
            assert error is None, (
                f"Phantom edge detected! Edge {edge_id} not found in graph: "
                f"{edge['subject']} --{edge['predicate']}--> {edge['object']}"
            )

    def test_symmetric_two_hop_validation(self, graph, bmt):
        """All edges in multi-hop symmetric query should exist in graph."""
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["CHEBI:6801"]},  # Metformin
                        "n1": {"categories": ["biolink:Gene"]},
                        "n2": {"ids": ["NCBIGene:3643"]},  # INSR
                    },
                    "edges": {
                        "e0": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:affects"],
                        },
                        "e1": {
                            "subject": "n2",  # Reverse direction
                            "object": "n1",
                            "predicates": ["biolink:interacts_with"],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, bmt=bmt, verbose=False)
        kg_edges = response["message"]["knowledge_graph"]["edges"]

        # Validate every edge
        for edge_id, edge in kg_edges.items():
            error = validate_edge_exists(
                graph,
                edge["subject"],
                edge["predicate"],
                edge["object"],
            )
            assert error is None, (
                f"Phantom edge detected! Edge {edge_id} not found in graph: "
                f"{edge['subject']} --{edge['predicate']}--> {edge['object']}"
            )
