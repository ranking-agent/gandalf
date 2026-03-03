"""Tests for TRAPI response structure and specific edge queries."""

from tests.search_fixtures import graph  # noqa: F401

from gandalf.search import lookup


class TestLookupResponseStructure:
    """Tests for verifying the response structure from lookup."""

    def test_response_contains_knowledge_graph(self, graph, bmt):
        """Response should contain knowledge_graph with nodes and edges."""
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

        response = lookup(graph, query, bmt=bmt, verbose=False)

        assert "message" in response
        assert "knowledge_graph" in response["message"]
        assert "nodes" in response["message"]["knowledge_graph"]
        assert "edges" in response["message"]["knowledge_graph"]

    def test_response_nodes_have_required_fields(self, graph, bmt):
        """Knowledge graph nodes should have id, category, and name."""
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

        response = lookup(graph, query, bmt=bmt, verbose=False)
        kg_nodes = response["message"]["knowledge_graph"]["nodes"]

        # Check Metformin node
        assert "CHEBI:6801" in kg_nodes
        metformin = kg_nodes["CHEBI:6801"]
        assert metformin["name"] == "Metformin"
        assert "biolink:SmallMolecule" in metformin["categories"]

    def test_response_edges_have_required_fields(self, graph, bmt):
        """Knowledge graph edges should have predicate, subject, object."""
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

        response = lookup(graph, query, bmt=bmt, verbose=False)
        kg_edges = response["message"]["knowledge_graph"]["edges"]

        # Should have 4 edges (treats includes descendants: ameliorates_condition, preventative_for_condition)
        assert len(kg_edges) == 4

        # All edges should have required fields and correct subject/object
        for edge in kg_edges.values():
            assert "predicate" in edge
            assert edge["subject"] == "CHEBI:6801"
            assert edge["object"] == "MONDO:0005148"

        # Verify all 3 predicates are present
        predicates = {edge["predicate"] for edge in kg_edges.values()}
        assert predicates == {
            "biolink:treats",
            "biolink:ameliorates_condition",
            "biolink:preventative_for_condition",
        }

        # Results should be aggregated: 1 result with all 3 edges in bindings
        results = response["message"]["results"]
        assert len(results) == 1
        edge_bindings = results[0]["analyses"][0]["edge_bindings"]["e0"]
        assert len(edge_bindings) == 4

    def test_results_have_node_and_edge_bindings(self, graph, bmt):
        """Each result should have node_bindings and edge_bindings."""
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

        response = lookup(graph, query, bmt=bmt, verbose=False)
        result = response["message"]["results"][0]

        assert "node_bindings" in result
        assert "analyses" in result
        assert "edge_bindings" in result["analyses"][0]

        # Check node bindings map to query graph nodes
        assert "n0" in result["node_bindings"]
        assert "n1" in result["node_bindings"]

        # Check edge bindings map to query graph edges
        assert "e0" in result["analyses"][0]["edge_bindings"]

        # Edge bindings should be a list containing multiple edges
        edge_bindings = result["analyses"][0]["edge_bindings"]["e0"]
        assert isinstance(edge_bindings, list)
        assert len(edge_bindings) == 4  # treats, ameliorates_condition, preventative_for_condition


class TestMetforminType2DiabetesEdges:
    """Tests specifically for Metformin to Type 2 Diabetes edges."""

    def test_metformin_treats_type2_diabetes(self, graph, bmt):
        """Query for biolink:treats returns 1 result with 4 edge bindings.

        Note: biolink:treats is a parent predicate that includes descendants
        ameliorates_condition and preventative_for_condition.
        Results are aggregated by unique node paths.
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

        response = lookup(graph, query, bmt=bmt, verbose=False)
        results = response["message"]["results"]

        # Results aggregated by node path: 1 result with 4 edge bindings
        assert len(results) == 1
        kg_edges = response["message"]["knowledge_graph"]["edges"]
        predicates = {edge["predicate"] for edge in kg_edges.values()}
        assert "biolink:treats" in predicates

        # Verify edge bindings contain all 4 edges
        edge_bindings = results[0]["analyses"][0]["edge_bindings"]["e0"]
        assert len(edge_bindings) == 4

    def test_metformin_ameliorates_type2_diabetes(self, graph, bmt):
        """Query for Metformin ameliorates_condition Type 2 Diabetes edge."""
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
                            "predicates": ["biolink:ameliorates_condition"],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, bmt=bmt, verbose=False)
        results = response["message"]["results"]

        assert len(results) == 1
        edge = list(response["message"]["knowledge_graph"]["edges"].values())[0]
        assert edge["predicate"] == "biolink:ameliorates_condition"
        assert edge["subject"] == "CHEBI:6801"
        assert edge["object"] == "MONDO:0005148"

    def test_metformin_prevents_type2_diabetes(self, graph, bmt):
        """Query for Metformin preventative_for_condition Type 2 Diabetes edge."""
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
                            "predicates": ["biolink:preventative_for_condition"],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, bmt=bmt, verbose=False)
        results = response["message"]["results"]

        assert len(results) == 1
        edge = list(response["message"]["knowledge_graph"]["edges"].values())[0]
        assert edge["predicate"] == "biolink:preventative_for_condition"
        assert edge["subject"] == "CHEBI:6801"
        assert edge["object"] == "MONDO:0005148"

    def test_all_metformin_to_type2_diabetes_edges(self, graph, bmt):
        """Query for all edges from Metformin to Type 2 Diabetes returns 1 result with 4 edge bindings."""
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
                            "predicates": [
                                "biolink:treats",
                                "biolink:ameliorates_condition",
                                "biolink:preventative_for_condition",
                            ],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, bmt=bmt, verbose=False)
        results = response["message"]["results"]

        # Results aggregated by node path: 1 result with 3 edge bindings
        assert len(results) == 1

        # Collect all predicates from knowledge graph
        kg_edges = response["message"]["knowledge_graph"]["edges"]
        predicates = {edge["predicate"] for edge in kg_edges.values()}
        assert predicates == {
            "biolink:treats",
            "biolink:ameliorates_condition",
            "biolink:preventative_for_condition",
        }

        # Verify edge bindings contain all 4 edges
        edge_bindings = results[0]["analyses"][0]["edge_bindings"]["e0"]
        assert len(edge_bindings) == 4
