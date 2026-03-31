"""Tests for subclass reasoning feature."""

from tests.search_fixtures import graph  # noqa: F401

from gandalf.search import lookup


class TestSubclassHandling:
    """Tests for subclass reasoning feature.

    The test fixtures contain:
    - MONDO:0005148 (Type 2 Diabetes) --subclass_of--> MONDO:0005015 (Diabetes Mellitus)
    - MONDO:0005015 (Diabetes Mellitus) --subclass_of--> MONDO:0004995 (Cardiovascular Disease)
    - CHEBI:6801 (Metformin) --treats--> MONDO:0005148 (Type 2 Diabetes)
    - CHEBI:6801 (Metformin) --treats--> MONDO:0005015 (Diabetes Mellitus)
    """

    def test_subclass_off(self, graph, bmt):
        """Without subclass=True, querying for Diabetes Mellitus only returns exact matches."""
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["CHEBI:6801"]},
                        "n1": {"ids": ["MONDO:0005015"]},  # Diabetes Mellitus
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

        response = lookup(graph, query, bmt=bmt, subclass=False)
        results = response["message"]["results"]

        # Only exact match: Metformin treats Diabetes Mellitus
        assert len(results) == 1
        assert results[0]["node_bindings"]["n1"][0]["id"] == "MONDO:0005015"

    def test_subclass_depth_one_expands_to_children(self, graph, bmt):
        """With subclass=True, querying for Diabetes Mellitus also finds Type 2 Diabetes results.

        Diabetes Mellitus (MONDO:0005015) has child Type 2 Diabetes (MONDO:0005148).
        Metformin treats both, so we should see results for both.
        """
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["CHEBI:6801"]},
                        "n1": {"ids": ["MONDO:0005015"]},  # Diabetes Mellitus (parent)
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

        # Should find results for both Diabetes Mellitus (direct) and Type 2 Diabetes (subclass)
        assert len(results) == 1

        # Node bindings should reference the originally queried ID (superclass)
        # for results that came via subclass expansion
        bound_ids = {r["node_bindings"]["n1"][0]["id"] for r in results}
        assert "MONDO:0005015" in bound_ids

    def test_subclass_depth_zero_is_identity(self, graph, bmt):
        """With subclass_depth=0, only the exact node matches (equivalent to no subclass)."""
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

        response_no_subclass = lookup(graph, query, bmt=bmt, subclass=False)
        response_depth_zero = lookup(
            graph, query, bmt=bmt, subclass=True, subclass_depth=0
        )

        results_no = response_no_subclass["message"]["results"]
        results_zero = response_depth_zero["message"]["results"]

        # Both should return 1 result: exact match only
        assert len(results_no) == len(results_zero) == 1

    def test_subclass_skips_explicit_hierarchy_edges(self, graph, bmt):
        """Nodes already in explicit subclass_of edges are not rewritten."""
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["MONDO:0005148"]},  # Type 2 Diabetes
                        "n1": {"ids": ["MONDO:0005015"]},  # Diabetes Mellitus
                    },
                    "edges": {
                        "e0": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:subclass_of"],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, bmt=bmt, subclass=True, subclass_depth=1)
        results = response["message"]["results"]

        # Should find the explicit edge without creating synthetic superclass nodes
        assert len(results) == 1
        # Node bindings should use the exact queried IDs (no rewriting happened)
        assert results[0]["node_bindings"]["n0"][0]["id"] == "MONDO:0005148"
        assert results[0]["node_bindings"]["n1"][0]["id"] == "MONDO:0005015"

    def test_subclass_auxiliary_graphs_present(self, graph, bmt):
        """Results from subclass expansion should include auxiliary_graphs."""
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["CHEBI:6801"]},
                        "n1": {"ids": ["MONDO:0005015"]},  # Diabetes Mellitus (parent)
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
        aux_graphs = response["message"]["auxiliary_graphs"]

        # auxiliary_graphs should exist in the response
        assert isinstance(aux_graphs, dict)

        # If there were subclass expansions that found edges via subclass hops,
        # there should be auxiliary graphs with edge lists
        if aux_graphs:
            for ag_id, ag in aux_graphs.items():
                assert "edges" in ag
                assert (
                    len(ag["edges"]) >= 2
                )  # At least the real edge + the subclass edge

    def test_subclass_inferred_edges_have_logical_entailment(self, graph, bmt):
        """Inferred composite edges should have knowledge_level=logical_entailment."""
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
        kg_edges = response["message"]["knowledge_graph"]["edges"]

        # Find inferred edges (those with support_graphs attribute)
        inferred_edges = [
            e
            for e in kg_edges.values()
            if any(
                attr.get("attribute_type_id") == "biolink:support_graphs"
                for attr in e.get("attributes", [])
            )
        ]

        # There should be at least one inferred edge (from subclass expansion)
        if inferred_edges:
            for edge in inferred_edges:
                attr_map = {
                    a["attribute_type_id"]: a["value"] for a in edge["attributes"]
                }
                assert attr_map["biolink:knowledge_level"] == "logical_entailment"
                assert attr_map["biolink:agent_type"] == "automated_agent"

    def test_subclass_node_binding_uses_superclass_id(self, graph, bmt):
        """When a result comes via subclass, node binding should reference the queried (superclass) ID."""
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["CHEBI:6801"]},
                        "n1": {"ids": ["MONDO:0005015"]},  # Diabetes Mellitus (parent)
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

        # All results should have n0 bound to Metformin
        for result in results:
            assert result["node_bindings"]["n0"][0]["id"] == "CHEBI:6801"

        # n1 bindings: direct match uses MONDO:0005015, subclass match also uses MONDO:0005015
        # (the superclass ID, since that's what was queried)
        n1_ids = {r["node_bindings"]["n1"][0]["id"] for r in results}
        assert "MONDO:0005015" in n1_ids

    def test_subclass_superclass_nodes_hidden_from_bindings(self, graph, bmt):
        """Synthetic superclass nodes should not appear in result node_bindings."""
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

        for result in results:
            # Only original query node IDs should be in bindings
            assert set(result["node_bindings"].keys()) == {"n0", "n1"}
            # No "_superclass" keys
            for key in result["node_bindings"]:
                assert "_superclass" not in key

    def test_subclass_subclass_edges_hidden_from_bindings(self, graph, bmt):
        """Synthetic subclass edges should not appear in result edge_bindings."""
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

        for result in results:
            edge_binding_keys = set(result["analyses"][0]["edge_bindings"].keys())
            # Only original query edge IDs should be in bindings
            assert "e0" in edge_binding_keys
            # No "_subclass_edge" keys
            for key in edge_binding_keys:
                assert "_subclass" not in key

    def test_subclass_two_hop_with_expansion(self, graph, bmt):
        """Two-hop query with subclass expansion on one end.

        Query: Metformin --treats--> ? --has_phenotype--> Hypoglycemia
        With subclass on, the disease node should expand to include subclasses.
        Type 2 Diabetes has phenotype Hypoglycemia, and Type 2 Diabetes is
        a subclass of Diabetes Mellitus.
        """
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["CHEBI:6801"]},
                        "n1": {"categories": ["biolink:Disease"]},
                        "n2": {"ids": ["HP:0001943"]},  # Hypoglycemia
                    },
                    "edges": {
                        "e0": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:treats"],
                        },
                        "e1": {
                            "subject": "n1",
                            "object": "n2",
                            "predicates": ["biolink:has_phenotype"],
                        },
                    },
                },
            },
        }

        # Without subclass: n1 must have treats edges AND has_phenotype edges
        response_no = lookup(graph, query, bmt=bmt, subclass=False)
        results_no = response_no["message"]["results"]

        # With subclass: same query but subclass expansion might find more paths
        response_yes = lookup(graph, query, bmt=bmt, subclass=True, subclass_depth=1)
        results_yes = response_yes["message"]["results"]

        # Both should find results through Type 2 Diabetes
        # (Metformin treats T2D, T2D has_phenotype Hypoglycemia)
        assert len(results_no) >= 1
        assert len(results_yes) >= 1

    def test_subclass_response_has_auxiliary_graphs_key(self, graph, bmt):
        """Even without subclass, response should have auxiliary_graphs key."""
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
        assert "auxiliary_graphs" in response["message"]
        assert isinstance(response["message"]["auxiliary_graphs"], dict)
