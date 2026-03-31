"""Regression tests: direct edges take priority over subclass-expanded edges.

When subclass expansion is enabled and a direct edge already connects the
originally-queried (superclass) nodes, subclass-expanded edges through child
nodes are superfluous.  The result should contain only the direct edge and
should NOT include subclass_of edges or child nodes in the knowledge graph.

Test graph relationships used:
- CHEBI:6801 (Metformin) --treats--> MONDO:0005148 (Type 2 Diabetes)
- CHEBI:6801 (Metformin) --treats--> MONDO:0005015 (Diabetes Mellitus)
- MONDO:0005148 --subclass_of--> MONDO:0005015
- MONDO:0005015 --subclass_of--> MONDO:0004995
- MONDO:0005148 --has_phenotype--> HP:0001943 (Hypoglycemia)
"""

from tests.search_fixtures import graph  # noqa: F401

from gandalf.search import lookup


def _get_inferred_edges(response):
    """Return KG edges that have a biolink:support_graphs attribute (inferred)."""
    return {
        eid: edge
        for eid, edge in response["message"]["knowledge_graph"]["edges"].items()
        if any(
            a.get("attribute_type_id") == "biolink:support_graphs"
            for a in edge.get("attributes", [])
        )
    }


class TestDirectEdgePriorityOverSubclass:
    """When a direct edge exists between the queried nodes, subclass-expanded
    edges through child nodes should be excluded from the result."""

    def test_no_subclass_of_edges_when_direct_exists(self, graph, bmt):
        """KG should not contain subclass_of edges when a direct treats edge exists.

        Query: CHEBI:6801 --treats--> MONDO:0005015 with subclass=True.
        Direct edge: CHEBI:6801 --treats--> MONDO:0005015 exists.
        Subclass path: CHEBI:6801 --treats--> MONDO:0005148 --subclass_of--> MONDO:0005015
        The subclass_of edge should NOT appear in the KG.
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

        kg_edges = response["message"]["knowledge_graph"]["edges"]
        subclass_edges = [
            e for e in kg_edges.values() if e["predicate"] == "biolink:subclass_of"
        ]
        assert len(subclass_edges) == 0, (
            f"Expected no subclass_of edges in KG when direct edge exists, "
            f"but found: {subclass_edges}"
        )

    def test_no_child_nodes_in_kg_when_direct_exists(self, graph, bmt):
        """KG should not contain subclass child nodes when a direct edge exists.

        With the direct edge CHEBI:6801 --treats--> MONDO:0005015, the child
        node MONDO:0005148 (Type 2 Diabetes) should not appear in the KG.
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

        kg_nodes = set(response["message"]["knowledge_graph"]["nodes"].keys())
        assert "MONDO:0005148" not in kg_nodes, (
            "Child node MONDO:0005148 should not appear in KG when a direct "
            "edge already connects the queried nodes"
        )

    def test_no_inferred_edges_when_direct_exists(self, graph, bmt):
        """No composite inferred edges should be created when a direct edge exists."""
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

        inferred = _get_inferred_edges(response)
        assert len(inferred) == 0, (
            f"Expected no inferred composite edges when direct edge exists, "
            f"but found {len(inferred)}: {list(inferred.keys())}"
        )

    def test_no_auxiliary_graphs_when_direct_exists(self, graph, bmt):
        """auxiliary_graphs should be empty when a direct edge exists."""
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

        aux_graphs = response["message"]["auxiliary_graphs"]
        assert len(aux_graphs) == 0, (
            f"Expected empty auxiliary_graphs when direct edge exists, "
            f"but found {len(aux_graphs)} entries"
        )

    def test_direct_edge_binding_is_simple(self, graph, bmt):
        """When a direct edge exists, edge binding should reference the real
        edge directly (not a composite/inferred edge)."""
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
        assert len(results) == 1

        kg_edges = response["message"]["knowledge_graph"]["edges"]
        for result in results:
            for eb in result["analyses"][0]["edge_bindings"]["e0"]:
                bound_edge = kg_edges[eb["id"]]
                # The bound edge should be the direct treats edge,
                # not a composite inferred edge
                assert bound_edge["predicate"] == "biolink:treats"
                assert bound_edge["subject"] == "CHEBI:6801"
                assert bound_edge["object"] == "MONDO:0005015"
                assert "attributes" not in bound_edge or not any(
                    a.get("attribute_type_id") == "biolink:support_graphs"
                    for a in bound_edge.get("attributes", [])
                )


class TestSubclassCompositeStillCreatedWhenNoDirectEdge:
    """When NO direct edge exists between the queried nodes, subclass-expanded
    edges should still produce composite inferred edges as before."""

    def test_composite_created_when_only_subclass_path_exists(self, graph, bmt):
        """Composite inferred edge should be created when no direct edge exists.

        Query: HP:0001943 --related_to--> MONDO:0005015 with subclass=True.
        No direct edge between HP:0001943 and MONDO:0005015 exists.
        Subclass path: MONDO:0005148 --has_phenotype--> HP:0001943 (inverse)
                     + MONDO:0005148 --subclass_of--> MONDO:0005015
        A composite inferred edge should be created.
        """
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["HP:0001943"]},  # Hypoglycemia
                        "n1": {"ids": ["MONDO:0005015"]},  # Diabetes Mellitus
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

        response = lookup(graph, query, bmt=bmt, subclass=True, subclass_depth=1)

        results = response["message"]["results"]
        assert len(results) >= 1

        inferred = _get_inferred_edges(response)
        assert len(inferred) > 0, (
            "Expected inferred composite edges when no direct edge exists "
            "and the path goes through subclass expansion"
        )

        # The inferred edge should connect the queried nodes
        for eid, edge in inferred.items():
            endpoints = {edge["subject"], edge["object"]}
            assert "HP:0001943" in endpoints, (
                f"Inferred edge {eid} missing HP:0001943: "
                f"{edge['subject']} -> {edge['object']}"
            )
            assert "MONDO:0005015" in endpoints, (
                f"Inferred edge {eid} missing MONDO:0005015: "
                f"{edge['subject']} -> {edge['object']}"
            )

    def test_subclass_of_edge_present_when_no_direct(self, graph, bmt):
        """subclass_of edge should appear in KG when no direct edge exists.

        The subclass_of edge is part of the support graph for the composite.
        """
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["HP:0001943"]},
                        "n1": {"ids": ["MONDO:0005015"]},
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

        response = lookup(graph, query, bmt=bmt, subclass=True, subclass_depth=1)

        kg_edges = response["message"]["knowledge_graph"]["edges"]
        subclass_edges = [
            e for e in kg_edges.values() if e["predicate"] == "biolink:subclass_of"
        ]
        assert len(subclass_edges) > 0, (
            "Expected subclass_of edges in KG when no direct edge exists "
            "and the result goes through subclass expansion"
        )

    def test_child_node_present_when_no_direct(self, graph, bmt):
        """Child node should appear in KG when no direct edge exists.

        MONDO:0005148 (Type 2 Diabetes) is the child that provides the path.
        """
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["HP:0001943"]},
                        "n1": {"ids": ["MONDO:0005015"]},
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

        response = lookup(graph, query, bmt=bmt, subclass=True, subclass_depth=1)

        kg_nodes = set(response["message"]["knowledge_graph"]["nodes"].keys())
        assert "MONDO:0005148" in kg_nodes, (
            "Child node MONDO:0005148 should appear in KG when no direct edge "
            "exists and the result goes through subclass expansion"
        )


class TestDirectEdgePriorityWithInverse:
    """Same tests but with queries where the edge is found via inverse lookup."""

    def test_inverse_no_subclass_when_direct_exists(self, graph, bmt):
        """Inverse edge: no subclass artifacts when direct edge exists.

        Query: MONDO:0005015 --related_to--> CHEBI:6801 (disease -> chemical).
        Stored edge: CHEBI:6801 --treats--> MONDO:0005015 (found via inverse).
        Direct match exists, so subclass-expanded paths should be dropped.
        """
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["MONDO:0005015"]},
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

        response = lookup(graph, query, bmt=bmt, subclass=True, subclass_depth=1)

        results = response["message"]["results"]
        assert len(results) >= 1

        # No subclass_of edges should be in KG
        kg_edges = response["message"]["knowledge_graph"]["edges"]
        subclass_edges = [
            e for e in kg_edges.values() if e["predicate"] == "biolink:subclass_of"
        ]
        assert len(subclass_edges) == 0, (
            f"Expected no subclass_of edges with inverse lookup when direct "
            f"edge exists, but found: {subclass_edges}"
        )

        # No inferred composite edges
        inferred = _get_inferred_edges(response)
        assert (
            len(inferred) == 0
        ), "Expected no inferred edges with inverse lookup when direct edge exists"

        # No child node in KG
        kg_nodes = set(response["message"]["knowledge_graph"]["nodes"].keys())
        assert "MONDO:0005148" not in kg_nodes, (
            "Child node MONDO:0005148 should not appear in KG with inverse "
            "lookup when direct edge exists"
        )
