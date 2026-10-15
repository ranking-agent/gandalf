"""Tests for queries using non-canonical predicates with subclass_of edges.

Non-canonical predicates are inverse predicates (e.g. biolink:treated_by is
the non-canonical inverse of biolink:treats).  When combined with subclass_of
edges and subclass expansion, the system must correctly resolve the inverse
lookup and return nodes in the right query-graph order.

Test graph relationships used:
- CHEBI:6801 (Metformin) --treats--> MONDO:0005148 (Type 2 Diabetes)
- CHEBI:6801 (Metformin) --treats--> MONDO:0005015 (Diabetes Mellitus)
- MONDO:0005148 --subclass_of--> MONDO:0005015 (Diabetes Mellitus)
- MONDO:0005015 --subclass_of--> MONDO:0004995 (Cardiovascular Disease)
- MONDO:0005148 --has_phenotype--> HP:0001943 (Hypoglycemia)
"""

from tests.search_fixtures import graph  # noqa: F401

from gandalf.search import lookup


class TestNonCanonicalPredicateWithSubclass:
    """Queries that pair a non-canonical (inverse) predicate with a
    subclass_of edge should return the correct nodes in the correct
    query-graph positions."""

    def test_treated_by_with_subclass_of_two_hop(self, graph, bmt):
        """Two-hop: Disease --treated_by--> Drug, Disease --subclass_of--> Parent.

        Query graph:
            n0 (MONDO:0005148, Type 2 Diabetes)
              --[e0: treated_by]--> n1 (Drug, unpinned)
            n0
              --[e1: subclass_of]--> n2 (MONDO:0005015, Diabetes Mellitus)

        treated_by is non-canonical (inverse of treats).
        The stored edge is CHEBI:6801 --treats--> MONDO:0005148, so e0
        must be resolved via inverse lookup.

        Expected result:
            n0 = MONDO:0005148, n1 = CHEBI:6801, n2 = MONDO:0005015
        """
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["MONDO:0005148"]},
                        "n1": {"categories": ["biolink:Drug"]},
                        "n2": {"ids": ["MONDO:0005015"]},
                    },
                    "edges": {
                        "e0": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:treated_by"],
                        },
                        "e1": {
                            "subject": "n0",
                            "object": "n2",
                            "predicates": ["biolink:subclass_of"],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, bmt=bmt)
        results = response["message"]["results"]

        # Should find exactly one result
        assert len(results) == 1
        result = results[0]

        # Verify node bindings are in the correct query-graph positions
        assert result["node_bindings"]["n0"][0]["id"] == "MONDO:0005148"
        assert result["node_bindings"]["n1"][0]["id"] == "CHEBI:6801"
        assert result["node_bindings"]["n2"][0]["id"] == "MONDO:0005015"

        # Verify edge bindings exist for both edges
        edge_bindings = result["analyses"][0]["edge_bindings"]
        assert "e0" in edge_bindings
        assert "e1" in edge_bindings
        assert len(edge_bindings["e0"]) >= 1
        assert len(edge_bindings["e1"]) >= 1

        # Verify knowledge graph contains the expected nodes
        kg_nodes = response["message"]["knowledge_graph"]["nodes"]
        assert "MONDO:0005148" in kg_nodes
        assert "CHEBI:6801" in kg_nodes
        assert "MONDO:0005015" in kg_nodes

    def test_treated_by_with_subclass_of_kg_edge_direction(self, graph, bmt):
        """KG edges should preserve stored direction regardless of query direction.

        Even though the query uses treated_by (non-canonical / inverse),
        the KG edge for the treats relationship should still have
        subject=CHEBI:6801 and object=MONDO:0005148 (the stored direction).
        """
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["MONDO:0005148"]},
                        "n1": {"categories": ["biolink:Drug"]},
                        "n2": {"ids": ["MONDO:0005015"]},
                    },
                    "edges": {
                        "e0": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:treated_by"],
                        },
                        "e1": {
                            "subject": "n0",
                            "object": "n2",
                            "predicates": ["biolink:subclass_of"],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, bmt=bmt)
        kg_edges = response["message"]["knowledge_graph"]["edges"]

        # Find the treats/ameliorates/preventative edges (matched via inverse)
        treat_edges = [
            e
            for e in kg_edges.values()
            if e["predicate"]
            in {
                "biolink:treats",
                "biolink:ameliorates_condition",
                "biolink:preventative_for_condition",
            }
        ]
        assert len(treat_edges) >= 1

        # KG edges are stored in canonical direction: drug -> disease
        for edge in treat_edges:
            assert edge["subject"] == "CHEBI:6801"
            assert edge["object"] == "MONDO:0005148"

        # Find the subclass_of edge
        subclass_edges = [
            e for e in kg_edges.values() if e["predicate"] == "biolink:subclass_of"
        ]
        assert len(subclass_edges) == 1
        assert subclass_edges[0]["subject"] == "MONDO:0005148"
        assert subclass_edges[0]["object"] == "MONDO:0005015"

    def test_phenotype_of_with_subclass_of_two_hop(self, graph, bmt):
        """Two-hop: Phenotype --phenotype_of--> Disease --subclass_of--> Parent.

        phenotype_of is non-canonical (inverse of has_phenotype).
        Stored edge: MONDO:0005148 --has_phenotype--> HP:0001943.

        Query graph:
            n0 (HP:0001943)
              --[e0: phenotype_of]--> n1 (Disease, unpinned)
            n1
              --[e1: subclass_of]--> n2 (MONDO:0005015)

        Expected: n0=HP:0001943, n1=MONDO:0005148, n2=MONDO:0005015
        """
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["HP:0001943"]},
                        "n1": {"categories": ["biolink:Disease"]},
                        "n2": {"ids": ["MONDO:0005015"]},
                    },
                    "edges": {
                        "e0": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:phenotype_of"],
                        },
                        "e1": {
                            "subject": "n1",
                            "object": "n2",
                            "predicates": ["biolink:subclass_of"],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, bmt=bmt)
        results = response["message"]["results"]

        assert len(results) == 1
        result = results[0]

        # Verify correct node ordering
        assert result["node_bindings"]["n0"][0]["id"] == "HP:0001943"
        assert result["node_bindings"]["n1"][0]["id"] == "MONDO:0005148"
        assert result["node_bindings"]["n2"][0]["id"] == "MONDO:0005015"

        # Verify both edges are bound
        edge_bindings = result["analyses"][0]["edge_bindings"]
        assert "e0" in edge_bindings
        assert "e1" in edge_bindings

    def test_superclass_of_non_canonical_one_hop(self, graph, bmt):
        """One-hop query using superclass_of (non-canonical inverse of subclass_of).

        Query: MONDO:0005015 --superclass_of--> ?
        Stored edge: MONDO:0005148 --subclass_of--> MONDO:0005015

        Expected: n0=MONDO:0005015, n1=MONDO:0005148
        """
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["MONDO:0005015"]},
                        "n1": {"categories": ["biolink:Disease"]},
                    },
                    "edges": {
                        "e0": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:superclass_of"],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, bmt=bmt)
        results = response["message"]["results"]

        # MONDO:0005148 (T2D) and MONDO:0004995 (CVD, via MONDO:0005015 subclass_of MONDO:0004995 inverse)
        # Actually only MONDO:0005148 subclass_of MONDO:0005015, so superclass_of finds MONDO:0005148
        assert len(results) >= 1

        # Verify n0 is always the parent (queried node)
        for result in results:
            assert result["node_bindings"]["n0"][0]["id"] == "MONDO:0005015"

        # MONDO:0005148 should be among the child nodes found
        child_ids = {r["node_bindings"]["n1"][0]["id"] for r in results}
        assert "MONDO:0005148" in child_ids

    def test_non_canonical_with_subclass_expansion(self, graph, bmt):
        """Non-canonical predicate query with subclass expansion enabled.

        Query: ? --treated_by--> CHEBI:6801, pinning n1 to MONDO:0005015
        with subclass=True so it expands to include MONDO:0005148.

        treated_by is non-canonical. The stored edges are:
        - CHEBI:6801 --treats--> MONDO:0005148
        - CHEBI:6801 --treats--> MONDO:0005015

        With subclass expansion on n0 (MONDO:0005015), it should find
        Type 2 Diabetes as a child and return results.
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
                            "predicates": ["biolink:treated_by"],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, bmt=bmt, subclass=True, subclass_depth=1)
        results = response["message"]["results"]

        assert len(results) >= 1

        # Node bindings should reference the queried IDs
        for result in results:
            assert result["node_bindings"]["n0"][0]["id"] == "MONDO:0005015"
            assert result["node_bindings"]["n1"][0]["id"] == "CHEBI:6801"

        # KG edges should exist and endpoints should be in KG nodes
        kg_nodes = set(response["message"]["knowledge_graph"]["nodes"].keys())
        kg_edges = response["message"]["knowledge_graph"]["edges"]
        for edge in kg_edges.values():
            assert edge["subject"] in kg_nodes
            assert edge["object"] in kg_nodes
