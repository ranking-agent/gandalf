"""Tests for TRAPI attribute constraint matching and filtering."""

import pytest

from gandalf.search.attribute_constraints import matches_attribute_constraints
from tests.search_fixtures import graph  # noqa: F401

from gandalf.search.lookup import lookup


# ---------------------------------------------------------------------------
# Unit tests for the matching function
# ---------------------------------------------------------------------------

class TestMatchesAttributeConstraints:
    """Unit tests for matches_attribute_constraints."""

    def test_empty_constraints_returns_true(self):
        attrs = [{"attribute_type_id": "biolink:p_value", "value": 0.05}]
        assert matches_attribute_constraints(attrs, []) is True
        assert matches_attribute_constraints(attrs, None) is True

    def test_no_attributes_with_constraints_returns_false(self):
        constraints = [{
            "id": "biolink:p_value",
            "name": "p-value",
            "operator": "<",
            "value": 0.05,
        }]
        assert matches_attribute_constraints([], constraints) is False
        assert matches_attribute_constraints(None, constraints) is False

    def test_equals_operator(self):
        attrs = [{"attribute_type_id": "biolink:knowledge_level", "value": "knowledge_assertion"}]
        constraint_match = [{
            "id": "biolink:knowledge_level",
            "name": "knowledge level",
            "operator": "==",
            "value": "knowledge_assertion",
        }]
        constraint_no_match = [{
            "id": "biolink:knowledge_level",
            "name": "knowledge level",
            "operator": "==",
            "value": "prediction",
        }]
        assert matches_attribute_constraints(attrs, constraint_match) is True
        assert matches_attribute_constraints(attrs, constraint_no_match) is False

    def test_greater_than_operator(self):
        attrs = [{"attribute_type_id": "biolink:p_value", "value": 0.03}]
        assert matches_attribute_constraints(attrs, [{
            "id": "biolink:p_value", "name": "p", "operator": ">", "value": 0.01,
        }]) is True
        assert matches_attribute_constraints(attrs, [{
            "id": "biolink:p_value", "name": "p", "operator": ">", "value": 0.05,
        }]) is False

    def test_less_than_operator(self):
        attrs = [{"attribute_type_id": "biolink:p_value", "value": 0.03}]
        assert matches_attribute_constraints(attrs, [{
            "id": "biolink:p_value", "name": "p", "operator": "<", "value": 0.05,
        }]) is True
        assert matches_attribute_constraints(attrs, [{
            "id": "biolink:p_value", "name": "p", "operator": "<", "value": 0.01,
        }]) is False

    def test_matches_operator_regex(self):
        attrs = [{"attribute_type_id": "biolink:description", "value": "Metformin treats diabetes"}]
        assert matches_attribute_constraints(attrs, [{
            "id": "biolink:description", "name": "desc", "operator": "matches",
            "value": "treats.*diabetes",
        }]) is True
        assert matches_attribute_constraints(attrs, [{
            "id": "biolink:description", "name": "desc", "operator": "matches",
            "value": "^prevents",
        }]) is False

    def test_strict_equals_operator(self):
        attrs = [{"attribute_type_id": "biolink:score", "value": 42}]
        # Same type and value
        assert matches_attribute_constraints(attrs, [{
            "id": "biolink:score", "name": "score", "operator": "===", "value": 42,
        }]) is True
        # Different type (float vs int)
        assert matches_attribute_constraints(attrs, [{
            "id": "biolink:score", "name": "score", "operator": "===", "value": 42.0,
        }]) is False

    def test_strict_equals_list_order(self):
        attrs = [{"attribute_type_id": "biolink:tags", "value": ["a", "b", "c"]}]
        assert matches_attribute_constraints(attrs, [{
            "id": "biolink:tags", "name": "tags", "operator": "===", "value": ["a", "b", "c"],
        }]) is True
        assert matches_attribute_constraints(attrs, [{
            "id": "biolink:tags", "name": "tags", "operator": "===", "value": ["c", "b", "a"],
        }]) is False

    def test_not_negation(self):
        attrs = [{"attribute_type_id": "biolink:knowledge_level", "value": "prediction"}]
        # "not prediction" should pass for "prediction" -> negated match -> False
        assert matches_attribute_constraints(attrs, [{
            "id": "biolink:knowledge_level", "name": "kl", "operator": "==",
            "value": "prediction", "not": True,
        }]) is False
        # "not knowledge_assertion" should pass for "prediction" -> no match -> negated -> True
        assert matches_attribute_constraints(attrs, [{
            "id": "biolink:knowledge_level", "name": "kl", "operator": "==",
            "value": "knowledge_assertion", "not": True,
        }]) is True

    def test_and_logic_multiple_constraints(self):
        attrs = [
            {"attribute_type_id": "biolink:p_value", "value": 0.03},
            {"attribute_type_id": "biolink:score", "value": 0.95},
        ]
        # Both pass
        constraints = [
            {"id": "biolink:p_value", "name": "p", "operator": "<", "value": 0.05},
            {"id": "biolink:score", "name": "score", "operator": ">", "value": 0.9},
        ]
        assert matches_attribute_constraints(attrs, constraints) is True

        # First passes, second fails
        constraints_fail = [
            {"id": "biolink:p_value", "name": "p", "operator": "<", "value": 0.05},
            {"id": "biolink:score", "name": "score", "operator": ">", "value": 0.99},
        ]
        assert matches_attribute_constraints(attrs, constraints_fail) is False

    def test_missing_attribute_fails(self):
        attrs = [{"attribute_type_id": "biolink:score", "value": 0.95}]
        assert matches_attribute_constraints(attrs, [{
            "id": "biolink:p_value", "name": "p", "operator": "<", "value": 0.05,
        }]) is False

    def test_numeric_comparison_with_list_value_or_logic(self):
        """Per TRAPI spec: with lists and > or <, at least one must be true (OR)."""
        attrs = [{"attribute_type_id": "biolink:score", "value": 5}]
        assert matches_attribute_constraints(attrs, [{
            "id": "biolink:score", "name": "s", "operator": ">", "value": [10, 3],
        }]) is True  # 5 > 3 is true
        assert matches_attribute_constraints(attrs, [{
            "id": "biolink:score", "name": "s", "operator": ">", "value": [10, 20],
        }]) is False  # neither

    def test_match_by_original_attribute_name(self):
        """Constraints should also match on original_attribute_name."""
        attrs = [{"attribute_type_id": "biolink:Attribute",
                  "original_attribute_name": "information_content", "value": 92.3}]
        assert matches_attribute_constraints(attrs, [{
            "id": "information_content", "name": "IC", "operator": ">", "value": 90,
        }]) is True

    def test_no_attributes_all_negated_returns_true(self):
        """If all constraints are negated and there are no attributes, they all pass."""
        constraints = [{
            "id": "biolink:p_value", "name": "p", "operator": "<",
            "value": 0.05, "not": True,
        }]
        assert matches_attribute_constraints([], constraints) is True


# ---------------------------------------------------------------------------
# Integration tests: node constraints filtering through lookup
# ---------------------------------------------------------------------------

class TestNodeConstraintsIntegration:
    """Test node attribute constraints in full TRAPI queries."""

    def test_node_constraint_filters_by_information_content(self, graph, bmt):
        """Node constraints with '>' on information_content should filter nodes.

        Metformin affects 4 genes with ICs: PPARG(92.3), INSR(88.7), GCK(81.2), TNF(94.5).
        Constraint IC > 90 should keep PPARG and TNF.
        """
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["CHEBI:6801"]},
                        "n1": {
                            "categories": ["biolink:Gene"],
                            "constraints": [{
                                "id": "information_content",
                                "name": "information content",
                                "operator": ">",
                                "value": 90,
                            }],
                        },
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

        response = lookup(graph, query, bmt=bmt, verbose=False)
        results = response["message"]["results"]

        assert len(results) == 2
        gene_ids = {r["node_bindings"]["n1"][0]["id"] for r in results}
        assert gene_ids == {"NCBIGene:5468", "NCBIGene:7124"}

    def test_node_constraint_less_than_ic(self, graph, bmt):
        """Node constraint with '<' on information_content.

        Metformin affects 4 genes with ICs: PPARG(92.3), INSR(88.7), GCK(81.2), TNF(94.5).
        Constraint IC < 85 should keep only GCK.
        """
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["CHEBI:6801"]},
                        "n1": {
                            "categories": ["biolink:Gene"],
                            "constraints": [{
                                "id": "information_content",
                                "name": "information content",
                                "operator": "<",
                                "value": 85,
                            }],
                        },
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

        response = lookup(graph, query, bmt=bmt, verbose=False)
        results = response["message"]["results"]

        assert len(results) == 1
        assert results[0]["node_bindings"]["n1"][0]["id"] == "NCBIGene:2645"

    def test_node_constraint_not_negation(self, graph, bmt):
        """Negated node constraint should exclude matching nodes.

        Genes associated with T2D: PPARG(IC=92.3), INSR(IC=88.7), GCK(IC=81.2).
        Constraint: NOT IC > 90 -> keeps INSR and GCK.
        """
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {
                            "categories": ["biolink:Gene"],
                            "constraints": [{
                                "id": "information_content",
                                "name": "IC",
                                "operator": ">",
                                "value": 90,
                                "not": True,
                            }],
                        },
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

        response = lookup(graph, query, bmt=bmt, verbose=False)
        results = response["message"]["results"]

        assert len(results) == 2
        gene_ids = {r["node_bindings"]["n0"][0]["id"] for r in results}
        assert gene_ids == {"NCBIGene:3643", "NCBIGene:2645"}

    def test_empty_constraints_no_filtering(self, graph, bmt):
        """Empty constraints list should not filter anything."""
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["CHEBI:6801"]},
                        "n1": {
                            "categories": ["biolink:Gene"],
                            "constraints": [],
                        },
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

        response = lookup(graph, query, bmt=bmt, verbose=False)
        results = response["message"]["results"]

        # All 4 genes should still be returned
        assert len(results) == 4

    def test_node_constraint_multiple_and_logic(self, graph, bmt):
        """Multiple node constraints use AND logic.

        Metformin affects 4 genes with ICs: PPARG(92.3), INSR(88.7), GCK(81.2), TNF(94.5).
        Constraint IC > 85 AND IC < 93 should keep only PPARG(92.3) and INSR(88.7).
        """
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["CHEBI:6801"]},
                        "n1": {
                            "categories": ["biolink:Gene"],
                            "constraints": [
                                {
                                    "id": "information_content",
                                    "name": "IC",
                                    "operator": ">",
                                    "value": 85,
                                },
                                {
                                    "id": "information_content",
                                    "name": "IC",
                                    "operator": "<",
                                    "value": 93,
                                },
                            ],
                        },
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

        response = lookup(graph, query, bmt=bmt, verbose=False)
        results = response["message"]["results"]

        assert len(results) == 2
        gene_ids = {r["node_bindings"]["n1"][0]["id"] for r in results}
        assert gene_ids == {"NCBIGene:5468", "NCBIGene:3643"}

    def test_node_constraint_filters_all_returns_empty(self, graph, bmt):
        """Constraint that no node satisfies should return empty results."""
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["CHEBI:6801"]},
                        "n1": {
                            "categories": ["biolink:Gene"],
                            "constraints": [{
                                "id": "information_content",
                                "name": "IC",
                                "operator": ">",
                                "value": 999,
                            }],
                        },
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

        response = lookup(graph, query, bmt=bmt, verbose=False)
        results = response["message"]["results"]
        assert len(results) == 0


# ---------------------------------------------------------------------------
# Integration tests: edge attribute_constraints filtering through lookup
# ---------------------------------------------------------------------------

class TestEdgeAttributeConstraintsIntegration:
    """Test edge attribute_constraints in full TRAPI queries."""

    def test_edge_constraint_knowledge_level_matches_all(self, graph, bmt):
        """All edges have knowledge_level=knowledge_assertion.

        Constraining to that value should keep all results unchanged.
        Metformin --affects--> Gene normally returns 4 genes.
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
                            "attribute_constraints": [{
                                "id": "biolink:knowledge_level",
                                "name": "knowledge level",
                                "operator": "==",
                                "value": "knowledge_assertion",
                            }],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, bmt=bmt, verbose=False)
        results = response["message"]["results"]
        assert len(results) == 4

    def test_edge_constraint_knowledge_level_no_match(self, graph, bmt):
        """No edges have knowledge_level=prediction, so all should be filtered out."""
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
                            "attribute_constraints": [{
                                "id": "biolink:knowledge_level",
                                "name": "knowledge level",
                                "operator": "==",
                                "value": "prediction",
                            }],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, bmt=bmt, verbose=False)
        results = response["message"]["results"]
        assert len(results) == 0

    def test_edge_constraint_publications_regex(self, graph, bmt):
        """Filter edges by publication PMID using regex.

        Metformin --affects--> Gene edges have these PMIDs:
          PPARG: PMID:23456789, INSR: PMID:11111111 & PMID:66666666,
          GCK: PMID:22222222, TNF: PMID:33333333

        Constraining publications to match '23456789' should keep only the
        PPARG edge, yielding 1 result.
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
                            "attribute_constraints": [{
                                "id": "biolink:publications",
                                "name": "publications",
                                "operator": "matches",
                                "value": "23456789",
                            }],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, bmt=bmt, verbose=False)
        results = response["message"]["results"]

        assert len(results) == 1
        gene_ids = {r["node_bindings"]["n1"][0]["id"] for r in results}
        assert gene_ids == {"NCBIGene:5468"}

    def test_edge_constraint_not_negation(self, graph, bmt):
        """Negated edge constraint should exclude matching edges.

        NOT knowledge_level == knowledge_assertion → no edges pass → 0 results.
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
                            "attribute_constraints": [{
                                "id": "biolink:knowledge_level",
                                "name": "knowledge level",
                                "operator": "==",
                                "value": "knowledge_assertion",
                                "not": True,
                            }],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, bmt=bmt, verbose=False)
        results = response["message"]["results"]
        assert len(results) == 0

    def test_edge_constraint_nonexistent_attribute(self, graph, bmt):
        """Constraining on an attribute that doesn't exist should filter all edges."""
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
                            "attribute_constraints": [{
                                "id": "biolink:fake_attribute",
                                "name": "fake",
                                "operator": "==",
                                "value": "anything",
                            }],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, bmt=bmt, verbose=False)
        results = response["message"]["results"]
        assert len(results) == 0

    def test_edge_constraint_empty_list_no_filtering(self, graph, bmt):
        """Empty attribute_constraints list should not filter anything."""
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
                            "attribute_constraints": [],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, bmt=bmt, verbose=False)
        results = response["message"]["results"]
        assert len(results) == 4

    def test_edge_and_node_constraints_combined(self, graph, bmt):
        """Both node constraints and edge attribute_constraints applied together.

        Metformin --affects--> Gene:
          Node constraint: IC > 90 → keeps PPARG(92.3) and TNF(94.5)
          Edge constraint: publications matches '23456789' → keeps only PPARG edge

        Combined: only PPARG survives.
        """
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["CHEBI:6801"]},
                        "n1": {
                            "categories": ["biolink:Gene"],
                            "constraints": [{
                                "id": "information_content",
                                "name": "IC",
                                "operator": ">",
                                "value": 90,
                            }],
                        },
                    },
                    "edges": {
                        "e0": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:affects"],
                            "attribute_constraints": [{
                                "id": "biolink:publications",
                                "name": "publications",
                                "operator": "matches",
                                "value": "23456789",
                            }],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, bmt=bmt, verbose=False)
        results = response["message"]["results"]

        assert len(results) == 1
        gene_ids = {r["node_bindings"]["n1"][0]["id"] for r in results}
        assert gene_ids == {"NCBIGene:5468"}

    def test_edge_constraint_backward_search(self, graph, bmt):
        """Edge attribute_constraints work in backward search (start unpinned).

        Gene --gene_associated_with_condition--> T2D:
          PPARG edge: pub PMID:34567890
          INSR edge: pub PMID:45678901
          GCK edge: pub PMID:67890123

        Constraining publications to match '34567890' should keep only PPARG.
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
                            "attribute_constraints": [{
                                "id": "biolink:publications",
                                "name": "publications",
                                "operator": "matches",
                                "value": "34567890",
                            }],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, bmt=bmt, verbose=False)
        results = response["message"]["results"]

        assert len(results) == 1
        assert results[0]["node_bindings"]["n0"][0]["id"] == "NCBIGene:5468"

    def test_edge_constraint_both_pinned(self, graph, bmt):
        """Edge attribute_constraints work when both ends are pinned.

        Metformin --treats--> T2D has 2 edges:
          drugcentral: pub PMID:12345678
          chembl: pub PMID:55555555

        Constraining publications to match '55555555' keeps only the chembl edge.
        The result still exists (1 result) but with only the chembl edge in the KG.
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
                            "attribute_constraints": [{
                                "id": "biolink:publications",
                                "name": "publications",
                                "operator": "matches",
                                "value": "55555555",
                            }],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, bmt=bmt, verbose=False)
        results = response["message"]["results"]

        assert len(results) == 1
        # Only the chembl edge should be in the KG
        kg_edges = response["message"]["knowledge_graph"]["edges"]
        treats_edges = [e for e in kg_edges.values() if e["predicate"] == "biolink:treats"]
        assert len(treats_edges) == 1
        pubs = [a for a in treats_edges[0].get("attributes", [])
                if a.get("original_attribute_name") == "publications"]
        assert len(pubs) == 1
        assert "PMID:55555555" in pubs[0]["value"]


# ---------------------------------------------------------------------------
# Integration tests: numeric edge attribute constraints (p_value, evidence_count)
# ---------------------------------------------------------------------------

class TestNumericEdgeConstraints:
    """Test numeric edge attribute constraints using p_value and evidence_count.

    The affects edges have these values:
        PPARG (unqualified):         p_value=0.001,  evidence_count=12
        INSR (activity/increased):   p_value=0.03,   evidence_count=5
        GCK (activity/decreased):    p_value=0.08,   evidence_count=2
        TNF (abundance/increased):   p_value=0.0005, evidence_count=20
        INSR (abundance/decreased):  p_value=0.04,   evidence_count=3
    """

    def test_p_value_less_than_filters_edges(self, graph, bmt):
        """p_value < 0.01 should keep only PPARG(0.001) and TNF(0.0005)."""
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
                            "attribute_constraints": [{
                                "id": "biolink:p_value",
                                "name": "p-value",
                                "operator": "<",
                                "value": 0.01,
                            }],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, bmt=bmt, verbose=False)
        results = response["message"]["results"]

        assert len(results) == 2
        gene_ids = {r["node_bindings"]["n1"][0]["id"] for r in results}
        assert gene_ids == {"NCBIGene:5468", "NCBIGene:7124"}

    def test_evidence_count_greater_than(self, graph, bmt):
        """evidence_count > 10 should keep PPARG(12) and TNF(20)."""
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
                            "attribute_constraints": [{
                                "id": "biolink:evidence_count",
                                "name": "evidence count",
                                "operator": ">",
                                "value": 10,
                            }],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, bmt=bmt, verbose=False)
        results = response["message"]["results"]

        assert len(results) == 2
        gene_ids = {r["node_bindings"]["n1"][0]["id"] for r in results}
        assert gene_ids == {"NCBIGene:5468", "NCBIGene:7124"}

    def test_p_value_and_evidence_count_combined(self, graph, bmt):
        """p_value < 0.05 AND evidence_count > 4 should keep PPARG and TNF.

        Edges passing p_value < 0.05:
            PPARG(0.001), INSR-activity(0.03), TNF(0.0005), INSR-abundance(0.04)
        Of those, evidence_count > 4:
            PPARG(12), INSR-activity(5), TNF(20)
        Both: PPARG, INSR (via activity edge), TNF → 3 gene results.
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
                            "attribute_constraints": [
                                {
                                    "id": "biolink:p_value",
                                    "name": "p-value",
                                    "operator": "<",
                                    "value": 0.05,
                                },
                                {
                                    "id": "biolink:evidence_count",
                                    "name": "evidence count",
                                    "operator": ">",
                                    "value": 4,
                                },
                            ],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, bmt=bmt, verbose=False)
        results = response["message"]["results"]

        assert len(results) == 3
        gene_ids = {r["node_bindings"]["n1"][0]["id"] for r in results}
        assert gene_ids == {"NCBIGene:5468", "NCBIGene:3643", "NCBIGene:7124"}

    def test_very_strict_p_value_filters_all(self, graph, bmt):
        """p_value < 0.0001 should filter all edges (lowest is 0.0005)."""
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
                            "attribute_constraints": [{
                                "id": "biolink:p_value",
                                "name": "p-value",
                                "operator": "<",
                                "value": 0.0001,
                            }],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, bmt=bmt, verbose=False)
        results = response["message"]["results"]
        assert len(results) == 0

    def test_evidence_count_equals(self, graph, bmt):
        """evidence_count == 20 should match only the TNF edge."""
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
                            "attribute_constraints": [{
                                "id": "biolink:evidence_count",
                                "name": "evidence count",
                                "operator": "==",
                                "value": 20,
                            }],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, bmt=bmt, verbose=False)
        results = response["message"]["results"]

        assert len(results) == 1
        assert results[0]["node_bindings"]["n1"][0]["id"] == "NCBIGene:7124"

    def test_p_value_backward_search(self, graph, bmt):
        """Numeric edge constraints work in backward search.

        Gene --gene_associated_with_condition--> T2D edges:
            PPARG: p_value=0.002, evidence_count=15
            INSR:  p_value=0.07,  evidence_count=4
            GCK:   p_value=0.15,  evidence_count=1

        p_value < 0.05 should keep only PPARG.
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
                            "attribute_constraints": [{
                                "id": "biolink:p_value",
                                "name": "p-value",
                                "operator": "<",
                                "value": 0.05,
                            }],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, bmt=bmt, verbose=False)
        results = response["message"]["results"]

        assert len(results) == 1
        assert results[0]["node_bindings"]["n0"][0]["id"] == "NCBIGene:5468"

    def test_evidence_count_two_hop_with_edge_constraint(self, graph, bmt):
        """Edge constraints on a two-hop query filter the constrained hop only.

        Metformin --affects--> Gene --gene_associated_with_condition--> T2D
        Constrain the second hop: evidence_count > 10 → only PPARG(15) passes.
        So only paths through PPARG survive.
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
                            "attribute_constraints": [{
                                "id": "biolink:evidence_count",
                                "name": "evidence count",
                                "operator": ">",
                                "value": 10,
                            }],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, bmt=bmt, verbose=False)
        results = response["message"]["results"]

        assert len(results) == 1
        assert results[0]["node_bindings"]["n1"][0]["id"] == "NCBIGene:5468"

    def test_node_and_numeric_edge_constraints_combined(self, graph, bmt):
        """Combine node IC constraint with numeric edge constraint.

        Metformin --affects--> Gene:
            Node IC > 90 keeps: PPARG(92.3), TNF(94.5)
            Edge p_value < 0.001 keeps: TNF(0.0005)
            Combined: only TNF survives.
        """
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["CHEBI:6801"]},
                        "n1": {
                            "categories": ["biolink:Gene"],
                            "constraints": [{
                                "id": "information_content",
                                "name": "IC",
                                "operator": ">",
                                "value": 90,
                            }],
                        },
                    },
                    "edges": {
                        "e0": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:affects"],
                            "attribute_constraints": [{
                                "id": "biolink:p_value",
                                "name": "p-value",
                                "operator": "<",
                                "value": 0.001,
                            }],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, bmt=bmt, verbose=False)
        results = response["message"]["results"]

        assert len(results) == 1
        assert results[0]["node_bindings"]["n1"][0]["id"] == "NCBIGene:7124"
