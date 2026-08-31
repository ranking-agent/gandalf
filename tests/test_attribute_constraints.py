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
        constraints = [
            {
                "id": "biolink:p_value",
                "name": "p-value",
                "operator": "<",
                "value": 0.05,
            }
        ]
        assert matches_attribute_constraints([], constraints) is False
        assert matches_attribute_constraints(None, constraints) is False

    def test_equals_operator(self):
        attrs = [
            {
                "attribute_type_id": "biolink:knowledge_level",
                "value": "knowledge_assertion",
            }
        ]
        constraint_match = [
            {
                "id": "biolink:knowledge_level",
                "name": "knowledge level",
                "operator": "==",
                "value": "knowledge_assertion",
            }
        ]
        constraint_no_match = [
            {
                "id": "biolink:knowledge_level",
                "name": "knowledge level",
                "operator": "==",
                "value": "prediction",
            }
        ]
        assert matches_attribute_constraints(attrs, constraint_match) is True
        assert matches_attribute_constraints(attrs, constraint_no_match) is False

    def test_greater_than_operator(self):
        attrs = [{"attribute_type_id": "biolink:p_value", "value": 0.03}]
        assert (
            matches_attribute_constraints(
                attrs,
                [
                    {
                        "id": "biolink:p_value",
                        "name": "p",
                        "operator": ">",
                        "value": 0.01,
                    }
                ],
            )
            is True
        )
        assert (
            matches_attribute_constraints(
                attrs,
                [
                    {
                        "id": "biolink:p_value",
                        "name": "p",
                        "operator": ">",
                        "value": 0.05,
                    }
                ],
            )
            is False
        )

    def test_less_than_operator(self):
        attrs = [{"attribute_type_id": "biolink:p_value", "value": 0.03}]
        assert (
            matches_attribute_constraints(
                attrs,
                [
                    {
                        "id": "biolink:p_value",
                        "name": "p",
                        "operator": "<",
                        "value": 0.05,
                    }
                ],
            )
            is True
        )
        assert (
            matches_attribute_constraints(
                attrs,
                [
                    {
                        "id": "biolink:p_value",
                        "name": "p",
                        "operator": "<",
                        "value": 0.01,
                    }
                ],
            )
            is False
        )

    def test_matches_operator_regex(self):
        attrs = [
            {
                "attribute_type_id": "biolink:description",
                "value": "Metformin treats diabetes",
            }
        ]
        assert (
            matches_attribute_constraints(
                attrs,
                [
                    {
                        "id": "biolink:description",
                        "name": "desc",
                        "operator": "matches",
                        "value": "treats.*diabetes",
                    }
                ],
            )
            is True
        )
        assert (
            matches_attribute_constraints(
                attrs,
                [
                    {
                        "id": "biolink:description",
                        "name": "desc",
                        "operator": "matches",
                        "value": "^prevents",
                    }
                ],
            )
            is False
        )

    def test_strict_equals_operator(self):
        attrs = [{"attribute_type_id": "biolink:score", "value": 42}]
        # Same type and value
        assert (
            matches_attribute_constraints(
                attrs,
                [
                    {
                        "id": "biolink:score",
                        "name": "score",
                        "operator": "===",
                        "value": 42,
                    }
                ],
            )
            is True
        )
        # Different type (float vs int)
        assert (
            matches_attribute_constraints(
                attrs,
                [
                    {
                        "id": "biolink:score",
                        "name": "score",
                        "operator": "===",
                        "value": 42.0,
                    }
                ],
            )
            is False
        )

    def test_strict_equals_list_order(self):
        attrs = [{"attribute_type_id": "biolink:tags", "value": ["a", "b", "c"]}]
        assert (
            matches_attribute_constraints(
                attrs,
                [
                    {
                        "id": "biolink:tags",
                        "name": "tags",
                        "operator": "===",
                        "value": ["a", "b", "c"],
                    }
                ],
            )
            is True
        )
        assert (
            matches_attribute_constraints(
                attrs,
                [
                    {
                        "id": "biolink:tags",
                        "name": "tags",
                        "operator": "===",
                        "value": ["c", "b", "a"],
                    }
                ],
            )
            is False
        )

    def test_equals_list_value_membership(self):
        # Per TRAPI, a list constraint value means OR / membership.
        attrs = [
            {
                "attribute_type_id": "biolink:agent_type",
                "value": "data_analysis_pipeline",
            }
        ]
        in_list = [
            {
                "id": "biolink:agent_type",
                "name": "agent_type",
                "operator": "==",
                "value": ["data_analysis_pipeline", "text_mining_agent"],
            }
        ]
        not_in_list = [
            {
                "id": "biolink:agent_type",
                "name": "agent_type",
                "operator": "==",
                "value": ["manual_agent", "automated_agent"],
            }
        ]
        assert matches_attribute_constraints(attrs, in_list) is True
        assert matches_attribute_constraints(attrs, not_in_list) is False

    def test_equals_list_value_negated(self):
        # Exact scenario from issue #22: NOT (agent_type == one of [...]).
        constraint = [
            {
                "id": "biolink:agent_type",
                "name": "agent_type",
                "operator": "==",
                "not": True,
                "value": ["data_analysis_pipeline", "text_mining_agent"],
            }
        ]
        # Value in the excluded list -> rejected.
        excluded = [
            {
                "attribute_type_id": "biolink:agent_type",
                "value": "text_mining_agent",
            }
        ]
        assert matches_attribute_constraints(excluded, constraint) is False
        # Value outside the excluded list -> passes.
        kept = [
            {
                "attribute_type_id": "biolink:agent_type",
                "value": "manual_agent",
            }
        ]
        assert matches_attribute_constraints(kept, constraint) is True

    def test_matches_list_value_membership(self):
        # `matches` with a list of patterns: OR semantics.
        attrs = [
            {
                "attribute_type_id": "biolink:description",
                "value": "Metformin treats diabetes",
            }
        ]
        any_match = [
            {
                "id": "biolink:description",
                "name": "desc",
                "operator": "matches",
                "value": ["^prevents", "treats.*diabetes"],
            }
        ]
        none_match = [
            {
                "id": "biolink:description",
                "name": "desc",
                "operator": "matches",
                "value": ["^prevents", "causes.*cancer"],
            }
        ]
        assert matches_attribute_constraints(attrs, any_match) is True
        assert matches_attribute_constraints(attrs, none_match) is False

    def test_not_negation(self):
        attrs = [
            {"attribute_type_id": "biolink:knowledge_level", "value": "prediction"}
        ]
        # "not prediction" should pass for "prediction" -> negated match -> False
        assert (
            matches_attribute_constraints(
                attrs,
                [
                    {
                        "id": "biolink:knowledge_level",
                        "name": "kl",
                        "operator": "==",
                        "value": "prediction",
                        "not": True,
                    }
                ],
            )
            is False
        )
        # "not knowledge_assertion" should pass for "prediction" -> no match -> negated -> True
        assert (
            matches_attribute_constraints(
                attrs,
                [
                    {
                        "id": "biolink:knowledge_level",
                        "name": "kl",
                        "operator": "==",
                        "value": "knowledge_assertion",
                        "not": True,
                    }
                ],
            )
            is True
        )

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
        assert (
            matches_attribute_constraints(
                attrs,
                [
                    {
                        "id": "biolink:p_value",
                        "name": "p",
                        "operator": "<",
                        "value": 0.05,
                    }
                ],
            )
            is False
        )

    def test_numeric_comparison_with_list_value_or_logic(self):
        """Per TRAPI spec: with lists and > or <, at least one must be true (OR)."""
        attrs = [{"attribute_type_id": "biolink:score", "value": 5}]
        assert (
            matches_attribute_constraints(
                attrs,
                [
                    {
                        "id": "biolink:score",
                        "name": "s",
                        "operator": ">",
                        "value": [10, 3],
                    }
                ],
            )
            is True
        )  # 5 > 3 is true
        assert (
            matches_attribute_constraints(
                attrs,
                [
                    {
                        "id": "biolink:score",
                        "name": "s",
                        "operator": ">",
                        "value": [10, 20],
                    }
                ],
            )
            is False
        )  # neither

    def test_match_by_original_attribute_name(self):
        """Constraints should also match on original_attribute_name."""
        attrs = [
            {
                "attribute_type_id": "biolink:Attribute",
                "original_attribute_name": "information_content",
                "value": 92.3,
            }
        ]
        assert (
            matches_attribute_constraints(
                attrs,
                [
                    {
                        "id": "information_content",
                        "name": "IC",
                        "operator": ">",
                        "value": 90,
                    }
                ],
            )
            is True
        )

    def test_no_attributes_all_negated_returns_true(self):
        """If all constraints are negated and there are no attributes, they all pass."""
        constraints = [
            {
                "id": "biolink:p_value",
                "name": "p",
                "operator": "<",
                "value": 0.05,
                "not": True,
            }
        ]
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
                            "constraints": [
                                {
                                    "id": "biolink:information_content",
                                    "name": "information content",
                                    "operator": ">",
                                    "value": 90,
                                }
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

        response = lookup(graph, query, bmt=bmt)
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
                            "constraints": [
                                {
                                    "id": "biolink:information_content",
                                    "name": "information content",
                                    "operator": "<",
                                    "value": 85,
                                }
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

        response = lookup(graph, query, bmt=bmt)
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
                            "constraints": [
                                {
                                    "id": "biolink:information_content",
                                    "name": "IC",
                                    "operator": ">",
                                    "value": 90,
                                    "not": True,
                                }
                            ],
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

        response = lookup(graph, query, bmt=bmt)
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

        response = lookup(graph, query, bmt=bmt)
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
                                    "id": "biolink:information_content",
                                    "name": "IC",
                                    "operator": ">",
                                    "value": 85,
                                },
                                {
                                    "id": "biolink:information_content",
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

        response = lookup(graph, query, bmt=bmt)
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
                            "constraints": [
                                {
                                    "id": "biolink:information_content",
                                    "name": "IC",
                                    "operator": ">",
                                    "value": 999,
                                }
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

        response = lookup(graph, query, bmt=bmt)
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
                            "attribute_constraints": [
                                {
                                    "id": "biolink:knowledge_level",
                                    "name": "knowledge level",
                                    "operator": "==",
                                    "value": "knowledge_assertion",
                                }
                            ],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, bmt=bmt)
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
                            "attribute_constraints": [
                                {
                                    "id": "biolink:knowledge_level",
                                    "name": "knowledge level",
                                    "operator": "==",
                                    "value": "prediction",
                                }
                            ],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, bmt=bmt)
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
                            "attribute_constraints": [
                                {
                                    "id": "biolink:publications",
                                    "name": "publications",
                                    "operator": "matches",
                                    "value": "23456789",
                                }
                            ],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, bmt=bmt)
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
                            "attribute_constraints": [
                                {
                                    "id": "biolink:knowledge_level",
                                    "name": "knowledge level",
                                    "operator": "==",
                                    "value": "knowledge_assertion",
                                    "not": True,
                                }
                            ],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, bmt=bmt)
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
                            "attribute_constraints": [
                                {
                                    "id": "biolink:fake_attribute",
                                    "name": "fake",
                                    "operator": "==",
                                    "value": "anything",
                                }
                            ],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, bmt=bmt)
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

        response = lookup(graph, query, bmt=bmt)
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
                            "constraints": [
                                {
                                    "id": "biolink:information_content",
                                    "name": "IC",
                                    "operator": ">",
                                    "value": 90,
                                }
                            ],
                        },
                    },
                    "edges": {
                        "e0": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:affects"],
                            "attribute_constraints": [
                                {
                                    "id": "biolink:publications",
                                    "name": "publications",
                                    "operator": "matches",
                                    "value": "23456789",
                                }
                            ],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, bmt=bmt)
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
                            "attribute_constraints": [
                                {
                                    "id": "biolink:publications",
                                    "name": "publications",
                                    "operator": "matches",
                                    "value": "34567890",
                                }
                            ],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, bmt=bmt)
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
                            "attribute_constraints": [
                                {
                                    "id": "biolink:publications",
                                    "name": "publications",
                                    "operator": "matches",
                                    "value": "55555555",
                                }
                            ],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, bmt=bmt)
        results = response["message"]["results"]

        assert len(results) == 1
        # Only the chembl edge should be in the KG
        kg_edges = response["message"]["knowledge_graph"]["edges"]
        treats_edges = [
            e for e in kg_edges.values() if e["predicate"] == "biolink:treats"
        ]
        assert len(treats_edges) == 1
        pubs = [
            a
            for a in treats_edges[0].get("attributes", [])
            if a.get("attribute_type_id") == "biolink:publications"
        ]
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
                            "attribute_constraints": [
                                {
                                    "id": "biolink:p_value",
                                    "name": "p-value",
                                    "operator": "<",
                                    "value": 0.01,
                                }
                            ],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, bmt=bmt)
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
                            "attribute_constraints": [
                                {
                                    "id": "biolink:evidence_count",
                                    "name": "evidence count",
                                    "operator": ">",
                                    "value": 10,
                                }
                            ],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, bmt=bmt)
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

        response = lookup(graph, query, bmt=bmt)
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
                            "attribute_constraints": [
                                {
                                    "id": "biolink:p_value",
                                    "name": "p-value",
                                    "operator": "<",
                                    "value": 0.0001,
                                }
                            ],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, bmt=bmt)
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
                            "attribute_constraints": [
                                {
                                    "id": "biolink:evidence_count",
                                    "name": "evidence count",
                                    "operator": "==",
                                    "value": 20,
                                }
                            ],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, bmt=bmt)
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
                            "attribute_constraints": [
                                {
                                    "id": "biolink:p_value",
                                    "name": "p-value",
                                    "operator": "<",
                                    "value": 0.05,
                                }
                            ],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, bmt=bmt)
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
                            "attribute_constraints": [
                                {
                                    "id": "biolink:evidence_count",
                                    "name": "evidence count",
                                    "operator": ">",
                                    "value": 10,
                                }
                            ],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, bmt=bmt)
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
                            "constraints": [
                                {
                                    "id": "biolink:information_content",
                                    "name": "IC",
                                    "operator": ">",
                                    "value": 90,
                                }
                            ],
                        },
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
                                    "value": 0.001,
                                }
                            ],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, bmt=bmt)
        results = response["message"]["results"]

        assert len(results) == 1
        assert results[0]["node_bindings"]["n1"][0]["id"] == "NCBIGene:7124"


# ---------------------------------------------------------------------------
# Unit tests: list-valued attributes and PubMed ID filtering
# ---------------------------------------------------------------------------


def _publications(*pmids):
    """Build a publications attribute holding the given identifiers."""
    return [{"attribute_type_id": "biolink:publications", "value": list(pmids)}]


def _pub_constraint(value, operator="==", negated=False):
    """Build a publications attribute constraint."""
    constraint = {
        "id": "biolink:publications",
        "name": "publications",
        "operator": operator,
        "value": value,
    }
    if negated:
        constraint["not"] = True
    return [constraint]


class TestListValuedAttributes:
    """A list-valued attribute is satisfied when ANY member satisfies the operator."""

    @pytest.mark.parametrize(
        "constraint_value,expected",
        [
            ("PMID:11111111", True),  # first member
            ("PMID:66666666", True),  # later member
            ("PMID:99999999", False),  # absent
            (["PMID:99999999", "PMID:66666666"], True),  # any-of, one present
            (["PMID:99999999", "PMID:88888888"], False),  # any-of, none present
            ([], False),  # empty any-of matches nothing
        ],
    )
    def test_equals_is_containment(self, constraint_value, expected):
        attrs = _publications("PMID:11111111", "PMID:66666666")
        assert (
            matches_attribute_constraints(attrs, _pub_constraint(constraint_value))
            is expected
        )

    def test_negated_equals_excludes_edges_citing_the_pmid(self):
        constraint = _pub_constraint("PMID:11111111", negated=True)
        assert (
            matches_attribute_constraints(_publications("PMID:11111111"), constraint)
            is False
        )
        assert (
            matches_attribute_constraints(_publications("PMID:22222222"), constraint)
            is True
        )

    def test_strict_equals_still_compares_the_whole_list(self):
        attrs = _publications("PMID:11111111", "PMID:66666666")
        # `===` keeps its exact type/value/order semantics: no containment.
        assert (
            matches_attribute_constraints(
                attrs, _pub_constraint("PMID:11111111", operator="===")
            )
            is False
        )
        assert (
            matches_attribute_constraints(
                attrs,
                _pub_constraint(["PMID:11111111", "PMID:66666666"], operator="==="),
            )
            is True
        )
        assert (
            matches_attribute_constraints(
                attrs,
                _pub_constraint(["PMID:66666666", "PMID:11111111"], operator="==="),
            )
            is False
        )

    def test_matches_applies_per_member(self):
        attrs = _publications("PMID:11111111", "PMID:66666666")
        assert (
            matches_attribute_constraints(
                attrs, _pub_constraint("66666666", operator="matches")
            )
            is True
        )
        # A pattern spanning two members must not match: each is tested alone.
        assert (
            matches_attribute_constraints(
                attrs, _pub_constraint("11111111.*66666666", operator="matches")
            )
            is False
        )

    def test_numeric_operators_apply_per_member(self):
        attrs = [{"attribute_type_id": "biolink:score", "value": [0.2, 0.9]}]
        assert (
            matches_attribute_constraints(
                attrs,
                [{"id": "biolink:score", "name": "s", "operator": ">", "value": 0.8}],
            )
            is True
        )
        assert (
            matches_attribute_constraints(
                attrs,
                [{"id": "biolink:score", "name": "s", "operator": ">", "value": 1.5}],
            )
            is False
        )

    def test_empty_publications_list_never_matches(self):
        assert (
            matches_attribute_constraints(
                _publications(), _pub_constraint("PMID:11111111")
            )
            is False
        )


class TestPubMedIdentifierForms:
    """PubMed IDs compare canonically across the spellings sources use."""

    @pytest.mark.parametrize(
        "constraint_value",
        [
            "PMID:23456789",
            "pmid:23456789",
            "PubMed:23456789",
            "pubmed:23456789",
            "23456789",
            23456789,
            "https://pubmed.ncbi.nlm.nih.gov/23456789",
            "https://pubmed.ncbi.nlm.nih.gov/23456789/",
            "http://www.ncbi.nlm.nih.gov/pubmed/23456789",
        ],
    )
    def test_constraint_forms_match_a_pmid_curie(self, constraint_value):
        attrs = _publications("PMID:23456789")
        assert (
            matches_attribute_constraints(attrs, _pub_constraint(constraint_value))
            is True
        )

    @pytest.mark.parametrize(
        "stored_value",
        [
            "PMID:23456789",
            "pubmed:23456789",
            "https://pubmed.ncbi.nlm.nih.gov/23456789",
            "http://www.ncbi.nlm.nih.gov/pubmed/23456789",
        ],
    )
    def test_stored_forms_match_a_bare_accession(self, stored_value):
        attrs = _publications(stored_value)
        assert matches_attribute_constraints(attrs, _pub_constraint("23456789")) is True

    @pytest.mark.parametrize(
        "constraint_value",
        ["PMID:2345678", "PMID:234567890", "PMC:23456789", "pubmed:99999999"],
    )
    def test_near_misses_do_not_match(self, constraint_value):
        """Canonicalisation must not turn a distinct ID into a match.

        A substring regex would accept ``PMID:2345678`` against
        ``PMID:23456789``; equality does not.
        """
        attrs = _publications("PMID:23456789")
        assert (
            matches_attribute_constraints(attrs, _pub_constraint(constraint_value))
            is False
        )

    @pytest.mark.parametrize(
        "attr_value,constraint_value,expected",
        [
            (12, 12, True),
            (12, "12", False),  # a bare number is not a PubMed ID on its own
            ("12", 12, False),
            (0.05, "0.05", False),
        ],
    )
    def test_non_publication_values_are_unaffected(
        self, attr_value, constraint_value, expected
    ):
        """Only an explicit PubMed identifier opts a comparison into
        canonicalisation, so ordinary attributes keep strict equality."""
        attrs = [{"attribute_type_id": "biolink:evidence_count", "value": attr_value}]
        constraint = [
            {
                "id": "biolink:evidence_count",
                "name": "evidence",
                "operator": "==",
                "value": constraint_value,
            }
        ]
        assert matches_attribute_constraints(attrs, constraint) is expected


# ---------------------------------------------------------------------------
# Integration tests: filtering edges on specific PubMed IDs
# ---------------------------------------------------------------------------


def _pmid_query(subject, obj, predicate, value, negated=False, operator="=="):
    """Build a one-hop TRAPI query with a publications constraint on the edge."""
    return {
        "message": {
            "query_graph": {
                "nodes": {"n0": subject, "n1": obj},
                "edges": {
                    "e0": {
                        "subject": "n0",
                        "object": "n1",
                        "predicates": [predicate],
                        "attribute_constraints": _pub_constraint(
                            value, operator=operator, negated=negated
                        ),
                    },
                },
            },
        },
    }


class TestPubMedEdgeFiltering:
    """Filter a query's edges down to specific PubMed IDs.

    The ``CHEBI:6801 --affects--> Gene`` fixture edges cite one PMID each:
    PPARG (NCBIGene:5468) PMID:23456789, INSR (NCBIGene:3643) PMID:11111111
    and PMID:66666666 on a second edge, GCK (NCBIGene:2645) PMID:22222222,
    TNF (NCBIGene:7124) PMID:33333333.
    """

    def test_single_pmid_keeps_only_that_edge(self, graph, bmt):
        query = _pmid_query(
            {"ids": ["CHEBI:6801"]},
            {"categories": ["biolink:Gene"]},
            "biolink:affects",
            "PMID:23456789",
        )
        results = lookup(graph, query, bmt=bmt)["message"]["results"]

        assert len(results) == 1
        assert results[0]["node_bindings"]["n1"][0]["id"] == "NCBIGene:5468"

    def test_several_pmids_keep_every_cited_edge(self, graph, bmt):
        query = _pmid_query(
            {"ids": ["CHEBI:6801"]},
            {"categories": ["biolink:Gene"]},
            "biolink:affects",
            ["PMID:23456789", "PMID:33333333"],
        )
        results = lookup(graph, query, bmt=bmt)["message"]["results"]

        gene_ids = {r["node_bindings"]["n1"][0]["id"] for r in results}
        assert gene_ids == {"NCBIGene:5468", "NCBIGene:7124"}

    def test_bare_accession_is_accepted(self, graph, bmt):
        query = _pmid_query(
            {"ids": ["CHEBI:6801"]},
            {"categories": ["biolink:Gene"]},
            "biolink:affects",
            "23456789",
        )
        results = lookup(graph, query, bmt=bmt)["message"]["results"]

        assert len(results) == 1
        assert results[0]["node_bindings"]["n1"][0]["id"] == "NCBIGene:5468"

    def test_unknown_pmid_yields_no_results(self, graph, bmt):
        query = _pmid_query(
            {"ids": ["CHEBI:6801"]},
            {"categories": ["biolink:Gene"]},
            "biolink:affects",
            "PMID:99999999",
        )
        assert lookup(graph, query, bmt=bmt)["message"]["results"] == []

    def test_negated_pmid_excludes_that_edge(self, graph, bmt):
        query = _pmid_query(
            {"ids": ["CHEBI:6801"]},
            {"categories": ["biolink:Gene"]},
            "biolink:affects",
            "PMID:23456789",
            negated=True,
        )
        results = lookup(graph, query, bmt=bmt)["message"]["results"]

        gene_ids = {r["node_bindings"]["n1"][0]["id"] for r in results}
        assert "NCBIGene:5468" not in gene_ids
        assert gene_ids == {"NCBIGene:3643", "NCBIGene:2645", "NCBIGene:7124"}

    def test_backward_search_filters_on_pmid(self, graph, bmt):
        """Start node unpinned: the constraint runs in the backward direction."""
        query = _pmid_query(
            {"categories": ["biolink:Gene"]},
            {"ids": ["MONDO:0005148"]},
            "biolink:gene_associated_with_condition",
            "PMID:34567890",
        )
        results = lookup(graph, query, bmt=bmt)["message"]["results"]

        assert len(results) == 1
        assert results[0]["node_bindings"]["n0"][0]["id"] == "NCBIGene:5468"

    def test_both_pinned_selects_one_of_two_parallel_edges(self, graph, bmt):
        """Metformin --treats--> T2D has a drugcentral edge (PMID:12345678)
        and a chembl edge (PMID:55555555); the PMID picks one."""
        query = _pmid_query(
            {"ids": ["CHEBI:6801"]},
            {"ids": ["MONDO:0005148"]},
            "biolink:treats",
            "PMID:55555555",
        )
        response = lookup(graph, query, bmt=bmt)

        assert len(response["message"]["results"]) == 1
        treats_edges = [
            e
            for e in response["message"]["knowledge_graph"]["edges"].values()
            if e["predicate"] == "biolink:treats"
        ]
        assert len(treats_edges) == 1
        pubs = [
            a["value"]
            for a in treats_edges[0]["attributes"]
            if a["attribute_type_id"] == "biolink:publications"
        ]
        assert pubs == [["PMID:55555555"]]

    def test_two_hop_query_filters_each_hop_on_its_pmid(self, graph, bmt):
        """A PMID constraint on each hop of Metformin -> Gene -> T2D."""
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
                            "attribute_constraints": _pub_constraint("PMID:23456789"),
                        },
                        "e1": {
                            "subject": "n1",
                            "object": "n2",
                            "predicates": ["biolink:gene_associated_with_condition"],
                            "attribute_constraints": _pub_constraint("PMID:34567890"),
                        },
                    },
                },
            },
        }
        results = lookup(graph, query, bmt=bmt)["message"]["results"]

        assert len(results) == 1
        assert results[0]["node_bindings"]["n1"][0]["id"] == "NCBIGene:5468"

    def test_two_hop_query_with_conflicting_pmids_yields_nothing(self, graph, bmt):
        """The second hop's PMID belongs to a different gene's edge."""
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
                            "attribute_constraints": _pub_constraint("PMID:23456789"),
                        },
                        "e1": {
                            "subject": "n1",
                            "object": "n2",
                            "predicates": ["biolink:gene_associated_with_condition"],
                            # PMID:45678901 is on the NCBIGene:3643 edge.
                            "attribute_constraints": _pub_constraint("PMID:45678901"),
                        },
                    },
                },
            },
        }
        assert lookup(graph, query, bmt=bmt)["message"]["results"] == []
