"""Tests for qualifier constraint matching and expansion."""

from tests.search_fixtures import graph  # noqa: F401

from gandalf.search import lookup
from gandalf.search.qualifiers import edge_matches_qualifier_constraints
from gandalf.search.expanders import QualifierExpander


class TestQualifierConstraintMatching:
    """Unit tests for the edge_matches_qualifier_constraints helper function."""

    def test_no_constraints_returns_true(self):
        """No qualifier constraints should match any edge."""
        edge_qualifiers = [
            {"qualifier_type_id": "biolink:object_aspect_qualifier", "qualifier_value": "activity"}
        ]
        assert edge_matches_qualifier_constraints(edge_qualifiers, None) is True
        assert edge_matches_qualifier_constraints(edge_qualifiers, []) is True

    def test_empty_qualifier_set_matches_any_edge(self):
        """Empty qualifier_set should match any edge."""
        edge_qualifiers = [
            {"qualifier_type_id": "biolink:object_aspect_qualifier", "qualifier_value": "activity"}
        ]
        constraints = [{"qualifier_set": []}]
        assert edge_matches_qualifier_constraints(edge_qualifiers, constraints) is True

    def test_single_qualifier_match(self):
        """Edge with matching single qualifier should match."""
        edge_qualifiers = [
            {"qualifier_type_id": "biolink:object_aspect_qualifier", "qualifier_value": "activity"}
        ]
        constraints = [
            {
                "qualifier_set": [
                    {"qualifier_type_id": "biolink:object_aspect_qualifier", "qualifier_value": "activity"}
                ]
            }
        ]
        assert edge_matches_qualifier_constraints(edge_qualifiers, constraints) is True

    def test_single_qualifier_no_match(self):
        """Edge with non-matching qualifier should not match."""
        edge_qualifiers = [
            {"qualifier_type_id": "biolink:object_aspect_qualifier", "qualifier_value": "activity"}
        ]
        constraints = [
            {
                "qualifier_set": [
                    {"qualifier_type_id": "biolink:object_aspect_qualifier", "qualifier_value": "abundance"}
                ]
            }
        ]
        assert edge_matches_qualifier_constraints(edge_qualifiers, constraints) is False

    def test_multiple_qualifiers_all_match(self):
        """Edge with all required qualifiers should match (AND semantics within set)."""
        edge_qualifiers = [
            {"qualifier_type_id": "biolink:object_aspect_qualifier", "qualifier_value": "activity"},
            {"qualifier_type_id": "biolink:object_direction_qualifier", "qualifier_value": "increased"},
        ]
        constraints = [
            {
                "qualifier_set": [
                    {"qualifier_type_id": "biolink:object_aspect_qualifier", "qualifier_value": "activity"},
                    {"qualifier_type_id": "biolink:object_direction_qualifier", "qualifier_value": "increased"},
                ]
            }
        ]
        assert edge_matches_qualifier_constraints(edge_qualifiers, constraints) is True

    def test_multiple_qualifiers_partial_match(self):
        """Edge with only some required qualifiers should not match."""
        edge_qualifiers = [
            {"qualifier_type_id": "biolink:object_aspect_qualifier", "qualifier_value": "activity"},
        ]
        constraints = [
            {
                "qualifier_set": [
                    {"qualifier_type_id": "biolink:object_aspect_qualifier", "qualifier_value": "activity"},
                    {"qualifier_type_id": "biolink:object_direction_qualifier", "qualifier_value": "increased"},
                ]
            }
        ]
        assert edge_matches_qualifier_constraints(edge_qualifiers, constraints) is False

    def test_or_semantics_between_qualifier_sets(self):
        """Edge matching any qualifier_set should match (OR semantics between sets)."""
        edge_qualifiers = [
            {"qualifier_type_id": "biolink:object_aspect_qualifier", "qualifier_value": "abundance"},
        ]
        constraints = [
            {
                "qualifier_set": [
                    {"qualifier_type_id": "biolink:object_aspect_qualifier", "qualifier_value": "activity"},
                ]
            },
            {
                "qualifier_set": [
                    {"qualifier_type_id": "biolink:object_aspect_qualifier", "qualifier_value": "abundance"},
                ]
            },
        ]
        assert edge_matches_qualifier_constraints(edge_qualifiers, constraints) is True

    def test_edge_with_no_qualifiers(self):
        """Edge with no qualifiers should not match constraints requiring qualifiers."""
        edge_qualifiers = []
        constraints = [
            {
                "qualifier_set": [
                    {"qualifier_type_id": "biolink:object_aspect_qualifier", "qualifier_value": "activity"},
                ]
            }
        ]
        assert edge_matches_qualifier_constraints(edge_qualifiers, constraints) is False

    def test_edge_with_extra_qualifiers_still_matches(self):
        """Edge with extra qualifiers beyond required should still match."""
        edge_qualifiers = [
            {"qualifier_type_id": "biolink:object_aspect_qualifier", "qualifier_value": "activity"},
            {"qualifier_type_id": "biolink:object_direction_qualifier", "qualifier_value": "increased"},
            {"qualifier_type_id": "biolink:qualified_predicate", "qualifier_value": "biolink:causes"},
        ]
        constraints = [
            {
                "qualifier_set": [
                    {"qualifier_type_id": "biolink:object_aspect_qualifier", "qualifier_value": "activity"},
                ]
            }
        ]
        assert edge_matches_qualifier_constraints(edge_qualifiers, constraints) is True

    def test_expanded_format_single_value_match(self):
        """Expanded format with qualifier_values (plural) should match if edge has any value."""
        edge_qualifiers = [
            {"qualifier_type_id": "biolink:object_aspect_qualifier", "qualifier_value": "activity"}
        ]
        # Expanded format: qualifier_values (plural) with list of acceptable values
        constraints = [
            {
                "qualifier_set": [
                    {
                        "qualifier_type_id": "biolink:object_aspect_qualifier",
                        "qualifier_values": ["activity", "abundance"],  # Edge has "activity"
                    }
                ]
            }
        ]
        assert edge_matches_qualifier_constraints(edge_qualifiers, constraints) is True

    def test_expanded_format_no_match(self):
        """Expanded format should not match if edge value is not in the list."""
        edge_qualifiers = [
            {"qualifier_type_id": "biolink:object_aspect_qualifier", "qualifier_value": "expression"}
        ]
        constraints = [
            {
                "qualifier_set": [
                    {
                        "qualifier_type_id": "biolink:object_aspect_qualifier",
                        "qualifier_values": ["activity", "abundance"],  # "expression" not in list
                    }
                ]
            }
        ]
        assert edge_matches_qualifier_constraints(edge_qualifiers, constraints) is False

    def test_expanded_format_multiple_types_all_match(self):
        """Expanded format with multiple qualifier types - all must match (AND semantics)."""
        edge_qualifiers = [
            {"qualifier_type_id": "biolink:object_aspect_qualifier", "qualifier_value": "activity"},
            {"qualifier_type_id": "biolink:object_direction_qualifier", "qualifier_value": "increased"},
        ]
        constraints = [
            {
                "qualifier_set": [
                    {
                        "qualifier_type_id": "biolink:object_aspect_qualifier",
                        "qualifier_values": ["activity", "abundance"],
                    },
                    {
                        "qualifier_type_id": "biolink:object_direction_qualifier",
                        "qualifier_values": ["increased", "decreased"],
                    },
                ]
            }
        ]
        assert edge_matches_qualifier_constraints(edge_qualifiers, constraints) is True

    def test_expanded_format_multiple_types_partial_match(self):
        """Expanded format with multiple qualifier types - partial match should fail."""
        edge_qualifiers = [
            {"qualifier_type_id": "biolink:object_aspect_qualifier", "qualifier_value": "activity"},
            {"qualifier_type_id": "biolink:object_direction_qualifier", "qualifier_value": "unchanged"},
        ]
        constraints = [
            {
                "qualifier_set": [
                    {
                        "qualifier_type_id": "biolink:object_aspect_qualifier",
                        "qualifier_values": ["activity", "abundance"],  # Matches
                    },
                    {
                        "qualifier_type_id": "biolink:object_direction_qualifier",
                        "qualifier_values": ["increased", "decreased"],  # "unchanged" not in list
                    },
                ]
            }
        ]
        assert edge_matches_qualifier_constraints(edge_qualifiers, constraints) is False


class TestQualifierExpander:
    """Tests for the QualifierExpander class which handles qualifier value hierarchy."""

    def test_get_value_descendants_unknown_value(self, bmt):
        """Unknown values should return just the original value."""
        expander = QualifierExpander(bmt)
        descendants = expander.get_value_descendants("unknown_value_xyz")
        assert "unknown_value_xyz" in descendants
        # May only have the original value if not in any enum
        assert len(descendants) >= 1

    def test_get_value_descendants_activity(self, bmt):
        """Activity value should include itself (may have no children)."""
        expander = QualifierExpander(bmt)
        descendants = expander.get_value_descendants("activity")
        assert "activity" in descendants

    def test_expand_qualifier_constraints_empty(self, bmt):
        """Empty constraints should return empty."""
        expander = QualifierExpander(bmt)
        result = expander.expand_qualifier_constraints([])
        assert result == []

    def test_expand_qualifier_constraints_none(self, bmt):
        """None constraints should return None."""
        expander = QualifierExpander(bmt)
        result = expander.expand_qualifier_constraints(None)
        assert result is None

    def test_expand_qualifier_constraints_empty_qualifier_set(self, bmt):
        """Empty qualifier_set should be preserved."""
        expander = QualifierExpander(bmt)
        constraints = [{"qualifier_set": []}]
        result = expander.expand_qualifier_constraints(constraints)
        assert len(result) == 1
        assert result[0]["qualifier_set"] == []

    def test_expand_qualifier_constraints_creates_qualifier_values(self, bmt):
        """Expansion should create qualifier_values (plural) format."""
        expander = QualifierExpander(bmt)
        constraints = [
            {
                "qualifier_set": [
                    {
                        "qualifier_type_id": "biolink:object_aspect_qualifier",
                        "qualifier_value": "activity",
                    }
                ]
            }
        ]
        result = expander.expand_qualifier_constraints(constraints)
        assert len(result) == 1
        assert len(result[0]["qualifier_set"]) == 1
        expanded_qualifier = result[0]["qualifier_set"][0]
        assert expanded_qualifier["qualifier_type_id"] == "biolink:object_aspect_qualifier"
        assert "qualifier_values" in expanded_qualifier
        assert "activity" in expanded_qualifier["qualifier_values"]

    def test_expand_qualifier_constraints_preserves_or_semantics(self, bmt):
        """Multiple qualifier_sets should be preserved (OR semantics)."""
        expander = QualifierExpander(bmt)
        constraints = [
            {
                "qualifier_set": [
                    {"qualifier_type_id": "biolink:object_aspect_qualifier", "qualifier_value": "activity"}
                ]
            },
            {
                "qualifier_set": [
                    {"qualifier_type_id": "biolink:object_aspect_qualifier", "qualifier_value": "abundance"}
                ]
            },
        ]
        result = expander.expand_qualifier_constraints(constraints)
        assert len(result) == 2

    def test_caching_works(self, bmt):
        """Repeated calls should use cache."""
        expander = QualifierExpander(bmt)
        # First call
        descendants1 = expander.get_value_descendants("activity")
        # Second call should use cache
        descendants2 = expander.get_value_descendants("activity")
        assert descendants1 == descendants2
        # Check cache was populated
        assert ("_all_", "activity") in expander._descendants_cache


class TestLookupWithQualifierConstraints:
    """Tests for lookup function with qualifier constraints."""

    def test_qualifier_constraint_filters_edges(self, graph, bmt):
        """Qualifier constraints should filter to only matching edges."""
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
                            "qualifier_constraints": [
                                {
                                    "qualifier_set": [
                                        {"qualifier_type_id": "biolink:object_aspect_qualifier", "qualifier_value": "activity"},
                                        {"qualifier_type_id": "biolink:object_direction_qualifier", "qualifier_value": "increased"},
                                    ]
                                }
                            ],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, bmt=bmt, verbose=False)
        results = response["message"]["results"]

        # Only NCBIGene:3643 (INSR) has qualifiers matching activity+increased
        assert len(results) == 1
        assert results[0]["node_bindings"]["n1"][0]["id"] == "NCBIGene:3643"

    def test_qualifier_constraint_decreased_direction(self, graph, bmt):
        """Query for edges with decreased direction qualifier."""
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
                            "qualifier_constraints": [
                                {
                                    "qualifier_set": [
                                        {"qualifier_type_id": "biolink:object_aspect_qualifier", "qualifier_value": "activity"},
                                        {"qualifier_type_id": "biolink:object_direction_qualifier", "qualifier_value": "decreased"},
                                    ]
                                }
                            ],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, bmt=bmt, verbose=False)
        results = response["message"]["results"]

        # Only NCBIGene:2645 (GCK) has qualifiers matching activity+decreased
        assert len(results) == 1
        assert results[0]["node_bindings"]["n1"][0]["id"] == "NCBIGene:2645"

    def test_qualifier_constraint_abundance_aspect(self, graph, bmt):
        """Query for edges with abundance aspect qualifier."""
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
                            "qualifier_constraints": [
                                {
                                    "qualifier_set": [
                                        {"qualifier_type_id": "biolink:object_aspect_qualifier", "qualifier_value": "abundance"},
                                        {"qualifier_type_id": "biolink:object_direction_qualifier", "qualifier_value": "increased"},
                                    ]
                                }
                            ],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, bmt=bmt, verbose=False)
        results = response["message"]["results"]

        # Only NCBIGene:7124 (TNF) has abundance increased qualifiers
        assert len(results) == 1
        assert results[0]["node_bindings"]["n1"][0]["id"] == "NCBIGene:7124"

    def test_qualifier_constraint_or_semantics(self, graph, bmt):
        """Multiple qualifier sets should use OR semantics."""
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
                            "qualifier_constraints": [
                                {
                                    "qualifier_set": [
                                        {"qualifier_type_id": "biolink:object_aspect_qualifier", "qualifier_value": "activity"},
                                        {"qualifier_type_id": "biolink:object_direction_qualifier", "qualifier_value": "increased"},
                                    ]
                                },
                                {
                                    "qualifier_set": [
                                        {"qualifier_type_id": "biolink:object_aspect_qualifier", "qualifier_value": "abundance"},
                                    ]
                                },
                            ],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, bmt=bmt, verbose=False)
        results = response["message"]["results"]

        # Should match both INSR (activity+increased) and TNF (abundance)
        assert len(results) == 2
        result_ids = {r["node_bindings"]["n1"][0]["id"] for r in results}
        assert result_ids == {"NCBIGene:3643", "NCBIGene:7124"}

    def test_no_qualifier_constraints_returns_all(self, graph, bmt):
        """Without qualifier constraints, all edges should match."""
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

        response = lookup(graph, query, bmt=bmt, verbose=False)
        results = response["message"]["results"]

        # Should return all 4 affects edges to genes:
        # PPARG (no qualifiers), INSR (activity+increased), GCK (activity+decreased), TNF (abundance+increased)
        assert len(results) == 4

    def test_qualifier_constraint_no_matches(self, graph, bmt):
        """Query with non-matching qualifier constraints should return 0 results."""
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
                            "qualifier_constraints": [
                                {
                                    "qualifier_set": [
                                        {"qualifier_type_id": "biolink:object_aspect_qualifier", "qualifier_value": "expression"},
                                    ]
                                }
                            ],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, bmt=bmt, verbose=False)
        results = response["message"]["results"]

        # No edges have expression qualifier
        assert len(results) == 0
