"""Tests for qualifier constraint matching and expansion.

TRAPI 2.0 writes a QualifierSetConstraint as a plain mapping of
``qualifier_type_id`` to the required ``qualifier_value``, and puts the list of
them at ``QEdge.constraints.qualifiers``.
"""

import pytest

from tests.search_fixtures import graph  # noqa: F401

from gandalf.search import lookup
from gandalf.search.qualifiers import edge_matches_qualifier_constraints
from gandalf.search.expanders import QualifierExpander

ASPECT = "biolink:object_aspect_qualifier"
DIRECTION = "biolink:object_direction_qualifier"
QUALIFIED_PREDICATE = "biolink:qualified_predicate"


def qualifiers(**by_type: str) -> list[dict]:
    """Build an edge's TRAPI Qualifier list from ``short_name=value`` pairs."""
    return [
        {"qualifier_type_id": f"biolink:object_{name}_qualifier", "qualifier_value": v}
        for name, v in by_type.items()
    ]


class TestQualifierConstraintMatching:
    """Unit tests for the edge_matches_qualifier_constraints helper function."""

    @pytest.mark.parametrize("constraints", [None, [], [{}]])
    def test_no_requirement_matches_any_edge(self, constraints):
        """Absent, empty, and requirement-free constraint lists match anything."""
        assert (
            edge_matches_qualifier_constraints(
                qualifiers(aspect="activity"), constraints
            )
            is True
        )

    @pytest.mark.parametrize(
        "edge_qualifiers, constraints, expected",
        [
            # Single type-value pair
            (qualifiers(aspect="activity"), [{ASPECT: "activity"}], True),
            (qualifiers(aspect="activity"), [{ASPECT: "abundance"}], False),
            # AND within one constraint
            (
                qualifiers(aspect="activity", direction="increased"),
                [{ASPECT: "activity", DIRECTION: "increased"}],
                True,
            ),
            (
                qualifiers(aspect="activity"),
                [{ASPECT: "activity", DIRECTION: "increased"}],
                False,
            ),
            # OR between constraints
            (
                qualifiers(aspect="abundance"),
                [{ASPECT: "activity"}, {ASPECT: "abundance"}],
                True,
            ),
            # An edge carrying extra qualifiers still satisfies the constraint
            (
                qualifiers(aspect="activity", direction="increased"),
                [{ASPECT: "activity"}],
                True,
            ),
            # An edge with no qualifiers cannot satisfy a requirement
            ([], [{ASPECT: "activity"}], False),
        ],
    )
    def test_matching(self, edge_qualifiers, constraints, expected):
        assert (
            edge_matches_qualifier_constraints(edge_qualifiers, constraints) is expected
        )

    @pytest.mark.parametrize(
        "edge_value, accepted, expected",
        [
            ("activity", ["activity", "abundance"], True),
            ("expression", ["activity", "abundance"], False),
        ],
    )
    def test_expanded_values_match_any_member(self, edge_value, accepted, expected):
        """A list value (what QualifierExpander produces) reads as "any of these"."""
        assert (
            edge_matches_qualifier_constraints(
                qualifiers(aspect=edge_value), [{ASPECT: accepted}]
            )
            is expected
        )

    @pytest.mark.parametrize(
        "direction, expected",
        [("increased", True), ("unchanged", False)],
    )
    def test_expanded_values_still_and_within_a_constraint(self, direction, expected):
        """Every type in a constraint must match, expanded values included."""
        constraints = [
            {ASPECT: ["activity", "abundance"], DIRECTION: ["increased", "decreased"]}
        ]
        assert (
            edge_matches_qualifier_constraints(
                qualifiers(aspect="activity", direction=direction), constraints
            )
            is expected
        )


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

    def test_expand_requirement_free_constraint(self, bmt):
        """A constraint that names no qualifier is preserved as-is."""
        expander = QualifierExpander(bmt)
        assert expander.expand_qualifier_constraints([{}]) == [{}]

    def test_expand_turns_values_into_value_lists(self, bmt):
        """Expansion replaces each value with the list of acceptable values."""
        expander = QualifierExpander(bmt)
        result = expander.expand_qualifier_constraints([{ASPECT: "activity"}])
        assert len(result) == 1
        assert "activity" in result[0][ASPECT]

    def test_expand_qualified_predicate_uses_predicate_hierarchy(self, bmt):
        """qualified_predicate values expand through the predicate hierarchy.

        Regression test for issue #21: a query for the parent predicate
        "biolink:contributes_to" should match edges whose qualified_predicate
        is the descendant "biolink:causes". Unlike enum-valued qualifiers,
        qualified_predicate values are Biolink predicate CURIEs and must be
        expanded via get_descendants rather than enum permissible values.
        """
        expander = QualifierExpander(bmt)
        result = expander.expand_qualifier_constraints(
            [{QUALIFIED_PREDICATE: "biolink:contributes_to"}]
        )
        values = result[0][QUALIFIED_PREDICATE]
        # Original value plus its predicate descendant
        assert "biolink:contributes_to" in values
        assert "biolink:causes" in values

    def test_expand_non_predicate_qualifier_still_uses_enum(self, bmt):
        """Enum-valued qualifiers must NOT use the predicate hierarchy.

        Guards against the qualified_predicate branch leaking into ordinary
        qualifiers: object_aspect_qualifier "activity" should expand via enum
        values, not predicate descendants.
        """
        expander = QualifierExpander(bmt)
        result = expander.expand_qualifier_constraints([{ASPECT: "activity"}])
        values = result[0][ASPECT]
        assert "activity" in values
        assert "biolink:causes" not in values

    def test_expand_qualifier_constraints_preserves_or_semantics(self, bmt):
        """Multiple constraints should be preserved (OR semantics)."""
        expander = QualifierExpander(bmt)
        result = expander.expand_qualifier_constraints(
            [{ASPECT: "activity"}, {ASPECT: "abundance"}]
        )
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


def affects_genes_query(qualifier_constraints=None) -> dict:
    """A CHEBI:6801 --affects--> Gene query, optionally qualifier-constrained."""
    qedge = {
        "subject": "n0",
        "object": "n1",
        "predicates": ["biolink:affects"],
    }
    if qualifier_constraints is not None:
        qedge["constraints"] = {"qualifiers": qualifier_constraints}
    return {
        "message": {
            "query_graph": {
                "nodes": {
                    "n0": {"ids": ["CHEBI:6801"]},
                    "n1": {"categories": ["biolink:Gene"]},
                },
                "edges": {"e0": qedge},
            },
        },
    }


class TestLookupWithQualifierConstraints:
    """Tests for lookup function with QEdge constraints.qualifiers."""

    @pytest.mark.parametrize(
        "qualifier_constraints, expected_genes",
        [
            # Only INSR has activity+increased
            (
                [{ASPECT: "activity", DIRECTION: "increased"}],
                {"NCBIGene:3643"},
            ),
            # Only GCK has activity+decreased
            (
                [{ASPECT: "activity", DIRECTION: "decreased"}],
                {"NCBIGene:2645"},
            ),
            # Only TNF has abundance+increased
            (
                [{ASPECT: "abundance", DIRECTION: "increased"}],
                {"NCBIGene:7124"},
            ),
            # OR between constraints: INSR (activity+increased) and TNF (abundance)
            (
                [{ASPECT: "activity", DIRECTION: "increased"}, {ASPECT: "abundance"}],
                {"NCBIGene:3643", "NCBIGene:7124"},
            ),
            # No edge carries an expression aspect
            ([{ASPECT: "expression"}], set()),
        ],
    )
    def test_qualifier_constraint_filters_edges(
        self, graph, bmt, qualifier_constraints, expected_genes
    ):
        """constraints.qualifiers should filter to only matching edges."""
        response = lookup(graph, affects_genes_query(qualifier_constraints), bmt=bmt)
        results = response["message"]["results"]

        assert {r["node_bindings"]["n1"]["ids"][0] for r in results} == expected_genes
        assert len(results) == len(expected_genes)

    def test_no_qualifier_constraints_returns_all(self, graph, bmt):
        """Without qualifier constraints, all edges should match."""
        response = lookup(graph, affects_genes_query(), bmt=bmt)

        # Should return all 4 affects edges to genes:
        # PPARG (no qualifiers), INSR (activity+increased),
        # GCK (activity+decreased), TNF (abundance+increased)
        assert len(response["message"]["results"]) == 4
