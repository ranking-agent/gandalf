"""Tests for the relabeled metagraph census script.

The counting core is exercised against a hand-built closure (a plain dataclass,
so no toolkit or network is involved) and a tiny synthetic edge list where every
expected number can be checked by hand.
"""

import json

import numpy as np
import pytest

from scripts.metagraph_census import (
    BiolinkClosure,
    CensusTables,
    GraphArrays,
    build_facets,
    build_qualifier_summary,
    edge_qualifier_signature,
    primary_source,
    qualified_edge_total,
    qualifier_signature,
    render_signature,
    summarize_annotations,
    _ancestors,
    _model_inverse,
    _predicate_metadata,
    build_category_tables,
    build_predicate_summary,
    build_wide_rows,
    choose_primary_category,
    group_stats,
    match_sets,
    relabel_to_pins,
    run_census,
)

# ---------------------------------------------------------------------------
# Fixtures: a four-predicate, four-category toy model
# ---------------------------------------------------------------------------

PREDICATE_ANCESTORS = {
    "biolink:related_to": ("biolink:related_to",),
    "biolink:affects": ("biolink:affects", "biolink:related_to"),
    "biolink:treats": ("biolink:treats", "biolink:affects", "biolink:related_to"),
    "biolink:ameliorates_condition": (
        "biolink:ameliorates_condition",
        "biolink:treats",
        "biolink:affects",
        "biolink:related_to",
    ),
    "biolink:affected_by": ("biolink:affected_by", "biolink:related_to"),
    "biolink:interacts_with": ("biolink:interacts_with", "biolink:related_to"),
}

CATEGORY_ANCESTORS = {
    "biolink:NamedThing": ("biolink:NamedThing",),
    "biolink:ChemicalEntity": ("biolink:ChemicalEntity", "biolink:NamedThing"),
    "biolink:Drug": (
        "biolink:Drug",
        "biolink:ChemicalEntity",
        "biolink:NamedThing",
    ),
    "biolink:Food": ("biolink:Food", "biolink:ChemicalEntity", "biolink:NamedThing"),
    "biolink:Disease": ("biolink:Disease", "biolink:NamedThing"),
}

# expression is_a abundance is_a activity_or_abundance, as in Biolink 4.3.2
QUALIFIER_ANCESTORS = {
    ("biolink:object_aspect_qualifier", "activity"): (
        "activity",
        "activity_or_abundance",
    ),
    ("biolink:object_aspect_qualifier", "abundance"): (
        "abundance",
        "activity_or_abundance",
    ),
    ("biolink:object_aspect_qualifier", "expression"): (
        "expression",
        "abundance",
        "activity_or_abundance",
    ),
    ("biolink:object_aspect_qualifier", "activity_or_abundance"): (
        "activity_or_abundance",
    ),
    ("biolink:object_direction_qualifier", "decreased"): ("decreased",),
    ("biolink:object_direction_qualifier", "increased"): ("increased",),
}

PREDICATE_META = {
    "biolink:related_to": {"in_model": True, "canonical": True, "symmetric": True},
    "biolink:affects": {
        "in_model": True,
        "canonical": True,
        "symmetric": False,
        "inverse": "biolink:affected_by",
    },
    "biolink:treats": {"in_model": True, "canonical": True, "symmetric": False},
    "biolink:ameliorates_condition": {
        "in_model": True,
        "canonical": True,
        "symmetric": False,
    },
    "biolink:affected_by": {
        "in_model": True,
        "canonical": False,
        "symmetric": False,
        "inverse": "biolink:affects",
    },
    "biolink:interacts_with": {
        "in_model": True,
        "canonical": True,
        "symmetric": True,
    },
}


@pytest.fixture
def closure():
    return BiolinkClosure(
        version="test",
        schema="test",
        predicate_ancestors=PREDICATE_ANCESTORS,
        category_ancestors=CATEGORY_ANCESTORS,
        predicate_meta=PREDICATE_META,
        predicate_depths={
            predicate: len(chain) - 1
            for predicate, chain in PREDICATE_ANCESTORS.items()
        },
        category_depths={
            category: len(chain) - 1 for category, chain in CATEGORY_ANCESTORS.items()
        },
        unmapped_predicates=(),
        unmapped_categories=(),
        qualifier_ancestors=QUALIFIER_ANCESTORS,
        qualifier_enums={
            "biolink:object_aspect_qualifier": "AspectEnum",
            "biolink:object_direction_qualifier": "DirectionQualifierEnum",
        },
    )


@pytest.fixture
def arrays():
    """Six edges over five nodes.

    ==============================  ===========================================
    nodes 0,1                       biolink:Drug
    node 2                          biolink:Food
    nodes 3,4                       biolink:Disease
    ==============================  ===========================================

    * 0 -ameliorates_condition-> 3
    * 0 -ameliorates_condition-> 4
    * 1 -ameliorates_condition-> 3
    * 1 -treats-> 3
    * 2 -affects-> 3
    * 3 -affected_by-> 0
    """
    categories = ["biolink:Drug", "biolink:Food", "biolink:Disease"]
    predicates = [
        "biolink:ameliorates_condition",
        "biolink:treats",
        "biolink:affects",
        "biolink:affected_by",
    ]
    return GraphArrays(
        subjects=np.array([0, 0, 1, 1, 2, 3], dtype=np.int32),
        objects=np.array([3, 4, 3, 3, 3, 0], dtype=np.int32),
        predicate_codes=np.array([0, 0, 0, 1, 2, 3], dtype=np.int32),
        predicates=predicates,
        node_category_codes=np.array([0, 0, 1, 2, 2], dtype=np.int32),
        categories=categories,
        num_nodes=5,
        source="synthetic",
    )


def row_for(census, subject, predicate, obj):
    """Look up one rollup row, or None."""
    return census.rollup_index.get((subject, predicate, obj))


# ---------------------------------------------------------------------------
# Counting core
# ---------------------------------------------------------------------------


def test_group_stats_counts_edges_and_distinct_endpoints():
    keys = np.array([7, 7, 7, 9, 9], dtype=np.int64)
    subjects = np.array([1, 1, 2, 5, 5], dtype=np.int32)
    objects = np.array([3, 4, 4, 6, 6], dtype=np.int32)

    unique, edges, subject_counts, object_counts = group_stats(keys, subjects, objects)

    assert unique.tolist() == [7, 9]
    assert edges.tolist() == [3, 2]
    assert subject_counts.tolist() == [2, 1]
    assert object_counts.tolist() == [2, 1]


def test_group_stats_handles_unsorted_input():
    keys = np.array([9, 7, 9, 7], dtype=np.int64)
    subjects = np.array([5, 1, 5, 2], dtype=np.int32)
    objects = np.array([6, 3, 7, 3], dtype=np.int32)

    unique, edges, subject_counts, object_counts = group_stats(keys, subjects, objects)

    assert unique.tolist() == [7, 9]
    assert edges.tolist() == [2, 2]
    assert subject_counts.tolist() == [2, 1]
    assert object_counts.tolist() == [1, 2]


def test_group_stats_empty():
    empty_keys = np.empty(0, dtype=np.int64)
    empty_nodes = np.empty(0, dtype=np.int32)

    results = group_stats(empty_keys, empty_nodes, empty_nodes)

    assert all(result.tolist() == [] for result in results)


# ---------------------------------------------------------------------------
# Category labelling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "declared,policy,expected",
    [
        (["biolink:ChemicalEntity", "biolink:Drug"], "most-specific", "biolink:Drug"),
        (
            ["biolink:ChemicalEntity", "biolink:Drug"],
            "first",
            "biolink:ChemicalEntity",
        ),
        (["biolink:Drug"], "most-specific", "biolink:Drug"),
        # Not a single chain: the deepest term wins, ties alphabetical.
        (["biolink:Drug", "biolink:Disease"], "most-specific", "biolink:Drug"),
        (["biolink:NamedThing"], "most-specific", "biolink:NamedThing"),
    ],
)
def test_choose_primary_category(closure, declared, policy, expected):
    assert choose_primary_category(declared, closure, policy) == expected


def test_relabel_to_pins_collapses_to_most_specific_pin(closure):
    categories = ["biolink:Drug", "biolink:Food", "biolink:Disease"]
    codes = np.array([0, 0, 1, 2, 2], dtype=np.int32)

    pinned_codes, pinned_categories = relabel_to_pins(
        closure, categories, codes, ["biolink:ChemicalEntity"], "leaf"
    )

    labels = [pinned_categories[code] for code in pinned_codes]
    assert labels == [
        "biolink:ChemicalEntity",
        "biolink:ChemicalEntity",
        "biolink:ChemicalEntity",
        "biolink:Disease",
        "biolink:Disease",
    ]


def test_relabel_to_pins_other_fallback_buckets_uncovered(closure):
    categories = ["biolink:Drug", "biolink:Disease"]
    codes = np.array([0, 1], dtype=np.int32)

    pinned_codes, pinned_categories = relabel_to_pins(
        closure, categories, codes, ["biolink:ChemicalEntity"], "other"
    )

    labels = [pinned_categories[code] for code in pinned_codes]
    assert labels == ["biolink:ChemicalEntity", "biolink:NamedThing"]


# ---------------------------------------------------------------------------
# Match sets
# ---------------------------------------------------------------------------


def test_match_sets_stored_takes_every_observed_descendant(closure):
    observed = ["biolink:ameliorates_condition", "biolink:treats", "biolink:affects"]

    forward, inverse = match_sets("biolink:affects", observed, closure, "stored")

    assert set(forward) == set(observed)
    assert inverse == []


def test_match_sets_query_adds_inverse_direction(closure):
    observed = [
        "biolink:ameliorates_condition",
        "biolink:affects",
        "biolink:affected_by",
    ]

    forward, inverse = match_sets("biolink:affects", observed, closure, "query")

    # affected_by is non-canonical, so it is only reachable in reverse.
    assert set(forward) == {"biolink:ameliorates_condition", "biolink:affects"}
    assert inverse == ["biolink:affected_by"]


def test_match_sets_query_related_to_matches_everything(closure):
    observed = ["biolink:affects", "biolink:affected_by"]

    forward, inverse = match_sets("biolink:related_to", observed, closure, "query")

    assert forward == observed
    assert inverse == observed


def test_match_sets_query_symmetric_matches_both_directions(closure):
    observed = ["biolink:interacts_with"]

    forward, inverse = match_sets("biolink:interacts_with", observed, closure, "query")

    assert forward == ["biolink:interacts_with"]
    assert inverse == ["biolink:interacts_with"]


# ---------------------------------------------------------------------------
# Biolink lookups
# ---------------------------------------------------------------------------


class _UnderscoreNamedSlotToolkit:
    """Stands in for BMT on a slot whose Biolink name contains underscores.

    ``SchemaView.inverse`` cannot resolve those (BMT hands it the de-underscored
    name) and dereferences ``None``, so ``get_inverse_predicate`` raises
    ``AttributeError`` -- exactly what this reproduces.
    """

    def get_inverse_predicate(self, predicate, formatted=False):
        raise AttributeError("'NoneType' object has no attribute 'name'")


def test_model_inverse_survives_unresolvable_slot_names():
    assert (
        _model_inverse(_UnderscoreNamedSlotToolkit(), "biolink:gene_fusion_with")
        is None
    )


@pytest.mark.integration
def test_every_biolink_predicate_and_category_resolves():
    """The real model, end to end: no term may blow up the closure lookups.

    ``biolink:gene_fusion_with`` and ``biolink:genetic_neighborhood_of`` are
    named with underscores in the 4.3.2 YAML and used to raise here.
    """
    from gandalf.biolink import make_toolkit

    toolkit = make_toolkit()
    predicates = toolkit.get_descendants(
        "related to", reflexive=True, formatted=True, mixin=True
    )
    categories = toolkit.get_descendants(
        "entity", reflexive=True, formatted=True, mixin=True
    )

    for predicate in predicates:
        assert _predicate_metadata(toolkit, predicate)["in_model"]
        assert _ancestors(toolkit, predicate, include_mixins=True)
        assert _ancestors(toolkit, predicate, include_mixins=False)
    for category in categories:
        assert _ancestors(toolkit, category, include_mixins=True)

    fusion = _predicate_metadata(toolkit, "biolink:gene_fusion_with")
    assert fusion["symmetric"] is True
    assert fusion["model_inverse"] is None


# ---------------------------------------------------------------------------
# Census
# ---------------------------------------------------------------------------


def test_leaf_rows_are_the_occurring_triples(arrays, closure):
    census = run_census(arrays, closure, arrays.node_category_codes, arrays.categories)

    leaves = {
        (row.subject_category, row.predicate, row.object_category): (
            row.edge_count,
            row.distinct_subjects,
            row.distinct_objects,
        )
        for row in census.leaf_rows
    }
    assert leaves == {
        ("biolink:Drug", "biolink:ameliorates_condition", "biolink:Disease"): (3, 2, 2),
        ("biolink:Drug", "biolink:treats", "biolink:Disease"): (1, 1, 1),
        ("biolink:Food", "biolink:affects", "biolink:Disease"): (1, 1, 1),
        ("biolink:Disease", "biolink:affected_by", "biolink:Drug"): (1, 1, 1),
    }
    assert census.total_edges == 6


def test_rollup_aggregates_descendants_without_double_counting_nodes(arrays, closure):
    census = run_census(arrays, closure, arrays.node_category_codes, arrays.categories)

    treats = row_for(census, "biolink:Drug", "biolink:treats", "biolink:Disease")
    assert (treats.edge_count, treats.distinct_subjects, treats.distinct_objects) == (
        4,
        2,
        2,
    )

    affects = row_for(census, "biolink:Drug", "biolink:affects", "biolink:Disease")
    assert affects.edge_count == 4  # affects itself only occurs Food -> Disease

    root = row_for(census, "biolink:Drug", "biolink:related_to", "biolink:Disease")
    assert root.edge_count == 4
    assert (
        row_for(
            census, "biolink:Food", "biolink:related_to", "biolink:Disease"
        ).edge_count
        == 1
    )


def test_rollup_row_metadata(arrays, closure):
    census = run_census(arrays, closure, arrays.node_category_codes, arrays.categories)

    rows = {
        (row["subject_category"], row["predicate"], row["object_category"]): row
        for row in census.rollup_rows
    }
    treats = rows[("biolink:Drug", "biolink:treats", "biolink:Disease")]
    assert treats["occurs_as_leaf"] == 1
    assert treats["n_forward_predicates"] == 2
    assert set(treats["forward_predicates"].split("|")) == {
        "biolink:ameliorates_condition",
        "biolink:treats",
    }
    assert treats["share_of_graph"] == pytest.approx(4 / 6)

    affects = rows[("biolink:Drug", "biolink:affects", "biolink:Disease")]
    assert affects["occurs_as_leaf"] == 0  # nothing is *labelled* affects here


def test_query_semantics_pull_in_the_inverse_direction(arrays, closure):
    census = run_census(
        arrays,
        closure,
        arrays.node_category_codes,
        arrays.categories,
        semantics="query",
    )

    # Drug -affects-> Disease: 4 stored edges, plus the Disease -affected_by->
    # Drug edge, which a qedge reaches in reverse.
    affects = row_for(census, "biolink:Drug", "biolink:affects", "biolink:Disease")
    assert affects.edge_count == 5
    assert affects.distinct_subjects == 2
    assert affects.distinct_objects == 2

    # ... and under stored semantics it is not there.
    stored = run_census(arrays, closure, arrays.node_category_codes, arrays.categories)
    assert (
        row_for(stored, "biolink:Drug", "biolink:affects", "biolink:Disease").edge_count
        == 4
    )


def test_query_semantics_related_to_covers_every_edge_both_ways(arrays, closure):
    census = run_census(
        arrays,
        closure,
        arrays.node_category_codes,
        arrays.categories,
        semantics="query",
    )

    total = sum(
        row["edge_count"]
        for row in census.rollup_rows
        if row["predicate"] == "biolink:related_to"
    )
    assert total == 2 * arrays.num_edges


def test_pinned_census_merges_categories(arrays, closure):
    pinned_codes, pinned_categories = relabel_to_pins(
        closure,
        arrays.categories,
        arrays.node_category_codes,
        ["biolink:ChemicalEntity"],
        "leaf",
    )
    census = run_census(arrays, closure, pinned_codes, pinned_categories)

    # Drug and Food collapse into one subject category, so the affects rollup
    # now covers all five forward edges into Disease.
    affects = row_for(
        census, "biolink:ChemicalEntity", "biolink:affects", "biolink:Disease"
    )
    assert affects.edge_count == 5
    assert affects.distinct_subjects == 3


# ---------------------------------------------------------------------------
# Derived tables
# ---------------------------------------------------------------------------


def test_wide_rows_carry_the_ancestor_rollup(arrays, closure):
    census = run_census(arrays, closure, arrays.node_category_codes, arrays.categories)

    rows = {
        (row["subject_category"], row["predicate"], row["object_category"]): row
        for row in build_wide_rows(census, closure)
    }
    row = rows[("biolink:Drug", "biolink:ameliorates_condition", "biolink:Disease")]
    rollup = json.loads(row["rollup_json"])

    assert rollup["biolink:ameliorates_condition"]["edges"] == 3
    assert rollup["biolink:treats"]["edges"] == 4
    assert rollup["biolink:related_to"]["edges"] == 4
    assert row["parent_predicate"] == "biolink:treats"
    assert row["edges_at_parent"] == 4
    assert row["dilution_at_parent"] == pytest.approx(4 / 3, abs=1e-3)
    assert row["share_of_related_to"] == pytest.approx(3 / 4)


def test_predicate_summary_separates_own_from_subtree(arrays, closure):
    census = run_census(arrays, closure, arrays.node_category_codes, arrays.categories)

    rows = {
        row["predicate"]: row
        for row in build_predicate_summary(census, closure, arrays.num_edges)
    }
    assert rows["biolink:ameliorates_condition"]["own_edge_count"] == 3
    assert rows["biolink:ameliorates_condition"]["subtree_edge_count"] == 3
    # treats is stored on one edge but matches four.
    assert rows["biolink:treats"]["own_edge_count"] == 1
    assert rows["biolink:treats"]["subtree_edge_count"] == 4
    assert rows["biolink:treats"]["own_share_of_subtree"] == pytest.approx(0.25)
    # A pure grouping term: no edge is labelled with it.
    assert rows["biolink:related_to"]["own_edge_count"] == 0
    assert rows["biolink:related_to"]["subtree_edge_count"] == arrays.num_edges


def test_category_tables_roll_leaves_into_ancestors(arrays, closure):
    summary, rollup = build_category_tables(
        arrays, closure, arrays.node_category_codes, arrays.categories
    )

    by_category = {row["category"]: row for row in summary}
    assert by_category["biolink:Drug"]["nodes"] == 2
    assert by_category["biolink:Drug"]["edges_as_subject"] == 4
    assert by_category["biolink:Disease"]["edges_as_object"] == 5

    rolled = {row["category"]: row for row in rollup}
    chemical = rolled["biolink:ChemicalEntity"]
    assert chemical["nodes"] == 3  # 2 drugs + 1 food
    assert chemical["n_member_categories"] == 2
    assert chemical["largest_member"] == "biolink:Drug"
    assert "biolink:Food:1" in chemical["member_breakdown"]
    assert rolled["biolink:NamedThing"]["nodes"] == arrays.num_nodes


def test_empty_census_is_not_an_error(closure):
    empty = GraphArrays(
        subjects=np.empty(0, dtype=np.int32),
        objects=np.empty(0, dtype=np.int32),
        predicate_codes=np.empty(0, dtype=np.int32),
        predicates=["biolink:affects"],
        node_category_codes=np.array([0], dtype=np.int32),
        categories=["biolink:Drug"],
        num_nodes=1,
        source="empty",
    )

    census = run_census(empty, closure, empty.node_category_codes, empty.categories)

    assert census.leaf_rows == []
    assert census.rollup_rows == []
    assert build_wide_rows(census, closure) == []


def test_census_tables_dataclass_defaults():
    census = CensusTables(leaf_rows=[], rollup_rows=[], rollup_index={})

    assert census.semantics == "stored"
    assert census.forward_sources == {}


# ---------------------------------------------------------------------------
# Qualifier and source facets
# ---------------------------------------------------------------------------


DECREASES_ACTIVITY = (
    ("biolink:object_aspect_qualifier", "activity"),
    ("biolink:object_direction_qualifier", "decreased"),
)
INCREASES_EXPRESSION = (
    ("biolink:object_aspect_qualifier", "expression"),
    ("biolink:object_direction_qualifier", "increased"),
)


@pytest.fixture
def qualified_arrays(arrays):
    """The same six edges, with qualifiers and sources on some of them.

    * edge 0 (Drug -ameliorates_condition-> Disease): decreases activity
    * edge 1 (Drug -ameliorates_condition-> Disease): increases expression
    * edge 2 (Drug -ameliorates_condition-> Disease): decreases activity
    * edges 3-5: unqualified
    """
    signatures = [(), DECREASES_ACTIVITY, INCREASES_EXPRESSION]
    arrays.qualifier_signatures = signatures
    arrays.qualifier_codes = np.array([1, 2, 1, 0, 0, 0], dtype=np.int32)
    arrays.primary_sources = ["infores:ctd", "infores:semmeddb"]
    arrays.source_codes = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
    return arrays


def test_qualifier_signature_normalizes_order_and_drops_partials():
    assert qualifier_signature(
        [
            {"qualifier_type_id": "b", "qualifier_value": "2"},
            {"qualifier_type_id": "a", "qualifier_value": "1"},
            {"qualifier_type_id": "c"},
        ]
    ) == (("a", "1"), ("b", "2"))
    assert qualifier_signature([]) == ()


def test_edge_qualifier_signature_reads_raw_kgx_and_normalized():
    raw = {
        "object_aspect_qualifier": "activity",
        "object_direction_qualifier": "decreased",
        "predicate": "biolink:affects",
    }
    normalized = {
        "qualifiers": [
            {
                "qualifier_type_id": "biolink:object_aspect_qualifier",
                "qualifier_value": "activity",
            },
            {
                "qualifier_type_id": "biolink:object_direction_qualifier",
                "qualifier_value": "decreased",
            },
        ]
    }

    assert edge_qualifier_signature(raw) == DECREASES_ACTIVITY
    assert edge_qualifier_signature(normalized) == DECREASES_ACTIVITY
    assert edge_qualifier_signature({"predicate": "biolink:affects"}) == ()


def test_edge_qualifier_signature_prefixes_qualified_predicate():
    assert edge_qualifier_signature({"qualified_predicate": "causes"}) == (
        ("biolink:qualified_predicate", "biolink:causes"),
    )


@pytest.mark.parametrize(
    "sources,expected",
    [
        (
            [
                {
                    "resource_id": "infores:agg",
                    "resource_role": "aggregator_knowledge_source",
                },
                {
                    "resource_id": "infores:ctd",
                    "resource_role": "primary_knowledge_source",
                },
            ],
            "infores:ctd",
        ),
        (
            [
                {
                    "resource_id": "infores:only",
                    "resource_role": "aggregator_knowledge_source",
                }
            ],
            "infores:only",
        ),
        ([], ""),
    ],
)
def test_primary_source(sources, expected):
    assert primary_source(sources) == expected


def test_render_signature_round_trips_a_conjunction():
    assert render_signature(DECREASES_ACTIVITY) == (
        "biolink:object_aspect_qualifier=activity|"
        "biolink:object_direction_qualifier=decreased"
    )


def test_qualified_edge_total_counts_edges_not_assertions(qualified_arrays):
    assert qualified_edge_total(qualified_arrays) == 3
    assert qualified_edge_total(arrays_without_qualifiers()) == 0


def arrays_without_qualifiers():
    return GraphArrays(
        subjects=np.array([0], dtype=np.int32),
        objects=np.array([0], dtype=np.int32),
        predicate_codes=np.array([0], dtype=np.int32),
        predicates=["biolink:affects"],
        node_category_codes=np.array([0], dtype=np.int32),
        categories=["biolink:Drug"],
        num_nodes=1,
        source="synthetic",
    )


def test_signature_facet_counts_conjunctions_not_marginals(qualified_arrays, closure):
    facets = build_facets(qualified_arrays, closure)
    census = run_census(
        qualified_arrays,
        closure,
        qualified_arrays.node_category_codes,
        qualified_arrays.categories,
        facets=facets,
    )

    rows = {
        (
            row["subject_category"],
            row["predicate"],
            row["object_category"],
            row["qualifier_signature"],
        ): row
        for row in census.facet_rows["qualifier_signature"]
    }
    key = (
        "biolink:Drug",
        "biolink:ameliorates_condition",
        "biolink:Disease",
        render_signature(DECREASES_ACTIVITY),
    )
    assert rows[key]["edge_count"] == 2
    assert rows[key]["n_qualifiers"] == 2
    assert rows[key]["share_of_triple"] == pytest.approx(2 / 3)
    # The triple's third edge carries the other signature.
    other = rows[
        (
            "biolink:Drug",
            "biolink:ameliorates_condition",
            "biolink:Disease",
            render_signature(INCREASES_EXPRESSION),
        )
    ]
    assert other["edge_count"] == 1


def test_qualifier_values_roll_up_through_the_enum(qualified_arrays, closure):
    facets = build_facets(qualified_arrays, closure)
    census = run_census(
        qualified_arrays,
        closure,
        qualified_arrays.node_category_codes,
        qualified_arrays.categories,
        facets=facets,
    )

    rows = {
        (row["predicate"], row["qualifier"]): row
        for row in census.facet_rows["qualifier"]
        if row["subject_category"] == "biolink:Drug"
        and row["object_category"] == "biolink:Disease"
    }
    leaf = rows[
        ("biolink:ameliorates_condition", "biolink:object_aspect_qualifier=activity")
    ]
    assert leaf["edge_count"] == 2
    assert leaf["is_leaf_value"] == 1

    # expression rolls into abundance, and both into activity_or_abundance,
    # so a qedge asking for activity_or_abundance matches all three edges.
    rolled = rows[
        (
            "biolink:ameliorates_condition",
            "biolink:object_aspect_qualifier=activity_or_abundance",
        )
    ]
    assert rolled["edge_count"] == 3
    assert rolled["is_leaf_value"] == 0

    abundance = rows[
        ("biolink:ameliorates_condition", "biolink:object_aspect_qualifier=abundance")
    ]
    assert abundance["edge_count"] == 1
    assert abundance["is_leaf_value"] == 0


def test_qualifier_facet_counted_at_ancestor_predicates_too(qualified_arrays, closure):
    facets = build_facets(qualified_arrays, closure)
    census = run_census(
        qualified_arrays,
        closure,
        qualified_arrays.node_category_codes,
        qualified_arrays.categories,
        facets=facets,
    )

    # A template asks for affects + a direction, so coverage has to be known
    # at the ancestor, not just at the leaf predicate.
    rows = {
        (row["predicate"], row["qualifier"]): row
        for row in census.facet_rows["qualifier"]
        if row["subject_category"] == "biolink:Drug"
    }
    assert (
        rows[("biolink:affects", "biolink:object_direction_qualifier=decreased")][
            "edge_count"
        ]
        == 2
    )
    assert (
        rows[("biolink:treats", "biolink:object_direction_qualifier=increased")][
            "edge_count"
        ]
        == 1
    )


def test_source_facet_splits_a_triple_by_provenance(qualified_arrays, closure):
    facets = build_facets(qualified_arrays, closure)
    census = run_census(
        qualified_arrays,
        closure,
        qualified_arrays.node_category_codes,
        qualified_arrays.categories,
        facets=facets,
    )

    rows = {
        (
            row["subject_category"],
            row["predicate"],
            row["object_category"],
            row["primary_source"],
        ): row
        for row in census.facet_rows["primary_source"]
    }
    drug_disease = ("biolink:Drug", "biolink:ameliorates_condition", "biolink:Disease")
    assert rows[(*drug_disease, "infores:ctd")]["edge_count"] == 3
    # Rolled up to related_to the same triple also picks up the treats edge,
    # which came from a different source.
    rolled = ("biolink:Drug", "biolink:related_to", "biolink:Disease")
    assert rows[(*rolled, "infores:ctd")]["edge_count"] == 3
    assert rows[(*rolled, "infores:semmeddb")]["edge_count"] == 1


def test_annotation_summary_columns(qualified_arrays, closure):
    facets = build_facets(qualified_arrays, closure)
    census = run_census(
        qualified_arrays,
        closure,
        qualified_arrays.node_category_codes,
        qualified_arrays.categories,
        facets=facets,
    )

    summary = summarize_annotations(census)
    entry = summary[
        ("biolink:Drug", "biolink:ameliorates_condition", "biolink:Disease")
    ]
    assert entry["qualified_edges"] == 3
    assert entry["qualified_share"] == pytest.approx(1.0)
    assert entry["n_qualifier_signatures"] == 2
    assert entry["top_primary_source"] == "infores:ctd"
    assert entry["top_primary_source_share"] == pytest.approx(1.0)


def test_qualifier_summary_counts_each_edge_once_per_value(qualified_arrays, closure):
    facets = build_facets(qualified_arrays, closure)
    census = run_census(
        qualified_arrays,
        closure,
        qualified_arrays.node_category_codes,
        qualified_arrays.categories,
        facets=facets,
    )

    rows = {
        (row["qualifier_type_id"], row["qualifier_value"]): row
        for row in build_qualifier_summary(
            qualified_arrays,
            facets,
            closure,
            qualified_arrays.node_category_codes,
            qualified_arrays.categories,
        )
    }
    assert rows[("biolink:object_aspect_qualifier", "activity")]["edge_count"] == 2
    assert (
        rows[("biolink:object_aspect_qualifier", "activity_or_abundance")]["edge_count"]
        == 3
    )
    assert rows[("biolink:object_direction_qualifier", "decreased")]["n_triples"] == 1
    assert rows[("biolink:object_aspect_qualifier", "activity")]["is_leaf_value"] == 1


def test_facets_absent_when_graph_has_no_annotations(arrays, closure):
    assert build_facets(arrays, closure) == []

    census = run_census(arrays, closure, arrays.node_category_codes, arrays.categories)

    assert census.facet_rows == {}
    assert summarize_annotations(census) == {}
    assert (
        build_qualifier_summary(
            arrays, [], closure, arrays.node_category_codes, arrays.categories
        )
        == []
    )
