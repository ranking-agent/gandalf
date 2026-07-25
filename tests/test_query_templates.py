"""Tests for the drug-for-disease query template portfolio.

The templates are only useful if they render to query graphs Gandalf accepts and
if their cost estimates track the census they are read from, so both are checked
here against a hand-built census with numbers that are easy to verify by eye.
"""

import pytest

from scripts.query_templates import (
    TEMPLATES,
    Census,
    Hop,
    Template,
    estimate,
)

DISEASE = "biolink:Disease"
PROTEIN = "biolink:Protein"
SMALL_MOLECULE = "biolink:SmallMolecule"
DIRECTION = "biolink:object_direction_qualifier"


@pytest.fixture
def census():
    """Two triples: 10 proteins per disease, 20 chemicals per protein."""
    return Census(
        rollup={
            (DISEASE, "biolink:associated_with", PROTEIN): {
                "edges": 1000,
                "subjects": 100,
                "objects": 500,
            },
            (SMALL_MOLECULE, "biolink:affects", PROTEIN): {
                "edges": 8000,
                "subjects": 400,
                "objects": 200,
            },
        },
        qualifier_values={
            (
                SMALL_MOLECULE,
                "biolink:affects",
                PROTEIN,
                DIRECTION,
                "decreased",
            ): {"edges": 2000, "subjects": 300, "objects": 100},
        },
        signatures={
            (
                SMALL_MOLECULE,
                "biolink:affects",
                PROTEIN,
                # Rendered sorted, exactly as metagraph_census writes it.
                f"biolink:object_aspect_qualifier=activity|{DIRECTION}=decreased",
            ): {"edges": 500, "subjects": 200, "objects": 50},
        },
    )


@pytest.fixture
def two_hop():
    return Template(
        name="test_two_hop",
        tier="A-mechanism",
        mechanism="test",
        categories={
            "n_disease": DISEASE,
            "n_protein": PROTEIN,
            "n_chem": SMALL_MOLECULE,
        },
        hops=(
            Hop("n_disease", "n_protein", ("biolink:associated_with",)),
            Hop(
                "n_chem", "n_protein", ("biolink:affects",), ((DIRECTION, "decreased"),)
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def test_render_pins_the_disease_and_builds_one_qedge_per_hop(two_hop):
    query_graph = two_hop.render("MONDO:0004979")["message"]["query_graph"]

    assert query_graph["nodes"]["n_disease"]["ids"] == ["MONDO:0004979"]
    assert query_graph["nodes"]["n_chem"]["categories"] == [SMALL_MOLECULE]
    assert "ids" not in query_graph["nodes"]["n_chem"]
    assert len(query_graph["edges"]) == 2
    assert query_graph["edges"]["e0"]["subject"] == "n_disease"
    assert query_graph["edges"]["e1"]["predicates"] == ["biolink:affects"]


def test_render_emits_trapi_qualifier_constraints(two_hop):
    edges = two_hop.render("MONDO:1")["message"]["query_graph"]["edges"]

    assert edges["e1"]["qualifier_constraints"] == [
        {
            "qualifier_set": [
                {
                    "qualifier_type_id": DIRECTION,
                    "qualifier_value": "decreased",
                }
            ]
        }
    ]
    assert "qualifier_constraints" not in edges["e0"]


@pytest.mark.parametrize("template", TEMPLATES, ids=lambda t: t.name)
def test_every_template_renders_a_connected_query_graph(template):
    query_graph = template.render("MONDO:0004979")["message"]["query_graph"]

    assert query_graph["nodes"][template.pinned]["ids"] == ["MONDO:0004979"]
    assert set(query_graph["nodes"]) == set(template.categories)

    # Every qnode must be reachable from the pinned node, or the query graph
    # has an orphan component Gandalf would reject.
    adjacency: dict[str, set[str]] = {key: set() for key in query_graph["nodes"]}
    for edge in query_graph["edges"].values():
        adjacency[edge["subject"]].add(edge["object"])
        adjacency[edge["object"]].add(edge["subject"])
    seen = {template.pinned}
    stack = [template.pinned]
    while stack:
        for neighbour in adjacency[stack.pop()]:
            if neighbour not in seen:
                seen.add(neighbour)
                stack.append(neighbour)
    assert seen == set(query_graph["nodes"])


# ---------------------------------------------------------------------------
# Costing
# ---------------------------------------------------------------------------


def test_estimate_multiplies_fanouts_along_the_path(two_hop, census):
    summary = estimate(two_hop, census)

    # 1000/100 = 10 proteins per disease, then the qualified row backwards:
    # 2000 edges / 100 distinct proteins = 20 chemicals per protein.
    assert summary["expected_paths"] == 200
    assert summary["disease_coverage"] == 100
    assert [hop["fanout"] for hop in summary["hops"]] == [10.0, 20.0]
    assert summary["missing_triples"] == []


def test_qualifier_constraint_changes_the_row_used(two_hop, census):
    unqualified = Template(
        name="unqualified",
        tier="B-broad",
        mechanism="test",
        categories=two_hop.categories,
        hops=(
            two_hop.hops[0],
            Hop("n_chem", "n_protein", ("biolink:affects",)),
        ),
    )

    # Unqualified reads the rollup row: 8000/200 = 40 per protein.
    assert estimate(unqualified, census)["expected_paths"] == 400
    # Constrained reads the value row: 2000/100 = 20 per protein.
    assert estimate(two_hop, census)["expected_paths"] == 200


def test_conjunction_reads_the_signature_table(two_hop, census):
    conjunction = Template(
        name="conjunction",
        tier="A-mechanism",
        mechanism="test",
        categories=two_hop.categories,
        hops=(
            two_hop.hops[0],
            Hop(
                "n_chem",
                "n_protein",
                ("biolink:affects",),
                (
                    (DIRECTION, "decreased"),
                    ("biolink:object_aspect_qualifier", "activity"),
                ),
            ),
        ),
    )

    # 500 edges / 50 distinct proteins = 10 chemicals per protein.
    assert estimate(conjunction, census)["expected_paths"] == 100


def test_missing_census_row_is_reported_not_guessed(two_hop, census):
    absent = Template(
        name="absent",
        tier="A-mechanism",
        mechanism="test",
        categories={"n_disease": DISEASE, "n_protein": PROTEIN},
        hops=(Hop("n_disease", "n_protein", ("biolink:no_such_predicate",)),),
    )

    summary = estimate(absent, census)

    assert summary["missing_triples"]
    assert summary["expected_paths"] == 1


def test_closing_edge_does_not_inflate_the_estimate(census):
    """A branching template's second witness constrains; it must not multiply."""
    branching = Template(
        name="branching",
        tier="D-branching",
        mechanism="test",
        categories={
            "n_disease": DISEASE,
            "n_protein_a": PROTEIN,
            "n_protein_b": PROTEIN,
            "n_chem": SMALL_MOLECULE,
        },
        hops=(
            Hop("n_disease", "n_protein_a", ("biolink:associated_with",)),
            Hop("n_disease", "n_protein_b", ("biolink:associated_with",)),
            Hop(
                "n_chem",
                "n_protein_a",
                ("biolink:affects",),
                ((DIRECTION, "decreased"),),
            ),
            Hop(
                "n_chem",
                "n_protein_b",
                ("biolink:affects",),
                ((DIRECTION, "decreased"),),
            ),
        ),
    )

    summary = estimate(branching, census)

    roles = [hop["role"] for hop in summary["hops"]]
    assert roles.count("closes a cycle (not multiplied)") == 1
    # 10 x 10 x 20, with the fourth hop constraining rather than expanding.
    assert summary["expected_paths"] == 2000


# ---------------------------------------------------------------------------
# Portfolio invariants
# ---------------------------------------------------------------------------


def test_portfolio_names_are_unique():
    names = [template.name for template in TEMPLATES]
    assert len(names) == len(set(names))


def test_leaky_templates_are_the_only_ones_touching_the_treats_family():
    for template in TEMPLATES:
        touches_treats = any(
            "treat" in predicate
            for hop in template.hops
            for predicate in hop.predicates
        )
        assert touches_treats == template.leaky, template.name


def test_mechanism_templates_pin_protein_not_gene():
    """Gene-disease biology lives on Protein in this graph; Gene is near-empty."""
    for template in TEMPLATES:
        if template.tier == "A-mechanism":
            assert PROTEIN in template.categories.values(), template.name
            assert "biolink:Gene" not in template.categories.values(), template.name


def test_conjunction_matches_signatures_as_a_subset():
    """A qualifier_set matches any edge carrying at least those qualifiers.

    Real signatures bundle aspect and direction with qualified_predicate and
    species context, so exact-signature matching would find nothing.
    """
    aspect = "biolink:object_aspect_qualifier"
    census = Census(
        rollup={},
        qualifier_values={},
        signatures={
            (
                SMALL_MOLECULE,
                "biolink:affects",
                PROTEIN,
                f"{aspect}=activity|{DIRECTION}=decreased"
                "|biolink:qualified_predicate=biolink:causes",
            ): {"edges": 900, "subjects": 90, "objects": 30},
            (
                SMALL_MOLECULE,
                "biolink:affects",
                PROTEIN,
                f"{aspect}=activity|{DIRECTION}=increased",
            ): {"edges": 100, "subjects": 10, "objects": 10},
        },
    )
    hop = Hop(
        "n_chem",
        "n_protein",
        ("biolink:affects",),
        ((aspect, "activity"), (DIRECTION, "decreased")),
    )
    template = Template(
        name="subset",
        tier="A-mechanism",
        mechanism="test",
        categories={"n_chem": SMALL_MOLECULE, "n_protein": PROTEIN},
        hops=(hop,),
        pinned="n_chem",
    )

    stats = census.stats(template, hop)

    # Only the decreased signature qualifies, despite its extra qualifier.
    assert stats == {"edges": 900, "subjects": 90, "objects": 30}


def test_conjunction_honours_the_qualifier_value_hierarchy():
    """activity_or_abundance must be satisfied by an edge qualified expression."""
    aspect = "biolink:object_aspect_qualifier"
    census = Census(
        rollup={},
        qualifier_values={},
        signatures={
            (
                SMALL_MOLECULE,
                "biolink:affects",
                PROTEIN,
                f"{aspect}=expression|{DIRECTION}=decreased",
            ): {"edges": 700, "subjects": 70, "objects": 20},
        },
        value_ancestors={
            (aspect, "expression"): (
                "expression",
                "abundance",
                "activity_or_abundance",
            ),
            (DIRECTION, "decreased"): ("decreased",),
        },
    )
    hop = Hop(
        "n_chem",
        "n_protein",
        ("biolink:affects",),
        ((aspect, "activity_or_abundance"), (DIRECTION, "decreased")),
    )
    template = Template(
        name="hierarchy",
        tier="A-mechanism",
        mechanism="test",
        categories={"n_chem": SMALL_MOLECULE, "n_protein": PROTEIN},
        hops=(hop,),
        pinned="n_chem",
    )

    assert census.stats(template, hop)["edges"] == 700
