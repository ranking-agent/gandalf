#!/usr/bin/env python3
"""Candidate query templates for "what drugs may treat this disease", with costs.

Renders a portfolio of TRAPI query graphs for a pinned disease CURIE and prices
each one against the metagraph census, so a fan-out can be budgeted before it is
fired rather than measured after it buries the ranker.

Every template here was chosen against census numbers from the 28.7M-edge
graph, not from general biomedical intuition.  Three findings did most of the
shaping:

**Gene biology lives on ``biolink:Protein``.**  ``Disease -associated_with->
Protein`` has 99,582 edges over 8,832 diseases; the same shape against
``biolink:Gene`` has 1,329 edges over 436.  A template written against Gene
returns essentially nothing, so every mechanism template here pins Protein.

**The disease side has no direction qualifiers.**  Only ``frequency_qualifier``
(HPO) appears with a Disease subject, so the textbook reversal template -- drug
*decreases* what disease *increases* -- is not expressible.  The directional
signal exists as predicates instead: ``Protein -causes|contributes_to->
Disease``.  ``causal_gene_inhibition`` below is the closest expressible
substitute.

**Qualifiers make the drug side cheaper *and* sharper.**  ``Protein
<-interacts_with- SmallMolecule`` fans out to ~148 chemicals per protein;
constraining ``object_direction_qualifier=decreased`` on ``affects`` cuts that
to ~21 while adding the mechanism claim.  That is a 7x cost reduction for a
stronger statement -- the rare case where precision is also the cheap option.

Usage::

    # Price the portfolio without rendering anything
    python scripts/query_templates.py --census census/

    # Render every template for one disease, ready to POST to Gandalf
    python scripts/query_templates.py --census census/ --disease MONDO:0004979 \\
        --out queries/

    # Only the templates that cannot see a treats edge
    python scripts/query_templates.py --census census/ --disease MONDO:0004979 \\
        --out queries/ --exclude-leaky

Estimates are first-order: the census gives mean fan-out per triple, so a
product over hops assumes independence and ignores the heavy tail.  A disease
with 200 associated proteins will blow past the estimate; that is what
``max_node_degree`` in each template's ``filter_config`` is for.  Treat the
numbers as a budget, not a promise.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

logger = logging.getLogger("query_templates")

DISEASE = "biolink:Disease"


# ---------------------------------------------------------------------------
# Template definition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Hop:
    """One qedge: a census triple plus optional qualifier constraints.

    ``subject``/``object`` are qnode keys.  The direction must match the census
    row the estimate is read from, since fan-out is not symmetric.
    """

    subject: str
    object: str
    predicates: tuple[str, ...]
    qualifiers: tuple[tuple[str, str], ...] = ()

    def qualifier_constraints(self) -> list[dict]:
        """TRAPI qualifier_constraints for this hop (AND within the set)."""
        if not self.qualifiers:
            return []
        return [
            {
                "qualifier_set": [
                    {"qualifier_type_id": type_id, "qualifier_value": value}
                    for type_id, value in self.qualifiers
                ]
            }
        ]


@dataclass(frozen=True)
class Template:
    """A named query shape, its mechanism claim, and how to execute it."""

    name: str
    tier: str
    mechanism: str
    categories: dict[str, str]
    hops: tuple[Hop, ...]
    pinned: str = "n_disease"
    leaky: bool = False
    filter_config: dict = field(default_factory=dict)
    notes: str = ""

    def render(self, disease_curie: str) -> dict:
        """Build the TRAPI request for a pinned disease."""
        nodes: dict[str, dict] = {}
        for key, category in self.categories.items():
            nodes[key] = {"categories": [category]}
        nodes[self.pinned]["ids"] = [disease_curie]

        edges = {}
        for index, hop in enumerate(self.hops):
            edge: dict = {
                "subject": hop.subject,
                "object": hop.object,
                "predicates": list(hop.predicates),
            }
            constraints = hop.qualifier_constraints()
            if constraints:
                edge["qualifier_constraints"] = constraints
            edges[f"e{index}"] = edge

        return {"message": {"query_graph": {"nodes": nodes, "edges": edges}}}


# ---------------------------------------------------------------------------
# The portfolio
# ---------------------------------------------------------------------------

CHEMICAL = "biolink:ChemicalEntity"
SMALL_MOLECULE = "biolink:SmallMolecule"
DRUG = "biolink:Drug"
PROTEIN = "biolink:Protein"
PATHWAY = "biolink:Pathway"
PHENOTYPE = "biolink:PhenotypicFeature"

DIRECTION = "biolink:object_direction_qualifier"
ASPECT = "biolink:object_aspect_qualifier"
MECHANISM = "biolink:causal_mechanism_qualifier"

# Predicates that assert the gene drives the disease, used where the census has
# no directional qualifier to offer.
CAUSAL_GENE = ("biolink:causes", "biolink:contributes_to")

TEMPLATES: tuple[Template, ...] = (
    # -- Tier A: mechanism, qualified --------------------------------------
    Template(
        name="target_inhibition_sm",
        tier="A-mechanism",
        mechanism="A small molecule decreases the activity or abundance of a "
        "protein associated with the disease.",
        categories={
            "n_disease": DISEASE,
            "n_protein": PROTEIN,
            "n_chem": SMALL_MOLECULE,
        },
        hops=(
            Hop("n_disease", "n_protein", ("biolink:associated_with",)),
            Hop(
                "n_chem",
                "n_protein",
                ("biolink:affects",),
                ((DIRECTION, "decreased"),),
            ),
        ),
        filter_config={"max_node_degree": 500},
        notes="The workhorse. 8,832 diseases have the entry hop; the qualifier "
        "cuts drug-side fan-out from ~148 to ~21 per protein.",
    ),
    Template(
        name="target_inhibition_drug",
        tier="A-mechanism",
        mechanism="An approved drug decreases the activity or abundance of a "
        "protein associated with the disease.",
        categories={"n_disease": DISEASE, "n_protein": PROTEIN, "n_chem": DRUG},
        hops=(
            Hop("n_disease", "n_protein", ("biolink:associated_with",)),
            Hop(
                "n_chem",
                "n_protein",
                ("biolink:affects",),
                ((DIRECTION, "decreased"),),
            ),
        ),
        filter_config={"max_node_degree": 500},
        notes="Same shape restricted to Drug (5,326 nodes): far smaller "
        "candidate pool, ~14 drugs per protein, and every hit is a real drug.",
    ),
    Template(
        name="target_activation_sm",
        tier="A-mechanism",
        mechanism="A small molecule increases the activity or abundance of a "
        "protein associated with the disease -- the loss-of-function case.",
        categories={
            "n_disease": DISEASE,
            "n_protein": PROTEIN,
            "n_chem": SMALL_MOLECULE,
        },
        hops=(
            Hop("n_disease", "n_protein", ("biolink:associated_with",)),
            Hop(
                "n_chem",
                "n_protein",
                ("biolink:affects",),
                ((DIRECTION, "increased"),),
            ),
        ),
        filter_config={"max_node_degree": 500},
        notes="Mirror of target_inhibition_sm. Fire both: without disease-side "
        "direction the graph cannot say which way the protein should move.",
    ),
    Template(
        name="causal_gene_inhibition",
        tier="A-mechanism",
        mechanism="A small molecule decreases a protein that causes or "
        "contributes to the disease.",
        categories={
            "n_disease": DISEASE,
            "n_protein": PROTEIN,
            "n_chem": SMALL_MOLECULE,
        },
        hops=(
            Hop("n_protein", "n_disease", CAUSAL_GENE),
            Hop(
                "n_chem",
                "n_protein",
                ("biolink:affects",),
                ((DIRECTION, "decreased"),),
            ),
        ),
        filter_config={"max_node_degree": 500},
        notes="The nearest thing to a reversal template this graph supports: "
        "causal direction from the predicate, drug direction from the "
        "qualifier. Narrower coverage (2,436 diseases) but the strongest claim.",
    ),
    Template(
        name="inhibition_mechanism_sm",
        tier="A-mechanism",
        mechanism="A small molecule inhibits (by declared mechanism) a protein "
        "associated with the disease.",
        categories={
            "n_disease": DISEASE,
            "n_protein": PROTEIN,
            "n_chem": SMALL_MOLECULE,
        },
        hops=(
            Hop("n_disease", "n_protein", ("biolink:associated_with",)),
            Hop(
                "n_chem",
                "n_protein",
                ("biolink:affects",),
                ((MECHANISM, "inhibition"),),
            ),
        ),
        filter_config={"max_node_degree": 500},
        notes="causal_mechanism_qualifier=inhibition covers 176,212 edges over "
        "95,309 chemicals -- pharmacology rather than perturbation readout.",
    ),
    # -- Tier B: broad, unqualified ----------------------------------------
    Template(
        name="target_binding_sm",
        tier="B-broad",
        mechanism="A small molecule physically binds a protein associated with "
        "the disease.",
        categories={
            "n_disease": DISEASE,
            "n_protein": PROTEIN,
            "n_chem": SMALL_MOLECULE,
        },
        hops=(
            Hop("n_disease", "n_protein", ("biolink:associated_with",)),
            Hop("n_protein", "n_chem", ("biolink:physically_interacts_with",)),
        ),
        filter_config={"max_node_degree": 300},
        notes="Recall play: 1.18M binding edges, ~148 chemicals per protein. "
        "No direction, so it cannot tell a helper from a harmer -- but it is "
        "the widest mechanistically defensible net.",
    ),
    Template(
        name="pathway_participation",
        tier="B-broad",
        mechanism="A chemical participates in a pathway that a "
        "disease-associated protein participates in.",
        categories={
            "n_disease": DISEASE,
            "n_protein": PROTEIN,
            "n_pathway": PATHWAY,
            "n_chem": SMALL_MOLECULE,
        },
        hops=(
            Hop("n_disease", "n_protein", ("biolink:associated_with",)),
            Hop("n_protein", "n_pathway", ("biolink:participates_in",)),
            Hop("n_pathway", "n_chem", ("biolink:has_participant",)),
        ),
        filter_config={"max_node_degree": 200},
        notes="Reaches drugs that miss the disease protein but hit its pathway. "
        "Pathways are hubs -- the degree cap is doing real work here.",
    ),
    Template(
        name="ppi_neighborhood",
        tier="B-broad",
        mechanism="A small molecule decreases a protein that physically "
        "interacts with a disease-associated protein.",
        categories={
            "n_disease": DISEASE,
            "n_protein": PROTEIN,
            "n_partner": PROTEIN,
            "n_chem": SMALL_MOLECULE,
        },
        hops=(
            Hop("n_disease", "n_protein", ("biolink:associated_with",)),
            Hop("n_protein", "n_partner", ("biolink:physically_interacts_with",)),
            Hop(
                "n_chem",
                "n_partner",
                ("biolink:affects",),
                ((DIRECTION, "decreased"),),
            ),
        ),
        filter_config={"max_node_degree": 100},
        notes="The most explosive template in the set (~46 PPI partners per "
        "protein). Keep the degree cap tight or drop it for dense diseases.",
    ),
    # -- Tier C: non-mechanistic signal ------------------------------------
    Template(
        name="phenotype_drug_bridge",
        tier="C-associative",
        mechanism="A drug is associated with a phenotype the disease presents.",
        categories={
            "n_disease": DISEASE,
            "n_phenotype": PHENOTYPE,
            "n_chem": DRUG,
        },
        hops=(
            Hop("n_disease", "n_phenotype", ("biolink:has_phenotype",)),
            Hop("n_phenotype", "n_chem", ("biolink:associated_with",)),
        ),
        notes="Only 380 phenotypes carry drug associations, so this fires for "
        "few diseases -- cheap enough to keep in the portfolio anyway.",
    ),
    Template(
        name="direct_association",
        tier="C-associative",
        mechanism="A drug is directly associated or correlated with the disease.",
        categories={"n_disease": DISEASE, "n_chem": DRUG},
        hops=(Hop("n_disease", "n_chem", ("biolink:associated_with",)),),
        notes="One hop, 1,906 diseases covered, ~16 drugs each. Not a mechanism "
        "-- a baseline every other template should beat.",
    ),
    # -- Tier D: routes through the treats family --------------------------
    Template(
        name="indication_transfer",
        tier="D-leaky",
        mechanism="A drug treats another disease that shares a phenotype with "
        "this one.",
        categories={
            "n_disease": DISEASE,
            "n_phenotype": PHENOTYPE,
            "n_other": DISEASE,
            "n_chem": DRUG,
        },
        hops=(
            Hop("n_disease", "n_phenotype", ("biolink:has_phenotype",)),
            Hop("n_other", "n_phenotype", ("biolink:has_phenotype",)),
            Hop(
                "n_chem",
                "n_other",
                ("biolink:treats_or_applied_or_studied_to_treat",),
            ),
        ),
        leaky=True,
        filter_config={"max_node_degree": 200},
        notes="Empirically strong, mechanism-free, and it reads treats edges -- "
        "so it will flatter itself against any ground truth drawn from the same "
        "indication data. Evaluate it in its own bucket.",
    ),
    Template(
        name="two_witness_inhibition",
        tier="D-branching",
        mechanism="A small molecule decreases two different proteins, both "
        "associated with the disease.",
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
        filter_config={"max_node_degree": 500},
        notes="Branching, not linear: two independent witnesses for the same "
        "chemical -- the precision lever. Two caveats, both confirmed by "
        "running it: TRAPI cannot say n_protein_a != n_protein_b, so results "
        "include degenerate pairs where both bind the same protein, and each "
        "genuine pair comes back twice (a,b) and (b,a). Drop the degenerate "
        "ones and de-duplicate unordered pairs downstream. The estimate below "
        "does not model the self-join, so the real count is well under it.",
    ),
)


# ---------------------------------------------------------------------------
# Costing against the census
# ---------------------------------------------------------------------------


@dataclass
class Census:
    """The census tables a template estimate needs."""

    rollup: dict[tuple[str, str, str], dict]
    qualifier_values: dict[tuple[str, str, str, str, str], dict]
    signatures: dict[tuple[str, str, str, str], dict]
    # (subject, predicate, object) -> [(qualifier pairs on the edge, stats)]
    signature_index: dict[tuple[str, str, str], list] = field(default_factory=dict)
    # qualifier value -> its reflexive ancestor values
    value_ancestors: dict[tuple[str, str], tuple[str, ...]] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        """Derive the signature index when only the flat table was supplied."""
        if self.signature_index or not self.signatures:
            return
        for (subject, predicate, obj, rendered), stats in self.signatures.items():
            pairs = {
                tuple(part.split("=", 1)) for part in rendered.split("|") if "=" in part
            }
            self.signature_index.setdefault((subject, predicate, obj), []).append(
                (pairs, stats)
            )

    def satisfies(self, edge_pairs: set, required: Sequence[tuple[str, str]]) -> bool:
        """Whether an edge's qualifiers satisfy a qualifier_set.

        Subset semantics, with the value hierarchy: a qedge asking for
        ``activity_or_abundance`` is satisfied by an edge qualified
        ``expression``, and extra qualifiers on the edge are irrelevant.
        """
        for type_id, value in required:
            if not any(
                edge_type == type_id
                and value
                in self.value_ancestors.get((edge_type, edge_value), (edge_value,))
                for edge_type, edge_value in edge_pairs
            ):
                return False
        return True

    @classmethod
    def load(cls, directory: Path) -> "Census":
        rollup: dict[tuple[str, str, str], dict] = {}
        with open(directory / "census_rollup.tsv", encoding="utf-8") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                triple_key = (
                    row["subject_category"],
                    row["predicate"],
                    row["object_category"],
                )
                rollup[triple_key] = {
                    "edges": int(row["edge_count"]),
                    "subjects": int(row["distinct_subjects"]),
                    "objects": int(row["distinct_objects"]),
                }

        qualifier_values: dict[tuple[str, str, str, str, str], dict] = {}
        path = directory / "qualifier_values.tsv"
        if path.exists():
            with open(path, encoding="utf-8") as handle:
                for row in csv.DictReader(handle, delimiter="\t"):
                    value_key = (
                        row["subject_category"],
                        row["predicate"],
                        row["object_category"],
                        row["qualifier_type_id"],
                        row["qualifier_value"],
                    )
                    qualifier_values[value_key] = {
                        "edges": int(row["edge_count"]),
                        "subjects": int(row["distinct_subjects"]),
                        "objects": int(row["distinct_objects"]),
                    }

        signatures: dict[tuple[str, str, str, str], dict] = {}
        signature_index: dict[tuple[str, str, str], list] = {}
        path = directory / "qualifier_signatures.tsv"
        if path.exists():
            with open(path, encoding="utf-8") as handle:
                for row in csv.DictReader(handle, delimiter="\t"):
                    triple = (
                        row["subject_category"],
                        row["predicate"],
                        row["object_category"],
                    )
                    rendered = row["qualifier_signature"]
                    stats = {
                        "edges": int(row["edge_count"]),
                        "subjects": int(row["distinct_subjects"]),
                        "objects": int(row["distinct_objects"]),
                    }
                    signatures[(*triple, rendered)] = stats
                    pairs = {
                        tuple(part.split("=", 1))
                        for part in rendered.split("|")
                        if "=" in part
                    }
                    signature_index.setdefault(triple, []).append((pairs, stats))

        value_ancestors: dict[tuple[str, str], tuple[str, ...]] = {}
        closure_path = directory / "biolink_closure.json"
        if closure_path.exists():
            with open(closure_path, encoding="utf-8") as handle:
                closure = json.load(handle)
            for entry in (closure.get("qualifiers") or {}).values():
                value_ancestors[
                    (entry["qualifier_type_id"], entry["qualifier_value"])
                ] = tuple(entry["ancestor_values"])

        return cls(
            rollup, qualifier_values, signatures, signature_index, value_ancestors
        )

    def stats(self, template: Template, hop: Hop) -> Optional[dict]:
        """Counts for one hop, honouring its qualifier constraints.

        A single qualifier constraint is read from the value table, which already
        unions every signature containing that value and rolls the value
        hierarchy up, exactly as the query would.

        A conjunction has to be summed over signatures instead, because TRAPI
        matches a qualifier_set as a *subset*: the census shows aspect and
        direction almost never occur alone -- they arrive bundled with
        ``qualified_predicate`` and species context -- so an exact signature
        match would find nothing.  Edge counts sum exactly; distinct endpoints
        cannot (one chemical may appear under several signatures), so the
        largest contributing signature is used, which biases the fan-out
        estimate high.  Over-budgeting is the safe direction.

        Multiple predicates take the largest matching row, since Gandalf ORs
        them.
        """
        subject_category = template.categories[hop.subject]
        object_category = template.categories[hop.object]

        best = None
        for predicate in hop.predicates:
            key = (subject_category, predicate, object_category)
            if len(hop.qualifiers) == 1:
                type_id, value = hop.qualifiers[0]
                found = self.qualifier_values.get((*key, type_id, value))
            elif hop.qualifiers:
                found = None
                matching = [
                    stats
                    for pairs, stats in self.signature_index.get(key, [])
                    if self.satisfies(pairs, hop.qualifiers)
                ]
                if matching:
                    found = {
                        "edges": sum(stats["edges"] for stats in matching),
                        "subjects": max(stats["subjects"] for stats in matching),
                        "objects": max(stats["objects"] for stats in matching),
                    }
            else:
                found = self.rollup.get(key)
            if found and (best is None or found["edges"] > best["edges"]):
                best = found
        return best


def estimate(template: Template, census: Census) -> dict:
    """Walk the query graph from the pinned node, multiplying fan-outs.

    Returns expected path count, the per-hop breakdown, and how many diseases
    can match the entry hop at all -- coverage being the number that decides
    whether a template is worth firing for an arbitrary disease.
    """
    known = {template.pinned: 1.0}
    per_hop = []
    expected = 1.0
    coverage: Optional[int] = None
    missing = []

    remaining = list(template.hops)
    while remaining:
        progressed = False
        for hop in list(remaining):
            forward = hop.subject in known
            backward = hop.object in known
            if not (forward or backward):
                continue
            remaining.remove(hop)
            progressed = True

            stats = census.stats(template, hop)
            if stats is None:
                missing.append(f"{hop.subject}->{hop.object} {hop.predicates}")
                continue

            if forward:
                fanout = stats["edges"] / max(stats["subjects"], 1)
                target, anchor_count = hop.object, stats["subjects"]
            else:
                fanout = stats["edges"] / max(stats["objects"], 1)
                target, anchor_count = hop.subject, stats["objects"]

            if coverage is None:
                coverage = anchor_count

            if target in known:
                # A closing edge constrains rather than expands; the join is not
                # modelled, so leave the running product alone and say so.
                per_hop.append(
                    {
                        "hop": f"{hop.subject} -> {hop.object}",
                        "predicates": list(hop.predicates),
                        "qualifiers": [f"{t}={v}" for t, v in hop.qualifiers],
                        "edges": stats["edges"],
                        "fanout": round(fanout, 1),
                        "role": "closes a cycle (not multiplied)",
                    }
                )
                continue

            known[target] = known.get(target, 1.0) * fanout
            expected *= fanout
            per_hop.append(
                {
                    "hop": f"{hop.subject} -> {hop.object}",
                    "predicates": list(hop.predicates),
                    "qualifiers": [f"{t}={v}" for t, v in hop.qualifiers],
                    "edges": stats["edges"],
                    "fanout": round(fanout, 1),
                    "role": "expands",
                }
            )
        if not progressed:
            break

    return {
        "template": template.name,
        "tier": template.tier,
        "leaky": template.leaky,
        "expected_paths": round(expected),
        "disease_coverage": coverage or 0,
        "hops": per_hop,
        "missing_triples": missing,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render and price drug-for-disease query templates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--census",
        type=Path,
        required=True,
        help="Census output directory (from metagraph_census.py)",
    )
    parser.add_argument("--disease", help="Disease CURIE to pin, e.g. MONDO:0004979")
    parser.add_argument(
        "--out", type=Path, help="Directory to write one TRAPI request per template"
    )
    parser.add_argument(
        "--exclude-leaky",
        action="store_true",
        help="Skip templates that read treats-family edges",
    )
    parser.add_argument(
        "--tier", action="append", default=[], help="Only these tiers (repeatable)"
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=0,
        help="Warn when the portfolio's expected paths exceed this total",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)

    census = Census.load(args.census)
    selected = [
        template
        for template in TEMPLATES
        if not (args.exclude_leaky and template.leaky)
        and (not args.tier or template.tier in args.tier)
    ]

    print(f"{'template':<26} {'tier':<14} {'exp.paths':>10} {'coverage':>9}  mechanism")
    print("-" * 110)
    total = 0
    estimates = []
    for template in selected:
        summary = estimate(template, census)
        estimates.append(summary)
        total += summary["expected_paths"]
        print(
            f"{template.name:<26} {template.tier:<14} "
            f"{summary['expected_paths']:>10,} {summary['disease_coverage']:>9,}  "
            f"{template.mechanism[:60]}"
        )
        for hop in summary["hops"]:
            qualifiers = (
                (" " + ",".join(hop["qualifiers"])) if hop["qualifiers"] else ""
            )
            print(
                f"{'':<26} {'':<14} {'':>10} {'':>9}    {hop['hop']}: "
                f"{','.join(p.replace('biolink:', '') for p in hop['predicates'])}"
                f"{qualifiers} -- {hop['edges']:,} edges, x{hop['fanout']}"
                f"{'' if hop['role'] == 'expands' else ' [' + hop['role'] + ']'}"
            )
        for gap in summary["missing_triples"]:
            print(f"{'':<26} {'':<14} {'':>10} {'':>9}    NO CENSUS ROW: {gap}")

    print("-" * 110)
    print(f"{'PORTFOLIO TOTAL':<26} {'':<14} {total:>10,} expected paths")
    if args.budget and total > args.budget:
        print(
            f"  over budget by {total - args.budget:,}: drop a Tier B template "
            "or tighten max_node_degree"
        )

    if args.disease:
        if not args.out:
            print("\n(--out not given; not writing requests)")
        else:
            args.out.mkdir(parents=True, exist_ok=True)
            for template in selected:
                request = template.render(args.disease)
                # Outside "message" so the query graph stays TRAPI-clean.
                request["_template"] = template.name
                path = args.out / f"{template.name}.json"
                with open(path, "w", encoding="utf-8") as handle:
                    json.dump(request, handle, indent=2)
            index = {
                "disease": args.disease,
                "templates": [
                    {
                        "name": template.name,
                        "tier": template.tier,
                        "leaky": template.leaky,
                        "mechanism": template.mechanism,
                        "notes": template.notes,
                        "filter_config": template.filter_config,
                        "file": f"{template.name}.json",
                        "estimate": summary,
                    }
                    for template, summary in zip(selected, estimates)
                ],
            }
            with open(args.out / "index.json", "w", encoding="utf-8") as handle:
                json.dump(index, handle, indent=2)
            print(f"\nwrote {len(selected)} requests + index.json to {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
