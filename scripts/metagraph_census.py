#!/usr/bin/env python3
"""Relabeled metagraph census for query-rule mining.

Emits one row per distinct ``(subject_category, predicate, object_category)``
triple that actually occurs in a Gandalf graph -- with edge count, distinct
subject count and distinct object count -- plus the same counts recomputed at
every Biolink *ancestor* of the predicate.  The ancestor rollup is what turns a
leaf census into a **query-granularity** census: it answers "if a qedge asks for
``biolink:affects``, how many edges / subjects / objects does it actually
match?"

Two steps, matching the spec:

1. **Biolink closure map** -- for every predicate and category observed in the
   graph, its full ancestor chain, read from the Biolink Model YAML for the
   version Gandalf targets (``settings.biolink_version``, pinned in
   ``gandalf/config.py``).  The closure is built with BMT from that tagged YAML,
   so the census uses exactly the hierarchy ``PredicateExpander`` uses at query
   time.
2. **Counts grouped by triple** -- exact (not sampled, not sketched) edge and
   distinct-endpoint counts, at leaf granularity and at every ancestor.

No TRAPI, no ground truth, no server: it reads the CSR arrays and the node store
directly (or raw KGX jsonl), so it skips the edge-property LMDB, meta_kg
regeneration and the plugin enrichers that ``CSRGraph.load_mmap`` would pull in.

Match semantics
---------------

``--match-semantics stored`` (default) rolls a leaf predicate up into every
ancestor it has: the honest structural view of the graph as stored.

``--match-semantics query`` mirrors what Gandalf would actually retrieve for a
qedge naming that ancestor, following ``PredicateExpander``:

* descendants are filtered to canonical-or-symmetric predicates;
* the inverse of the queried predicate is matched in the reverse direction, so
  those edges are counted with subject and object swapped;
* ``biolink:related_to`` matches every edge in both directions.

An edge whose predicate is symmetric matches in both orientations and is
therefore counted once per orientation -- which is what a query would return.

Everything is counted exactly -- no sampling, no sketches.  A synthetic 8M-edge,
2M-node graph with 20 predicates censuses in ~11s (stored) / ~17s (query) in
~1 GB of RAM; adding the qualifier and source facets below takes it to ~56s and
~1.4 GB on a deliberately dense worst case where every triple carries every
signature.  A 38M-edge Translator graph is therefore minutes, not hours, and
``--skip-annotations`` buys back most of it when only the predicate census is
wanted.

Qualifiers and provenance
-------------------------

Every triple is also broken down by the qualifiers and the primary knowledge
source on its edges, because a mechanism template stands or falls on those.
"Drug affects gene" matches the drug that helps and the drug that harms
identically; "drug decreases the activity of a gene whose expression the disease
increases" does not.  Three things make that breakdown query-granular:

* the **signature** table counts whole conjunctions, since a TRAPI
  ``qualifier_set`` ANDs its members -- 40% of edges carrying a direction and
  60% carrying an aspect says nothing about how many carry both;
* **qualifier values roll up their enum hierarchy** (``expression`` ->
  ``abundance`` -> ``activity_or_abundance``), mirroring ``QualifierExpander``,
  which expands a queried value down to its descendants;
* both are counted at **every ancestor predicate**, not just leaf ones, because
  a template asks for ``biolink:affects`` plus a qualifier, not for
  ``biolink:ameliorates_condition``.

All of it is read from the interned hot-path pools (two int32 arrays and a small
pickle), so it costs a fraction of the node scan.  Knowledge level and
publications live in the cold-path LMDB and are deliberately not read.

Outputs (tab-separated with a header row, written to ``--out``):

===========================  ==================================================
``manifest.json``            run provenance, graph totals, unmapped terms
``biolink_closure.json``     ancestor chains for every observed term, including
                             qualifier values
``census_leaf.tsv``          one row per occurring triple (leaf predicate)
``census_rollup.tsv``        one row per (subj_cat, ancestor_pred, obj_cat)
``census_wide.tsv``          leaf rows + the rollup as a JSON column
``predicate_summary.tsv``    per-predicate totals, own vs. whole subtree
``category_summary.tsv``     per-category node/edge counts (leaf categories)
``category_rollup.tsv``      per-ancestor-category totals + member breakdown
``qualifier_signatures.tsv`` per triple, each whole qualifier conjunction
``qualifier_values.tsv``     per triple, each qualifier value and value ancestor
``qualifier_summary.tsv``    graph-wide totals per qualifier value
``source_census.tsv``        per triple, each primary knowledge source
``census_pinned_*.tsv``      the same census with categories collapsed to
                             ``--pin-category`` values (optional)
===========================  ==================================================

Examples::

    # Census of a built graph
    python scripts/metagraph_census.py --graph data/processed/gandalf_mmap --out census/

    # Straight off a KGX dump, before building
    python scripts/metagraph_census.py --edges edges.jsonl --nodes nodes.jsonl --out census/

    # What a qedge would really match, not just what is stored
    python scripts/metagraph_census.py --graph graph_mmap/ --out census/ \\
        --match-semantics query

    # Is biolink:ChemicalEntity a clean drug-side pin?
    python scripts/metagraph_census.py --graph graph_mmap/ --out census/ \\
        --pin-category biolink:ChemicalEntity --pin-category biolink:Gene \\
        --pin-category biolink:Disease --pin-category biolink:PhenotypicFeature
"""

from __future__ import annotations

import argparse
import array
import csv
import gzip
import json
import logging
import pickle
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, Sequence

import numpy as np

# Allow `python scripts/metagraph_census.py` from a checkout, not just an
# installed gandalf: running a file directly puts scripts/ on sys.path, not the
# repository root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger("metagraph_census")

DEFAULT_CATEGORY = "biolink:NamedThing"
ROOT_PREDICATE = "biolink:related_to"

_SCHEMA_URL = (
    "https://raw.githubusercontent.com/biolink/biolink-model/"
    "refs/tags/v{version}/biolink-model.yaml"
)
_PREDICATE_MAP_URL = (
    "https://raw.githubusercontent.com/biolink/biolink-model/"
    "refs/tags/v{version}/predicate_mapping.yaml"
)


# ---------------------------------------------------------------------------
# Step 1: Biolink closure map
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BiolinkClosure:
    """Ancestor chains for every predicate and category seen in the graph.

    Chains are *reflexive* and start with the term itself, e.g.
    ``predicate_ancestors["biolink:ameliorates_condition"]`` is
    ``("biolink:ameliorates_condition", "biolink:treats", "biolink:affects",
    "biolink:related_to_at_instance_level", "biolink:related_to",
    "biolink:treats_or_applied_or_studied_to_treat")``.  Mixin parents are
    included by default because BMT includes them when expanding a query
    predicate to its descendants.

    ``depths`` are measured on the ``is_a`` chain only (mixins excluded), so
    "how deep is this term" stays a tree distance rather than a DAG artifact.
    """

    version: str
    schema: str
    predicate_ancestors: dict[str, tuple[str, ...]]
    category_ancestors: dict[str, tuple[str, ...]]
    predicate_meta: dict[str, dict]
    predicate_depths: dict[str, int]
    category_depths: dict[str, int]
    unmapped_predicates: tuple[str, ...]
    unmapped_categories: tuple[str, ...]
    # (qualifier_type_id, qualifier_value) -> reflexive ancestor values, most
    # specific first.  Values live in Biolink enums rather than the class or
    # slot hierarchy, except ``biolink:qualified_predicate`` whose values are
    # predicate CURIEs -- which is exactly how QualifierExpander treats them.
    qualifier_ancestors: dict[tuple[str, str], tuple[str, ...]] = field(
        default_factory=dict
    )
    # qualifier_type_id -> the enum its values are drawn from ("" if unknown)
    qualifier_enums: dict[str, str] = field(default_factory=dict)
    unmapped_qualifier_values: tuple[tuple[str, str], ...] = ()

    def qualifier_value_ancestors(self, type_id: str, value: str) -> tuple[str, ...]:
        """Reflexive ancestor values for one qualifier assertion.

        A qedge asking for ``activity_or_abundance`` matches edges qualified
        with ``activity``, so rolling an edge's value up to its ancestors is
        what makes the qualifier census query-granular.
        """
        return self.qualifier_ancestors.get((type_id, value), (value,))

    def predicate_depth(self, predicate: str) -> int:
        """``is_a`` distance from the predicate root (-1 if not in the model)."""
        return self.predicate_depths.get(predicate, -1)

    def category_depth(self, category: str) -> int:
        """``is_a`` distance from ``biolink:Entity`` (-1 if not in the model)."""
        return self.category_depths.get(category, -1)

    def is_canonical_or_symmetric(self, predicate: str) -> bool:
        """Whether a query would expand *into* this predicate.

        Mirrors ``PredicateExpander.get_filtered_descendants``, which keeps only
        canonical or symmetric descendants.
        """
        meta = self.predicate_meta.get(predicate, {})
        return bool(meta.get("canonical") or meta.get("symmetric"))

    def inverse(self, predicate: str) -> Optional[str]:
        """The predicate's inverse, if the model defines one."""
        return self.predicate_meta.get(predicate, {}).get("inverse")

    def to_json(self) -> dict:
        """Serializable form, written to ``biolink_closure.json``."""
        return {
            "biolink_version": self.version,
            "schema": self.schema,
            "predicates": {
                predicate: {
                    "ancestors": list(chain),
                    "depth": self.predicate_depth(predicate),
                    **self.predicate_meta.get(predicate, {}),
                }
                for predicate, chain in sorted(self.predicate_ancestors.items())
            },
            "categories": {
                category: {
                    "ancestors": list(chain),
                    "depth": self.category_depth(category),
                }
                for category, chain in sorted(self.category_ancestors.items())
            },
            "qualifiers": {
                f"{type_id} = {value}": {
                    "qualifier_type_id": type_id,
                    "qualifier_value": value,
                    "enum": self.qualifier_enums.get(type_id, ""),
                    "ancestor_values": list(chain),
                }
                for (type_id, value), chain in sorted(self.qualifier_ancestors.items())
            },
            "unmapped_predicates": list(self.unmapped_predicates),
            "unmapped_categories": list(self.unmapped_categories),
            "unmapped_qualifier_values": [
                {"qualifier_type_id": type_id, "qualifier_value": value}
                for type_id, value in self.unmapped_qualifier_values
            ],
        }


def _make_toolkit(version: Optional[str], schema: Optional[str]):
    """Build a BMT Toolkit for *version*, or from an explicit *schema* URL/path.

    With neither argument this defers to :func:`gandalf.biolink.make_toolkit`,
    i.e. whatever Biolink version Gandalf itself targets.
    """
    from bmt.toolkit import Toolkit

    if schema:
        logger.info("Building Biolink closure from schema %s", schema)
        return Toolkit(schema=schema), schema
    if version:
        url = _SCHEMA_URL.format(version=version)
        logger.info("Building Biolink closure from biolink-model v%s", version)
        return (
            Toolkit(
                schema=url,
                predicate_map=_PREDICATE_MAP_URL.format(version=version),
            ),
            url,
        )

    from gandalf.biolink import make_toolkit
    from gandalf.config import settings

    logger.info(
        "Building Biolink closure from the version Gandalf targets (v%s)",
        settings.biolink_version or "bmt default",
    )
    url = (
        _SCHEMA_URL.format(version=settings.biolink_version)
        if settings.biolink_version
        else "bmt built-in default schema"
    )
    return make_toolkit(), url


def build_closure_map(
    predicates: Sequence[str],
    categories: Sequence[str],
    qualifiers: Sequence[tuple[str, str]] = (),
    *,
    version: Optional[str] = None,
    schema: Optional[str] = None,
    include_mixins: bool = True,
) -> BiolinkClosure:
    """Look up the full ancestor chain of every observed term.

    Covers all three vocabularies a template draws on: predicates, categories,
    and ``(qualifier_type_id, qualifier_value)`` assertions.

    Terms the model does not know (typos, retired predicates, non-Biolink
    categories) keep a single-element chain and are reported in ``unmapped_*``
    rather than silently dropped -- an unmapped predicate is a data-quality
    finding, not something to hide.

    Ancestors that no edge uses are added to the map too, so a rollup row for
    ``biolink:affects`` carries its own depth and flags even when nothing in the
    graph is labelled ``biolink:affects``.
    """
    toolkit, schema_used = _make_toolkit(version, schema)
    resolved_version = toolkit.get_model_version()

    predicate_ancestors: dict[str, tuple[str, ...]] = {}
    predicate_meta: dict[str, dict] = {}
    predicate_depths: dict[str, int] = {}
    unmapped_predicates: list[str] = []

    def add_predicate(predicate: str) -> tuple[str, ...]:
        chain = _ancestors(toolkit, predicate, include_mixins)
        if chain is None:
            unmapped_predicates.append(predicate)
            chain = (predicate,)
            predicate_depths[predicate] = -1
        else:
            isa_chain = _ancestors(toolkit, predicate, include_mixins=False) or ()
            predicate_depths[predicate] = len(isa_chain) - 1
        predicate_ancestors[predicate] = chain
        predicate_meta[predicate] = _predicate_metadata(toolkit, predicate)
        return chain

    category_ancestors: dict[str, tuple[str, ...]] = {}
    category_depths: dict[str, int] = {}
    unmapped_categories: list[str] = []

    def add_category(category: str) -> tuple[str, ...]:
        chain = _ancestors(toolkit, category, include_mixins)
        if chain is None:
            unmapped_categories.append(category)
            chain = (category,)
            category_depths[category] = -1
        else:
            isa_chain = _ancestors(toolkit, category, include_mixins=False) or ()
            category_depths[category] = len(isa_chain) - 1
        category_ancestors[category] = chain
        return chain

    for predicate in sorted(set(predicates)):
        if predicate and predicate not in predicate_ancestors:
            add_predicate(predicate)
    for category in sorted(set(categories)):
        if category and category not in category_ancestors:
            add_category(category)

    for chain in list(predicate_ancestors.values()):
        for ancestor in chain[1:]:
            if ancestor not in predicate_ancestors:
                add_predicate(ancestor)
    for chain in list(category_ancestors.values()):
        for ancestor in chain[1:]:
            if ancestor not in category_ancestors:
                add_category(ancestor)

    enums_by_type: dict[str, set[str]] = {}
    qualifier_ancestors: dict[tuple[str, str], tuple[str, ...]] = {}
    unmapped_qualifier_values: list[tuple[str, str]] = []
    for type_id, value in sorted(set(qualifiers)):
        enum_names = _qualifier_enums(toolkit, type_id, value)
        enums_by_type.setdefault(type_id, set()).update(enum_names)
        value_chain = _qualifier_value_ancestors(
            toolkit, type_id, value, enum_names, predicate_ancestors
        )
        if value_chain is None:
            unmapped_qualifier_values.append((type_id, value))
            value_chain = (value,)
        qualifier_ancestors[(type_id, value)] = value_chain
    qualifier_enums = {
        type_id: "|".join(sorted(names)) for type_id, names in enums_by_type.items()
    }

    logger.info(
        "  closure: %d predicates (%d unmapped), %d categories (%d unmapped), "
        "%d qualifier values (%d unmapped)",
        len(predicate_ancestors),
        len(unmapped_predicates),
        len(category_ancestors),
        len(unmapped_categories),
        len(qualifier_ancestors),
        len(unmapped_qualifier_values),
    )
    return BiolinkClosure(
        version=resolved_version,
        schema=schema_used,
        predicate_ancestors=predicate_ancestors,
        category_ancestors=category_ancestors,
        predicate_meta=predicate_meta,
        predicate_depths=predicate_depths,
        category_depths=category_depths,
        unmapped_predicates=tuple(sorted(set(unmapped_predicates))),
        unmapped_categories=tuple(sorted(set(unmapped_categories))),
        qualifier_ancestors=qualifier_ancestors,
        qualifier_enums=qualifier_enums,
        unmapped_qualifier_values=tuple(sorted(set(unmapped_qualifier_values))),
    )


def _ancestors(toolkit, term: str, include_mixins: bool) -> Optional[tuple[str, ...]]:
    """Reflexive ancestor chain for *term*, the term itself first.

    Returns ``None`` when the model has no such element.
    """
    if toolkit.get_element(term) is None:
        return None
    chain = toolkit.get_ancestors(
        term, reflexive=True, formatted=True, mixin=include_mixins
    )
    ordered = list(dict.fromkeys(chain))
    if term in ordered:
        ordered.remove(term)
    return (term, *ordered)


def _predicate_metadata(toolkit, predicate: str) -> dict:
    """Flags that decide whether a predicate is usable in a query rule.

    ``inverse`` is gated on ``has_inverse`` because that is what
    ``PredicateExpander.get_inverse`` gates on, and query semantics must mirror
    Gandalf rather than the model in the abstract.  The distinction is real: the
    model declares the ``inverse`` slot only on the non-canonical side, so
    ``biolink:affected_by`` has an inverse but ``biolink:affects`` does not --
    meaning a qedge for ``affects`` does *not* reach stored ``affected_by``
    edges.  ``model_inverse`` records the pairing anyway, since knowing the two
    predicates are two faces of one relation matters when mining rules.
    """
    element = toolkit.get_element(predicate)
    if element is None:
        return {"in_model": False}
    annotations = getattr(element, "annotations", None) or {}
    canonical = annotations.get("canonical_predicate")
    model_inverse = _model_inverse(toolkit, predicate)
    return {
        "in_model": True,
        "symmetric": bool(getattr(element, "symmetric", False)),
        "canonical": bool(getattr(canonical, "value", canonical)),
        "mixin": bool(getattr(element, "mixin", False)),
        "abstract": bool(getattr(element, "abstract", False)),
        "deprecated": bool(getattr(element, "deprecated", None)),
        "inverse": model_inverse if toolkit.has_inverse(predicate) else None,
        "model_inverse": model_inverse,
    }


QUALIFIED_PREDICATE = "biolink:qualified_predicate"


def _qualifier_enums(toolkit, type_id: str, value: str) -> list[str]:
    """Enums that admit *value* for this qualifier type.

    The slot's own ``range`` is the fast path, but the interesting qualifier
    slots are abstract and declare no range -- ``object_aspect_qualifier`` gets
    its range from per-predicate ``slot_usage`` -- so the fallback scans all 26
    enums for the value.  That is exactly what ``QualifierExpander`` does at
    query time, and it is cheap here: a graph has hundreds of distinct
    ``(type, value)`` pairs, not millions.
    """
    if type_id == QUALIFIED_PREDICATE:
        return []
    element = toolkit.get_element(type_id)
    range_name = getattr(element, "range", None) if element is not None else None
    all_enums = toolkit.view.all_enums()
    if (
        range_name
        and range_name in all_enums
        and toolkit.is_permissible_value_of_enum(enum_name=range_name, value=value)
    ):
        return [str(range_name)]
    return [
        str(enum_name)
        for enum_name in all_enums
        if toolkit.is_permissible_value_of_enum(enum_name=enum_name, value=value)
    ]


def _qualifier_value_ancestors(
    toolkit,
    type_id: str,
    value: str,
    enum_names: Sequence[str],
    predicate_ancestors: dict[str, tuple[str, ...]],
) -> Optional[tuple[str, ...]]:
    """Reflexive ancestor chain for a qualifier value, most specific first.

    ``qualified_predicate`` values are predicate CURIEs and climb the predicate
    hierarchy; everything else climbs its enum's permissible-value hierarchy
    (``expression`` -> ``abundance`` -> ``activity_or_abundance``).  When a
    value appears in several enums the chains are unioned, matching
    ``QualifierExpander``, which searches every enum.  Returns ``None`` when the
    model cannot place the value at all.
    """
    if type_id == QUALIFIED_PREDICATE:
        return predicate_ancestors.get(value) or _ancestors(
            toolkit, value, include_mixins=True
        )

    if not enum_names:
        return None
    ordered: list[str] = []
    for enum_name in enum_names:
        ordered.extend(
            str(ancestor)
            for ancestor in toolkit.get_permissible_value_ancestors(
                permissible_value=value, enum_name=enum_name
            )
            or ()
        )
    ordered = [ancestor for ancestor in dict.fromkeys(ordered) if ancestor != value]
    return (value, *ordered)


def _model_inverse(toolkit, predicate: str) -> Optional[str]:
    """The inverse the model declares for *predicate*, or ``None``.

    Guarded against an upstream defect rather than a Gandalf one: BMT resolves a
    CURIE by turning underscores into spaces (``biolink:foo_bar`` -> ``foo
    bar``), but a couple of Biolink slots are literally named with underscores
    (``gene_fusion_with`` and ``genetic_neighborhood_of`` in 4.3.2).  For those,
    ``SchemaView.inverse`` fails to resolve the element and then dereferences
    ``None``, raising ``AttributeError``.  Neither predicate declares an
    inverse, so treating the failure as "no inverse" loses nothing -- and it is
    the same defensive stance ``PredicateExpander`` takes around BMT calls.
    """
    try:
        inverse: Optional[str] = toolkit.get_inverse_predicate(
            predicate, formatted=True
        )
        return inverse
    except AttributeError:
        logger.debug(
            "  BMT could not resolve an inverse for %s "
            "(biolink slot name contains underscores); treating as no inverse",
            predicate,
        )
        return None


# ---------------------------------------------------------------------------
# Graph loading -- CSR mmap directory or raw KGX jsonl
# ---------------------------------------------------------------------------


@dataclass
class GraphArrays:
    """Edge list and node labels as flat integer arrays.

    ``subjects``/``objects`` hold node indices, ``predicate_codes`` indexes
    ``predicates``, and ``node_category_codes`` maps a node index to its single
    *primary* category (an index into ``categories``).
    """

    subjects: np.ndarray
    objects: np.ndarray
    predicate_codes: np.ndarray
    predicates: list[str]
    node_category_codes: np.ndarray
    categories: list[str]
    num_nodes: int
    source: str
    multi_category_nodes: int = 0
    dangling_endpoints: int = 0
    # Per-edge annotations, aligned with the edge arrays.  ``qualifier_codes``
    # indexes ``qualifier_signatures`` (the full conjunction of qualifiers on an
    # edge, which is what a TRAPI qualifier_set matches); ``source_codes``
    # indexes ``primary_sources``.  Both are None when the graph does not carry
    # them or the census was told to skip them.
    qualifier_codes: Optional[np.ndarray] = None
    qualifier_signatures: list[tuple[tuple[str, str], ...]] = field(
        default_factory=list
    )
    source_codes: Optional[np.ndarray] = None
    primary_sources: list[str] = field(default_factory=list)

    @property
    def num_edges(self) -> int:
        return int(self.subjects.shape[0])

    @property
    def qualifier_pairs(self) -> list[tuple[str, str]]:
        """Every distinct ``(qualifier_type_id, qualifier_value)`` observed."""
        return sorted(
            {pair for signature in self.qualifier_signatures for pair in signature}
        )


class _SignatureInterner:
    """Interns a tuple-valued per-edge annotation to a small integer code."""

    def __init__(self) -> None:
        self.values: list = []
        self._index: dict = {}

    def intern(self, value):
        code = self._index.get(value)
        if code is None:
            code = len(self.values)
            self._index[value] = code
            self.values.append(value)
        return code


def qualifier_signature(qualifiers) -> tuple[tuple[str, str], ...]:
    """Normalize an edge's qualifier list into a sorted, hashable signature.

    >>> qualifier_signature([
    ...     {"qualifier_type_id": "biolink:object_direction_qualifier",
    ...      "qualifier_value": "decreased"},
    ...     {"qualifier_type_id": "biolink:object_aspect_qualifier",
    ...      "qualifier_value": "activity"}])
    (('biolink:object_aspect_qualifier', 'activity'), ('biolink:object_direction_qualifier', 'decreased'))
    """
    if not qualifiers:
        return ()
    pairs = []
    for qualifier in qualifiers:
        type_id = qualifier.get("qualifier_type_id")
        value = qualifier.get("qualifier_value")
        if type_id and value is not None:
            pairs.append((str(type_id), str(value)))
    return tuple(sorted(set(pairs)))


def primary_source(sources) -> str:
    """The ``primary_knowledge_source`` resource_id for an edge.

    Falls back to the first source when no entry carries the primary role, and
    to ``""`` when the edge has no sources at all -- both are worth seeing in
    the census rather than silently bucketing as unknown.
    """
    if not sources:
        return ""
    for source in sources:
        if source.get("resource_role") == "primary_knowledge_source":
            return str(source.get("resource_id") or "")
    return str(sources[0].get("resource_id") or "")


class _CategoryComboInterner:
    """Interns the *list* of categories a node declares.

    Real KGs have millions of nodes but only a few thousand distinct category
    combinations, so interning the combo lets us defer the choice of primary
    category until after the Biolink closure is built -- the policy needs the
    hierarchy, and the hierarchy is built from the observed vocabulary.
    """

    def __init__(self) -> None:
        self.combos: list[tuple[str, ...]] = []
        self._index: dict[tuple[str, ...], int] = {}

    def intern(self, categories: Sequence[str]) -> int:
        combo = tuple(categories) if categories else (DEFAULT_CATEGORY,)
        code = self._index.get(combo)
        if code is None:
            code = len(self.combos)
            self._index[combo] = code
            self.combos.append(combo)
        return code

    @property
    def vocabulary(self) -> set[str]:
        return {category for combo in self.combos for category in combo}


def choose_primary_category(
    combo: Sequence[str], closure: BiolinkClosure, policy: str
) -> str:
    """Reduce a node's declared categories to the one the census labels it with.

    ``most-specific`` (default) prefers the category that has every other
    declared category among its ancestors -- for ``[ChemicalEntity, Drug,
    NamedThing]`` that is ``Drug``.  When the declared set is not a single
    chain (a node labelled both ``Drug`` and ``Protein``), the deepest term
    wins, ties broken alphabetically so runs are reproducible.

    ``first`` reproduces ``CSRGraph._build_node_categories`` (``categories[0]``),
    which is what the served meta_kg uses.

    >>> closure = BiolinkClosure(
    ...     "test", "test",
    ...     {}, {"biolink:Drug": ("biolink:Drug", "biolink:ChemicalEntity"),
    ...          "biolink:ChemicalEntity": ("biolink:ChemicalEntity",)},
    ...     {}, {}, {"biolink:Drug": 2, "biolink:ChemicalEntity": 1}, (), ())
    >>> choose_primary_category(
    ...     ["biolink:ChemicalEntity", "biolink:Drug"], closure, "most-specific")
    'biolink:Drug'
    >>> choose_primary_category(
    ...     ["biolink:ChemicalEntity", "biolink:Drug"], closure, "first")
    'biolink:ChemicalEntity'
    """
    if policy == "first":
        return combo[0]

    declared = set(combo)
    covering = [
        category
        for category in combo
        if declared <= set(closure.category_ancestors.get(category, (category,)))
    ]
    candidates = covering or list(combo)
    return min(
        candidates, key=lambda category: (-closure.category_depth(category), category)
    )


def _resolve_primary_categories(
    interner: _CategoryComboInterner, closure: BiolinkClosure, policy: str
) -> tuple[np.ndarray, list[str]]:
    """Map every interned combo to a primary-category code."""
    categories: list[str] = []
    category_index: dict[str, int] = {}
    combo_to_code = np.empty(len(interner.combos), dtype=np.int32)

    for combo_code, combo in enumerate(interner.combos):
        primary = choose_primary_category(combo, closure, policy)
        code = category_index.get(primary)
        if code is None:
            code = len(categories)
            category_index[primary] = code
            categories.append(primary)
        combo_to_code[combo_code] = code

    return combo_to_code, categories


ClosureBuilder = Callable[
    [Sequence[str], Sequence[str], Sequence[tuple[str, str]]], BiolinkClosure
]


def node_categories(record: dict) -> list[str]:
    """Categories for a node, however the record spells them.

    ``gandalf.normalize.normalize_node`` reads KGX's singular ``category`` and
    files anything else under ``attributes``, so a graph built from a dump that
    used the plural ``categories`` key ends up with an empty ``categories``
    field and the real list stashed in an attribute.  Checking both keeps the
    census from silently labelling such a graph as all ``biolink:NamedThing``.
    """
    categories = record.get("categories") or record.get("category") or []
    if isinstance(categories, str):
        categories = [categories]
    if categories:
        return list(categories)
    for attribute in record.get("attributes") or []:
        if attribute.get("original_attribute_name") in ("categories", "category"):
            value = attribute.get("value") or []
            return [value] if isinstance(value, str) else list(value)
    return []


def load_from_mmap(
    graph_dir: Path,
    closure_builder: ClosureBuilder,
    policy: str,
    annotations: bool = True,
) -> tuple[GraphArrays, BiolinkClosure]:
    """Read a saved Gandalf mmap graph directory.

    Reads only what the census needs: the forward CSR arrays, the predicate
    vocabulary from ``metadata.pkl``, node categories from the node store, and
    -- when *annotations* is set -- the interned qualifier and source pools.

    Those pools are the cheap half of the graph: ``edge_quals_idx.npy`` and
    ``edge_sources_idx.npy`` are int32 arrays reordered by the same permutation
    as the forward CSR (``loader.py`` calls ``prop_builder.reorder(sort_order)``),
    so position *i* in them is the same edge as position *i* in
    ``fwd_targets.npy``.  Knowledge level and publications live in the cold-path
    LMDB instead and would cost a full scan, so they are deliberately not read.
    """
    metadata_path = graph_dir / "metadata.pkl"
    if not metadata_path.exists():
        raise FileNotFoundError(f"{metadata_path} not found -- is this a graph dir?")

    with open(metadata_path, "rb") as handle:
        metadata = pickle.load(handle)

    num_nodes = int(metadata["num_nodes"])
    predicate_to_idx = metadata["predicate_to_idx"]
    predicates = [""] * (max(predicate_to_idx.values(), default=-1) + 1)
    for predicate, idx in predicate_to_idx.items():
        predicates[idx] = predicate

    logger.info("Reading CSR arrays from %s", graph_dir)
    fwd_offsets = np.load(graph_dir / "fwd_offsets.npy")
    objects = np.array(np.load(graph_dir / "fwd_targets.npy", mmap_mode="r"))
    predicate_codes = np.array(np.load(graph_dir / "fwd_predicates.npy", mmap_mode="r"))
    subjects = np.repeat(np.arange(num_nodes, dtype=np.int32), np.diff(fwd_offsets))
    logger.info("  %s nodes, %s edges", f"{num_nodes:,}", f"{len(objects):,}")

    interner = _CategoryComboInterner()
    node_combo_codes = np.full(num_nodes, interner.intern([DEFAULT_CATEGORY]), np.int32)
    multi_category_nodes = 0

    node_store_path = graph_dir / "node_store.lmdb"
    if node_store_path.exists():
        from gandalf.node_store import NodeStore

        logger.info("  Reading node categories from %s", node_store_path.name)
        with NodeStore(node_store_path, readonly=True) as store:
            for count, (node_idx, props) in enumerate(store.iter_properties(), 1):
                categories = node_categories(props) or [DEFAULT_CATEGORY]
                if len(categories) > 1:
                    multi_category_nodes += 1
                if node_idx < num_nodes:
                    node_combo_codes[node_idx] = interner.intern(categories)
                if count % 2_000_000 == 0:
                    logger.info("    %s nodes read", f"{count:,}")
    else:
        logger.info("  Legacy graph: reading node categories from metadata.pkl")
        for node_idx, props in (metadata.get("node_properties") or {}).items():
            categories = node_categories(props) or [DEFAULT_CATEGORY]
            if len(categories) > 1:
                multi_category_nodes += 1
            node_combo_codes[int(node_idx)] = interner.intern(categories)

    qualifier_codes, qualifier_signatures, source_codes, primary_sources = (
        _read_edge_annotations(graph_dir, len(objects))
        if annotations
        else (None, [], None, [])
    )

    closure = closure_builder(
        predicates,
        sorted(interner.vocabulary),
        sorted({pair for sig in qualifier_signatures for pair in sig}),
    )
    combo_to_code, categories = _resolve_primary_categories(interner, closure, policy)

    arrays = GraphArrays(
        subjects=subjects,
        objects=objects.astype(np.int32, copy=False),
        predicate_codes=predicate_codes.astype(np.int32, copy=False),
        predicates=predicates,
        node_category_codes=combo_to_code[node_combo_codes],
        categories=categories,
        num_nodes=num_nodes,
        source=str(graph_dir),
        multi_category_nodes=multi_category_nodes,
        qualifier_codes=qualifier_codes,
        qualifier_signatures=qualifier_signatures,
        source_codes=source_codes,
        primary_sources=primary_sources,
    )
    return arrays, closure


def _read_edge_annotations(
    graph_dir: Path, num_edges: int
) -> tuple[Optional[np.ndarray], list, Optional[np.ndarray], list]:
    """Read the interned qualifier and source pools for every edge.

    Returns ``(qualifier_codes, qualifier_signatures, source_codes,
    primary_sources)``.  The stored pools are re-interned rather than used
    directly: two pool entries can normalize to the same signature (qualifier
    order differs, or two source chains share a primary), and the census wants
    one code per distinct *meaning*.
    """
    pools_path = graph_dir / "edge_property_pools.pkl"
    quals_idx_path = graph_dir / "edge_quals_idx.npy"
    sources_idx_path = graph_dir / "edge_sources_idx.npy"
    if not (
        pools_path.exists() and quals_idx_path.exists() and sources_idx_path.exists()
    ):
        logger.warning(
            "  No edge property pools in %s; skipping qualifier/source census",
            graph_dir,
        )
        return None, [], None, []

    logger.info("  Reading qualifier and source pools")
    with open(pools_path, "rb") as handle:
        pools = pickle.load(handle)

    qualifier_interner = _SignatureInterner()
    quals_pool_map = np.array(
        [
            qualifier_interner.intern(qualifier_signature(entry))
            for entry in pools["quals_pool"]
        ],
        dtype=np.int32,
    )
    source_interner = _SignatureInterner()
    sources_pool_map = np.array(
        [
            source_interner.intern(primary_source(entry))
            for entry in pools["sources_pool"]
        ],
        dtype=np.int32,
    )

    quals_idx = np.load(quals_idx_path, mmap_mode="r")
    sources_idx = np.load(sources_idx_path, mmap_mode="r")
    if len(quals_idx) != num_edges or len(sources_idx) != num_edges:
        raise ValueError(
            f"edge property index length ({len(quals_idx)}, {len(sources_idx)}) "
            f"does not match edge count ({num_edges}); graph directory is inconsistent"
        )

    qualifier_codes = quals_pool_map[np.asarray(quals_idx)]
    source_codes = sources_pool_map[np.asarray(sources_idx)]
    logger.info(
        "    %s distinct qualifier signatures, %s distinct primary sources",
        f"{len(qualifier_interner.values):,}",
        f"{len(source_interner.values):,}",
    )
    return (
        qualifier_codes,
        qualifier_interner.values,
        source_codes,
        source_interner.values,
    )


def _open_maybe_gzip(path: Path):
    """Open a jsonl file, transparently handling ``.gz``."""
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def _json_loader() -> Callable[[str], Any]:
    """The fastest JSON line parser available (orjson is ~3x json)."""
    from importlib.util import find_spec

    if find_spec("orjson") is not None:
        import orjson

        return orjson.loads
    return json.loads


def _iter_json_lines(path: Path) -> Iterator[dict]:
    """Stream one parsed object per line."""
    loads = _json_loader()
    with _open_maybe_gzip(path) as handle:
        for line in handle:
            if line.strip():
                yield loads(line)


def edge_qualifier_signature(edge: dict) -> tuple[tuple[str, str], ...]:
    """Qualifier signature for a KGX edge record, normalized or raw.

    ``gandalf.normalize._extract_qualifiers`` lifts raw KGX's top-level
    ``object_aspect_qualifier``-style fields into a TRAPI ``qualifiers`` list
    and prefixes the type with ``biolink:``.  A raw dump has not been through
    that yet, so recognize both spellings -- otherwise the census reports zero
    qualifier coverage on exactly the input where you most want to check it.

    >>> edge_qualifier_signature(
    ...     {"object_aspect_qualifier": "activity", "qualified_predicate": "causes"})
    (('biolink:object_aspect_qualifier', 'activity'), ('biolink:qualified_predicate', 'biolink:causes'))
    """
    if edge.get("qualifiers"):
        return qualifier_signature(edge["qualifiers"])
    pairs = []
    for key, value in edge.items():
        if value is None or not (
            key.endswith("_qualifier") or key == "qualified_predicate"
        ):
            continue
        text = value if isinstance(value, str) else json.dumps(value)
        if key == "qualified_predicate" and not text.startswith("biolink:"):
            text = f"biolink:{text}"
        pairs.append((f"biolink:{key}", text))
    return tuple(sorted(set(pairs)))


def edge_primary_source(edge: dict) -> str:
    """Primary knowledge source for a KGX edge record, normalized or raw."""
    if edge.get("sources"):
        return primary_source(edge["sources"])
    return str(edge.get("primary_knowledge_source") or "")


def load_from_jsonl(
    edges_path: Path,
    nodes_path: Optional[Path],
    closure_builder: ClosureBuilder,
    policy: str,
    annotations: bool = True,
) -> tuple[GraphArrays, BiolinkClosure]:
    """Read KGX ``nodes.jsonl`` / ``edges.jsonl`` without building a graph.

    Node IDs are interned to indices as they are encountered.  Edge endpoints
    absent from the node file get ``biolink:NamedThing`` and are counted, so a
    dangling-endpoint problem shows up in the manifest instead of quietly
    inflating the NamedThing rows.

    Qualifiers and sources are read per record when *annotations* is set, from
    either the normalized form or raw KGX's top-level fields.
    """
    interner = _CategoryComboInterner()
    default_combo = interner.intern([DEFAULT_CATEGORY])
    node_id_to_idx: dict[str, int] = {}
    node_combo_codes = array.array("i")
    multi_category_nodes = 0

    if nodes_path is not None:
        logger.info("Reading nodes from %s", nodes_path)
        for count, node in enumerate(_iter_json_lines(nodes_path), 1):
            node_id = node.get("id")
            if not node_id or node_id in node_id_to_idx:
                continue
            categories = node_categories(node) or [DEFAULT_CATEGORY]
            if len(categories) > 1:
                multi_category_nodes += 1
            node_id_to_idx[node_id] = len(node_combo_codes)
            node_combo_codes.append(interner.intern(categories))
            if count % 2_000_000 == 0:
                logger.info("  %s nodes read", f"{count:,}")

    logger.info("Reading edges from %s", edges_path)
    predicates: list[str] = []
    predicate_index: dict[str, int] = {}
    subjects = array.array("i")
    objects = array.array("i")
    predicate_codes = array.array("i")
    qualifier_codes = array.array("i")
    source_codes = array.array("i")
    qualifier_interner = _SignatureInterner()
    source_interner = _SignatureInterner()
    dangling_endpoints = 0

    def node_index(node_id: str) -> int:
        nonlocal dangling_endpoints
        idx = node_id_to_idx.get(node_id)
        if idx is None:
            idx = len(node_combo_codes)
            node_id_to_idx[node_id] = idx
            node_combo_codes.append(default_combo)
            dangling_endpoints += 1
        return idx

    for count, edge in enumerate(_iter_json_lines(edges_path), 1):
        predicate = edge["predicate"]
        predicate_code = predicate_index.get(predicate)
        if predicate_code is None:
            predicate_code = len(predicates)
            predicate_index[predicate] = predicate_code
            predicates.append(predicate)

        subjects.append(node_index(edge["subject"]))
        objects.append(node_index(edge["object"]))
        predicate_codes.append(predicate_code)
        if annotations:
            qualifier_codes.append(
                qualifier_interner.intern(edge_qualifier_signature(edge))
            )
            source_codes.append(source_interner.intern(edge_primary_source(edge)))
        if count % 5_000_000 == 0:
            logger.info("  %s edges read", f"{count:,}")

    if dangling_endpoints:
        logger.warning(
            "  %s edge endpoints had no node record; labelled %s",
            f"{dangling_endpoints:,}",
            DEFAULT_CATEGORY,
        )

    closure = closure_builder(
        predicates,
        sorted(interner.vocabulary),
        sorted({pair for sig in qualifier_interner.values for pair in sig}),
    )
    combo_to_code, categories = _resolve_primary_categories(interner, closure, policy)

    arrays = GraphArrays(
        subjects=np.frombuffer(subjects, dtype=np.int32),
        objects=np.frombuffer(objects, dtype=np.int32),
        predicate_codes=np.frombuffer(predicate_codes, dtype=np.int32),
        predicates=predicates,
        node_category_codes=combo_to_code[
            np.frombuffer(node_combo_codes, dtype=np.int32)
        ],
        categories=categories,
        num_nodes=len(node_combo_codes),
        source=str(edges_path),
        multi_category_nodes=multi_category_nodes,
        dangling_endpoints=dangling_endpoints,
        qualifier_codes=(
            np.frombuffer(qualifier_codes, dtype=np.int32) if annotations else None
        ),
        qualifier_signatures=qualifier_interner.values,
        source_codes=(
            np.frombuffer(source_codes, dtype=np.int32) if annotations else None
        ),
        primary_sources=source_interner.values,
    )
    logger.info(
        "  %s nodes, %s edges", f"{arrays.num_nodes:,}", f"{arrays.num_edges:,}"
    )
    return arrays, closure


# ---------------------------------------------------------------------------
# Step 2: exact grouped counts
# ---------------------------------------------------------------------------


def group_stats(
    keys: np.ndarray, subjects: np.ndarray, objects: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Exact per-group edge, distinct-subject and distinct-object counts.

    Distinct counts are exact, not sketched: ``(key, endpoint)`` is packed into
    a single int64, sorted, and deduplicated.  Packing rather than
    ``np.lexsort`` matters at graph scale -- one value sort per endpoint instead
    of an argsort over two int64 columns is ~15x faster on 8M edges, with no
    per-group Python objects either way.

    Returns ``(unique_keys, edge_counts, distinct_subjects, distinct_objects)``
    with ``unique_keys`` ascending; the three count arrays are aligned to it.

    >>> keys = np.array([1, 1, 1, 2], dtype=np.int64)
    >>> subjects = np.array([10, 10, 11, 12], dtype=np.int32)
    >>> objects = np.array([20, 21, 21, 22], dtype=np.int32)
    >>> [a.tolist() for a in group_stats(keys, subjects, objects)]
    [[1, 2], [3, 1], [2, 1], [2, 1]]
    """
    if keys.shape[0] == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty.copy(), empty.copy(), empty.copy()

    # Dense codes rather than the raw keys: a facet key such as
    # (triple, qualifier combo) is already a product of three vocabularies, and
    # packing *that* against a node index would overflow int64 on a large graph.
    # Densifying first bounds the key by the number of groups actually present.
    unique_keys, dense, edge_counts = np.unique(
        keys, return_inverse=True, return_counts=True
    )
    dense = dense.astype(np.int64, copy=False).reshape(-1)
    return (
        unique_keys,
        edge_counts.astype(np.int64),
        _distinct_per_key(dense, subjects),
        _distinct_per_key(dense, objects),
    )


def _distinct_per_key(keys: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Distinct *values* per key, aligned with ``np.unique(keys)``.

    Every key in *keys* survives deduplication, so the result lines up with
    ``np.unique(keys)`` element for element without a join.
    """
    span = int(values.max()) + 1
    if int(keys.max()) * span + span > np.iinfo(np.int64).max:
        raise OverflowError(
            f"cannot pack key {int(keys.max())} with node span {span} into int64"
        )
    packed = keys * span + values
    packed.sort()
    first_of_pair = np.concatenate(([True], packed[1:] != packed[:-1]))
    _, counts = np.unique(packed[first_of_pair] // span, return_counts=True)
    return counts.astype(np.int64)


@dataclass
class TripleCounts:
    """Counts for one ``(subject_category, predicate, object_category)`` row."""

    subject_category: str
    predicate: str
    object_category: str
    edge_count: int
    distinct_subjects: int
    distinct_objects: int


@dataclass
class CensusTables:
    """Everything the census computes for one categorical labelling."""

    leaf_rows: list[TripleCounts]
    rollup_rows: list[dict]
    rollup_index: dict[tuple[str, str, str], TripleCounts]
    forward_sources: dict[str, list[str]] = field(default_factory=dict)
    inverse_sources: dict[str, list[str]] = field(default_factory=dict)
    total_edges: int = 0
    semantics: str = "stored"
    # facet name -> rows, one per (triple, facet value) that occurs
    facet_rows: dict[str, list[dict[str, Any]]] = field(default_factory=dict)


@dataclass
class Facet:
    """A per-edge annotation censused alongside every triple.

    ``codes`` indexes ``labels``.  When ``rows`` is None the facet is 1:1 with
    edges (an edge has exactly one qualifier signature, one primary source).
    When ``rows`` is given, the facet is one-to-many and ``rows[i]`` is the edge
    that row *i* belongs to -- that is how a single edge contributes to a row
    for each of its qualifiers, and for each ancestor of each qualifier value.
    """

    name: str
    labels: list[str]
    codes: np.ndarray
    rows: Optional[np.ndarray] = None
    columns: dict[str, list[Any]] = field(default_factory=dict)

    def row_columns(self, code: int) -> dict[str, Any]:
        """Extra output columns describing one facet value."""
        return {name: values[code] for name, values in self.columns.items()}


def render_signature(signature: Sequence[tuple[str, str]]) -> str:
    """Render a qualifier signature the way a qualifier_set reads.

    >>> render_signature((("biolink:object_aspect_qualifier", "activity"),))
    'biolink:object_aspect_qualifier=activity'
    >>> render_signature(())
    '(unqualified)'
    """
    if not signature:
        return "(unqualified)"
    return "|".join(f"{type_id}={value}" for type_id, value in signature)


def build_facets(arrays: GraphArrays, closure: BiolinkClosure) -> list[Facet]:
    """Build the per-edge facets the census reports alongside each triple.

    Three of them:

    ``qualifier_signature``
        The whole conjunction of qualifiers on an edge.  This is the one that
        answers "can I write this template", because a TRAPI ``qualifier_set``
        ANDs its members -- knowing that 40% of edges carry a direction and 60%
        carry an aspect says nothing about how many carry *both*.

    ``qualifier``
        One row per ``(qualifier_type_id, qualifier_value)``, expanded to the
        value's ancestors as well, since ``QualifierExpander`` expands a queried
        value *down* to its descendants: an edge qualified ``expression`` is
        matched by a qedge asking for ``abundance``.  Rows are therefore
        query-granular, and an edge can appear under several of them.

    ``primary_source``
        Provenance, for pricing quality constraints inside a template.
    """
    facets: list[Facet] = []

    if arrays.qualifier_codes is not None and arrays.qualifier_signatures:
        facets.append(
            Facet(
                name="qualifier_signature",
                labels=[
                    render_signature(signature)
                    for signature in arrays.qualifier_signatures
                ],
                codes=arrays.qualifier_codes,
                columns={
                    "n_qualifiers": [
                        len(signature) for signature in arrays.qualifier_signatures
                    ],
                },
            )
        )
        facets.append(_qualifier_value_facet(arrays, closure))

    if arrays.source_codes is not None and arrays.primary_sources:
        facets.append(
            Facet(
                name="primary_source",
                labels=[source or "(none)" for source in arrays.primary_sources],
                codes=arrays.source_codes,
            )
        )

    return facets


def _qualifier_value_facet(arrays: GraphArrays, closure: BiolinkClosure) -> Facet:
    """Expand each edge into one row per qualifier value *and value ancestor*.

    Only called when the graph carries qualifiers, so ``qualifier_codes`` is set.
    """
    assert arrays.qualifier_codes is not None
    labels: list[str] = []
    label_index: dict[tuple[str, str], int] = {}
    type_ids: list[str] = []
    values: list[str] = []
    is_leaf: list[int] = []

    def label_code(type_id: str, value: str, leaf: bool) -> int:
        key = (type_id, value)
        code = label_index.get(key)
        if code is None:
            code = len(labels)
            label_index[key] = code
            labels.append(f"{type_id}={value}")
            type_ids.append(type_id)
            values.append(value)
            is_leaf.append(int(leaf))
        elif leaf:
            is_leaf[code] = 1
        return code

    # signature code -> the label codes every edge with that signature emits
    signature_labels: list[np.ndarray] = []
    for signature in arrays.qualifier_signatures:
        emitted: set[int] = set()
        for type_id, value in signature:
            for depth, ancestor in enumerate(
                closure.qualifier_value_ancestors(type_id, value)
            ):
                emitted.add(label_code(type_id, ancestor, depth == 0))
        signature_labels.append(np.array(sorted(emitted), dtype=np.int32))

    # Expand signature by signature rather than edge by edge: there are
    # thousands of distinct signatures but tens of millions of edges, so this
    # is a few thousand vectorized repeats instead of a per-edge Python loop.
    edge_codes = arrays.qualifier_codes
    order = np.argsort(edge_codes, kind="stable")
    grouped = edge_codes[order]
    starts = (
        np.flatnonzero(np.concatenate(([True], grouped[1:] != grouped[:-1])))
        if grouped.size
        else np.empty(0, dtype=np.int64)
    )
    ends = np.concatenate((starts[1:], [grouped.size])) if starts.size else starts

    row_chunks: list[np.ndarray] = []
    code_chunks: list[np.ndarray] = []
    for start, end in zip(starts, ends):
        labels_for_signature = signature_labels[int(grouped[start])]
        if labels_for_signature.size == 0:
            continue
        edges = order[start:end].astype(np.int32, copy=False)
        row_chunks.append(np.repeat(edges, labels_for_signature.size))
        code_chunks.append(np.tile(labels_for_signature, edges.size))

    rows = np.concatenate(row_chunks) if row_chunks else np.empty(0, dtype=np.int32)
    codes = np.concatenate(code_chunks) if code_chunks else np.empty(0, dtype=np.int32)

    return Facet(
        name="qualifier",
        labels=labels,
        codes=codes,
        rows=rows,
        columns={
            "qualifier_type_id": type_ids,
            "qualifier_value": values,
            "is_leaf_value": is_leaf,
        },
    )


def match_sets(
    ancestor: str,
    observed: Sequence[str],
    closure: BiolinkClosure,
    semantics: str,
) -> tuple[list[str], list[str]]:
    """Observed predicates a qedge naming *ancestor* would match.

    Returns ``(forward, inverse)``: predicates matched in the stored direction,
    and predicates matched with subject/object swapped.

    Under ``stored`` semantics this is simply every observed descendant of the
    ancestor, matched forward.  Under ``query`` semantics it follows
    ``PredicateExpander.expand_predicates``: the queried term always matches,
    its *descendants* are filtered to canonical-or-symmetric, the inverse term
    (and its filtered descendants) matches in reverse, a symmetric term matches
    its own subtree in both directions, and ``biolink:related_to`` matches
    everything both ways.
    """
    if semantics == "stored":
        return _descendants(ancestor, observed, closure, filtered=False), []

    if ancestor == ROOT_PREDICATE:
        return list(observed), list(observed)

    forward = _descendants(ancestor, observed, closure, filtered=True)
    inverse_predicate = closure.inverse(ancestor)
    inverse = (
        _descendants(inverse_predicate, observed, closure, filtered=True)
        if inverse_predicate
        else []
    )
    if closure.predicate_meta.get(ancestor, {}).get("symmetric"):
        inverse = sorted(set(inverse) | set(forward))
    return forward, inverse


def _descendants(
    term: str, observed: Sequence[str], closure: BiolinkClosure, filtered: bool
) -> list[str]:
    """Observed predicates under *term*, term included.

    With ``filtered=True`` the strict descendants are restricted to
    canonical-or-symmetric predicates while *term* itself is always kept -- the
    asymmetry is deliberate, and matches ``expand_predicates``, which seeds its
    expansion with the queried predicate before filtering what it adds.
    """
    return [
        predicate
        for predicate in observed
        if predicate == term
        or (
            term in closure.predicate_ancestors.get(predicate, (predicate,))
            and (not filtered or closure.is_canonical_or_symmetric(predicate))
        )
    ]


def run_census(
    arrays: GraphArrays,
    closure: BiolinkClosure,
    category_codes: np.ndarray,
    categories: list[str],
    semantics: str = "stored",
    facets: Sequence[Facet] = (),
) -> CensusTables:
    """Census one labelling of the graph: leaf triples plus the predicate rollup.

    *category_codes* maps node index -> index into *categories*; passing a
    relabelled array (see :func:`relabel_to_pins`) re-runs the whole census at a
    different node granularity without re-reading the graph.

    Each facet in *facets* is counted for every triple in the rollup, not just
    for leaf triples -- a template asks for ``biolink:affects`` with a direction
    qualifier, so qualifier coverage has to be known at ancestor granularity too.
    """
    subject_categories = category_codes[arrays.subjects].astype(np.int64)
    object_categories = category_codes[arrays.objects].astype(np.int64)
    num_categories = len(categories)
    num_predicates = len(arrays.predicates)
    total_edges = arrays.num_edges

    pair_keys = subject_categories * num_categories + object_categories
    swapped_pair_keys = object_categories * num_categories + subject_categories
    del subject_categories, object_categories

    logger.info("Counting leaf triples...")
    unique_keys, edge_counts, distinct_subjects, distinct_objects = group_stats(
        pair_keys * num_predicates + arrays.predicate_codes,
        arrays.subjects,
        arrays.objects,
    )

    leaf_rows = [
        TripleCounts(
            subject_category=categories[int(key) // num_predicates // num_categories],
            predicate=arrays.predicates[int(key) % num_predicates],
            object_category=categories[int(key) // num_predicates % num_categories],
            edge_count=int(edges),
            distinct_subjects=int(subs),
            distinct_objects=int(objs),
        )
        for key, edges, subs, objs in zip(
            unique_keys, edge_counts, distinct_subjects, distinct_objects
        )
    ]
    logger.info("  %s distinct occurring triples", f"{len(leaf_rows):,}")

    observed = sorted(
        {arrays.predicates[code] for code in np.unique(arrays.predicate_codes).tolist()}
    )
    ancestors = sorted(
        {
            ancestor
            for predicate in observed
            for ancestor in closure.predicate_ancestors.get(predicate, (predicate,))
        }
        | ({ROOT_PREDICATE} if semantics == "query" else set())
    )
    predicate_code_of = {
        predicate: code for code, predicate in enumerate(arrays.predicates)
    }
    leaf_index = {
        (row.subject_category, row.predicate, row.object_category) for row in leaf_rows
    }

    logger.info(
        "Rolling up over %s ancestor predicates (%s semantics)...",
        f"{len(ancestors):,}",
        semantics,
    )
    rollup_rows: list[dict[str, Any]] = []
    rollup_index: dict[tuple[str, str, str], TripleCounts] = {}
    forward_sources: dict[str, list[str]] = {}
    inverse_sources: dict[str, list[str]] = {}
    facet_rows: dict[str, list[dict[str, Any]]] = {facet.name: [] for facet in facets}

    for ancestor in ancestors:
        forward, inverse = match_sets(ancestor, observed, closure, semantics)
        forward_sources[ancestor] = forward
        inverse_sources[ancestor] = inverse
        if not forward and not inverse:
            continue

        selector = np.zeros(num_predicates, dtype=bool)
        selector[[predicate_code_of[p] for p in forward]] = True
        forward_mask = selector[arrays.predicate_codes]

        keys = [pair_keys[forward_mask]]
        subjects = [arrays.subjects[forward_mask]]
        objects = [arrays.objects[forward_mask]]
        inverse_mask = None

        if inverse:
            selector = np.zeros(num_predicates, dtype=bool)
            selector[[predicate_code_of[p] for p in inverse]] = True
            inverse_mask = selector[arrays.predicate_codes]
            # A reverse-direction match binds the edge's object to the qedge's
            # subject, so swap both the key and the endpoints.
            keys.append(swapped_pair_keys[inverse_mask])
            subjects.append(arrays.objects[inverse_mask])
            objects.append(arrays.subjects[inverse_mask])

        unique_keys, edge_counts, distinct_subjects, distinct_objects = group_stats(
            np.concatenate(keys), np.concatenate(subjects), np.concatenate(objects)
        )

        for key, edges, subs, objs in zip(
            unique_keys, edge_counts, distinct_subjects, distinct_objects
        ):
            subject_category = categories[int(key) // num_categories]
            object_category = categories[int(key) % num_categories]
            row = TripleCounts(
                subject_category=subject_category,
                predicate=ancestor,
                object_category=object_category,
                edge_count=int(edges),
                distinct_subjects=int(subs),
                distinct_objects=int(objs),
            )
            rollup_index[(subject_category, ancestor, object_category)] = row
            rollup_rows.append(
                {
                    "subject_category": subject_category,
                    "predicate": ancestor,
                    "object_category": object_category,
                    "edge_count": row.edge_count,
                    "distinct_subjects": row.distinct_subjects,
                    "distinct_objects": row.distinct_objects,
                    "predicate_depth": closure.predicate_depth(ancestor),
                    "occurs_as_leaf": int(
                        (subject_category, ancestor, object_category) in leaf_index
                    ),
                    "n_forward_predicates": len(forward),
                    "n_inverse_predicates": len(inverse),
                    "forward_predicates": "|".join(forward),
                    "inverse_predicates": "|".join(inverse),
                    "share_of_graph": (
                        round(row.edge_count / total_edges, 8) if total_edges else 0.0
                    ),
                }
            )

        for facet in facets:
            facet_rows[facet.name].extend(
                _facet_rows_for_predicate(
                    facet,
                    ancestor,
                    forward_mask,
                    inverse_mask,
                    pair_keys,
                    swapped_pair_keys,
                    arrays,
                    categories,
                    rollup_index,
                )
            )

    for rows in facet_rows.values():
        rows.sort(key=lambda row: -row["edge_count"])
    rollup_rows.sort(key=lambda row: -row["edge_count"])
    return CensusTables(
        leaf_rows=sorted(leaf_rows, key=lambda row: -row.edge_count),
        rollup_rows=rollup_rows,
        rollup_index=rollup_index,
        forward_sources=forward_sources,
        inverse_sources=inverse_sources,
        total_edges=total_edges,
        semantics=semantics,
        facet_rows=facet_rows,
    )


def _facet_rows_for_predicate(
    facet: Facet,
    predicate: str,
    forward_mask: np.ndarray,
    inverse_mask: Optional[np.ndarray],
    pair_keys: np.ndarray,
    swapped_pair_keys: np.ndarray,
    arrays: GraphArrays,
    categories: list[str],
    rollup_index: dict[tuple[str, str, str], TripleCounts],
) -> list[dict[str, Any]]:
    """Count one facet's values within every triple matched by *predicate*."""
    num_categories = len(categories)
    num_labels = max(len(facet.labels), 1)

    keys: list[np.ndarray] = []
    codes: list[np.ndarray] = []
    subjects: list[np.ndarray] = []
    objects: list[np.ndarray] = []

    for mask, key_source, subject_source, object_source in (
        (forward_mask, pair_keys, arrays.subjects, arrays.objects),
        (inverse_mask, swapped_pair_keys, arrays.objects, arrays.subjects),
    ):
        if mask is None:
            continue
        if facet.rows is None:
            keys.append(key_source[mask])
            codes.append(facet.codes[mask])
            subjects.append(subject_source[mask])
            objects.append(object_source[mask])
        else:
            selected = mask[facet.rows]
            edges = facet.rows[selected]
            keys.append(key_source[edges])
            codes.append(facet.codes[selected])
            subjects.append(subject_source[edges])
            objects.append(object_source[edges])

    if not keys:
        return []
    pair_key = np.concatenate(keys)
    if pair_key.size == 0:
        return []

    unique_keys, edge_counts, distinct_subjects, distinct_objects = group_stats(
        pair_key * num_labels + np.concatenate(codes),
        np.concatenate(subjects),
        np.concatenate(objects),
    )

    rows: list[dict[str, Any]] = []
    for key, edges, subs, objs in zip(
        unique_keys, edge_counts, distinct_subjects, distinct_objects
    ):
        code = int(key) % num_labels
        pair = int(key) // num_labels
        subject_category = categories[pair // num_categories]
        object_category = categories[pair % num_categories]
        triple = rollup_index.get((subject_category, predicate, object_category))
        triple_edges = triple.edge_count if triple else 0
        rows.append(
            {
                "subject_category": subject_category,
                "predicate": predicate,
                "object_category": object_category,
                facet.name: facet.labels[code],
                **facet.row_columns(code),
                "edge_count": int(edges),
                "distinct_subjects": int(subs),
                "distinct_objects": int(objs),
                "triple_edge_count": triple_edges,
                "share_of_triple": (
                    round(int(edges) / triple_edges, 6) if triple_edges else 0.0
                ),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Derived tables
# ---------------------------------------------------------------------------


def build_wide_rows(
    census: CensusTables, closure: BiolinkClosure
) -> list[dict[str, Any]]:
    """Leaf rows plus the ancestor rollup as a column: the query-granularity view.

    ``rollup_json`` maps each ancestor of the row's predicate to the counts a
    qedge naming *that* ancestor would match between the same two categories.
    ``dilution_at_parent`` is how much bigger the parent's match set is: 1.0
    means the parent buys nothing extra here, 50 means asking for the parent
    pulls in 50x the edges -- which is exactly the signal for whether a grammar
    class can be one term or needs leaf enumeration.
    """
    rows: list[dict[str, Any]] = []
    for leaf in census.leaf_rows:
        chain = closure.predicate_ancestors.get(leaf.predicate, (leaf.predicate,))
        rollup = {}
        for ancestor in chain:
            counts = census.rollup_index.get(
                (leaf.subject_category, ancestor, leaf.object_category)
            )
            if counts is not None:
                rollup[ancestor] = {
                    "edges": counts.edge_count,
                    "subjects": counts.distinct_subjects,
                    "objects": counts.distinct_objects,
                }
        parent = chain[1] if len(chain) > 1 else ""
        parent_edges = rollup.get(parent, {}).get("edges")
        root_edges = rollup.get(ROOT_PREDICATE, {}).get("edges")
        rows.append(
            {
                "subject_category": leaf.subject_category,
                "predicate": leaf.predicate,
                "object_category": leaf.object_category,
                "edge_count": leaf.edge_count,
                "distinct_subjects": leaf.distinct_subjects,
                "distinct_objects": leaf.distinct_objects,
                "predicate_depth": closure.predicate_depth(leaf.predicate),
                "parent_predicate": parent,
                "edges_at_parent": parent_edges if parent_edges is not None else "",
                "dilution_at_parent": (
                    round(parent_edges / leaf.edge_count, 3)
                    if parent_edges and leaf.edge_count
                    else ""
                ),
                "edges_at_related_to": root_edges if root_edges is not None else "",
                "share_of_related_to": (
                    round(leaf.edge_count / root_edges, 6) if root_edges else ""
                ),
                "ancestor_chain": "|".join(chain),
                "rollup_json": json.dumps(rollup, separators=(",", ":")),
            }
        )
    return rows


def build_predicate_summary(
    census: CensusTables, closure: BiolinkClosure, total_edges: int
) -> list[dict[str, Any]]:
    """Per-predicate totals: the first pass at "which predicates are noise".

    ``subtree_edge_count`` is the total matched at that predicate including all
    observed descendants; ``own_edge_count`` excludes them.  A predicate whose
    own count dominates its subtree is a real leaf; one with volume only via
    descendants is a grouping term, and one with volume of its own but no
    descendants and no specificity (``biolink:related_to`` and friends) is the
    noise floor.
    """
    own: dict[str, dict[str, Any]] = {}
    for triple in census.leaf_rows:
        entry = own.setdefault(
            triple.predicate,
            {"edges": 0, "triples": 0, "subject_cats": set(), "object_cats": set()},
        )
        entry["edges"] += triple.edge_count
        entry["triples"] += 1
        entry["subject_cats"].add(triple.subject_category)
        entry["object_cats"].add(triple.object_category)

    subtree: dict[str, int] = {}
    for rollup_row in census.rollup_rows:
        predicate = rollup_row["predicate"]
        subtree[predicate] = subtree.get(predicate, 0) + rollup_row["edge_count"]

    rows: list[dict[str, Any]] = []
    for predicate in sorted(set(own) | set(subtree)):
        entry = own.get(predicate) or {}
        meta = closure.predicate_meta.get(predicate, {})
        own_edges = entry.get("edges", 0)
        subtree_edges = subtree.get(predicate, 0)
        rows.append(
            {
                "predicate": predicate,
                "own_edge_count": own_edges,
                "own_share_of_graph": (
                    round(own_edges / total_edges, 8) if total_edges else 0.0
                ),
                "subtree_edge_count": subtree_edges,
                "subtree_share_of_graph": (
                    round(subtree_edges / total_edges, 8) if total_edges else 0.0
                ),
                "own_share_of_subtree": (
                    round(own_edges / subtree_edges, 6) if subtree_edges else 0.0
                ),
                "n_triples": entry.get("triples", 0),
                "n_subject_categories": len(entry.get("subject_cats", ())),
                "n_object_categories": len(entry.get("object_cats", ())),
                "n_matched_predicates": len(census.forward_sources.get(predicate, [])),
                "depth": closure.predicate_depth(predicate),
                "in_model": int(bool(meta.get("in_model"))),
                "canonical": int(bool(meta.get("canonical"))),
                "symmetric": int(bool(meta.get("symmetric"))),
                "mixin": int(bool(meta.get("mixin"))),
                "deprecated": int(bool(meta.get("deprecated"))),
                "inverse": meta.get("inverse") or "",
                "model_inverse": meta.get("model_inverse") or "",
            }
        )
    rows.sort(key=lambda row: -row["subtree_edge_count"])
    return rows


def build_category_tables(
    arrays: GraphArrays,
    closure: BiolinkClosure,
    category_codes: np.ndarray,
    categories: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Per-category node/edge counts, and the same rolled up to ancestors.

    The rollup answers "is ``biolink:ChemicalEntity`` a clean drug-side pin?" --
    ``member_breakdown`` names every leaf category the pin would drag in,
    largest first, so exposures and food are visible by name rather than by
    inference.  Leaf categories are disjoint (one primary category per node), so
    the ancestor sums are exact, not double-counted.
    """
    num_categories = len(categories)
    node_counts = np.bincount(category_codes, minlength=num_categories)
    edges_as_subject = np.bincount(
        category_codes[arrays.subjects], minlength=num_categories
    )
    edges_as_object = np.bincount(
        category_codes[arrays.objects], minlength=num_categories
    )
    subject_nodes = np.unique(arrays.subjects)
    object_nodes = np.unique(arrays.objects)
    nodes_as_subject = np.bincount(
        category_codes[subject_nodes], minlength=num_categories
    )
    nodes_as_object = np.bincount(
        category_codes[object_nodes], minlength=num_categories
    )
    connected = np.bincount(
        category_codes[np.union1d(subject_nodes, object_nodes)],
        minlength=num_categories,
    )

    summary: list[dict[str, Any]] = [
        {
            "category": category,
            "depth": closure.category_depth(category),
            "nodes": int(node_counts[code]),
            "nodes_with_edges": int(connected[code]),
            "nodes_as_subject": int(nodes_as_subject[code]),
            "nodes_as_object": int(nodes_as_object[code]),
            "edges_as_subject": int(edges_as_subject[code]),
            "edges_as_object": int(edges_as_object[code]),
            "ancestor_chain": "|".join(
                closure.category_ancestors.get(category, (category,))
            ),
        }
        for code, category in enumerate(categories)
    ]
    summary.sort(key=lambda row: -row["nodes"])

    members: dict[str, list[str]] = {}
    for category in categories:
        for ancestor in closure.category_ancestors.get(category, (category,)):
            members.setdefault(ancestor, []).append(category)

    by_name = {row["category"]: row for row in summary}
    rollup: list[dict[str, Any]] = []
    for ancestor, member_categories in sorted(members.items()):
        member_rows = sorted(
            (by_name[category] for category in member_categories),
            key=lambda row: -row["nodes"],
        )
        total_nodes = sum(row["nodes"] for row in member_rows)
        rollup.append(
            {
                "category": ancestor,
                "depth": closure.category_depth(ancestor),
                "n_member_categories": len(member_rows),
                "nodes": total_nodes,
                "nodes_with_edges": sum(row["nodes_with_edges"] for row in member_rows),
                "edges_as_subject": sum(row["edges_as_subject"] for row in member_rows),
                "edges_as_object": sum(row["edges_as_object"] for row in member_rows),
                "largest_member": member_rows[0]["category"] if member_rows else "",
                "largest_member_share": (
                    round(member_rows[0]["nodes"] / total_nodes, 4)
                    if total_nodes
                    else 0.0
                ),
                "member_breakdown": "|".join(
                    f"{row['category']}:{row['nodes']}" for row in member_rows
                ),
            }
        )
    rollup.sort(key=lambda row: -row["nodes"])
    return summary, rollup


def relabel_to_pins(
    closure: BiolinkClosure,
    categories: list[str],
    category_codes: np.ndarray,
    pins: Sequence[str],
    fallback: str,
) -> tuple[np.ndarray, list[str]]:
    """Collapse every category to its most specific matching pin.

    A category with no pin among its ancestors keeps its own name
    (``fallback="leaf"``) or is bucketed as ``biolink:NamedThing``
    (``fallback="other"``), which turns "how much of the graph do my pins
    actually cover?" into a single row.
    """
    pin_set = set(pins)
    new_categories: list[str] = []
    new_index: dict[str, int] = {}
    remap = np.empty(len(categories), dtype=np.int32)

    for code, category in enumerate(categories):
        label = next(
            (
                ancestor
                for ancestor in closure.category_ancestors.get(category, (category,))
                if ancestor in pin_set
            ),
            category if fallback == "leaf" else DEFAULT_CATEGORY,
        )
        new_code = new_index.get(label)
        if new_code is None:
            new_code = len(new_categories)
            new_index[label] = new_code
            new_categories.append(label)
        remap[code] = new_code

    return remap[category_codes], new_categories


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

LEAF_FIELDS = [
    "subject_category",
    "predicate",
    "object_category",
    "edge_count",
    "distinct_subjects",
    "distinct_objects",
]

# Per-triple annotation columns appended to the leaf and wide tables when the
# graph carries qualifiers and sources.
ANNOTATION_FIELDS = [
    "qualified_edges",
    "qualified_share",
    "n_qualifier_signatures",
    "top_qualifier_signature",
    "top_qualifier_signature_share",
    "n_primary_sources",
    "top_primary_source",
    "top_primary_source_share",
]

FACET_FIELDS = {
    "qualifier_signature": [
        "subject_category",
        "predicate",
        "object_category",
        "qualifier_signature",
        "n_qualifiers",
        "edge_count",
        "distinct_subjects",
        "distinct_objects",
        "triple_edge_count",
        "share_of_triple",
    ],
    "qualifier": [
        "subject_category",
        "predicate",
        "object_category",
        "qualifier",
        "qualifier_type_id",
        "qualifier_value",
        "is_leaf_value",
        "edge_count",
        "distinct_subjects",
        "distinct_objects",
        "triple_edge_count",
        "share_of_triple",
    ],
    "primary_source": [
        "subject_category",
        "predicate",
        "object_category",
        "primary_source",
        "edge_count",
        "distinct_subjects",
        "distinct_objects",
        "triple_edge_count",
        "share_of_triple",
    ],
}

FACET_FILES = {
    "qualifier_signature": "qualifier_signatures.tsv",
    "qualifier": "qualifier_values.tsv",
    "primary_source": "source_census.tsv",
}

ROLLUP_FIELDS = LEAF_FIELDS + [
    "predicate_depth",
    "occurs_as_leaf",
    "n_forward_predicates",
    "n_inverse_predicates",
    "forward_predicates",
    "inverse_predicates",
    "share_of_graph",
]

WIDE_FIELDS = LEAF_FIELDS + [
    "predicate_depth",
    "parent_predicate",
    "edges_at_parent",
    "dilution_at_parent",
    "edges_at_related_to",
    "share_of_related_to",
    "ancestor_chain",
    "rollup_json",
]


def qualified_edge_total(arrays: GraphArrays) -> int:
    """How many edges carry at least one qualifier."""
    if arrays.qualifier_codes is None or not arrays.qualifier_signatures:
        return 0
    unqualified = np.array(
        [not signature for signature in arrays.qualifier_signatures], dtype=bool
    )
    return int(np.count_nonzero(~unqualified[arrays.qualifier_codes]))


def summarize_annotations(
    census: CensusTables,
) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Condense the facet rows into per-triple columns for the leaf/wide tables.

    Keeps the headline numbers a template author checks first -- what fraction
    of a triple's edges carry any qualifier at all, how concentrated the
    qualifier signatures are, and whether one source supplies everything.
    """
    signature_rows = census.facet_rows.get("qualifier_signature") or []
    source_rows = census.facet_rows.get("primary_source") or []
    if not signature_rows and not source_rows:
        return {}

    summary: dict[tuple[str, str, str], dict[str, Any]] = {}

    for row in signature_rows:
        key = (row["subject_category"], row["predicate"], row["object_category"])
        entry = summary.setdefault(
            key,
            {
                "qualified_edges": 0,
                "qualified_share": 0.0,
                "n_qualifier_signatures": 0,
                "top_qualifier_signature": "",
                "top_qualifier_signature_share": 0.0,
                "n_primary_sources": 0,
                "top_primary_source": "",
                "top_primary_source_share": 0.0,
            },
        )
        if row["n_qualifiers"] == 0:
            continue
        entry["qualified_edges"] += row["edge_count"]
        entry["n_qualifier_signatures"] += 1
        if row["share_of_triple"] > entry["top_qualifier_signature_share"]:
            entry["top_qualifier_signature"] = row["qualifier_signature"]
            entry["top_qualifier_signature_share"] = row["share_of_triple"]

    for row in source_rows:
        key = (row["subject_category"], row["predicate"], row["object_category"])
        entry = summary.setdefault(
            key,
            {
                "qualified_edges": 0,
                "qualified_share": 0.0,
                "n_qualifier_signatures": 0,
                "top_qualifier_signature": "",
                "top_qualifier_signature_share": 0.0,
                "n_primary_sources": 0,
                "top_primary_source": "",
                "top_primary_source_share": 0.0,
            },
        )
        entry["n_primary_sources"] += 1
        if row["share_of_triple"] > entry["top_primary_source_share"]:
            entry["top_primary_source"] = row["primary_source"]
            entry["top_primary_source_share"] = row["share_of_triple"]

    for key, entry in summary.items():
        triple = census.rollup_index.get(key)
        if triple and triple.edge_count:
            entry["qualified_share"] = round(
                entry["qualified_edges"] / triple.edge_count, 6
            )
    return summary


def build_qualifier_summary(
    arrays: GraphArrays,
    facets: Sequence[Facet],
    closure: BiolinkClosure,
    category_codes: np.ndarray,
    categories: Sequence[str],
) -> list[dict[str, Any]]:
    """Graph-wide view of every qualifier assertion that occurs.

    Answers "does this graph even have direction qualifiers, and on what" before
    you write a template that depends on them.

    Counted from the per-edge facet rather than by summing the census tables:
    an ancestor predicate can also be a leaf predicate (``treats`` is both a
    parent of ``ameliorates_condition`` and a label in its own right), so
    adding up per-triple rows would count the same edge under both.
    """
    facet = next((f for f in facets if f.name == "qualifier"), None)
    if facet is None or facet.rows is None or facet.codes.size == 0:
        return []

    num_labels = max(len(facet.labels), 1)
    num_categories = len(categories)
    num_predicates = len(arrays.predicates)
    total_edges = arrays.num_edges

    leaf_keys = (
        category_codes[arrays.subjects].astype(np.int64) * num_categories
        + category_codes[arrays.objects]
    ) * num_predicates + arrays.predicate_codes

    codes = facet.codes.astype(np.int64)
    edges_per_label = np.bincount(codes, minlength=num_labels)
    triples_per_label = np.bincount(
        np.unique(leaf_keys[facet.rows] * num_labels + codes) % num_labels,
        minlength=num_labels,
    )
    predicates_per_label = np.bincount(
        np.unique(
            arrays.predicate_codes[facet.rows].astype(np.int64) * num_labels + codes
        )
        % num_labels,
        minlength=num_labels,
    )

    rows = [
        {
            "qualifier_type_id": facet.columns["qualifier_type_id"][code],
            "qualifier_value": facet.columns["qualifier_value"][code],
            "enum": closure.qualifier_enums.get(
                facet.columns["qualifier_type_id"][code], ""
            ),
            "is_leaf_value": facet.columns["is_leaf_value"][code],
            "ancestor_values": "|".join(
                closure.qualifier_value_ancestors(
                    facet.columns["qualifier_type_id"][code],
                    facet.columns["qualifier_value"][code],
                )
            ),
            "edge_count": int(edges_per_label[code]),
            "share_of_graph": (
                round(int(edges_per_label[code]) / total_edges, 8)
                if total_edges
                else 0.0
            ),
            "n_triples": int(triples_per_label[code]),
            "n_predicates": int(predicates_per_label[code]),
        }
        for code in range(num_labels)
        if edges_per_label[code]
    ]
    rows.sort(key=lambda row: (row["qualifier_type_id"], -row["edge_count"]))
    return rows


def write_tsv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    """Write *rows* as a tab-separated file with a header."""
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(fieldnames), delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    logger.info("  wrote %s (%s rows)", path, f"{len(rows):,}")


def write_census_tables(
    out_dir: Path, census: CensusTables, closure: BiolinkClosure, prefix: str = "census"
) -> None:
    """Write the leaf, rollup, wide and facet tables for one labelling."""
    annotations = summarize_annotations(census)
    extra_fields = ANNOTATION_FIELDS if annotations else []

    write_tsv(
        out_dir / f"{prefix}_leaf.tsv",
        [
            {
                "subject_category": row.subject_category,
                "predicate": row.predicate,
                "object_category": row.object_category,
                "edge_count": row.edge_count,
                "distinct_subjects": row.distinct_subjects,
                "distinct_objects": row.distinct_objects,
                **annotations.get(
                    (row.subject_category, row.predicate, row.object_category), {}
                ),
            }
            for row in census.leaf_rows
        ],
        LEAF_FIELDS + extra_fields,
    )
    write_tsv(out_dir / f"{prefix}_rollup.tsv", census.rollup_rows, ROLLUP_FIELDS)

    wide_rows = build_wide_rows(census, closure)
    for row in wide_rows:
        row.update(
            annotations.get(
                (row["subject_category"], row["predicate"], row["object_category"]), {}
            )
        )
    write_tsv(out_dir / f"{prefix}_wide.tsv", wide_rows, WIDE_FIELDS + extra_fields)

    for name, rows in census.facet_rows.items():
        if not rows:
            continue
        filename = FACET_FILES.get(name, f"{name}.tsv")
        write_tsv(
            out_dir / (filename if prefix == "census" else f"{prefix}_{filename}"),
            rows,
            FACET_FIELDS.get(name, list(rows[0])),
        )


def print_summary(
    census: CensusTables,
    predicate_summary: Sequence[dict],
    category_rollup: Sequence[dict],
    qualifier_summary: Sequence[dict],
    top: int,
) -> None:
    """Print the headline numbers the census exists to answer."""
    total = census.total_edges
    print()
    print("=" * 78)
    print(f"METAGRAPH CENSUS ({census.semantics} semantics)")
    print("=" * 78)
    print(f"edges: {total:,}   occurring triples: {len(census.leaf_rows):,}")

    print(f"\nTop {top} triples by edge count:")
    for triple in census.leaf_rows[:top]:
        print(
            f"  {triple.edge_count:>12,}  {triple.subject_category} "
            f"-{triple.predicate}-> {triple.object_category} "
            f"({triple.distinct_subjects:,} subj / {triple.distinct_objects:,} obj)"
        )

    print(f"\nTop {top} predicates by subtree edge count:")
    if census.semantics == "query":
        print(
            "  (query semantics count each matched orientation, so a share can "
            "exceed 100%)"
        )
    for predicate_row in predicate_summary[:top]:
        print(
            f"  {predicate_row['subtree_edge_count']:>12,} "
            f"({predicate_row['subtree_share_of_graph']:>7.2%})  "
            f"{predicate_row['predicate']}"
            f"  [own {predicate_row['own_edge_count']:,}, "
            f"{predicate_row['n_matched_predicates']} matched predicates, "
            f"depth {predicate_row['depth']}]"
        )

    thin = [triple for triple in census.leaf_rows if triple.edge_count < 1000]
    print(
        f"\nTriples with <1,000 edges: {len(thin):,} of {len(census.leaf_rows):,} "
        f"({len(thin) / max(len(census.leaf_rows), 1):.1%}), "
        f"{sum(triple.edge_count for triple in thin):,} edges total -- "
        "anything here is an anecdote, not a template."
    )

    print(f"\nTop {top} category subtrees by node count:")
    for category_row in category_rollup[:top]:
        print(
            f"  {category_row['nodes']:>12,}  {category_row['category']} "
            f"({category_row['n_member_categories']} leaf categories, largest "
            f"{category_row['largest_member']} at "
            f"{category_row['largest_member_share']:.0%})"
        )

    if qualifier_summary:
        qualified = sum(
            row["edge_count"] for row in qualifier_summary if row["is_leaf_value"]
        )
        print(
            f"\nQualifier coverage: {qualified:,} qualifier assertions over "
            f"{total:,} edges, {len({row['qualifier_type_id'] for row in qualifier_summary})} "
            "qualifier types"
        )
        print(f"Top {top} qualifier values by edge count:")
        for qualifier_row in sorted(
            qualifier_summary, key=lambda row: -row["edge_count"]
        )[:top]:
            leaf = "" if qualifier_row["is_leaf_value"] else " (rollup)"
            print(
                f"  {qualifier_row['edge_count']:>12,}  "
                f"{qualifier_row['qualifier_type_id']}="
                f"{qualifier_row['qualifier_value']}{leaf} "
                f"across {qualifier_row['n_triples']:,} triples"
            )
    else:
        print(
            "\nNo qualifiers found: every mechanism template that depends on "
            "direction or aspect is unavailable on this graph."
        )
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Relabeled metagraph census: triple counts + Biolink rollup",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples::" + (__doc__ or "").split("Examples::")[-1],
    )
    source = parser.add_argument_group("graph source (exactly one)")
    source.add_argument(
        "--graph", "-g", type=Path, help="Path to a saved Gandalf mmap graph directory"
    )
    source.add_argument("--edges", "-e", type=Path, help="KGX edges.jsonl[.gz]")
    source.add_argument("--nodes", "-n", type=Path, help="KGX nodes.jsonl[.gz]")

    parser.add_argument(
        "--out",
        "-o",
        type=Path,
        default=Path("census"),
        help="Output directory (default: ./census)",
    )
    parser.add_argument(
        "--biolink-version",
        help="Biolink version for the closure "
        "(default: the version Gandalf targets, gandalf_biolink_version)",
    )
    parser.add_argument(
        "--biolink-schema",
        help="Explicit biolink-model.yaml URL or local path "
        "(overrides --biolink-version; useful offline)",
    )
    parser.add_argument(
        "--no-mixins",
        action="store_true",
        help="Exclude mixin parents from ancestor chains "
        "(default: include them, matching BMT's query-time expansion)",
    )
    parser.add_argument(
        "--skip-annotations",
        action="store_true",
        help="Skip the qualifier and source census (they are read from the "
        "interned edge pools and are cheap, so this is rarely worth it)",
    )
    parser.add_argument(
        "--match-semantics",
        choices=("stored", "query"),
        default="stored",
        help="stored: roll up into every ancestor (structure as stored). "
        "query: mirror PredicateExpander -- canonical-or-symmetric descendants "
        "only, inverses matched in reverse, related_to matches everything",
    )
    parser.add_argument(
        "--category-policy",
        choices=("most-specific", "first"),
        default="most-specific",
        help="How to pick a node's single category when it declares several "
        "(default: most-specific; 'first' matches the served meta_kg)",
    )
    parser.add_argument(
        "--pin-category",
        action="append",
        default=[],
        metavar="CURIE",
        help="Also emit a census with categories collapsed to this pin; "
        "repeatable, e.g. --pin-category biolink:ChemicalEntity",
    )
    parser.add_argument(
        "--pin-fallback",
        choices=("leaf", "other"),
        default="leaf",
        help="What to do with categories no pin covers (default: keep the leaf)",
    )
    parser.add_argument(
        "--top", type=int, default=20, help="Rows per stdout summary section"
    )
    parser.add_argument("--log-level", default="INFO", help="Default: INFO")

    args = parser.parse_args(argv)
    if not args.graph and not args.edges:
        parser.error("one of --graph or --edges is required")
    if args.graph and args.edges:
        parser.error("--graph and --edges are mutually exclusive")
    if args.nodes and not args.edges:
        parser.error("--nodes only applies to the --edges (KGX jsonl) source")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )
    started = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=True)

    def closure_builder(predicates, categories, qualifiers=()):
        return build_closure_map(
            predicates,
            categories,
            qualifiers,
            version=args.biolink_version,
            schema=args.biolink_schema,
            include_mixins=not args.no_mixins,
        )

    annotations = not args.skip_annotations
    if args.graph:
        arrays, closure = load_from_mmap(
            args.graph, closure_builder, args.category_policy, annotations
        )
    else:
        arrays, closure = load_from_jsonl(
            args.edges, args.nodes, closure_builder, args.category_policy, annotations
        )

    closure_path = args.out / "biolink_closure.json"
    with open(closure_path, "w", encoding="utf-8") as handle:
        json.dump(closure.to_json(), handle, indent=2, sort_keys=True)
    logger.info("  wrote %s", closure_path)

    facets = build_facets(arrays, closure)
    census = run_census(
        arrays,
        closure,
        arrays.node_category_codes,
        arrays.categories,
        semantics=args.match_semantics,
        facets=facets,
    )
    write_census_tables(args.out, census, closure)

    qualifier_summary = build_qualifier_summary(
        arrays, facets, closure, arrays.node_category_codes, arrays.categories
    )
    if qualifier_summary:
        write_tsv(
            args.out / "qualifier_summary.tsv",
            qualifier_summary,
            list(qualifier_summary[0]),
        )

    predicate_summary = build_predicate_summary(census, closure, arrays.num_edges)
    write_tsv(
        args.out / "predicate_summary.tsv",
        predicate_summary,
        list(predicate_summary[0]) if predicate_summary else ["predicate"],
    )

    category_summary, category_rollup = build_category_tables(
        arrays, closure, arrays.node_category_codes, arrays.categories
    )
    write_tsv(
        args.out / "category_summary.tsv",
        category_summary,
        list(category_summary[0]) if category_summary else ["category"],
    )
    write_tsv(
        args.out / "category_rollup.tsv",
        category_rollup,
        list(category_rollup[0]) if category_rollup else ["category"],
    )

    pinned_census = None
    if args.pin_category:
        logger.info("Re-running census with %d category pins", len(args.pin_category))
        pinned_codes, pinned_categories = relabel_to_pins(
            closure,
            arrays.categories,
            arrays.node_category_codes,
            args.pin_category,
            args.pin_fallback,
        )
        pinned_census = run_census(
            arrays,
            closure,
            pinned_codes,
            pinned_categories,
            semantics=args.match_semantics,
            facets=facets,
        )
        write_census_tables(args.out, pinned_census, closure, prefix="census_pinned")

    non_canonical_edges = sum(
        row.edge_count
        for row in census.leaf_rows
        if not closure.is_canonical_or_symmetric(row.predicate)
    )
    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": arrays.source,
        "source_kind": "mmap" if args.graph else "kgx-jsonl",
        "biolink_version": closure.version,
        "biolink_schema": closure.schema,
        "include_mixins": not args.no_mixins,
        "match_semantics": args.match_semantics,
        "category_policy": args.category_policy,
        "pins": args.pin_category,
        "pin_fallback": args.pin_fallback,
        "nodes": arrays.num_nodes,
        "edges": arrays.num_edges,
        "predicates": len({p for p in arrays.predicates if p}),
        "categories": len(arrays.categories),
        "multi_category_nodes": arrays.multi_category_nodes,
        "dangling_endpoints": arrays.dangling_endpoints,
        "occurring_triples": len(census.leaf_rows),
        "rollup_rows": len(census.rollup_rows),
        "annotations": annotations,
        "qualifier_signatures": len(arrays.qualifier_signatures),
        "qualified_edges": qualified_edge_total(arrays),
        "qualifier_types": sorted({type_id for type_id, _ in arrays.qualifier_pairs}),
        "primary_sources": len(arrays.primary_sources),
        "unmapped_qualifier_values": [
            {"qualifier_type_id": type_id, "qualifier_value": value}
            for type_id, value in closure.unmapped_qualifier_values
        ],
        "edges_on_non_canonical_predicates": non_canonical_edges,
        "unmapped_predicates": list(closure.unmapped_predicates),
        "unmapped_categories": list(closure.unmapped_categories),
        "pinned_occurring_triples": (
            len(pinned_census.leaf_rows) if pinned_census else None
        ),
        "runtime_seconds": round(time.perf_counter() - started, 2),
    }
    manifest_path = args.out / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    logger.info("  wrote %s", manifest_path)

    print_summary(
        census, predicate_summary, category_rollup, qualifier_summary, args.top
    )
    if non_canonical_edges:
        offenders = sorted(
            {
                row.predicate
                for row in census.leaf_rows
                if not closure.is_canonical_or_symmetric(row.predicate)
            }
        )
        print(
            f"NOTE: {non_canonical_edges:,} edges "
            f"({non_canonical_edges / max(arrays.num_edges, 1):.1%}) sit on "
            f"predicates that are neither canonical nor symmetric "
            f"({', '.join(offenders[:5])}"
            f"{', ...' if len(offenders) > 5 else ''}). Gandalf only expands a "
            "qedge into canonical-or-symmetric descendants, so these are "
            "reachable only by naming them (or their inverse) directly"
            + (
                " -- --match-semantics query counts exactly that.\n"
                if args.match_semantics == "stored"
                else ".\n"
            )
        )
    logger.info(
        "Census complete in %.1fs -> %s", time.perf_counter() - started, args.out
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
