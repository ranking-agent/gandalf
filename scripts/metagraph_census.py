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
~1 GB of RAM, so a 38M-edge Translator graph is a couple of minutes and a few
GB, dominated by the node-store scan.

Outputs (tab-separated with a header row, written to ``--out``):

===========================  ==================================================
``manifest.json``            run provenance, graph totals, unmapped terms
``biolink_closure.json``     ancestor chains for every observed term
``census_leaf.tsv``          one row per occurring triple (leaf predicate)
``census_rollup.tsv``        one row per (subj_cat, ancestor_pred, obj_cat)
``census_wide.tsv``          leaf rows + the rollup as a JSON column
``predicate_summary.tsv``    per-predicate totals, own vs. whole subtree
``category_summary.tsv``     per-category node/edge counts (leaf categories)
``category_rollup.tsv``      per-ancestor-category totals + member breakdown
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
            "unmapped_predicates": list(self.unmapped_predicates),
            "unmapped_categories": list(self.unmapped_categories),
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
    *,
    version: Optional[str] = None,
    schema: Optional[str] = None,
    include_mixins: bool = True,
) -> BiolinkClosure:
    """Look up the full ancestor chain of every observed predicate and category.

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

    logger.info(
        "  closure: %d predicates (%d unmapped), %d categories (%d unmapped)",
        len(predicate_ancestors),
        len(unmapped_predicates),
        len(category_ancestors),
        len(unmapped_categories),
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

    @property
    def num_edges(self) -> int:
        return int(self.subjects.shape[0])


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


ClosureBuilder = Callable[[Sequence[str], Sequence[str]], BiolinkClosure]


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
    graph_dir: Path, closure_builder: ClosureBuilder, policy: str
) -> tuple[GraphArrays, BiolinkClosure]:
    """Read a saved Gandalf mmap graph directory.

    Reads only what the census needs: the forward CSR arrays, the predicate
    vocabulary from ``metadata.pkl`` and node categories from the node store.
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

    closure = closure_builder(predicates, sorted(interner.vocabulary))
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
    )
    return arrays, closure


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


def load_from_jsonl(
    edges_path: Path,
    nodes_path: Optional[Path],
    closure_builder: ClosureBuilder,
    policy: str,
) -> tuple[GraphArrays, BiolinkClosure]:
    """Read KGX ``nodes.jsonl`` / ``edges.jsonl`` without building a graph.

    Node IDs are interned to indices as they are encountered.  Edge endpoints
    absent from the node file get ``biolink:NamedThing`` and are counted, so a
    dangling-endpoint problem shows up in the manifest instead of quietly
    inflating the NamedThing rows.
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
        if count % 5_000_000 == 0:
            logger.info("  %s edges read", f"{count:,}")

    if dangling_endpoints:
        logger.warning(
            "  %s edge endpoints had no node record; labelled %s",
            f"{dangling_endpoints:,}",
            DEFAULT_CATEGORY,
        )

    closure = closure_builder(predicates, sorted(interner.vocabulary))
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

    unique_keys, edge_counts = np.unique(keys, return_counts=True)
    return (
        unique_keys,
        edge_counts.astype(np.int64),
        _distinct_per_key(keys, subjects),
        _distinct_per_key(keys, objects),
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
) -> CensusTables:
    """Census one labelling of the graph: leaf triples plus the predicate rollup.

    *category_codes* maps node index -> index into *categories*; passing a
    relabelled array (see :func:`relabel_to_pins`) re-runs the whole census at a
    different node granularity without re-reading the graph.
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

    rollup_rows.sort(key=lambda row: -row["edge_count"])
    return CensusTables(
        leaf_rows=sorted(leaf_rows, key=lambda row: -row.edge_count),
        rollup_rows=rollup_rows,
        rollup_index=rollup_index,
        forward_sources=forward_sources,
        inverse_sources=inverse_sources,
        total_edges=total_edges,
        semantics=semantics,
    )


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
    """Write the leaf, rollup and wide tables for one labelling."""
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
            }
            for row in census.leaf_rows
        ],
        LEAF_FIELDS,
    )
    write_tsv(out_dir / f"{prefix}_rollup.tsv", census.rollup_rows, ROLLUP_FIELDS)
    write_tsv(
        out_dir / f"{prefix}_wide.tsv", build_wide_rows(census, closure), WIDE_FIELDS
    )


def print_summary(
    census: CensusTables,
    predicate_summary: Sequence[dict],
    category_rollup: Sequence[dict],
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

    def closure_builder(predicates, categories):
        return build_closure_map(
            predicates,
            categories,
            version=args.biolink_version,
            schema=args.biolink_schema,
            include_mixins=not args.no_mixins,
        )

    if args.graph:
        arrays, closure = load_from_mmap(
            args.graph, closure_builder, args.category_policy
        )
    else:
        arrays, closure = load_from_jsonl(
            args.edges, args.nodes, closure_builder, args.category_policy
        )

    closure_path = args.out / "biolink_closure.json"
    with open(closure_path, "w", encoding="utf-8") as handle:
        json.dump(closure.to_json(), handle, indent=2, sort_keys=True)
    logger.info("  wrote %s", closure_path)

    census = run_census(
        arrays,
        closure,
        arrays.node_category_codes,
        arrays.categories,
        semantics=args.match_semantics,
    )
    write_census_tables(args.out, census, closure)

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

    print_summary(census, predicate_summary, category_rollup, args.top)
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
