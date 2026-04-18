"""Derive per-node metadata fields from a ``PublicationsIndex``.

This layer is source-agnostic: it only reads from the index, never from
PubTator directly.  The public entry point for node publication counts,
``derive_and_ingest_node_pub_counts``, aggregates PMIDs across each node's
``equivalent_identifiers`` (so MESH:D008687 contributes to the same
Metformin count as CHEBI:6801) and feeds the result straight into
``ingest_node_pub_counts_from_iter``, reusing its fail-loud validation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterator, Optional, Set, Tuple

from itertools import combinations

from gandalf.metadata.pair_intersections import (
    InMemoryPairCountAccumulator,
    PairCountAccumulator,
    ingest_pair_intersections_from_accumulator,
)
from gandalf.metadata.publications.index import PublicationsIndex
from gandalf.metadata.pub_counts import ingest_node_pub_counts_from_iter
from gandalf.node_store import NodeStore

logger = logging.getLogger(__name__)


def iter_node_equivalents(
    nodes_jsonl: Path,
) -> Iterator[Tuple[str, Set[str]]]:
    """Yield ``(node_id, {node_id, *equivalent_identifiers})`` for each node.

    The set always includes ``node_id`` itself so lookups stay uniform.
    """
    nodes_jsonl = Path(nodes_jsonl)
    with open(nodes_jsonl, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            node_id = obj.get("id")
            if not isinstance(node_id, str):
                raise ValueError(
                    f"{nodes_jsonl}:{lineno}: missing or non-string 'id' field"
                )
            equivalents = set(obj.get("equivalent_identifiers") or [])
            equivalents.add(node_id)
            yield node_id, equivalents


def collect_tracked_curies(nodes_jsonl: Path) -> Set[str]:
    """Return the union of ``id`` + ``equivalent_identifiers`` across every node.

    Used as the ``tracked_curies`` filter when building a
    ``PublicationsIndex`` so the index stores only CURIEs relevant to the
    graph.  For a 10M-node graph with ~5 equivalents each, this is a
    ~50M-string set that still fits comfortably in RAM.
    """
    tracked: Set[str] = set()
    for _, equivalents in iter_node_equivalents(nodes_jsonl):
        tracked.update(equivalents)
    return tracked


def iter_node_pub_counts(
    index: PublicationsIndex, nodes_jsonl: Path
) -> Iterator[Tuple[str, int]]:
    """Yield ``(node_id, pub_count)`` for every node in ``nodes_jsonl``.

    For each node the count is the size of the union of PMIDs associated
    with any of the node's CURIEs (``id`` + ``equivalent_identifiers``).
    Nodes not mentioned anywhere in the index yield ``count = 0``.
    """
    for node_id, equivalents in iter_node_equivalents(nodes_jsonl):
        # Union across every equivalent — a single publication that
        # mentions both CHEBI:6801 and DRUGBANK:DB00331 must be counted
        # once, not twice.
        pmids: Set[int] = set()
        for curie in equivalents:
            pmids.update(index.iter_pmids(curie))
        yield node_id, len(pmids)


def derive_and_ingest_node_pub_counts(
    graph_dir: Path,
    index_path: Path,
    nodes_jsonl: Path,
    source_label: Optional[str] = None,
) -> Path:
    """Derive per-node pub counts from ``index_path`` and write them to the graph.

    Args:
        graph_dir: Directory of the built Gandalf graph.
        index_path: Path to the ``PublicationsIndex`` LMDB directory.
        nodes_jsonl: Original nodes JSONL used to build the graph — needed
            for ``equivalent_identifiers``.
        source_label: Optional provenance string recorded in the manifest.
            Defaults to a description of the inputs.

    Returns:
        Path to the written ``node_pub_counts.npy``.
    """
    graph_dir = Path(graph_dir)
    index_path = Path(index_path)
    nodes_jsonl = Path(nodes_jsonl)

    source = source_label or (
        f"derive(index={index_path}, nodes={nodes_jsonl})"
    )
    logger.info("Deriving node pub counts: %s", source)

    with PublicationsIndex(index_path, readonly=True) as index:
        stats = index.stats()
        logger.info(
            "  PublicationsIndex stats: %s curie_to_pmid, %s pmid_to_curie",
            f"{stats['curie_to_pmid_entries']:,}",
            f"{stats['pmid_to_curie_entries']:,}",
        )
        return ingest_node_pub_counts_from_iter(
            graph_dir,
            iter_node_pub_counts(index, nodes_jsonl),
            source=source,
        )


# ---------------------------------------------------------------------------
# Pair intersections
# ---------------------------------------------------------------------------


def build_curie_to_node_idx(
    nodes_jsonl: Path, node_store: NodeStore
) -> dict:
    """Build a ``curie -> node_idx`` map covering ids + equivalent_identifiers.

    Held transiently in memory during pair-intersection derivation.  For a
    10M-node graph at ~5 equivalents per node this is ~50M entries — big
    but tractable on a build machine.  When a CURIE maps to more than one
    node (rare — typically points to a data quality issue in the node
    dictionary) the first binding wins and subsequent collisions are
    logged.
    """
    mapping: dict = {}
    collisions = 0
    for node_id, equivalents in iter_node_equivalents(nodes_jsonl):
        idx = node_store.get_node_idx(node_id)
        if idx is None:
            # The node_id in nodes.jsonl isn't in the graph — which only
            # happens if the jsonl and graph are out of sync.  Fail loud.
            raise ValueError(
                f"nodes.jsonl references {node_id!r} but graph has no such node"
            )
        for curie in equivalents:
            existing = mapping.get(curie)
            if existing is None:
                mapping[curie] = idx
            elif existing != idx:
                collisions += 1
    if collisions:
        logger.warning(
            "  Curie->node collisions: %s equivalent_identifier values bound "
            "to more than one node; first binding kept.",
            f"{collisions:,}",
        )
    return mapping


def accumulate_pair_counts(
    index: PublicationsIndex,
    curie_to_node_idx: dict,
    accumulator: Optional[PairCountAccumulator] = None,
) -> PairCountAccumulator:
    """Stream PMIDs, enumerate node-index pairs per PMID, sum into the accumulator.

    For each PMID we:
      1. Collect the CURIEs mentioned in that PMID.
      2. Map each to its ``node_idx`` (skipping CURIEs not in the graph).
      3. Dedupe — a PMID that mentions both CHEBI:6801 and DRUGBANK:DB00331
         maps them to the same Metformin node, which must only count once.
      4. Enumerate unordered pairs of distinct node indices, each +1.

    Returns the accumulator for chaining.
    """
    if accumulator is None:
        accumulator = InMemoryPairCountAccumulator()

    pmid_count = 0
    for pmid, curies in index.iter_pmid_groups():
        # Map to node indices, dedupe (a single PMID with two synonyms of the
        # same node still contributes only one vote to any pair involving it).
        indices: set = set()
        for curie in curies:
            idx = curie_to_node_idx.get(curie)
            if idx is not None:
                indices.add(idx)
        if len(indices) < 2:
            pmid_count += 1
            continue
        for a, b in combinations(sorted(indices), 2):
            accumulator.add_pair(a, b, 1)
        pmid_count += 1
        if pmid_count % 1_000_000 == 0:
            logger.info(
                "  pair accumulator: %s pmids processed, %s unique pairs so far",
                f"{pmid_count:,}",
                f"{len(accumulator):,}",
            )

    logger.info(
        "  pair accumulator: %s pmids total, %s unique pairs",
        f"{pmid_count:,}",
        f"{len(accumulator):,}",
    )
    return accumulator


def derive_and_ingest_pair_intersections(
    graph_dir: Path,
    index_path: Path,
    nodes_jsonl: Path,
    accumulator: Optional[PairCountAccumulator] = None,
    source_label: Optional[str] = None,
) -> Path:
    """Derive pair-intersection counts from ``index_path`` and write them to the graph.

    ``accumulator`` defaults to an in-memory dict; pass a different
    implementation (e.g. SQLite-backed) for large-scale runs.
    """
    graph_dir = Path(graph_dir)
    index_path = Path(index_path)
    nodes_jsonl = Path(nodes_jsonl)

    source = source_label or (
        f"derive(index={index_path}, nodes={nodes_jsonl})"
    )
    logger.info("Deriving pair intersections: %s", source)

    node_store = NodeStore(graph_dir / "node_store.lmdb", readonly=True)
    try:
        curie_to_idx = build_curie_to_node_idx(nodes_jsonl, node_store)
        logger.info("  curie->node_idx map: %s entries", f"{len(curie_to_idx):,}")
    finally:
        node_store.close()

    with PublicationsIndex(index_path, readonly=True) as index:
        stats = index.stats()
        logger.info(
            "  PublicationsIndex stats: %s curie_to_pmid, %s pmid_to_curie",
            f"{stats['curie_to_pmid_entries']:,}",
            f"{stats['pmid_to_curie_entries']:,}",
        )
        accumulator = accumulate_pair_counts(
            index, curie_to_idx, accumulator=accumulator
        )

    return ingest_pair_intersections_from_accumulator(
        graph_dir, accumulator, source=source
    )
