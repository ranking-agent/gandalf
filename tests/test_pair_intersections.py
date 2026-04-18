"""Tests for the pair intersection CSR store and derive-from-publications path."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from gandalf.graph import CSRGraph
from gandalf.loader import build_graph_from_jsonl
from gandalf.metadata import (
    InMemoryPairCountAccumulator,
    PairIntersections,
    load_manifest,
)
from gandalf.metadata.pair_intersections import (
    ingest_pair_intersections_from_accumulator,
    write_pair_intersections,
)
from gandalf.metadata.publications import (
    PublicationsIndex,
    derive_and_ingest_pair_intersections,
)

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
NODES_FILE = os.path.join(FIXTURES_DIR, "nodes.jsonl")
EDGES_FILE = os.path.join(FIXTURES_DIR, "edges.jsonl")


# ---------------------------------------------------------------------------
# InMemoryPairCountAccumulator
# ---------------------------------------------------------------------------


class TestInMemoryAccumulator:
    def test_self_pairs_ignored(self):
        acc = InMemoryPairCountAccumulator()
        acc.add_pair(5, 5, 10)
        assert len(acc) == 0
        assert list(acc.iter_sorted()) == []

    def test_normalizes_to_a_less_than_b(self):
        acc = InMemoryPairCountAccumulator()
        acc.add_pair(7, 3, 2)
        acc.add_pair(3, 7, 5)
        assert list(acc.iter_sorted()) == [(3, 7, 7)]

    def test_sums_duplicates(self):
        acc = InMemoryPairCountAccumulator()
        for _ in range(4):
            acc.add_pair(1, 2)
        acc.add_pair(1, 2, 10)
        assert list(acc.iter_sorted()) == [(1, 2, 14)]

    def test_sorts_by_a_then_b(self):
        acc = InMemoryPairCountAccumulator()
        acc.add_pair(2, 5, 1)
        acc.add_pair(0, 9, 1)
        acc.add_pair(2, 3, 1)
        acc.add_pair(0, 1, 1)
        assert list(acc.iter_sorted()) == [
            (0, 1, 1),
            (0, 9, 1),
            (2, 3, 1),
            (2, 5, 1),
        ]


# ---------------------------------------------------------------------------
# CSR writer + reader
# ---------------------------------------------------------------------------


class TestWriteAndReadCSR:
    @pytest.fixture
    def graph_dir(self, tmp_path):
        graph = build_graph_from_jsonl(EDGES_FILE, NODES_FILE)
        out = tmp_path / "graph_mmap"
        graph.save_mmap(str(out))
        return out

    def _write(self, graph_dir, num_nodes, pairs, source="test"):
        return write_pair_intersections(
            graph_dir=graph_dir,
            num_nodes=num_nodes,
            num_pairs=len(pairs),
            sorted_pairs=iter(pairs),
            source=source,
        )

    def test_roundtrip_basic(self, graph_dir):
        self._write(
            graph_dir,
            num_nodes=6,
            pairs=[(0, 2, 5), (0, 4, 1), (1, 3, 2), (3, 5, 7)],
        )
        store = PairIntersections(graph_dir)
        assert store.num_nodes == 6
        assert store.num_pairs == 4
        assert store.count(0, 2) == 5
        assert store.count(2, 0) == 5  # order-agnostic
        assert store.count(0, 4) == 1
        assert store.count(1, 3) == 2
        assert store.count(3, 5) == 7
        # Missing pairs
        assert store.count(0, 5) == 0
        assert store.count(2, 4) == 0
        assert store.count(5, 5) == 0  # self

    def test_out_of_range_indices(self, graph_dir):
        self._write(graph_dir, num_nodes=5, pairs=[(0, 1, 1)])
        store = PairIntersections(graph_dir)
        assert store.count(-1, 0) == 0
        assert store.count(0, 5) == 0
        assert store.count(5, 7) == 0

    def test_neighbors_returns_row(self, graph_dir):
        self._write(
            graph_dir,
            num_nodes=8,
            pairs=[(2, 3, 10), (2, 5, 20), (2, 7, 30), (4, 6, 5)],
        )
        store = PairIntersections(graph_dir)
        ns, cs = store.neighbors(2)
        assert list(ns) == [3, 5, 7]
        assert list(cs) == [10, 20, 30]
        ns2, _ = store.neighbors(3)  # 3's pairs are under lower node, not stored here
        assert list(ns2) == []

    def test_manifest_recorded(self, graph_dir):
        self._write(graph_dir, num_nodes=4, pairs=[(0, 2, 3)], source="unit-test")
        manifest = load_manifest(graph_dir)
        entry = manifest.get_field("pair_intersections")
        assert entry is not None
        assert entry["num_pairs"] == 1
        assert entry["total_shared_publications"] == 3
        assert entry["max_count"] == 3
        assert entry["source"] == "unit-test"

    def test_rejects_unsorted_stream(self, graph_dir):
        with pytest.raises(ValueError, match="not strictly sorted"):
            self._write(graph_dir, num_nodes=5, pairs=[(0, 2, 1), (0, 1, 1)])

    def test_rejects_a_not_less_than_b(self, graph_dir):
        with pytest.raises(ValueError, match="must have a < b"):
            self._write(graph_dir, num_nodes=5, pairs=[(3, 3, 1)])
        with pytest.raises(ValueError, match="must have a < b"):
            self._write(graph_dir, num_nodes=5, pairs=[(4, 2, 1)])

    def test_rejects_zero_count(self, graph_dir):
        with pytest.raises(ValueError, match="count must be > 0"):
            self._write(graph_dir, num_nodes=5, pairs=[(0, 1, 0)])

    def test_rejects_out_of_range_node(self, graph_dir):
        with pytest.raises(ValueError, match="out of range"):
            self._write(graph_dir, num_nodes=3, pairs=[(1, 5, 1)])

    def test_empty_store(self, graph_dir):
        write_pair_intersections(
            graph_dir=graph_dir,
            num_nodes=5,
            num_pairs=0,
            sorted_pairs=iter([]),
            source="empty",
        )
        store = PairIntersections(graph_dir)
        assert store.num_pairs == 0
        assert store.count(0, 1) == 0


# ---------------------------------------------------------------------------
# Accumulator -> ingest
# ---------------------------------------------------------------------------


class TestIngestFromAccumulator:
    @pytest.fixture
    def graph_dir(self, tmp_path):
        graph = build_graph_from_jsonl(EDGES_FILE, NODES_FILE)
        out = tmp_path / "graph_mmap"
        graph.save_mmap(str(out))
        return out

    def test_ingest_roundtrip(self, graph_dir):
        acc = InMemoryPairCountAccumulator()
        acc.add_pair(1, 4, 2)
        acc.add_pair(1, 4, 1)  # sum to 3
        acc.add_pair(2, 3, 5)
        ingest_pair_intersections_from_accumulator(graph_dir, acc, source="test")

        graph = CSRGraph.load_mmap(graph_dir)
        store = graph.scoring.pair_intersections
        assert store.count(1, 4) == 3
        assert store.count(2, 3) == 5
        assert store.count(4, 1) == 3


# ---------------------------------------------------------------------------
# Derive from PublicationsIndex
# ---------------------------------------------------------------------------


class TestDerivePairIntersections:
    @pytest.fixture
    def graph_dir(self, tmp_path):
        graph = build_graph_from_jsonl(EDGES_FILE, NODES_FILE)
        out = tmp_path / "graph_mmap"
        graph.save_mmap(str(out))
        return out

    @pytest.fixture
    def index_path(self, tmp_path):
        """Publications index with known co-occurrences.

        Expected shared-pub counts (nodes identified by their fixture ids):
          * CHEBI:6801 <-> MONDO:0005148 : PMIDs 1, 10  -> 2
          * CHEBI:6801 <-> NCBIGene:5468 : PMID 10      -> 1
          * MONDO:0005148 <-> NCBIGene:5468 : PMID 10   -> 1
          * CHEBI:6801 (via DRUGBANK alias) <-> HP:0001943: PMID 20 -> 1
          * NCBIGene:5468 (via HGNC alias) <-> MONDO:0005148: PMID 30 -> 1
            (which adds to the 10 pair above -> final 2)
        """
        path = tmp_path / "index.lmdb"
        PublicationsIndex.build(
            path,
            iter(
                [
                    (1, "CHEBI:6801"),
                    (1, "MONDO:0005148"),
                    (10, "CHEBI:6801"),
                    (10, "MONDO:0005148"),
                    (10, "NCBIGene:5468"),
                    (20, "DRUGBANK:DB00331"),  # equivalent of CHEBI:6801
                    (20, "HP:0001943"),
                    (30, "HGNC:9236"),          # equivalent of NCBIGene:5468
                    (30, "MONDO:0005148"),
                    (40, "GO:0006006"),          # singleton — no pairs emitted
                ]
            ),
        )
        return path

    def test_end_to_end(self, graph_dir, index_path):
        derive_and_ingest_pair_intersections(
            graph_dir=graph_dir,
            index_path=index_path,
            nodes_jsonl=NODES_FILE,
        )

        graph = CSRGraph.load_mmap(graph_dir)
        store = graph.scoring.pair_intersections

        def idx(node_id: str) -> int:
            return graph.get_node_idx(node_id)

        met = idx("CHEBI:6801")
        t2d = idx("MONDO:0005148")
        pparg = idx("NCBIGene:5468")
        hypo = idx("HP:0001943")
        glu = idx("CHEBI:17234")
        pathway = idx("GO:0006006")

        assert store.count(met, t2d) == 2       # PMIDs 1 and 10
        assert store.count(met, pparg) == 1     # PMID 10
        assert store.count(t2d, pparg) == 2     # PMIDs 10 and 30 (via HGNC alias)
        assert store.count(met, hypo) == 1      # PMID 20 via DRUGBANK alias
        assert store.count(glu, pathway) == 0   # only mentioned alone (PMID 40)

        # Manifest recorded
        manifest = load_manifest(graph_dir)
        entry = manifest.get_field("pair_intersections")
        assert entry is not None
        assert entry["num_pairs"] == 4

    def test_missing_manifest_returns_none(self, graph_dir):
        graph = CSRGraph.load_mmap(graph_dir)
        # Nothing ingested yet
        assert graph.scoring is None or graph.scoring.pair_intersections is None
