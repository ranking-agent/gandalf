"""Tests for the publications index + PubTator parser + derive-node-counts path."""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

import numpy as np
import pytest

from gandalf.graph import CSRGraph
from gandalf.loader import build_graph_from_jsonl
from gandalf.metadata import load_manifest
from gandalf.metadata.publications import (
    PublicationsIndex,
    collect_tracked_curies,
    derive_and_ingest_node_pub_counts,
    iter_node_equivalents,
    iter_node_pub_counts,
    iter_pubtator_annotations,
)
from gandalf.metadata.publications.pubtator import _normalize_concept_id

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
NODES_FILE = os.path.join(FIXTURES_DIR, "nodes.jsonl")
EDGES_FILE = os.path.join(FIXTURES_DIR, "edges.jsonl")


# ---------------------------------------------------------------------------
# PubTator parser
# ---------------------------------------------------------------------------


class TestPubTatorParser:
    def test_normalizes_bare_gene_ids(self):
        assert _normalize_concept_id("5468", "Gene") == "NCBIGene:5468"

    def test_normalizes_bare_species_ids(self):
        assert _normalize_concept_id("9606", "Species") == "NCBITaxon:9606"

    def test_passes_through_prefixed_curies(self):
        assert _normalize_concept_id("MESH:D008687", "Chemical") == "MESH:D008687"

    def test_skips_dashes_and_empties(self):
        assert _normalize_concept_id("-", "Gene") is None
        assert _normalize_concept_id("", "Gene") is None

    def test_unknown_entity_type_without_prefix_is_skipped(self):
        assert _normalize_concept_id("1234", "UnknownType") is None

    def test_iter_yields_all_annotations(self, tmp_path):
        path = tmp_path / "annotations.tsv"
        path.write_text(
            "\n".join(
                [
                    "12345|t|Metformin treats diabetes.",
                    "12345|a|Metformin is a drug.",
                    "12345\t0\t9\tMetformin\tChemical\tMESH:D008687",
                    "12345\t20\t28\tdiabetes\tDisease\tMESH:D003920",
                    "12345\t0\t8\tPPARG\tGene\t5468",
                    "",
                    "67890\t0\t5\tINSR\tGene\t3643;5468",
                ]
            )
            + "\n"
        )
        pairs = list(iter_pubtator_annotations(path))
        assert (12345, "MESH:D008687") in pairs
        assert (12345, "MESH:D003920") in pairs
        assert (12345, "NCBIGene:5468") in pairs
        assert (67890, "NCBIGene:3643") in pairs
        assert (67890, "NCBIGene:5468") in pairs

    def test_gzip_input(self, tmp_path):
        path = tmp_path / "annotations.tsv.gz"
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write("111\t0\t5\tTNF\tGene\t7124\n")
        pairs = list(iter_pubtator_annotations(path))
        assert pairs == [(111, "NCBIGene:7124")]

    def test_tracked_curies_filters(self, tmp_path):
        path = tmp_path / "annotations.tsv"
        path.write_text(
            "1\t0\t5\tA\tGene\t5468\n" "1\t0\t5\tB\tGene\t9999\n",
            encoding="utf-8",
        )
        pairs = list(
            iter_pubtator_annotations(path, tracked_curies={"NCBIGene:5468"})
        )
        assert pairs == [(1, "NCBIGene:5468")]

    def test_entity_type_filter(self, tmp_path):
        path = tmp_path / "annotations.tsv"
        path.write_text(
            "\n".join(
                [
                    "1\t0\t5\tA\tGene\t5468",
                    "1\t0\t5\tB\tChemical\tMESH:D001",
                ]
            )
            + "\n"
        )
        pairs = list(iter_pubtator_annotations(path, entity_types={"Gene"}))
        assert pairs == [(1, "NCBIGene:5468")]


# ---------------------------------------------------------------------------
# PublicationsIndex
# ---------------------------------------------------------------------------


class TestPublicationsIndex:
    def _build(self, tmp_path, pairs, tracked=None):
        path = tmp_path / "index.lmdb"
        PublicationsIndex.build(path, iter(pairs), tracked_curies=tracked)
        return path

    def test_build_and_query_both_directions(self, tmp_path):
        path = self._build(
            tmp_path,
            [
                (1, "CHEBI:6801"),
                (1, "MONDO:0005148"),
                (2, "CHEBI:6801"),
                (3, "NCBIGene:5468"),
            ],
        )
        with PublicationsIndex(path, readonly=True) as idx:
            assert idx.count_pmids("CHEBI:6801") == 2
            assert idx.pmid_set("CHEBI:6801") == {1, 2}
            assert idx.pmid_set("MONDO:0005148") == {1}
            assert idx.pmid_set("NCBIGene:5468") == {3}
            assert idx.count_pmids("DOES_NOT_EXIST") == 0
            assert list(idx.iter_curies_for_pmid(1)) == sorted(
                ["CHEBI:6801", "MONDO:0005148"]
            )

    def test_dupsort_dedupes_repeat_pairs(self, tmp_path):
        path = self._build(
            tmp_path,
            [(1, "CHEBI:6801"), (1, "CHEBI:6801"), (1, "CHEBI:6801")],
        )
        with PublicationsIndex(path, readonly=True) as idx:
            assert idx.count_pmids("CHEBI:6801") == 1
            assert idx.pmid_set("CHEBI:6801") == {1}

    def test_tracked_curies_filter_applied_at_build(self, tmp_path):
        path = self._build(
            tmp_path,
            [(1, "CHEBI:6801"), (1, "OUT:OF_GRAPH"), (2, "CHEBI:6801")],
            tracked={"CHEBI:6801"},
        )
        with PublicationsIndex(path, readonly=True) as idx:
            assert idx.pmid_set("CHEBI:6801") == {1, 2}
            assert idx.count_pmids("OUT:OF_GRAPH") == 0

    def test_overwrite_replaces_store(self, tmp_path):
        path = tmp_path / "index.lmdb"
        PublicationsIndex.build(path, iter([(1, "A")]))
        with pytest.raises(FileExistsError):
            PublicationsIndex.build(path, iter([(2, "B")]))
        PublicationsIndex.build(path, iter([(2, "B")]), overwrite=True)
        with PublicationsIndex(path, readonly=True) as idx:
            assert idx.count_pmids("A") == 0
            assert idx.pmid_set("B") == {2}

    def test_iter_pmid_groups_returns_all_pmids(self, tmp_path):
        path = self._build(
            tmp_path,
            [(1, "A"), (1, "B"), (2, "B"), (2, "C")],
        )
        with PublicationsIndex(path, readonly=True) as idx:
            groups = dict(idx.iter_pmid_groups())
        assert set(groups[1]) == {"A", "B"}
        assert set(groups[2]) == {"B", "C"}


# ---------------------------------------------------------------------------
# Derive
# ---------------------------------------------------------------------------


class TestDeriveNodeEquivalents:
    def test_yields_id_plus_equivalents(self):
        pairs = list(iter_node_equivalents(NODES_FILE))
        by_id = dict(pairs)
        assert "CHEBI:6801" in by_id["CHEBI:6801"]
        assert "DRUGBANK:DB00331" in by_id["CHEBI:6801"]
        # Every node should include its own id in the equivalent set.
        for node_id, equivalents in pairs:
            assert node_id in equivalents

    def test_collect_tracked_curies_is_union(self):
        tracked = collect_tracked_curies(NODES_FILE)
        assert "CHEBI:6801" in tracked
        assert "DRUGBANK:DB00331" in tracked
        assert "HGNC:9236" in tracked
        assert "MONDO:0005148" in tracked


class TestDeriveNodePubCounts:
    @pytest.fixture
    def graph_dir(self, tmp_path):
        graph = build_graph_from_jsonl(EDGES_FILE, NODES_FILE)
        out = tmp_path / "graph_mmap"
        graph.save_mmap(str(out))
        return out

    @pytest.fixture
    def index_path(self, tmp_path):
        """Build a tiny publications index keyed by our fixture CURIEs."""
        path = tmp_path / "index.lmdb"
        PublicationsIndex.build(
            path,
            iter(
                [
                    (1, "CHEBI:6801"),           # Metformin
                    (1, "MONDO:0005148"),         # Type 2 Diabetes (co-mention)
                    (2, "CHEBI:6801"),
                    (3, "DRUGBANK:DB00331"),      # equivalent of CHEBI:6801
                    (4, "HGNC:9236"),             # equivalent of NCBIGene:5468 (PPARG)
                    (5, "NCBIGene:5468"),
                    (6, "MONDO:0005148"),
                ]
            ),
        )
        return path

    def test_iter_counts_unions_equivalents(self, index_path):
        with PublicationsIndex(index_path, readonly=True) as idx:
            counts = dict(iter_node_pub_counts(idx, NODES_FILE))
        # Metformin's id + DRUGBANK equivalent should union pmids {1,2,3}.
        assert counts["CHEBI:6801"] == 3
        # PPARG via NCBIGene:5468 + HGNC:9236 -> pmids {4,5}.
        assert counts["NCBIGene:5468"] == 2
        # Type 2 Diabetes -> pmids {1,6}.
        assert counts["MONDO:0005148"] == 2
        # Untracked nodes -> 0.
        assert counts["HP:0001943"] == 0
        assert counts["GO:0006006"] == 0

    def test_derive_and_ingest_writes_array_and_manifest(
        self, graph_dir, index_path
    ):
        out = derive_and_ingest_node_pub_counts(
            graph_dir=graph_dir,
            index_path=index_path,
            nodes_jsonl=NODES_FILE,
        )
        assert out.exists()

        manifest = load_manifest(graph_dir)
        entry = manifest.get_field("node_pub_counts")
        assert entry is not None
        assert entry["dtype"] == "uint32"
        assert "derive(" in entry["source"]

        graph = CSRGraph.load_mmap(graph_dir)
        counts = graph.scoring.node_pub_counts
        assert counts.dtype == np.uint32
        met_idx = graph.get_node_idx("CHEBI:6801")
        pparg_idx = graph.get_node_idx("NCBIGene:5468")
        hypo_idx = graph.get_node_idx("HP:0001943")
        assert counts[met_idx] == 3
        assert counts[pparg_idx] == 2
        assert counts[hypo_idx] == 0

    def test_derive_rejects_malformed_nodes_file(
        self, graph_dir, index_path, tmp_path
    ):
        bad = tmp_path / "bad_nodes.jsonl"
        bad.write_text(
            json.dumps({"name": "no id here"}) + "\n", encoding="utf-8"
        )
        with pytest.raises(ValueError, match="missing or non-string 'id'"):
            derive_and_ingest_node_pub_counts(
                graph_dir=graph_dir,
                index_path=index_path,
                nodes_jsonl=bad,
            )
