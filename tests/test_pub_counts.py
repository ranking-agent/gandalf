"""Tests for the pre-computed pub count ingest + load path."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from gandalf.graph import CSRGraph
from gandalf.loader import build_graph_from_jsonl
from gandalf.metadata import ScoringMetadata, load_manifest
from gandalf.metadata.pub_counts import (
    PubCountIngestError,
    ingest_edge_pub_counts,
    ingest_node_pub_counts,
)

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
NODES_FILE = os.path.join(FIXTURES_DIR, "nodes.jsonl")
EDGES_FILE = os.path.join(FIXTURES_DIR, "edges.jsonl")


@pytest.fixture
def graph_dir(tmp_path):
    """Build the fixture graph and save it as a mmap directory."""
    graph = build_graph_from_jsonl(EDGES_FILE, NODES_FILE)
    out = tmp_path / "graph_mmap"
    graph.save_mmap(str(out))
    return out


def _write_jsonl(path: Path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row))
            f.write("\n")


def _collect_node_ids(graph_dir: Path):
    from gandalf.node_store import NodeStore

    store = NodeStore(graph_dir / "node_store.lmdb", readonly=True)
    try:
        return [(nid, idx) for nid, idx in store.iter_id_to_idx()]
    finally:
        store.close()


def _collect_edge_ids(graph_dir: Path):
    """Return list of (edge_id, edge_idx) tuples from the graph's edge_ids.lmdb."""
    import struct

    import lmdb

    env = lmdb.open(
        str(graph_dir / "edge_ids.lmdb"),
        readonly=True,
        max_dbs=0,
        map_size=256 * 1024 * 1024 * 1024,
        readahead=False,
        lock=False,
    )
    out = []
    try:
        with env.begin(buffers=True) as txn:
            for k, v in txn.cursor():
                idx = struct.unpack(">I", bytes(k))[0]
                out.append((bytes(v).decode("utf-8"), idx))
    finally:
        env.close()
    return out


class TestNodePubCounts:
    def test_happy_path_roundtrip(self, graph_dir, tmp_path):
        node_ids = _collect_node_ids(graph_dir)
        input_path = tmp_path / "node_counts.jsonl"
        _write_jsonl(
            input_path,
            [{"node_id": nid, "count": 10 + idx} for nid, idx in node_ids],
        )

        out_path = ingest_node_pub_counts(graph_dir, input_path)
        assert out_path.exists()

        manifest = load_manifest(graph_dir)
        assert manifest.has_field("node_pub_counts")
        entry = manifest.get_field("node_pub_counts")
        assert entry["dtype"] == "uint32"
        assert entry["shape"] == [len(node_ids)]
        assert entry["nonzero"] == len(node_ids)

        counts = np.load(out_path)
        assert counts.dtype == np.uint32
        for nid, idx in node_ids:
            assert counts[idx] == 10 + idx

    def test_exposed_on_graph_after_load(self, graph_dir, tmp_path):
        node_ids = _collect_node_ids(graph_dir)
        input_path = tmp_path / "node_counts.jsonl"
        _write_jsonl(
            input_path,
            [{"node_id": nid, "count": idx * 2} for nid, idx in node_ids],
        )
        ingest_node_pub_counts(graph_dir, input_path)

        graph = CSRGraph.load_mmap(graph_dir)
        assert graph.scoring is not None
        assert isinstance(graph.scoring, ScoringMetadata)
        counts = graph.scoring.node_pub_counts
        assert counts is not None
        assert counts.dtype == np.uint32
        for nid, idx in node_ids:
            assert counts[idx] == idx * 2
        assert graph.scoring.edge_pub_counts is None  # not ingested

    def test_unknown_node_id_raises(self, graph_dir, tmp_path):
        node_ids = _collect_node_ids(graph_dir)
        rows = [{"node_id": nid, "count": 1} for nid, _ in node_ids]
        rows.append({"node_id": "NOT:A_REAL_NODE", "count": 1})
        input_path = tmp_path / "node_counts.jsonl"
        _write_jsonl(input_path, rows)

        with pytest.raises(PubCountIngestError, match="unknown node_id"):
            ingest_node_pub_counts(graph_dir, input_path)

    def test_duplicate_node_id_raises(self, graph_dir, tmp_path):
        node_ids = _collect_node_ids(graph_dir)
        rows = [{"node_id": nid, "count": 1} for nid, _ in node_ids]
        rows.append({"node_id": node_ids[0][0], "count": 999})
        input_path = tmp_path / "node_counts.jsonl"
        _write_jsonl(input_path, rows)

        with pytest.raises(PubCountIngestError, match="duplicate node_id"):
            ingest_node_pub_counts(graph_dir, input_path)

    def test_missing_node_raises(self, graph_dir, tmp_path):
        node_ids = _collect_node_ids(graph_dir)
        rows = [{"node_id": nid, "count": 1} for nid, _ in node_ids[:-1]]
        input_path = tmp_path / "node_counts.jsonl"
        _write_jsonl(input_path, rows)

        with pytest.raises(PubCountIngestError, match="does not cover every node"):
            ingest_node_pub_counts(graph_dir, input_path)

    def test_bad_count_type_raises(self, graph_dir, tmp_path):
        node_ids = _collect_node_ids(graph_dir)
        rows = [{"node_id": nid, "count": 1} for nid, _ in node_ids]
        rows[0]["count"] = "many"
        input_path = tmp_path / "node_counts.jsonl"
        _write_jsonl(input_path, rows)

        with pytest.raises(PubCountIngestError, match="'count' must be an integer"):
            ingest_node_pub_counts(graph_dir, input_path)

    def test_negative_count_raises(self, graph_dir, tmp_path):
        node_ids = _collect_node_ids(graph_dir)
        rows = [{"node_id": nid, "count": 1} for nid, _ in node_ids]
        rows[0]["count"] = -5
        input_path = tmp_path / "node_counts.jsonl"
        _write_jsonl(input_path, rows)

        with pytest.raises(PubCountIngestError, match="out of uint32 range"):
            ingest_node_pub_counts(graph_dir, input_path)


class TestEdgePubCounts:
    def test_happy_path_roundtrip(self, graph_dir, tmp_path):
        edge_ids = _collect_edge_ids(graph_dir)
        input_path = tmp_path / "edge_counts.jsonl"
        _write_jsonl(
            input_path,
            [{"edge_id": eid, "count": idx + 1} for eid, idx in edge_ids],
        )

        out_path = ingest_edge_pub_counts(graph_dir, input_path)
        assert out_path.exists()

        manifest = load_manifest(graph_dir)
        entry = manifest.get_field("edge_pub_counts")
        assert entry is not None
        assert entry["shape"] == [len(edge_ids)]

        counts = np.load(out_path)
        assert counts.dtype == np.uint32
        for eid, idx in edge_ids:
            assert counts[idx] == idx + 1

    def test_exposed_on_graph_after_load(self, graph_dir, tmp_path):
        edge_ids = _collect_edge_ids(graph_dir)
        input_path = tmp_path / "edge_counts.jsonl"
        _write_jsonl(
            input_path,
            [{"edge_id": eid, "count": 7} for eid, _ in edge_ids],
        )
        ingest_edge_pub_counts(graph_dir, input_path)

        graph = CSRGraph.load_mmap(graph_dir)
        counts = graph.scoring.edge_pub_counts
        assert counts is not None
        assert counts.dtype == np.uint32
        assert counts.shape == (len(edge_ids),)
        assert np.all(counts == 7)

    def test_unknown_edge_id_raises(self, graph_dir, tmp_path):
        edge_ids = _collect_edge_ids(graph_dir)
        rows = [{"edge_id": eid, "count": 1} for eid, _ in edge_ids]
        rows.append({"edge_id": "not-a-real-edge", "count": 1})
        input_path = tmp_path / "edge_counts.jsonl"
        _write_jsonl(input_path, rows)

        with pytest.raises(PubCountIngestError, match="unknown edge_id"):
            ingest_edge_pub_counts(graph_dir, input_path)

    def test_duplicate_edge_id_raises(self, graph_dir, tmp_path):
        edge_ids = _collect_edge_ids(graph_dir)
        rows = [{"edge_id": eid, "count": 1} for eid, _ in edge_ids]
        rows.append({"edge_id": edge_ids[0][0], "count": 999})
        input_path = tmp_path / "edge_counts.jsonl"
        _write_jsonl(input_path, rows)

        with pytest.raises(PubCountIngestError, match="duplicate edge_id"):
            ingest_edge_pub_counts(graph_dir, input_path)

    def test_missing_edge_raises(self, graph_dir, tmp_path):
        edge_ids = _collect_edge_ids(graph_dir)
        rows = [{"edge_id": eid, "count": 1} for eid, _ in edge_ids[:-1]]
        input_path = tmp_path / "edge_counts.jsonl"
        _write_jsonl(input_path, rows)

        with pytest.raises(PubCountIngestError, match="does not cover every edge"):
            ingest_edge_pub_counts(graph_dir, input_path)


class TestScoringMetadataMissing:
    def test_graph_without_manifest_has_scoring_none(self, graph_dir):
        graph = CSRGraph.load_mmap(graph_dir)
        assert graph.scoring is None

    def test_scoring_load_returns_none_for_empty_manifest_dir(self, tmp_path):
        assert ScoringMetadata.load(tmp_path) is None


class TestBothTogether:
    def test_node_and_edge_both_present_in_manifest(self, graph_dir, tmp_path):
        node_ids = _collect_node_ids(graph_dir)
        edge_ids = _collect_edge_ids(graph_dir)
        node_input = tmp_path / "node_counts.jsonl"
        edge_input = tmp_path / "edge_counts.jsonl"
        _write_jsonl(node_input, [{"node_id": nid, "count": 3} for nid, _ in node_ids])
        _write_jsonl(edge_input, [{"edge_id": eid, "count": 5} for eid, _ in edge_ids])

        ingest_node_pub_counts(graph_dir, node_input)
        ingest_edge_pub_counts(graph_dir, edge_input)

        manifest = load_manifest(graph_dir)
        assert manifest.has_field("node_pub_counts")
        assert manifest.has_field("edge_pub_counts")

        graph = CSRGraph.load_mmap(graph_dir)
        assert graph.scoring is not None
        assert graph.scoring.node_pub_counts is not None
        assert graph.scoring.edge_pub_counts is not None
        assert np.all(graph.scoring.node_pub_counts == 3)
        assert np.all(graph.scoring.edge_pub_counts == 5)
