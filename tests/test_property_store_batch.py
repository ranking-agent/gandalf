"""Parity tests for batched node/edge property retrieval.

These cover the single-transaction, sorted batch reads added to speed up
attribute retrieval for large result sets.  The contract is simple: a batch
read must return exactly what N individual reads would, for any (including
scattered or partially-missing) set of indices.
"""

import os

import pytest

from gandalf.lmdb_store import LMDBPropertyStore
from gandalf.loader import build_graph_from_jsonl
from gandalf.node_store import NodeStore

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
NODES_FILE = os.path.join(FIXTURES_DIR, "nodes.jsonl")
EDGES_FILE = os.path.join(FIXTURES_DIR, "edges.jsonl")


# ---------------------------------------------------------------------------
# Standalone store-level parity
# ---------------------------------------------------------------------------


def test_node_store_get_batch_matches_individual(tmp_path):
    id_to_idx = {f"CURIE:{i}": i for i in range(10)}
    props = {i: {"name": f"node-{i}", "categories": [f"cat-{i}"]} for i in range(10)}
    store = NodeStore.build(tmp_path / "nodes.lmdb", id_to_idx, props)

    # Scattered order, with a duplicate and an out-of-range (missing) index.
    query = [7, 1, 7, 4, 99, 0]
    batch = store.get_batch(query)

    for idx in query:
        if idx in props:
            assert batch[idx] == store.get_properties(idx)
        else:
            assert idx not in batch  # missing indices are simply absent


def test_node_store_get_node_ids_batch_matches_individual(tmp_path):
    id_to_idx = {f"CURIE:{i}": i for i in range(10)}
    props = {i: {"name": f"node-{i}"} for i in range(10)}
    store = NodeStore.build(tmp_path / "nodes.lmdb", id_to_idx, props)

    query = [5, 2, 9, 200, 0]
    batch = store.get_node_ids_batch(query)

    for idx in query:
        assert batch.get(idx) == store.get_node_id(idx)


def test_lmdb_store_get_batch_matches_individual(tmp_path):
    edges = {
        i: {"attributes": [{"attribute_type_id": "x", "value": i}]} for i in range(10)
    }

    def edge_iter():
        for i in range(10):
            yield i, edges[i]

    store = LMDBPropertyStore.build(tmp_path / "edges.lmdb", edge_iter(), num_edges=10)

    query = [8, 3, 8, 0, 123]
    batch = store.get_batch(query)

    for idx in query:
        if idx in edges:
            assert batch[idx] == store.get(idx)
        else:
            assert idx not in batch


# ---------------------------------------------------------------------------
# Graph-level parity (LMDB-backed via save_mmap/load_mmap)
# ---------------------------------------------------------------------------


@pytest.fixture
def lmdb_graph(tmp_path):
    """A graph reloaded from mmap, so node_store/lmdb_store/edge_ids are LMDB-backed."""
    from gandalf.graph import CSRGraph

    graph = build_graph_from_jsonl(EDGES_FILE, NODES_FILE)
    mmap_dir = tmp_path / "graph"
    graph.save_mmap(mmap_dir)
    return CSRGraph.load_mmap(mmap_dir)


def test_graph_node_property_and_id_batch_parity(lmdb_graph):
    assert lmdb_graph.node_store is not None  # confirm LMDB-backed path
    node_indices = list(range(lmdb_graph.num_nodes))

    props_batch = lmdb_graph.get_all_node_properties_batch(node_indices)
    id_batch = lmdb_graph.get_node_ids_batch(node_indices)

    for idx in node_indices:
        assert props_batch.get(idx, {}) == lmdb_graph.get_all_node_properties(idx)
        assert id_batch.get(idx) == lmdb_graph.get_node_id(idx)


def test_graph_edge_id_batch_parity(lmdb_graph):
    edge_indices = list(range(len(lmdb_graph.fwd_predicates)))
    id_batch = lmdb_graph.get_edge_ids_batch(edge_indices)

    for idx in edge_indices:
        assert id_batch.get(idx) == lmdb_graph.get_edge_id(idx)


def test_graph_edge_properties_with_prefetched_detail_parity(lmdb_graph):
    edge_indices = list(range(len(lmdb_graph.fwd_predicates)))
    detail_map = lmdb_graph.lmdb_store.get_batch(edge_indices)

    for idx in edge_indices:
        from_individual = lmdb_graph.get_edge_properties_by_index(idx)
        from_prefetched = lmdb_graph.get_edge_properties_by_index(
            idx, lmdb_detail=detail_map.get(idx, {})
        )
        assert from_individual == from_prefetched
