"""Edge attributes are stored as the JSON array a TRAPI response carries.

A full response can then pass the stored bytes to the serializer as they are,
instead of decoding them into Python objects only to encode them again.
"""

import pickle

import orjson
import pytest

from tests.search_fixtures import graph  # noqa: F401

from gandalf.graph import CSRGraph, GraphFormatError
from gandalf.lmdb_store import EDGE_ATTRIBUTES_FORMAT, LMDBPropertyStore

ATTRIBUTES = {
    0: [{"attribute_type_id": "biolink:publications", "value": ["PMID:1", "PMID:2"]}],
    1: [],
    2: [
        {
            "attribute_type_id": "biolink:has_evidence",
            "value": "ECO:0000305",
            "attribute_source": "infores:x",
            "original_attribute_name": "evidence",
        },
        {"attribute_type_id": "biolink:score", "value": 0.25},
    ],
}


@pytest.fixture
def store(tmp_path):
    edges = ((idx, {"attributes": attrs}) for idx, attrs in ATTRIBUTES.items())
    return LMDBPropertyStore.build(tmp_path / "edges.lmdb", edges, num_edges=3)


def test_store_holds_each_edges_attributes_as_json(store):
    raw = store.get_json_batch([2, 0, 1, 2, 99])
    assert set(raw) == {0, 1, 2}
    for idx, attrs in ATTRIBUTES.items():
        assert raw[idx] == orjson.dumps(attrs)


def test_parsed_reads_match_the_json(store):
    batch = store.get_batch([0, 1, 2, 99])
    assert 99 not in batch
    for idx, attrs in ATTRIBUTES.items():
        assert store.get(idx) == {"attributes": attrs}
        assert batch[idx] == {"attributes": attrs}
    assert store.get(99) == {}


def test_built_graph_stores_attributes_as_json(graph):  # noqa: F811
    num_edges = len(graph.fwd_targets)
    raw = graph.lmdb_store.get_json_batch(range(num_edges))
    assert raw, "the fixture graph should have edge attributes"
    for idx, data in raw.items():
        attributes = orjson.loads(data)
        assert isinstance(attributes, list)
        assert graph.lmdb_store.get(idx) == {"attributes": attributes}


def test_saved_graph_records_the_format_and_loads(graph, tmp_path):  # noqa: F811
    graph.save_mmap(tmp_path / "graph")
    with open(tmp_path / "graph" / "metadata.pkl", "rb") as f:
        assert pickle.load(f)["edge_attributes_format"] == EDGE_ATTRIBUTES_FORMAT
    loaded = CSRGraph.load_mmap(tmp_path / "graph")
    num_edges = len(graph.fwd_targets)
    assert loaded.lmdb_store.get_json_batch(
        range(num_edges)
    ) == graph.lmdb_store.get_json_batch(range(num_edges))


def test_graph_from_before_json_attributes_is_refused(graph, tmp_path):  # noqa: F811
    graph.save_mmap(tmp_path / "graph")
    path = tmp_path / "graph" / "metadata.pkl"
    with open(path, "rb") as f:
        metadata = pickle.load(f)
    del metadata["edge_attributes_format"]
    with open(path, "wb") as f:
        pickle.dump(metadata, f)
    with pytest.raises(GraphFormatError, match="Rebuild the graph"):
        CSRGraph.load_mmap(tmp_path / "graph")
