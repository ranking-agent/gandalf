"""Edge IDs are stored as one memory-mapped blob of JSON strings.

See ``gandalf.edge_id_store``.  A response writer copies an ID's stored JSON
into its output as it is, so the stored bytes must be exactly what orjson
makes of the ID, and decoding them must give the ID back.
"""

import pickle

import numpy as np
import orjson
import pytest

from tests.search_fixtures import graph  # noqa: F401

from gandalf.config import settings
from gandalf.edge_id_store import (
    BLOB_FILE,
    EDGE_IDS_FORMAT,
    EdgeIdStore,
    decode_id,
    ids_unique,
)
from gandalf.graph import CSRGraph, GraphFormatError

IDS = [
    "infores:x/1",
    None,
    "urn:uuid:0b6f4f8e-1c2d-4e5f-8a9b-0c1d2e3f4a5b",
    "",
    'quote " and backslash \\',
    "control \n\t\x01",
    "unicode é 中 🧬",
    42,
    "/slash/",
]


def _expected(edge_id):
    """What the store hands back for *edge_id*: its str, None for no ID."""
    return str(edge_id) if edge_id not in (None, "") else None


#: IDs whose JSON needs no escape: the store then decodes without parsing.
PLAIN_IDS = [i for i in IDS if orjson.dumps(str(i)).count(b"\\") == 0]


@pytest.mark.parametrize("ids, escaped", [(IDS, True), (PLAIN_IDS, False)])
@pytest.mark.parametrize("chunk", [1, 2, 1000])
def test_round_trip(ids, escaped, chunk):
    store = EdgeIdStore.from_ids(ids, chunk=chunk)
    assert store.escaped is escaped
    assert len(store) == len(ids)
    expected = [_expected(i) for i in ids]
    assert [store.get(i) for i in range(len(ids))] == expected
    assert store.get_batch(range(len(ids))) == {
        i: eid for i, eid in enumerate(expected) if eid
    }
    spans = [span for span in store.get_json_batch(np.arange(len(ids))) if span]
    assert store.decode(spans) == [eid for eid in expected if eid]


def test_json_is_what_orjson_writes():
    store = EdgeIdStore.from_ids(IDS)
    order = np.array([6, 0, 1, 4, 4])
    assert store.get_json_batch(order) == [
        orjson.dumps(_expected(IDS[i])) if _expected(IDS[i]) else b"" for i in order
    ]
    assert store.lengths(order).tolist() == [
        len(span) for span in store.get_json_batch(order)
    ]


@pytest.mark.parametrize("edge_id", [i for i in IDS if isinstance(i, str) and i])
def test_decode_id(edge_id):
    assert decode_id(orjson.dumps(edge_id)) == edge_id


@pytest.mark.parametrize(
    "ids, unique",
    [
        ([], True),
        (["a", "b", "c"], True),
        (["a", None, "", None, ""], True),
        (["a", "b", "a"], False),
        ([f"e{i}" for i in range(10_000)] + ["e9999"], False),
    ],
)
def test_ids_unique(ids, unique):
    assert ids_unique(ids) is unique
    assert EdgeIdStore.from_ids(ids).unique is unique


@pytest.mark.parametrize("in_memory", [False, True])
def test_save_and_load(tmp_path, in_memory):
    store = EdgeIdStore.from_ids(IDS)
    store.save(tmp_path)
    loaded = EdgeIdStore.load(tmp_path, store.metadata(), in_memory=in_memory)
    everything = np.arange(len(IDS))
    assert loaded.get_json_batch(everything) == store.get_json_batch(everything)
    assert loaded.offsets.tolist() == store.offsets.tolist()
    loaded.close()


def test_load_edges_without_ids(tmp_path):
    """No edge has an ID: the blob is empty, which mmap cannot map."""
    store = EdgeIdStore.from_ids([None, ""])
    store.save(tmp_path)
    assert (tmp_path / BLOB_FILE).stat().st_size == 0
    loaded = EdgeIdStore.load(tmp_path, store.metadata())
    assert loaded.get_batch([0, 1]) == {}
    assert loaded.lengths(np.array([0, 1])).tolist() == [0, 0]


# ---------------------------------------------------------------------------
# In a saved graph
# ---------------------------------------------------------------------------


def _metadata(directory) -> dict:
    with open(directory / "metadata.pkl", "rb") as f:
        return pickle.load(f)


@pytest.mark.parametrize("in_memory", [False, True])
def test_saved_graph_keeps_its_ids(
    graph, tmp_path, monkeypatch, in_memory
):  # noqa: F811
    monkeypatch.setattr(settings, "load_mmaps_into_memory", in_memory)
    graph.save_mmap(tmp_path)
    metadata = _metadata(tmp_path)
    assert metadata["edge_ids_format"] == EDGE_IDS_FORMAT
    assert metadata["edge_ids_unique"] is graph.edge_id_store.unique
    assert metadata["edge_ids_escaped"] is graph.edge_id_store.escaped
    loaded = CSRGraph.load_mmap(tmp_path)
    num_edges = len(graph.fwd_targets)
    assert loaded.edge_id_store.unique is graph.edge_id_store.unique
    assert loaded.get_edge_ids_batch(range(num_edges)) == graph.get_edge_ids_batch(
        range(num_edges)
    )
    assert all(loaded.get_edge_id(i) == graph.get_edge_id(i) for i in range(num_edges))


def test_duplicate_ids_are_recorded(graph, tmp_path, monkeypatch):  # noqa: F811
    ids = [graph.get_edge_id(i) for i in range(len(graph.fwd_targets))]
    ids[1] = ids[0]
    monkeypatch.setattr(graph, "edge_id_store", EdgeIdStore.from_ids(ids))
    graph.save_mmap(tmp_path)
    assert _metadata(tmp_path)["edge_ids_unique"] is False
    assert CSRGraph.load_mmap(tmp_path).edge_id_store.unique is False


def test_graph_with_ids_in_lmdb_is_refused(graph, tmp_path):  # noqa: F811
    graph.save_mmap(tmp_path)
    metadata = _metadata(tmp_path)
    del metadata["edge_ids_format"]
    with open(tmp_path / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
    with pytest.raises(GraphFormatError, match="Rebuild the graph"):
        CSRGraph.load_mmap(tmp_path)
