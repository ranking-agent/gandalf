"""Node attributes are stored as the JSON array a TRAPI response carries.

A full response built for the server hands each node's stored bytes to the
serializer as they are, so a response must come out byte-identical to one
built from a node store holding them as msgpack arrays (as stores written
before this did), and every other reader must still see plain lists.
"""

import json
import shutil

import lmdb
import msgpack
import orjson
import pytest

from gandalf.execute import serialize_response
from gandalf.graph import CSRGraph
from gandalf.loader import build_graph_from_jsonl
from gandalf.node_store import (
    NodeStore,
    decode_properties,
    rewrite_attributes_as_json,
)
from gandalf.plugins.literature_cooccurrence_annotator import _attach_node_counts
from gandalf.search.lookup import lookup
from gandalf.trapi import to_fragments
from tests.search_fixtures import EDGES_FILE, NODES_FILE


def _downgrade(node_store_path):
    """Rewrite a node store's records the way older builds stored them."""
    env = lmdb.open(str(node_store_path), max_dbs=3)
    db = env.open_db(b"properties")
    with env.begin(write=True, db=db) as txn:
        records = [(bytes(k), decode_properties(v)) for k, v in txn.cursor()]
        for key, props in records:
            txn.put(key, msgpack.packb(props, use_bin_type=True))
    env.close()


@pytest.fixture(scope="module")
def graph_dirs(tmp_path_factory):
    base = tmp_path_factory.mktemp("graphs")
    new = base / "new"
    build_graph_from_jsonl(EDGES_FILE, NODES_FILE).save_mmap(new)
    old = base / "old"
    shutil.copytree(new, old)
    _downgrade(old / "node_store.lmdb")
    return new, old


def _load(path):
    return CSRGraph.load_mmap(path, serve_metadata=False)


def _close(graph):
    for store in (graph.node_store, graph.lmdb_store, graph.source_urls_store):
        if store is not None:
            store.close()


def _server_bytes(graph, bmt, query, **kwargs):
    """A response's message as the server builds and serializes it."""
    response = lookup(
        graph,
        query,
        bmt=bmt,
        attributes_as_json=True,
        results_as_json=True,
        **kwargs,
    )
    to_fragments(response)
    return serialize_response({"message": response["message"]})


def _one_hop(pinned):
    return {
        "message": {
            "query_graph": {
                "nodes": {"a": {"ids": [pinned]}, "b": {}},
                "edges": {"e": {"subject": "a", "object": "b"}},
            }
        }
    }


def _fixture_node_ids():
    with open(NODES_FILE) as f:
        return [json.loads(line)["id"] for line in f if line.strip()]


def test_new_store_keeps_attributes_as_json(graph_dirs):
    new, old = graph_dirs
    with NodeStore(new / "node_store.lmdb") as store:
        raw = store.get_batch([0, 1], attributes_as_json=True)
        decoded = store.get_batch([0, 1])
    for idx in (0, 1):
        assert type(raw[idx]["attributes"]) is bytes
        assert orjson.loads(raw[idx]["attributes"]) == decoded[idx]["attributes"]
    with NodeStore(old / "node_store.lmdb") as store:
        assert (
            type(store.get_batch([0], attributes_as_json=True)[0]["attributes"]) is list
        )


@pytest.mark.parametrize("pinned", _fixture_node_ids())
@pytest.mark.parametrize("subclass", [True, False])
def test_full_responses_match_the_older_store(graph_dirs, bmt, pinned, subclass):
    new, old = graph_dirs
    outputs = []
    for path in (new, old):
        graph = _load(path)
        outputs.append(
            _server_bytes(
                graph, bmt, _one_hop(pinned), dehydrated=False, subclass=subclass
            )
        )
        _close(graph)
    assert outputs[0] == outputs[1]
    nodes = json.loads(outputs[0])["message"]["knowledge_graph"]["nodes"]
    assert all(isinstance(node["attributes"], list) for node in nodes.values())


def test_decoded_lookups_still_give_lists(graph_dirs, bmt):
    graph = _load(graph_dirs[0])
    response = lookup(graph, _one_hop("CHEBI:6801"), bmt=bmt, dehydrated=False)
    _close(graph)
    nodes = response["message"]["knowledge_graph"]["nodes"]
    assert nodes and all(type(n["attributes"]) is list for n in nodes.values())


def test_rewrite_converts_an_older_store(graph_dirs, tmp_path):
    new, old = graph_dirs
    copy = tmp_path / "node_store.lmdb"
    shutil.copytree(old / "node_store.lmdb", copy)
    with NodeStore(copy) as store:
        before = dict(store.iter_properties())
        ids_before = dict(store.iter_id_to_idx())
    assert rewrite_attributes_as_json(copy) == len(before)
    assert rewrite_attributes_as_json(copy) == 0
    with NodeStore(copy) as store:
        assert dict(store.iter_properties()) == before
        assert dict(store.iter_id_to_idx()) == ids_before
        raw = store.get_batch(list(before), attributes_as_json=True)
    assert all(type(p["attributes"]) is bytes for p in raw.values())


def test_annotators_can_add_to_stored_attributes():
    nodes = {
        "A": {"attributes": b'[{"attribute_type_id":"biolink:x","value":1}]'},
        "B": {},
    }
    _attach_node_counts(nodes, {"A": 3, "B": 4}, "infores:test")
    assert [a["value"] for a in nodes["A"]["attributes"]] == [1, 3]
    assert [a["value"] for a in nodes["B"]["attributes"]] == [4]


def test_server_path_carries_node_attributes_as_stored_json(graph_dirs, bmt):
    graph = _load(graph_dirs[0])
    response = lookup(
        graph,
        _one_hop("CHEBI:6801"),
        bmt=bmt,
        dehydrated=False,
        attributes_as_json=True,
    )
    _close(graph)
    nodes = response["message"]["knowledge_graph"]["nodes"]
    assert nodes and all(type(n["attributes"]) is bytes for n in nodes.values())
