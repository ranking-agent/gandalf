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


# ---------------------------------------------------------------------------
# JSON mode: lookup(attributes_as_json=True) as the server runs it
# ---------------------------------------------------------------------------

from gandalf.search import lookup  # noqa: E402
from gandalf.trapi import (  # noqa: E402
    AttributesJSON,
    attributes_to_fragments,
    edge_attributes,
)

METFORMIN = "CHEBI:6801"
T2D = "MONDO:0005148"
DIABETES = "MONDO:0005015"


def _query(nodes: dict, edges: dict) -> dict:
    return {"message": {"query_graph": {"nodes": nodes, "edges": edges}}}


def _edge(subject: str, obj: str, predicate: str) -> dict:
    return {"subject": subject, "object": obj, "predicates": [predicate]}


QUERIES = {
    # single-path and multi-path results over two edge columns
    "two_hop": _query(
        {
            "n0": {"ids": [METFORMIN]},
            "n1": {"categories": ["biolink:Gene"]},
            "n2": {"ids": [T2D]},
        },
        {
            "e0": _edge("n0", "n1", "biolink:affects"),
            "e1": _edge("n1", "n2", "biolink:gene_associated_with_condition"),
        },
    ),
    # direct and inferred (subclass) edges
    "subclass": _query(
        {"n0": {"ids": [METFORMIN]}, "n1": {"ids": [DIABETES]}},
        {"e0": _edge("n0", "n1", "biolink:treats")},
    ),
    # several predicates between the same pair; found through inverses
    "related_to": _query(
        {"n0": {"ids": [T2D]}, "n1": {"ids": [METFORMIN]}},
        {"e0": _edge("n0", "n1", "biolink:related_to")},
    ),
}


def _message_bytes(response: dict) -> bytes:
    attributes_to_fragments(response)
    return orjson.dumps(response["message"])


@pytest.mark.parametrize("dehydrated", [False, True])
@pytest.mark.parametrize("name", list(QUERIES))
def test_json_mode_serializes_to_the_same_bytes(
    graph, bmt, name, dehydrated  # noqa: F811
):
    query = QUERIES[name]
    parsed = lookup(graph, query, bmt=bmt, dehydrated=dehydrated)
    as_json = lookup(
        graph, query, bmt=bmt, dehydrated=dehydrated, attributes_as_json=True
    )
    encoded = [
        edge
        for edge in as_json["message"]["knowledge_graph"]["edges"].values()
        if type(edge.get("attributes")) is AttributesJSON
    ]
    if dehydrated:
        assert not encoded, "a dehydrated response carries no attributes"
    else:
        assert encoded, "a full response should carry stored JSON"
    assert _message_bytes(as_json) == orjson.dumps(parsed["message"])


def test_edge_attributes_reads_and_edits_in_place(graph, bmt):  # noqa: F811
    query = QUERIES["two_hop"]
    parsed = lookup(graph, query, bmt=bmt)
    as_json = lookup(graph, query, bmt=bmt, attributes_as_json=True)
    added = {"attribute_type_id": "biolink:has_count", "value": 7}

    for response in (parsed, as_json):
        edges = response["message"]["knowledge_graph"]["edges"]
        first = next(iter(edges.values()))
        edge_attributes(first).append(added)

    parsed_edges = parsed["message"]["knowledge_graph"]["edges"]
    json_edges = as_json["message"]["knowledge_graph"]["edges"]
    for edge_id, edge in json_edges.items():
        assert edge_attributes(edge) == parsed_edges[edge_id].get("attributes", [])
    assert _message_bytes(as_json) == orjson.dumps(parsed["message"])


@pytest.fixture
def server(graph, bmt, monkeypatch):  # noqa: F811
    monkeypatch.setenv("GANDALF_SKIP_PRELOAD", "true")
    monkeypatch.setenv("GANDALF_OTEL_ENABLED", "false")
    from gandalf import server as gandalf_server

    monkeypatch.setattr(gandalf_server, "GRAPH", graph)
    monkeypatch.setattr(gandalf_server, "BMT", bmt)
    return gandalf_server


def test_server_serves_the_same_message(server, graph, bmt):  # noqa: F811
    query = QUERIES["two_hop"]
    rendered = server.sync_lookup(request=dict(query), profile=None)
    served = orjson.loads(rendered.body)["message"]
    assert served == lookup(graph, query, bmt=bmt)["message"]


def test_server_annotators_read_edge_attributes(server, monkeypatch):
    seen = []
    added = {"attribute_type_id": "biolink:has_count", "value": 7}

    def annotate(response, graph, config):
        for edge in response["message"]["knowledge_graph"]["edges"].values():
            attributes = edge_attributes(edge)
            seen.append(type(attributes))
            attributes.append(added)

    monkeypatch.setattr(server, "annotate_response", annotate)
    query = dict(QUERIES["two_hop"], parameters={"annotator_config": {"any": {}}})
    rendered = server.sync_lookup(request=query, profile=None)
    assert seen and set(seen) == {list}
    edges = orjson.loads(rendered.body)["message"]["knowledge_graph"]["edges"]
    assert all(edge["attributes"][-1] == added for edge in edges.values())


def test_validating_server_gets_attributes_as_lists(server, monkeypatch):
    monkeypatch.setattr(server, "_validate", True)
    response = server.sync_lookup(request=dict(QUERIES["two_hop"]), profile=None)
    edges = response["message"]["knowledge_graph"]["edges"]
    assert all(isinstance(e.get("attributes", []), list) for e in edges.values())


def test_server_default_serializes_a_stray_wrapper(server):
    data = orjson.dumps(
        {"attributes": AttributesJSON(b'[{"attribute_type_id":"biolink:x"}]')},
        default=server._orjson_default,
    )
    assert data == b'{"attributes":[{"attribute_type_id":"biolink:x"}]}'
