"""Single-path results written straight to JSON serialize to the same bytes.

``lookup(results_as_json=True)`` writes each single-path result from a
template instead of building it as dicts.  Every query below must serialize
to exactly the bytes the dict path gives -- results, their order, and the
knowledge graph's key order -- whatever the block size.
"""

import importlib

import orjson
import pytest

import gandalf.execute as gandalf_execute

from gandalf.execute import orjson_default

from tests.search_fixtures import bare_request, graph  # noqa: F401
from tests.test_single_path_fast_path import (  # noqa: F401
    DIABETES,
    HYPOGLYCEMIA,
    INSR,
    METFORMIN,
    PPARG,
    T2D,
    TARGET,
    TWO_HOP_MIXED,
    UMBRELLA,
    _edge,
    _query,
    multi_child_graph,
)

from gandalf.search import lookup
from gandalf.trapi import response_results, to_fragments

lookup_module = importlib.import_module("gandalf.search.lookup")

QUERIES = {
    "two_hop_mixed": TWO_HOP_MIXED,
    "direct_and_inferred": _query(
        {"n0": {"ids": [METFORMIN]}, "n1": {"ids": [DIABETES]}},
        {"e0": _edge("n0", "n1", "biolink:treats")},
    ),
    "inferred_only": _query(
        {"n0": {"ids": [DIABETES]}, "n1": {"ids": [HYPOGLYCEMIA]}},
        {"e0": _edge("n0", "n1", "biolink:has_phenotype")},
    ),
    "batch": _query(
        {"n0": {"ids": [METFORMIN]}, "n1": {"categories": ["biolink:Gene"]}},
        {"e0": _edge("n0", "n1", "biolink:affects")},
    ),
    "all": _query(
        {
            "n0": {"ids": [METFORMIN]},
            "n1": {"ids": [PPARG, INSR], "set_interpretation": "ALL"},
        },
        {"e0": _edge("n0", "n1", "biolink:affects")},
    ),
    "collate": _query(
        {
            "n0": {"ids": [METFORMIN]},
            "n1": {"categories": ["biolink:Gene"], "set_interpretation": "COLLATE"},
        },
        {"e0": _edge("n0", "n1", "biolink:affects")},
    ),
    "symmetric": _query(
        {"n0": {"ids": [INSR]}, "n1": {"categories": ["biolink:Gene"]}},
        {"e0": _edge("n0", "n1", "biolink:interacts_with")},
    ),
    "inverse": _query(
        {"n0": {"ids": [T2D]}, "n1": {"ids": [METFORMIN]}},
        {"e0": _edge("n0", "n1", "biolink:treated_by")},
    ),
    "related_to": _query(
        {"n0": {"ids": [METFORMIN]}, "n1": {"ids": [T2D]}},
        {"e0": _edge("n0", "n1", "biolink:related_to")},
    ),
    "two_hop_related_to": _query(
        {"n0": {"ids": [METFORMIN]}, "n1": {}, "n2": {"ids": [T2D]}},
        {
            "e0": _edge("n0", "n1", "biolink:related_to"),
            "e1": _edge("n1", "n2", "biolink:related_to"),
        },
    ),
}

SIBLING_QUERIES = {
    "sibling_one_hop": _query(
        {"n0": {"ids": [UMBRELLA]}, "n1": {"ids": [TARGET]}},
        {"e0": _edge("n0", "n1", "biolink:has_phenotype")},
    ),
    "sibling_two_hop": _query(
        {
            "n0": {"ids": ["CHEBI:9999"]},
            "n1": {"ids": [TARGET]},
            "n2": {"ids": [UMBRELLA]},
        },
        {
            "e0": _edge("n0", "n1", "biolink:treats"),
            "e1": _edge("n2", "n1", "biolink:has_phenotype"),
        },
    ),
}


def _served(response: dict) -> tuple:
    """The message as the server serializes it, plus the KG key order."""
    to_fragments(response)
    message = response["message"]
    return (
        orjson.dumps(message),
        list(message["knowledge_graph"]["nodes"]),
        list(message["knowledge_graph"]["edges"]),
        list(message.get("auxiliary_graphs", {})),
    )


def _check(monkeypatch, graph, query, bmt, block, **kwargs):  # noqa: F811
    kwargs["attributes_as_json"] = True
    reference = lookup(graph, query, bmt=bmt, **kwargs)
    monkeypatch.setattr(lookup_module, "_RESULTS_BLOCK", block)
    # Small chunks, so results are joined and sliced across chunk boundaries.
    monkeypatch.setattr(lookup_module, "_JSON_CHUNK", 3)
    response = lookup(graph, query, bmt=bmt, results_as_json=True, **kwargs)
    written = [r for r in response["message"]["results"] if type(r) is bytes]
    assert _served(response) == _served(reference)
    return written


@pytest.mark.parametrize("block", [1, 2, 4096])
@pytest.mark.parametrize("dehydrated", [False, True])
@pytest.mark.parametrize("subclass", [False, True])
@pytest.mark.parametrize("name", list(QUERIES))
def test_same_bytes_as_dicts(
    monkeypatch, graph, bmt, name, subclass, dehydrated, block  # noqa: F811
):
    written = _check(
        monkeypatch,
        graph,
        QUERIES[name],
        bmt,
        block,
        subclass=subclass,
        dehydrated=dehydrated,
    )
    if name in ("all", "collate"):
        assert not written, "ALL/COLLATE queries stay on dicts"


@pytest.mark.parametrize("block", [1, 4096])
@pytest.mark.parametrize("dehydrated", [False, True])
@pytest.mark.parametrize("name", list(SIBLING_QUERIES))
def test_same_bytes_with_sibling_subclasses(
    monkeypatch, multi_child_graph, bmt, name, dehydrated, block  # noqa: F811
):
    _check(
        monkeypatch,
        multi_child_graph,
        SIBLING_QUERIES[name],
        bmt,
        block,
        subclass=True,
        subclass_depth=1,
        dehydrated=dehydrated,
    )


def test_mixed_results_keep_their_order(monkeypatch, graph, bmt):  # noqa: F811
    """Written results and dict results interleave in the list, in order."""
    monkeypatch.setattr(lookup_module, "_RESULTS_BLOCK", 4096)
    response = lookup(graph, TWO_HOP_MIXED, bmt=bmt, results_as_json=True)
    kinds = [type(r) for r in response["message"]["results"]]
    assert bytes in kinds and dict in kinds


# ---------------------------------------------------------------------------
# Edges without IDs, and IDs shared between edges: left to the loop
# ---------------------------------------------------------------------------


@pytest.fixture
def counting_uuids(monkeypatch):
    """Deterministic uuid4s, so both modes name ID-less edges alike."""
    import uuid

    def reset():
        counter = iter(range(10**6))
        monkeypatch.setattr(
            lookup_module.uuid, "uuid4", lambda: uuid.UUID(int=next(counter))
        )

    return reset


def test_edges_without_ids(monkeypatch, graph, bmt, counting_uuids):  # noqa: F811
    """An edge without an ID gets a fresh one per result, so its results go
    to the loop; the others are still written as JSON."""
    get_ids = graph.get_edge_ids_batch

    def some_missing(indices):
        ids = get_ids(indices)
        return {idx: eid for idx, eid in ids.items() if idx % 2}

    monkeypatch.setattr(graph, "get_edge_ids_batch", some_missing)
    query = QUERIES["batch"]
    counting_uuids()
    reference = lookup(graph, query, bmt=bmt, attributes_as_json=True)
    counting_uuids()
    response = lookup(
        graph, query, bmt=bmt, attributes_as_json=True, results_as_json=True
    )
    kinds = {type(r) for r in response["message"]["results"]}
    assert kinds == {bytes, dict}
    assert _served(response) == _served(reference)


def test_shared_edge_ids_fall_back_to_dicts(monkeypatch, graph, bmt):  # noqa: F811
    """Two edges under one ID would share a KG entry: every result is then
    built by the loop, as without results_as_json.  The two need not both be
    in results that could be written as JSON."""
    get_ids = graph.get_edge_ids_batch

    def shared(indices):
        return {
            idx: "shared" if i < 2 else eid
            for i, (idx, eid) in enumerate(get_ids(indices).items())
        }

    monkeypatch.setattr(graph, "get_edge_ids_batch", shared)
    query = QUERIES["batch"]
    reference = lookup(graph, query, bmt=bmt, attributes_as_json=True)
    response = lookup(
        graph, query, bmt=bmt, attributes_as_json=True, results_as_json=True
    )
    assert all(type(r) is dict for r in response["message"]["results"])
    assert _served(response) == _served(reference)


# ---------------------------------------------------------------------------
# Reading results: response_results, and the server
# ---------------------------------------------------------------------------


def test_response_results_decodes_in_place(graph, bmt):  # noqa: F811
    query = TWO_HOP_MIXED
    reference = lookup(graph, query, bmt=bmt, attributes_as_json=True)
    response = lookup(
        graph, query, bmt=bmt, attributes_as_json=True, results_as_json=True
    )
    results = response_results(response)
    assert all(type(r) is dict for r in results)
    assert response["message"]["results"] is results
    assert results == reference["message"]["results"]

    # Changes to the decoded results are served.
    results[0]["node_bindings"]["n0"] = {"ids": ["CHEBI:changed"]}
    served = orjson.loads(_served(response)[0])
    assert served["results"][0]["node_bindings"]["n0"] == {"ids": ["CHEBI:changed"]}


@pytest.fixture
def server(graph, bmt, monkeypatch):  # noqa: F811
    monkeypatch.setenv("GANDALF_SKIP_PRELOAD", "true")
    monkeypatch.setenv("GANDALF_OTEL_ENABLED", "false")
    from gandalf import server as gandalf_server

    monkeypatch.setattr(gandalf_server, "GRAPH", graph)
    monkeypatch.setattr(gandalf_server, "BMT", bmt)
    return gandalf_server


def test_server_serves_the_same_bytes(server, graph, bmt):  # noqa: F811
    rendered = server.sync_lookup(
        bare_request(), body=dict(TWO_HOP_MIXED), profile=None
    )
    served = orjson.dumps(orjson.loads(rendered.body)["message"])
    reference = lookup(graph, TWO_HOP_MIXED, bmt=bmt)
    assert served == orjson.dumps(reference["message"])


def test_server_annotators_read_results(server, monkeypatch):
    seen = []

    def annotate(response, graph, config):
        for result in response_results(response):
            seen.append(type(result))
            result["analyses"][0]["score"] = 0.5

    monkeypatch.setattr(gandalf_execute, "annotate_response", annotate)
    query = dict(TWO_HOP_MIXED, parameters={"annotator_config": {"any": {}}})
    rendered = server.sync_lookup(bare_request(), body=query, profile=None)
    assert seen and set(seen) == {dict}
    results = orjson.loads(rendered.body)["message"]["results"]
    assert len(results) == len(seen)
    assert all(r["analyses"][0]["score"] == 0.5 for r in results)


def test_validating_server_gets_result_dicts(server, monkeypatch):
    monkeypatch.setattr(server, "_validate", True)
    response = server.sync_lookup(
        bare_request(), body=dict(TWO_HOP_MIXED), profile=None
    )
    assert all(type(r) is dict for r in response["message"]["results"])


def test_server_default_serializes_a_stray_block(server):
    block = b'{"node_bindings":{}},{"node_bindings":{}}'
    data = orjson.dumps({"results": [block]}, default=orjson_default)
    assert data == b'{"results":[{"node_bindings":{}},{"node_bindings":{}}]}'
