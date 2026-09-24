"""lookup(serialize_results=True) must serialize exactly like the dict path.

The server asks ``lookup`` for its results pre-serialized to JSON, skipping a
dict per result.  These tests hold that path to byte-for-byte equality with
serializing the dict results, across the query shapes that exercise every
branch of result assembly.
"""

import os

import orjson
import pytest

from gandalf import build_graph_from_jsonl
from gandalf.search.lookup import lookup
from gandalf.trapi import SerializedResults, orjson_default

_FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


@pytest.fixture(scope="module")
def graph():
    return build_graph_from_jsonl(
        os.path.join(_FIXTURES_DIR, "edges.jsonl"),
        os.path.join(_FIXTURES_DIR, "nodes.jsonl"),
    )


@pytest.fixture(scope="module")
def subclass_graph():
    """A disease hierarchy whose children, not the parent, carry the edges."""
    return build_graph_from_jsonl(
        os.path.join(_FIXTURES_DIR, "subclass_multi_edges.jsonl"),
        os.path.join(_FIXTURES_DIR, "subclass_multi_nodes.jsonl"),
    )


def _query(nodes: dict, edges: dict) -> dict:
    return {"message": {"query_graph": {"nodes": nodes, "edges": edges}}}


def _edge(subject: str, obj: str, predicate: str) -> dict:
    return {"subject": subject, "object": obj, "predicates": [predicate]}


# (id, graph fixture name, query, lookup kwargs)
CASES = [
    (
        "batch_one_hop",
        "graph",
        _query(
            {"n0": {"ids": ["CHEBI:6801"]}, "n1": {"categories": ["biolink:Gene"]}},
            {"e0": _edge("n0", "n1", "biolink:affects")},
        ),
        {"subclass": False},
    ),
    (
        "subclass_expansion",
        "graph",
        _query(
            {"n0": {"ids": ["CHEBI:6801"]}, "n1": {"ids": ["MONDO:0005015"]}},
            {"e0": _edge("n0", "n1", "biolink:treats")},
        ),
        {"subclass": True, "subclass_depth": 1},
    ),
    (
        "subclass_inferred_edges",
        "subclass_graph",
        _query(
            {"n0": {"ids": ["MONDO:0011122"]}, "n1": {}},
            {"e0": _edge("n0", "n1", "biolink:has_phenotype")},
        ),
        {"subclass": True, "subclass_depth": 1},
    ),
    (
        "set_interpretation_all",
        "graph",
        _query(
            {
                "n0": {"ids": ["CHEBI:6801"]},
                "n1": {
                    "ids": ["NCBIGene:5468", "NCBIGene:3643"],
                    "set_interpretation": "ALL",
                },
            },
            {"e0": _edge("n0", "n1", "biolink:affects")},
        ),
        {"subclass": False},
    ),
    (
        "set_interpretation_collate",
        "graph",
        _query(
            {
                "n0": {"ids": ["CHEBI:6801"]},
                "n1": {
                    "categories": ["biolink:Gene"],
                    "set_interpretation": "COLLATE",
                },
            },
            {"e0": _edge("n0", "n1", "biolink:affects")},
        ),
        {"subclass": False},
    ),
    (
        "two_hop_related_to",
        "graph",
        _query(
            {"n0": {"ids": ["CHEBI:6801"]}, "n1": {}, "n2": {}},
            {
                "e0": _edge("n0", "n1", "biolink:related_to"),
                "e1": _edge("n1", "n2", "biolink:related_to"),
            },
        ),
        {},
    ),
    (
        "dehydrated",
        "graph",
        _query(
            {"n0": {"ids": ["CHEBI:6801"]}, "n1": {}},
            {"e0": _edge("n0", "n1", "biolink:related_to")},
        ),
        {"dehydrated": True},
    ),
]


@pytest.mark.parametrize(
    "fixture_name, query, kwargs",
    [case[1:] for case in CASES],
    ids=[case[0] for case in CASES],
)
def test_serialized_results_match_dict_results(
    request, bmt, fixture_name, query, kwargs
):
    graph = request.getfixturevalue(fixture_name)
    as_dicts = lookup(graph, query, bmt=bmt, **kwargs)
    serialized = lookup(graph, query, bmt=bmt, serialize_results=True, **kwargs)

    results = serialized["message"]["results"]
    assert isinstance(results, SerializedResults)
    assert len(results) == len(as_dicts["message"]["results"]) > 0
    # The whole message, not just the results: the rest of the response is
    # built alongside them and must not change either.
    assert orjson.dumps(serialized["message"], default=orjson_default) == (
        orjson.dumps(as_dicts["message"], default=orjson_default)
    )
    assert results.to_list() == as_dicts["message"]["results"]


def test_empty_result_set_stays_a_list(graph, bmt):
    """No paths means no results to write; ``results`` is the usual ``[]``."""
    query = _query(
        {"n0": {"ids": ["CHEBI:6801"]}, "n1": {"ids": ["NCBIGene:7124"]}},
        {"e0": _edge("n0", "n1", "biolink:treats")},
    )
    response = lookup(graph, query, bmt=bmt, subclass=False, serialize_results=True)
    assert response["message"]["results"] == []
