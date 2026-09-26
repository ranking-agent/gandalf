"""Single-path results written straight to JSON serialize to the same bytes.

``lookup(results_as_json=True)`` writes each single-path result from a
template instead of building it as dicts.  Every query below must serialize
to exactly the bytes the dict path gives -- results, their order, and the
knowledge graph's key order -- whatever the block size.
"""

import importlib

import orjson
import pytest

from tests.search_fixtures import graph  # noqa: F401
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
from gandalf.trapi import attributes_to_fragments

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
    attributes_to_fragments(response)
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
