"""The single-path fast path in _build_response changes nothing in a response.

Each query runs with the fast path off (every result built by the general
per-group loop) and on, and the two messages must serialize to the same
bytes, with knowledge-graph nodes and edges and auxiliary graphs in the same
key order.  The benchmark fingerprint ignores order, so this is the check
that holds result, binding and knowledge-graph order fixed.
"""

import importlib
import os

import orjson
import pytest

from tests.search_fixtures import graph  # noqa: F401

from gandalf.loader import build_graph_from_jsonl
from gandalf.search import lookup

# ``gandalf.search`` re-exports a ``lookup`` function under the module's name.
lookup_module = importlib.import_module("gandalf.search.lookup")

_FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")

METFORMIN = "CHEBI:6801"
T2D = "MONDO:0005148"
DIABETES = "MONDO:0005015"
PPARG = "NCBIGene:5468"
INSR = "NCBIGene:3643"
HYPOGLYCEMIA = "HP:0001943"
UMBRELLA = "MONDO:0011122"
TARGET = "MONDO:0044444"


@pytest.fixture
def multi_child_graph():
    """One superclass with two children, each with the same phenotype."""
    return build_graph_from_jsonl(
        os.path.join(_FIXTURES_DIR, "subclass_multi_edges.jsonl"),
        os.path.join(_FIXTURES_DIR, "subclass_multi_nodes.jsonl"),
    )


def _query(nodes: dict, edges: dict) -> dict:
    return {"message": {"query_graph": {"nodes": nodes, "edges": edges}}}


def _edge(subject: str, obj: str, predicate: str) -> dict:
    return {"subject": subject, "object": obj, "predicates": [predicate]}


def _serialized(response: dict) -> tuple:
    message = response["message"]
    return (
        orjson.dumps(message),
        list(message["knowledge_graph"]["nodes"]),
        list(message["knowledge_graph"]["edges"]),
        list(message.get("auxiliary_graphs", {})),
    )


def _both_ways(monkeypatch, graph, query, bmt, **kwargs) -> dict:
    """Assert the response is identical with the fast path off and on."""
    monkeypatch.setattr(lookup_module, "_SINGLE_PATH_FAST", False)
    reference = lookup(graph, query, bmt=bmt, **kwargs)
    monkeypatch.setattr(lookup_module, "_SINGLE_PATH_FAST", True)
    response = lookup(graph, query, bmt=bmt, **kwargs)
    assert response["message"]["results"], "the case should have results"
    assert _serialized(response) == _serialized(reference)
    for edge_id, edge in response["message"]["knowledge_graph"]["edges"].items():
        leaked = [key for key in edge if key.startswith("_")]
        assert not leaked, f"KG edge {edge_id} leaks internal markers {leaked}"
    return response


def _binding_sizes(response: dict) -> set[int]:
    """How many edge IDs each result's edge bindings hold."""
    return {
        len(binding["ids"])
        for result in response["message"]["results"]
        for binding in result["analyses"][0]["edge_bindings"].values()
    }


# Metformin -affects-> gene -associated-> T2D.  INSR is reached by two
# `affects` edges, so its result is built from two paths over two edge
# columns, while PPARG's and GCK's come from one path each: fast-path and
# general-loop results interleave in one response.
TWO_HOP_MIXED = _query(
    {
        "n0": {"ids": [METFORMIN]},
        "n1": {"categories": ["biolink:Gene"]},
        "n2": {"ids": [T2D]},
    },
    {
        "e0": _edge("n0", "n1", "biolink:affects"),
        "e1": _edge("n1", "n2", "biolink:gene_associated_with_condition"),
    },
)


@pytest.mark.parametrize("dehydrated", [False, True])
@pytest.mark.parametrize("subclass", [False, True])
def test_multi_path_and_single_path_results_mixed(
    monkeypatch, graph, bmt, dehydrated, subclass
):
    response = _both_ways(
        monkeypatch,
        graph,
        TWO_HOP_MIXED,
        bmt,
        subclass=subclass,
        dehydrated=dehydrated,
    )
    # INSR's two `affects` edges differ only in qualifiers and sources, so a
    # dehydrated response, which dedups on (subject, predicate, object)
    # alone, binds one of them.
    assert _binding_sizes(response) == ({1} if dehydrated else {1, 2})


@pytest.mark.parametrize("dehydrated", [False, True])
@pytest.mark.parametrize(
    "query",
    [
        # Metformin treats Diabetes Mellitus directly and via Type 2
        # Diabetes: one result, built from two paths.
        _query(
            {"n0": {"ids": [METFORMIN]}, "n1": {"ids": [DIABETES]}},
            {"e0": _edge("n0", "n1", "biolink:treats")},
        ),
        # Diabetes Mellitus has Hypoglycemia only via Type 2 Diabetes: one
        # result, built from a single path through a subclass edge, which
        # must still get its inferred edge.
        _query(
            {"n0": {"ids": [DIABETES]}, "n1": {"ids": [HYPOGLYCEMIA]}},
            {"e0": _edge("n0", "n1", "biolink:has_phenotype")},
        ),
    ],
    ids=["direct_and_inferred", "inferred_only"],
)
def test_subclass_inferred_edges(monkeypatch, graph, bmt, dehydrated, query):
    response = _both_ways(
        monkeypatch, graph, query, bmt, subclass=True, dehydrated=dehydrated
    )
    assert response["message"].get("auxiliary_graphs")


@pytest.mark.parametrize("dehydrated", [False, True])
@pytest.mark.parametrize(
    "query",
    [
        _query(
            {"n0": {"ids": [UMBRELLA]}, "n1": {"ids": [TARGET]}},
            {"e0": _edge("n0", "n1", "biolink:has_phenotype")},
        ),
        _query(
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
    ],
    ids=["one_hop", "two_hop"],
)
def test_sibling_subclass_derivations(
    monkeypatch, multi_child_graph, bmt, dehydrated, query
):
    response = _both_ways(
        monkeypatch,
        multi_child_graph,
        query,
        bmt,
        subclass=True,
        subclass_depth=1,
        dehydrated=dehydrated,
    )
    assert response["message"].get("auxiliary_graphs")


@pytest.mark.parametrize("dehydrated", [False, True])
@pytest.mark.parametrize(
    "query",
    [
        # BATCH: one result per gene
        _query(
            {"n0": {"ids": [METFORMIN]}, "n1": {"categories": ["biolink:Gene"]}},
            {"e0": _edge("n0", "n1", "biolink:affects")},
        ),
        # ALL: one result binding both required genes
        _query(
            {
                "n0": {"ids": [METFORMIN]},
                "n1": {"ids": [PPARG, INSR], "set_interpretation": "ALL"},
            },
            {"e0": _edge("n0", "n1", "biolink:affects")},
        ),
        # COLLATE: one result binding every gene found
        _query(
            {
                "n0": {"ids": [METFORMIN]},
                "n1": {
                    "categories": ["biolink:Gene"],
                    "set_interpretation": "COLLATE",
                },
            },
            {"e0": _edge("n0", "n1", "biolink:affects")},
        ),
        # Symmetric predicate, queried against the stored direction
        _query(
            {"n0": {"ids": [INSR]}, "n1": {"categories": ["biolink:Gene"]}},
            {"e0": _edge("n0", "n1", "biolink:interacts_with")},
        ),
        # Inverse predicate: found through the stored `treats` edges
        _query(
            {"n0": {"ids": [T2D]}, "n1": {"ids": [METFORMIN]}},
            {"e0": _edge("n0", "n1", "biolink:treated_by")},
        ),
        # related_to: several predicates between the same pair, one result
        _query(
            {"n0": {"ids": [METFORMIN]}, "n1": {"ids": [T2D]}},
            {"e0": _edge("n0", "n1", "biolink:related_to")},
        ),
        # Multi-hop related_to
        _query(
            {
                "n0": {"ids": [METFORMIN]},
                "n1": {},
                "n2": {"ids": [T2D]},
            },
            {
                "e0": _edge("n0", "n1", "biolink:related_to"),
                "e1": _edge("n1", "n2", "biolink:related_to"),
            },
        ),
    ],
    ids=[
        "batch",
        "all",
        "collate",
        "symmetric",
        "inverse",
        "related_to",
        "two_hop_related_to",
    ],
)
@pytest.mark.parametrize("subclass", [False, True])
def test_query_shapes(monkeypatch, graph, bmt, dehydrated, query, subclass):
    _both_ways(monkeypatch, graph, query, bmt, subclass=subclass, dehydrated=dehydrated)


def test_single_id_bindings_are_shared(graph, bmt):
    """A ``{"ids": [x]}`` binding is one object per ID, reused by every
    result that binds x, so nothing may change a binding in place."""
    query = _query(
        {"n0": {"ids": [METFORMIN]}, "n1": {"categories": ["biolink:Gene"]}},
        {"e0": _edge("n0", "n1", "biolink:affects")},
    )
    results = lookup(graph, query, bmt=bmt, subclass=False)["message"]["results"]
    assert len(results) > 1
    first = results[0]["node_bindings"]["n0"]
    assert first == {"ids": [METFORMIN]}
    assert all(r["node_bindings"]["n0"] is first for r in results)
