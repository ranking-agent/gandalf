"""A dehydrated response's nodes come from the memory-mapped names and
category index, and are exactly what reading each node's record gives.
"""

import json

import pytest

from gandalf.execute import serialize_response
from gandalf.graph import CSRGraph
from gandalf.loader import build_graph_from_jsonl
from gandalf.search.lookup import lookup
from tests.search_fixtures import EDGES_FILE, NODES_FILE

# Names a store must reproduce exactly: empty, absent, escaped, non-ASCII.
TRICKY_NODES = [
    {"id": "HUB:1", "name": "hub", "category": ["biolink:Gene"]},
    {"id": "X:empty", "name": "", "category": ["biolink:Gene"]},
    {"id": "X:noname", "category": ["biolink:Disease"]},
    {"id": "X:nocat", "name": "no category"},
    {"id": "X:quote", "name": 'say "hi" \\ there', "category": ["biolink:Gene"]},
    {"id": "X:unicode", "name": "café – β", "category": ["biolink:Protein"]},
    {
        "id": "X:many",
        "name": "many categories",
        "category": ["biolink:Gene", "biolink:GeneOrGeneProduct", "biolink:Protein"],
    },
]


def _write_graph(tmp_path, nodes, out):
    nodes_file = tmp_path / "nodes.jsonl"
    edges_file = tmp_path / "edges.jsonl"
    nodes_file.write_text("\n".join(json.dumps(n) for n in nodes) + "\n")
    edges = [
        {
            "subject": "HUB:1",
            "predicate": "biolink:related_to",
            "object": n["id"],
            "primary_knowledge_source": "infores:test",
        }
        for n in nodes[1:]
    ]
    edges_file.write_text("\n".join(json.dumps(e) for e in edges) + "\n")
    graph = build_graph_from_jsonl(str(edges_file), str(nodes_file))
    graph.save_mmap(out)
    return out


def _message(graph, bmt, query, **kwargs):
    response = lookup(graph, query, bmt=bmt, **kwargs)
    return serialize_response({"message": response["message"]})


def _both_ways(saved, bmt, query, **kwargs):
    """The response with the names store, and with it switched off."""
    graph = CSRGraph.load_mmap(saved, serve_metadata=False)
    assert graph.node_name_store is not None
    fast = _message(graph, bmt, query, **kwargs)
    graph.node_name_store = None
    slow = _message(graph, bmt, query, **kwargs)
    for store in (graph.node_store, graph.lmdb_store, graph.source_urls_store):
        if store is not None:
            store.close()
    return fast, slow


def _one_hop(pinned, categories=None):
    other = {"categories": categories} if categories else {}
    return {
        "message": {
            "query_graph": {
                "nodes": {"a": {"ids": [pinned]}, "b": other},
                "edges": {"e": {"subject": "a", "object": "b"}},
            }
        }
    }


def test_tricky_names_and_categories_match_the_records(tmp_path, bmt):
    saved = _write_graph(tmp_path, TRICKY_NODES, tmp_path / "graph")
    fast, slow = _both_ways(saved, bmt, _one_hop("HUB:1"), dehydrated=True)
    assert fast == slow
    nodes = json.loads(fast)["message"]["knowledge_graph"]["nodes"]
    assert len(nodes) == len(TRICKY_NODES)
    assert nodes["X:empty"]["name"] == ""
    assert "name" not in nodes["X:noname"]
    assert nodes["X:quote"]["name"] == 'say "hi" \\ there'


@pytest.fixture(scope="module")
def fixture_graph_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("graph") / "graph"
    build_graph_from_jsonl(EDGES_FILE, NODES_FILE).save_mmap(out)
    return out


def _fixture_node_ids():
    with open(NODES_FILE) as f:
        return [json.loads(line)["id"] for line in f if line.strip()]


@pytest.mark.parametrize("pinned", _fixture_node_ids())
@pytest.mark.parametrize("subclass", [True, False])
def test_dehydrated_responses_match_the_records(
    fixture_graph_dir, bmt, pinned, subclass
):
    fast, slow = _both_ways(
        fixture_graph_dir,
        bmt,
        _one_hop(pinned),
        dehydrated=True,
        subclass=subclass,
    )
    assert fast == slow


def test_full_responses_do_not_use_the_names(fixture_graph_dir, bmt):
    fast, slow = _both_ways(
        fixture_graph_dir, bmt, _one_hop("CHEBI:6801"), dehydrated=False
    )
    assert fast == slow
    nodes = json.loads(fast)["message"]["knowledge_graph"]["nodes"]
    assert all("attributes" in node for node in nodes.values())


def test_specific_categories_are_cached_per_bmt(fixture_graph_dir, bmt):
    graph = CSRGraph.load_mmap(fixture_graph_dir, serve_metadata=False)
    table = graph.most_specific_categories(bmt)
    table[0] = ["biolink:Gene"]
    assert graph.most_specific_categories(bmt) is table
    assert graph.most_specific_categories(object()) == {}
