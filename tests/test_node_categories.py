"""The per-node category index a saved graph carries (``gandalf.node_categories``)."""

import pickle

import numpy as np
import pytest

from gandalf.graph import CSRGraph
from gandalf.node_categories import (
    METADATA_KEY,
    NODE_SETS_FILE,
    NodeCategoryIndex,
    add_to_graph,
    main,
)
from gandalf.search import lookup
from tests.search_fixtures import graph  # noqa: F401


@pytest.fixture
def graph_dir(graph, tmp_path):
    graph.save_mmap(tmp_path / "graph")
    return tmp_path / "graph"


def _genes_treated_query():
    return {
        "message": {
            "query_graph": {
                "nodes": {
                    "drug": {"ids": ["CHEBI:6801"]},
                    "other": {"categories": ["biolink:Gene"]},
                },
                "edges": {
                    "e": {
                        "subject": "drug",
                        "object": "other",
                        "predicates": ["biolink:affects"],
                    }
                },
            }
        }
    }


def test_index_holds_every_nodes_categories(graph_dir):
    mapped = CSRGraph.load_mmap(graph_dir)
    index = mapped.node_categories
    assert len(index.set_of_node) == mapped.num_nodes
    for node_idx in range(mapped.num_nodes):
        stored = mapped.get_all_node_properties(node_idx)["categories"]
        assert list(index.category_sets[index.set_of_node[node_idx]]) == stored


@pytest.mark.parametrize(
    "categories",
    [
        ["biolink:Gene"],
        ["biolink:Disease", "biolink:Drug"],
        ["biolink:NoSuchCategory"],
    ],
)
def test_any_of_agrees_with_the_node_store(graph_dir, categories):
    mapped = CSRGraph.load_mmap(graph_dir)
    nodes = np.arange(mapped.num_nodes)
    expected = [
        any(
            category in mapped.get_all_node_properties(node_idx)["categories"]
            for category in categories
        )
        for node_idx in nodes.tolist()
    ]
    assert mapped.node_categories.any_of(nodes, categories).tolist() == expected


def test_a_node_without_properties_has_no_categories():
    index = NodeCategoryIndex.build(3, {1: {"categories": ["biolink:Gene"]}})
    assert index.any_of(np.arange(3), ["biolink:Gene"]).tolist() == [False, True, False]


def _remove_index(graph_dir):
    """Leave *graph_dir* as a graph saved before the index existed."""
    (graph_dir / NODE_SETS_FILE).unlink()
    with open(graph_dir / "metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    del metadata[METADATA_KEY]
    with open(graph_dir / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)


def test_a_graph_saved_without_the_index_still_answers_the_same(graph_dir, bmt):
    """A graph from before the index existed loads, and filters by category
    from the node store instead."""
    with_index = lookup(CSRGraph.load_mmap(graph_dir), _genes_treated_query(), bmt=bmt)

    _remove_index(graph_dir)
    legacy = CSRGraph.load_mmap(graph_dir)
    assert legacy.node_categories is None
    without_index = lookup(legacy, _genes_treated_query(), bmt=bmt)

    assert with_index["message"]["results"]
    assert without_index["message"]["results"] == with_index["message"]["results"]
    assert (
        without_index["message"]["knowledge_graph"]
        == with_index["message"]["knowledge_graph"]
    )


@pytest.mark.parametrize(
    "add", [add_to_graph, lambda directory: main([str(directory)])]
)
def test_an_older_graph_can_be_given_the_index(graph_dir, add):
    """Without a rebuild, and identical to the one a build writes."""
    built = CSRGraph.load_mmap(graph_dir).node_categories
    _remove_index(graph_dir)

    add(graph_dir)

    added = CSRGraph.load_mmap(graph_dir).node_categories
    assert added.category_sets == built.category_sets
    assert added.set_of_node.tolist() == built.set_of_node.tolist()
    assert added.set_of_node.dtype == built.set_of_node.dtype


def test_resaving_a_mapped_graph_keeps_the_index(graph_dir, tmp_path):
    again = tmp_path / "again"
    CSRGraph.load_mmap(graph_dir).save_mmap(again)
    first = CSRGraph.load_mmap(graph_dir).node_categories
    second = CSRGraph.load_mmap(again).node_categories
    assert second.category_sets == first.category_sets
    assert second.set_of_node.tolist() == first.set_of_node.tolist()
