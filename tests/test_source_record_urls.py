"""``source_record_urls`` live in the cold store, not the interned source lists.

A record URL names one edge's row at its source, so it is unique per edge:
kept in the interned source lists it made every list unique and the pools
cost 2.8 GB of private memory per process on the Translator graph.  The
loader now stores the URLs per edge in ``edge_source_urls.lmdb`` and every
full response puts them back exactly where they were.
"""

import json

import orjson
import pytest

from gandalf import enrich_knowledge_graph, lookup
from gandalf.graph import CSRGraph
from gandalf.loader import build_graph_from_jsonl
from gandalf.trapi import to_fragments

NODES = [
    {"id": "CHEBI:1", "name": "drug", "category": ["biolink:SmallMolecule"]},
    {"id": "NCBIGene:1", "name": "gene", "category": ["biolink:Gene"]},
    {"id": "NCBIGene:2", "name": "gene2", "category": ["biolink:Gene"]},
    {"id": "MONDO:1", "name": "disease", "category": ["biolink:Disease"]},
]

# Three edges from one source, each with its own record URL (plus one
# without): all four must share ONE interned source list.
EDGES = [
    {
        "id": f"e{i}",
        "subject": "CHEBI:1",
        "predicate": "biolink:affects",
        "object": obj,
        "sources": [
            {
                "resource_id": "infores:ctd",
                "resource_role": "primary_knowledge_source",
                **(
                    {"source_record_urls": [f"https://ctd.example/{i}"]}
                    if i < 3
                    else {}
                ),
            },
            {
                "resource_id": "infores:agg",
                "resource_role": "aggregator_knowledge_source",
                "upstream_resource_ids": ["infores:ctd"],
            },
        ],
        "knowledge_level": "knowledge_assertion",
        "agent_type": "manual_agent",
        "publications": [f"PMID:{i}"],
    }
    for i, obj in enumerate(["NCBIGene:1", "NCBIGene:2", "MONDO:1", "MONDO:1"])
]
# make the fourth edge a different predicate so it is a distinct edge
EDGES[3]["predicate"] = "biolink:treats"

QUERY = {
    "message": {
        "query_graph": {
            "nodes": {
                "n0": {"ids": ["CHEBI:1"]},
                "n1": {"categories": ["biolink:Gene"]},
            },
            "edges": {
                "e0": {
                    "subject": "n0",
                    "object": "n1",
                    "predicates": ["biolink:affects"],
                }
            },
        }
    }
}


@pytest.fixture
def kgx(tmp_path):
    nodes = tmp_path / "nodes.jsonl"
    edges = tmp_path / "edges.jsonl"
    nodes.write_text("\n".join(json.dumps(n) for n in NODES) + "\n")
    edges.write_text("\n".join(json.dumps(e) for e in EDGES) + "\n")
    return edges, nodes


@pytest.fixture
def graph(kgx):
    return build_graph_from_jsonl(*kgx)


def _affects_edges(response: dict) -> dict:
    return {
        e["object"]: e
        for e in response["message"]["knowledge_graph"]["edges"].values()
        if e["predicate"] == "biolink:affects"
    }


def test_urls_leave_the_pool_and_dedup_holds(graph):
    stats = graph.edge_properties.dedup_stats()
    assert stats["total_edges"] == 4
    assert stats["unique_sources"] == 1  # not one per edge
    for i in range(4):
        for source in graph.edge_properties.get_sources(i):
            assert "source_record_urls" not in source
    assert graph.source_urls_store is not None
    stored = graph.source_urls_store.get_json_batch(range(4))
    assert len(stored) == 3
    assert all(
        orjson.loads(v)[0][0] == 1 for v in stored.values()
    )  # the ctd source, after the gandalf aggregator


def test_full_response_carries_each_edges_urls(graph, bmt):
    edges = _affects_edges(lookup(graph, QUERY, bmt=bmt))
    assert set(edges) == {"NCBIGene:1", "NCBIGene:2"}
    for obj, edge in edges.items():
        ctd = [s for s in edge["sources"] if s["resource_id"] == "infores:ctd"]
        assert len(ctd) == 1
        n = {"NCBIGene:1": 0, "NCBIGene:2": 1}[obj]
        assert ctd[0]["source_record_urls"] == [f"https://ctd.example/{n}"]
        # the key sits last, where the loader had it before the split
        assert list(ctd[0]) == ["resource_id", "resource_role", "source_record_urls"]
        agg = [s for s in edge["sources"] if s["resource_id"] == "infores:agg"][0]
        assert "source_record_urls" not in agg
    # the interned list was not changed in place
    assert all(
        "source_record_urls" not in s for s in graph.edge_properties.get_sources(0)
    )


def test_dehydrated_response_omits_sources_still(graph, bmt):
    edges = _affects_edges(lookup(graph, QUERY, bmt=bmt, dehydrated=True))
    assert edges and all("sources" not in e for e in edges.values())


def test_json_fast_paths_serve_the_same_bytes(graph, bmt):
    plain = lookup(graph, QUERY, bmt=bmt)
    fast = lookup(graph, QUERY, bmt=bmt, attributes_as_json=True, results_as_json=True)
    to_fragments(fast)
    a = orjson.dumps(plain)
    b = orjson.dumps(fast)
    assert (
        orjson.loads(a)["message"]["knowledge_graph"]
        == orjson.loads(b)["message"]["knowledge_graph"]
    )
    assert b"https://ctd.example/0" in b


def test_survives_a_save_load_round_trip(graph, bmt, tmp_path):
    graph.save_mmap(tmp_path / "g")
    loaded = CSRGraph.load_mmap(tmp_path / "g")
    assert loaded.source_urls_store is not None
    assert loaded.edge_properties.dedup_stats()["unique_sources"] == 1
    edges = _affects_edges(lookup(loaded, QUERY, bmt=bmt))
    assert edges["NCBIGene:2"]["sources"][1]["source_record_urls"] == [
        "https://ctd.example/1"
    ]
    assert loaded.load_memory.stages["edge property pools"] < 16 * 1024


def test_rehydration_puts_the_urls_back(graph, bmt):
    dehydrated = lookup(graph, QUERY, bmt=bmt, dehydrated=True)
    request = {"message": dehydrated["message"], "parameters": {"rehydrate": True}}
    enrich_knowledge_graph(request, graph)
    edges = _affects_edges(request)
    assert edges["NCBIGene:1"]["sources"][1]["source_record_urls"] == [
        "https://ctd.example/0"
    ]
    assert "source_record_urls" not in edges["NCBIGene:1"]["sources"][0]


def test_graph_without_urls_has_no_store(tmp_path):
    from tests.search_fixtures import EDGES_FILE, NODES_FILE

    plain = build_graph_from_jsonl(EDGES_FILE, NODES_FILE)
    assert plain.source_urls_store is None
    plain.save_mmap(tmp_path / "g")
    assert not (tmp_path / "g" / "edge_source_urls.lmdb").exists()
    assert CSRGraph.load_mmap(tmp_path / "g").source_urls_store is None
