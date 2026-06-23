"""Tests for the pluggable graph sources and the normalized-record contract."""

import os

import pytest

from gandalf.loader import _build_graph_from_source, build_graph_from_jsonl
from gandalf.normalize import normalize_edge, normalize_node
from gandalf.sources import KGXJsonlSource
from gandalf.sources.base import (
    SourceValidationError,
    validate_normalized_edge,
    validate_normalized_node,
)
from gandalf.sources.mongo import MongoSource

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
NODES_FILE = os.path.join(FIXTURES_DIR, "nodes.jsonl")
EDGES_FILE = os.path.join(FIXTURES_DIR, "edges.jsonl")

NUM_NODES = 11
NUM_EDGES = 20


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------


def test_normalize_edge_shape():
    raw = {
        "id": "e1",
        "subject": "CHEBI:1",
        "predicate": "biolink:treats",
        "object": "MONDO:1",
        "sources": [
            {"resource_role": "primary_knowledge_source", "resource_id": "infores:x"}
        ],
        "publications": ["PMID:1"],
    }
    edge = normalize_edge(raw)
    assert edge["subject"] == "CHEBI:1"
    assert edge["object"] == "MONDO:1"
    assert edge["predicate"] == "biolink:treats"
    assert edge["id"] == "e1"
    assert isinstance(edge["sources"], list)
    assert isinstance(edge["qualifiers"], list)
    assert isinstance(edge["attributes"], list)
    # gandalf aggregator is prepended
    assert edge["sources"][0]["resource_id"] == "infores:gandalf"
    # publications became an attribute
    assert any(a["original_attribute_name"] == "publications" for a in edge["attributes"])
    validate_normalized_edge(edge)


def test_normalize_edge_is_pure():
    """normalize_edge must not mutate its input (no in-place reverse)."""
    raw = {
        "subject": "A",
        "predicate": "biolink:related_to",
        "object": "B",
        "primary_knowledge_source": "infores:p",
        "aggregator_knowledge_source": ["infores:a1", "infores:a2"],
    }
    first = normalize_edge(raw)
    assert raw["aggregator_knowledge_source"] == ["infores:a1", "infores:a2"]
    second = normalize_edge(raw)
    assert first == second


def test_normalize_node_renames_category_to_categories():
    raw = {
        "id": "CHEBI:1",
        "name": "Metformin",
        "category": ["biolink:Drug"],
        "information_content": 85.5,
    }
    node = normalize_node(raw)
    assert node["id"] == "CHEBI:1"
    assert node["name"] == "Metformin"
    assert node["categories"] == ["biolink:Drug"]
    assert "category" not in node
    assert any(
        a["original_attribute_name"] == "information_content" for a in node["attributes"]
    )
    validate_normalized_node(node)


# ---------------------------------------------------------------------------
# KGXJsonlSource
# ---------------------------------------------------------------------------


def test_kgx_source_counts():
    source = KGXJsonlSource(EDGES_FILE, NODES_FILE)
    assert sum(1 for _ in source.iter_edge_triples()) == NUM_EDGES
    assert sum(1 for _ in source.iter_edges()) == NUM_EDGES
    assert sum(1 for _ in source.iter_nodes()) == NUM_NODES


def test_kgx_source_is_re_iterable():
    """Each iter_* call returns a fresh generator (loader iterates edges twice)."""
    source = KGXJsonlSource(EDGES_FILE, NODES_FILE)
    triples_a = list(source.iter_edge_triples())
    triples_b = list(source.iter_edge_triples())
    assert triples_a == triples_b
    assert len(triples_a) == NUM_EDGES


def test_kgx_source_triples_match_edges_order():
    source = KGXJsonlSource(EDGES_FILE, NODES_FILE)
    triples = list(source.iter_edge_triples())
    edges = list(source.iter_edges())
    for (s, o, p), edge in zip(triples, edges):
        assert (s, o, p) == (edge["subject"], edge["object"], edge["predicate"])


def test_kgx_source_no_node_file():
    source = KGXJsonlSource(EDGES_FILE, None)
    assert list(source.iter_nodes()) == []
    assert sum(1 for _ in source.iter_edges()) == NUM_EDGES


def test_kgx_source_yields_normalized_nodes():
    source = KGXJsonlSource(EDGES_FILE, NODES_FILE)
    nodes = list(source.iter_nodes())
    assert all("categories" in n for n in nodes)
    assert all("category" not in n for n in nodes)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _valid_edge():
    return {
        "subject": "A",
        "object": "B",
        "predicate": "biolink:treats",
        "id": "e1",
        "sources": [
            {
                "resource_id": "infores:x",
                "resource_role": "primary_knowledge_source",
                "upstream_resource_ids": [],
            }
        ],
        "qualifiers": [],
        "attributes": [],
    }


def _valid_node():
    return {"id": "A", "name": "n", "categories": ["biolink:Drug"], "attributes": []}


def test_validate_edge_happy_path():
    validate_normalized_edge(_valid_edge())


def test_validate_edge_missing_subject():
    edge = _valid_edge()
    del edge["subject"]
    with pytest.raises(SourceValidationError):
        validate_normalized_edge(edge)


def test_validate_edge_sources_not_list():
    edge = _valid_edge()
    edge["sources"] = {"resource_id": "x"}
    with pytest.raises(SourceValidationError):
        validate_normalized_edge(edge)


def test_validate_edge_source_missing_upstream():
    edge = _valid_edge()
    edge["sources"][0].pop("upstream_resource_ids")
    with pytest.raises(SourceValidationError):
        validate_normalized_edge(edge)


def test_validate_edge_qualifier_value_not_str():
    edge = _valid_edge()
    edge["qualifiers"] = [
        {"qualifier_type_id": "biolink:object_aspect_qualifier", "qualifier_value": ["a", "b"]}
    ]
    with pytest.raises(SourceValidationError):
        validate_normalized_edge(edge)


def test_validate_node_happy_path():
    validate_normalized_node(_valid_node())


def test_validate_node_missing_id():
    node = _valid_node()
    del node["id"]
    with pytest.raises(SourceValidationError):
        validate_normalized_node(node)


def test_validate_node_categories_not_list():
    node = _valid_node()
    node["categories"] = "biolink:Drug"
    with pytest.raises(SourceValidationError):
        validate_normalized_node(node)


# ---------------------------------------------------------------------------
# MongoSource (via fakes — pymongo not required)
# ---------------------------------------------------------------------------


class _FakeCursor:
    """Minimal cursor: supports projection, .sort(), and iteration."""

    def __init__(self, docs, projection=None):
        self._docs = [dict(d) for d in docs]  # copy so callers can pop _id
        self._projection = projection

    def sort(self, key, direction=1):
        self._docs.sort(key=lambda d: d[key], reverse=(direction < 0))
        return self

    def __iter__(self):
        for d in self._docs:
            if self._projection:
                fields = {k for k, v in self._projection.items() if v}
                yield {k: v for k, v in d.items() if k in fields}
            else:
                yield d


class _FakeCollection:
    def __init__(self, docs):
        self._docs = docs

    def find(self, _filter=None, projection=None):
        return _FakeCursor(self._docs, projection)


def _normalized_fixture_docs():
    """Build already-normalized Mongo-style docs from the jsonl fixtures.

    Mimics the upstream pipeline: normalize, then store with an _id.
    """
    import orjson

    edges = []
    with open(EDGES_FILE, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            doc = normalize_edge(orjson.loads(line))
            doc["_id"] = i
            edges.append(doc)
    nodes = []
    with open(NODES_FILE, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            doc = normalize_node(orjson.loads(line))
            doc["_id"] = i
            nodes.append(doc)
    return nodes, edges


def _fake_mongo_source():
    nodes, edges = _normalized_fixture_docs()
    return MongoSource.from_collections(_FakeCollection(nodes), _FakeCollection(edges))


def test_mongo_source_counts_and_strips_id():
    source = _fake_mongo_source()
    triples = list(source.iter_edge_triples())
    edges = list(source.iter_edges())
    nodes = list(source.iter_nodes())
    assert len(triples) == NUM_EDGES
    assert len(edges) == NUM_EDGES
    assert len(nodes) == NUM_NODES
    assert all("_id" not in e for e in edges)
    assert all("_id" not in n for n in nodes)


def test_mongo_source_triples_and_edges_same_order():
    source = _fake_mongo_source()
    triples = list(source.iter_edge_triples())
    edges = list(source.iter_edges())
    for (s, o, p), edge in zip(triples, edges):
        assert (s, o, p) == (edge["subject"], edge["object"], edge["predicate"])


def test_mongo_source_invalid_doc_raises():
    nodes, edges = _normalized_fixture_docs()
    # Corrupt one edge: list-valued qualifier_value
    edges[0]["qualifiers"] = [
        {"qualifier_type_id": "biolink:object_aspect_qualifier", "qualifier_value": ["x"]}
    ]
    source = MongoSource.from_collections(
        _FakeCollection(nodes), _FakeCollection(edges)
    )
    with pytest.raises(SourceValidationError):
        list(source.iter_edges())


def test_mongo_build_matches_jsonl_build():
    """A graph built from normalized Mongo docs matches the jsonl-built graph."""
    jsonl_graph = build_graph_from_jsonl(EDGES_FILE, NODES_FILE)
    mongo_graph = _build_graph_from_source(_fake_mongo_source())

    assert mongo_graph.num_nodes == jsonl_graph.num_nodes
    assert len(mongo_graph.fwd_targets) == len(jsonl_graph.fwd_targets)
    assert mongo_graph.node_id_to_idx == jsonl_graph.node_id_to_idx
    assert mongo_graph.predicate_to_idx == jsonl_graph.predicate_to_idx
