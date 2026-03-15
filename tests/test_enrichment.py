"""Unit tests for gandalf.enrichment module."""

import os

import pytest

from gandalf.enrichment import enrich_knowledge_graph
from gandalf.loader import build_graph_from_jsonl
from gandalf.search import lookup

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
NODES_FILE = os.path.join(FIXTURES_DIR, "nodes.jsonl")
EDGES_FILE = os.path.join(FIXTURES_DIR, "edges.jsonl")


@pytest.fixture
def graph():
    """Build a graph from test fixtures."""
    return build_graph_from_jsonl(EDGES_FILE, NODES_FILE)


def _one_hop_query():
    return {
        "message": {
            "query_graph": {
                "nodes": {
                    "n0": {"ids": ["CHEBI:6801"]},
                    "n1": {"ids": ["MONDO:0005148"]},
                },
                "edges": {
                    "e0": {
                        "subject": "n0",
                        "object": "n1",
                        "predicates": ["biolink:treats"],
                    },
                },
            },
        },
    }


class TestEnrichNodes:
    """Tests for node property enrichment."""

    def test_nodes_get_name(self, graph, bmt):
        """Nodes should have their 'name' property populated."""
        response = lookup(graph, _one_hop_query(), bmt=bmt)
        msg = response["message"]

        enrich_knowledge_graph(msg, graph)

        node = msg["knowledge_graph"]["nodes"]["CHEBI:6801"]
        assert node["name"] == "Metformin"

        node2 = msg["knowledge_graph"]["nodes"]["MONDO:0005148"]
        assert node2["name"] == "Type 2 Diabetes"

    def test_nodes_get_categories(self, graph, bmt):
        """Nodes should have their 'categories' property populated."""
        response = lookup(graph, _one_hop_query(), bmt=bmt)
        msg = response["message"]

        enrich_knowledge_graph(msg, graph)

        node = msg["knowledge_graph"]["nodes"]["CHEBI:6801"]
        assert "biolink:SmallMolecule" in node["categories"]

    def test_nodes_attributes_contain_equivalent_identifiers(self, graph, bmt):
        """Node attributes should include 'equivalent_identifiers' as a TRAPI attribute."""
        response = lookup(graph, _one_hop_query(), bmt=bmt)
        msg = response["message"]

        enrich_knowledge_graph(msg, graph)

        node = msg["knowledge_graph"]["nodes"]["CHEBI:6801"]
        print(node)
        eq_attrs = [
            a
            for a in node["attributes"]
            if a["original_attribute_name"] == "equivalent_identifiers"
        ]
        assert len(eq_attrs) == 1
        assert "DRUGBANK:DB00331" in eq_attrs[0]["value"]

    def test_nodes_attributes_contain_information_content(self, graph, bmt):
        """Node attributes should include 'information_content' as a TRAPI attribute."""
        response = lookup(graph, _one_hop_query(), bmt=bmt)
        msg = response["message"]

        enrich_knowledge_graph(msg, graph)

        node = msg["knowledge_graph"]["nodes"]["CHEBI:6801"]
        ic_attrs = [
            a
            for a in node["attributes"]
            if a["original_attribute_name"] == "information_content"
        ]
        assert len(ic_attrs) == 1
        assert ic_attrs[0]["value"] == 85.5

    def test_nodes_have_attributes_list(self, graph, bmt):
        """Every node should have an 'attributes' list (possibly empty)."""
        response = lookup(graph, _one_hop_query(), bmt=bmt)
        msg = response["message"]

        enrich_knowledge_graph(msg, graph)

        for node_id, node in msg["knowledge_graph"]["nodes"].items():
            assert "attributes" in node, f"Node {node_id} missing 'attributes'"
            assert isinstance(node["attributes"], list)

    def test_does_not_overwrite_existing_node_properties(self, graph, bmt):
        """Enrichment should not overwrite properties already present."""
        response = lookup(graph, _one_hop_query(), bmt=bmt)
        msg = response["message"]

        # Pre-set a custom name
        node = msg["knowledge_graph"]["nodes"]["CHEBI:6801"]
        node["name"] = "CustomName"

        enrich_knowledge_graph(msg, graph)

        assert node["name"] == "CustomName"


class TestEnrichEdges:
    """Tests for edge property enrichment."""

    def test_edges_have_sources(self, graph, bmt):
        """Edges should have a 'sources' list after enrichment."""
        response = lookup(graph, _one_hop_query(), bmt=bmt)
        msg = response["message"]

        enrich_knowledge_graph(msg, graph)

        for edge_id, edge in msg["knowledge_graph"]["edges"].items():
            assert "sources" in edge, f"Edge {edge_id} missing 'sources'"
            assert isinstance(edge["sources"], list)

    def test_edges_have_qualifiers(self, graph, bmt):
        """Edges should have a 'qualifiers' list after enrichment."""
        response = lookup(graph, _one_hop_query(), bmt=bmt)
        msg = response["message"]

        enrich_knowledge_graph(msg, graph)

        for edge_id, edge in msg["knowledge_graph"]["edges"].items():
            assert "qualifiers" in edge, f"Edge {edge_id} missing 'qualifiers'"
            assert isinstance(edge["qualifiers"], list)

    def test_edges_have_publications_in_attributes(self, graph, bmt):
        """Edge publications should appear as TRAPI attributes after enrichment."""
        response = lookup(graph, _one_hop_query(), bmt=bmt)
        msg = response["message"]

        enrich_knowledge_graph(msg, graph)

        # At least one edge should carry a publications attribute
        found_pubs = False
        for edge_id, edge in msg["knowledge_graph"]["edges"].items():
            assert "attributes" in edge, f"Edge {edge_id} missing 'attributes'"
            pub_attrs = [
                a
                for a in edge["attributes"]
                if a["attribute_type_id"] == "biolink:publications"
            ]
            if pub_attrs:
                assert isinstance(pub_attrs[0]["value"], list)
                found_pubs = True
        assert found_pubs, "No edge had a publications attribute"

    def test_edges_have_attributes(self, graph, bmt):
        """Edges should have an 'attributes' list after enrichment."""
        response = lookup(graph, _one_hop_query(), bmt=bmt)
        msg = response["message"]

        enrich_knowledge_graph(msg, graph)

        for edge_id, edge in msg["knowledge_graph"]["edges"].items():
            assert "attributes" in edge, f"Edge {edge_id} missing 'attributes'"
            assert isinstance(edge["attributes"], list)

    def test_does_not_overwrite_existing_edge_properties(self, graph, bmt):
        """Enrichment should not overwrite properties already present."""
        response = lookup(graph, _one_hop_query(), bmt=bmt)
        msg = response["message"]

        # Pick the first edge and pre-set sources
        first_edge_id = next(iter(msg["knowledge_graph"]["edges"]))
        edge = msg["knowledge_graph"]["edges"][first_edge_id]
        edge["sources"] = [
            {
                "resource_id": "infores:custom",
                "resource_role": "primary_knowledge_source",
            }
        ]

        enrich_knowledge_graph(msg, graph)

        assert edge["sources"] == [
            {
                "resource_id": "infores:custom",
                "resource_role": "primary_knowledge_source",
            }
        ]


class TestEnrichEdgesWithQualifiers:
    """Test enrichment with edges that have qualifiers (parallel edges)."""

    def test_qualified_edge_enrichment(self, graph, bmt):
        """Edges with qualifiers should be enriched correctly."""
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["CHEBI:6801"]},
                        "n1": {"categories": ["biolink:Gene"]},
                    },
                    "edges": {
                        "e0": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:affects"],
                            "qualifier_constraints": [
                                {
                                    "qualifier_set": [
                                        {
                                            "qualifier_type_id": "biolink:object_aspect_qualifier",
                                            "qualifier_value": "activity",
                                        }
                                    ]
                                }
                            ],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, bmt=bmt)
        msg = response["message"]

        enrich_knowledge_graph(msg, graph)

        # All edges should have sources, qualifiers, etc.
        for edge_id, edge in msg["knowledge_graph"]["edges"].items():
            assert "sources" in edge
            assert "qualifiers" in edge
            assert "attributes" in edge


class TestEnrichEmptyMessage:
    """Test enrichment handles edge cases gracefully."""

    def test_empty_knowledge_graph(self, graph):
        """Empty knowledge graph should not error."""
        msg = {"knowledge_graph": {"nodes": {}, "edges": {}}}
        result = enrich_knowledge_graph(msg, graph)
        assert result is msg
        assert msg["knowledge_graph"]["nodes"] == {}
        assert msg["knowledge_graph"]["edges"] == {}

    def test_missing_knowledge_graph(self, graph):
        """Message with no knowledge_graph key should not error."""
        msg = {}
        result = enrich_knowledge_graph(msg, graph)
        assert result is msg

    def test_returns_same_message(self, graph, bmt):
        """enrich_knowledge_graph should return the same dict it was passed."""
        response = lookup(graph, _one_hop_query(), bmt=bmt)
        msg = response["message"]

        result = enrich_knowledge_graph(msg, graph)
        assert result is msg
