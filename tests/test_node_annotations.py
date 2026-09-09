"""Tests for build-time node annotation (gandalf.node_annotations)."""

import os

import pytest

from gandalf.enrichment import enrich_knowledge_graph
from gandalf.graph import CSRGraph
from gandalf.loader import build_graph_from_jsonl
from gandalf.node_annotations import (
    BIOTHINGS_ANNOTATIONS_ATTRIBUTE_TYPE_ID,
    annotatable_curies,
    attach_annotations,
    clean_annotation,
    fetch_annotations,
)
from gandalf.search import lookup

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
NODES_FILE = os.path.join(FIXTURES_DIR, "nodes.jsonl")
EDGES_FILE = os.path.join(FIXTURES_DIR, "edges.jsonl")


def _one_hop_query():
    """A CHEBI:6801 -treats-> MONDO:0005148 query over the fixture graph."""
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


def annotations_of(node_properties, node_id_to_idx, curie):
    """Return the annotation attached to *curie*, or None if it has none."""
    properties = node_properties.get(node_id_to_idx[curie], {})
    for attribute in properties.get("attributes", []):
        if attribute["attribute_type_id"] == BIOTHINGS_ANNOTATIONS_ATTRIBUTE_TYPE_ID:
            return attribute["value"]
    return None


class TestCleanAnnotation:
    """The Annotator's empty / not-found results must not reach the graph."""

    @pytest.mark.parametrize(
        "result",
        [
            {},
            [],
            None,
            "",
            {"query": "FOO:1", "notfound": True},
            [{"query": "FOO:1", "notfound": True}],
            [
                {
                    "query": "PMID:1",
                    "notfound": True,
                    "skipped": True,
                    "reason": "source_unavailable_for_backend",
                }
            ],
        ],
    )
    def test_no_annotation_available(self, result):
        assert clean_annotation(result) is None

    def test_dict_annotation_passes_through(self):
        annotation = {"symbol": "TP53", "summary": "tumour suppressor"}
        assert clean_annotation(annotation) == annotation

    def test_list_annotation_keeps_only_hits(self):
        result = [
            {"query": "NCBIGene:7157", "symbol": "TP53"},
            {"query": "NCBIGene:7157", "notfound": True},
        ]
        assert clean_annotation(result) == [
            {"query": "NCBIGene:7157", "symbol": "TP53"}
        ]


class TestAttachAnnotations:
    """Annotations land on nodes as a single TRAPI attribute."""

    def test_appends_attribute_to_existing_node(self):
        node_properties = {
            0: {"name": "TP53", "categories": ["biolink:Gene"], "attributes": []}
        }
        annotated = attach_annotations(
            node_properties,
            {"NCBIGene:7157": 0},
            {"NCBIGene:7157": {"symbol": "TP53"}},
        )

        assert annotated == 1
        assert node_properties[0]["attributes"] == [
            {
                "attribute_type_id": BIOTHINGS_ANNOTATIONS_ATTRIBUTE_TYPE_ID,
                "value": {"symbol": "TP53"},
            }
        ]

    def test_preserves_existing_attributes(self):
        existing = {"attribute_type_id": "biolink:xref", "value": ["HGNC:11998"]}
        node_properties = {
            0: {"name": "TP53", "categories": [], "attributes": [existing]}
        }

        attach_annotations(
            node_properties, {"NCBIGene:7157": 0}, {"NCBIGene:7157": {"symbol": "TP53"}}
        )

        assert node_properties[0]["attributes"][0] == existing
        assert len(node_properties[0]["attributes"]) == 2

    def test_creates_properties_for_node_without_any(self):
        """A node in the edge file but not the node file still gets annotated."""
        node_properties = {}
        attach_annotations(
            node_properties, {"NCBIGene:7157": 3}, {"NCBIGene:7157": {"symbol": "TP53"}}
        )

        assert node_properties[3]["name"] is None
        assert node_properties[3]["categories"] == []
        assert node_properties[3]["attributes"][0]["value"] == {"symbol": "TP53"}

    def test_reannotating_replaces_rather_than_duplicates(self):
        node_properties = {0: {"name": "TP53", "categories": [], "attributes": []}}
        attach_annotations(
            node_properties, {"NCBIGene:7157": 0}, {"NCBIGene:7157": {"symbol": "old"}}
        )
        attach_annotations(
            node_properties, {"NCBIGene:7157": 0}, {"NCBIGene:7157": {"symbol": "new"}}
        )

        attributes = node_properties[0]["attributes"]
        assert len(attributes) == 1
        assert attributes[0]["value"] == {"symbol": "new"}

    def test_ignores_curies_outside_the_graph(self):
        node_properties = {}
        annotated = attach_annotations(
            node_properties, {"NCBIGene:7157": 0}, {"NCBIGene:1017": {"symbol": "CDK2"}}
        )

        assert annotated == 0
        assert node_properties == {}


class TestAnnotatableCuries:
    """Only prefixes the Annotator knows are worth sending to it."""

    def test_keeps_supported_prefixes_and_drops_the_rest(self):
        pytest.importorskip("biothings_annotator")
        node_ids = [
            "NCBIGene:1017",
            "CHEBI:45783",
            "MONDO:0004979",
            "HP:0001943",
            "GO:0006006",  # no Annotator source for GO
            "not-a-curie",
        ]
        assert annotatable_curies(node_ids) == [
            "NCBIGene:1017",
            "CHEBI:45783",
            "MONDO:0004979",
            "HP:0001943",
        ]

    def test_deduplicates(self):
        pytest.importorskip("biothings_annotator")
        assert annotatable_curies(["NCBIGene:1017", "NCBIGene:1017"]) == [
            "NCBIGene:1017"
        ]

    def test_graph_node_ids_are_filtered(self):
        """The fixture graph's GO node is never sent to the Annotator."""
        pytest.importorskip("biothings_annotator")
        graph = build_graph_from_jsonl(EDGES_FILE, NODES_FILE)
        curies = annotatable_curies(graph.node_id_to_idx.keys())

        assert "GO:0006006" not in curies
        assert "NCBIGene:7124" in curies


class TestAnnotationsReachTrapiResponses:
    """A build-time annotation must survive all the way to a TRAPI node.

    The annotation payload below is supplied directly rather than fetched, so
    the test covers gandalf's own plumbing (property storage, mmap
    serialization, enrichment) without depending on the Annotator service.
    """

    ANNOTATION = {
        "query": "MONDO:0005148",
        "mondo": {"mondo": "MONDO:0005148", "label": "type 2 diabetes mellitus"},
        "disease_ontology": {"doid": "DOID:9352"},
    }

    @pytest.fixture
    def annotated_graph(self):
        graph = build_graph_from_jsonl(EDGES_FILE, NODES_FILE)
        attach_annotations(
            graph.node_properties,
            graph.node_id_to_idx,
            {"MONDO:0005148": self.ANNOTATION},
        )
        return graph

    def test_annotation_reaches_the_trapi_node(self, annotated_graph, bmt):
        response = lookup(annotated_graph, _one_hop_query(), bmt=bmt)
        message = enrich_knowledge_graph(response["message"], annotated_graph)

        node = message["knowledge_graph"]["nodes"]["MONDO:0005148"]
        annotations = [
            attribute["value"]
            for attribute in node["attributes"]
            if attribute["attribute_type_id"] == BIOTHINGS_ANNOTATIONS_ATTRIBUTE_TYPE_ID
        ]
        assert annotations == [self.ANNOTATION]

    def test_annotation_survives_mmap_round_trip(self, annotated_graph, tmp_path):
        graph_dir = tmp_path / "graph_mmap"
        annotated_graph.save_mmap(graph_dir)
        reloaded = CSRGraph.load_mmap(graph_dir)

        properties = reloaded.get_all_node_properties(
            reloaded.get_node_idx("MONDO:0005148")
        )
        assert {
            "attribute_type_id": BIOTHINGS_ANNOTATIONS_ATTRIBUTE_TYPE_ID,
            "value": self.ANNOTATION,
        } in properties["attributes"]


@pytest.mark.integration
class TestAnnotatorService:
    """Exercises the live Translator Annotator service (network required)."""

    def test_fetch_annotations_returns_only_available_annotations(self):
        pytest.importorskip("biothings_annotator")
        annotations = fetch_annotations(
            ["NCBIGene:1017", "MONDO:0004979", "GO:0006006"]
        )

        assert "NCBIGene:1017" in annotations
        assert "GO:0006006" not in annotations  # unsupported prefix, never queried
        assert annotations["NCBIGene:1017"]

    def test_build_graph_with_annotations(self):
        pytest.importorskip("biothings_annotator")
        graph = build_graph_from_jsonl(EDGES_FILE, NODES_FILE, annotate_nodes=True)

        annotation = annotations_of(
            graph.node_properties, graph.node_id_to_idx, "NCBIGene:7124"
        )
        assert annotation is not None

        # Unsupported prefixes are left exactly as the node file had them.
        assert (
            annotations_of(graph.node_properties, graph.node_id_to_idx, "GO:0006006")
            is None
        )
