"""Tests for build-time node annotation (gandalf.node_annotations)."""

import os

import pytest

from gandalf.biolink import NAMED_THING
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
from tests.test_trapi_conformance import assert_valid_trapi

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
        """A node in the edge file but not the node file still gets annotated.

        The entry must match the shape the loader builds: TRAPI 2.0 admits no
        nulls, so an unknown name is absent rather than None, and the required
        categories fall back to the Biolink root class.
        """
        node_properties = {}
        attach_annotations(
            node_properties, {"NCBIGene:7157": 3}, {"NCBIGene:7157": {"symbol": "TP53"}}
        )

        assert "name" not in node_properties[3]
        assert node_properties[3]["categories"] == [NAMED_THING]
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

    def test_annotated_response_is_valid_trapi(self, annotated_graph, bmt):
        """The annotation attribute must not cost the response its conformance.

        ``biothings_annotations`` is not a Biolink CURIE -- it is the
        Annotator service's own attribute_type_id -- so it is worth pinning
        that TRAPI 2.0 still admits a response carrying it.
        """
        response = lookup(annotated_graph, _one_hop_query(), bmt=bmt)
        enrich_knowledge_graph(response["message"], annotated_graph)

        assert_valid_trapi(response)

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


# ---------------------------------------------------------------------------
# Resilience: retries, bisecting, outage, cache
# ---------------------------------------------------------------------------
#
# The Annotator fans out to BioThings APIs that answer 500 now and then, for
# a while or for one CURIE.  The policy that absorbs that is exercised here
# against callables that fail on purpose; the live service is covered by the
# integration tests above.

import asyncio  # noqa: E402

from gandalf.node_annotations import (  # noqa: E402
    AnnotationCache,
    AnnotatorUnavailable,
    _run_annotation,
    annotate_resilient,
    annotate_with_retry,
)


class _Service:
    """An Annotator stand-in: answers {curie: {"symbol": curie}}, with faults."""

    def __init__(
        self, *, fail_first: int = 0, poison: set = frozenset(), down: bool = False
    ):
        self.fail_first = fail_first  # the first N calls fail outright
        self.poison = set(poison)  # any batch containing one of these fails
        self.down = down
        self.calls: list = []

    async def __call__(self, batch):
        self.calls.append(list(batch))
        if self.down or len(self.calls) <= self.fail_first or self.poison & set(batch):
            raise RuntimeError("500 Internal Server Error from mydisease.info")
        return {c: {"symbol": c} for c in batch}


async def _no_sleep(seconds):
    _no_sleep.waited.append(seconds)


_no_sleep.waited = []


@pytest.fixture(autouse=True)
def _reset_sleep():
    _no_sleep.waited = []


def test_retry_backs_off_then_succeeds():
    service = _Service(fail_first=2)
    result = asyncio.run(
        annotate_with_retry(
            service, ["A:1", "A:2"], attempts=5, backoff_seconds=2.0, sleep=_no_sleep
        )
    )
    assert result == {"A:1": {"symbol": "A:1"}, "A:2": {"symbol": "A:2"}}
    assert len(service.calls) == 3
    assert _no_sleep.waited == [2.0, 4.0]


def test_retry_gives_up_with_the_last_error():
    service = _Service(down=True)
    with pytest.raises(RuntimeError, match="500"):
        asyncio.run(
            annotate_with_retry(
                service, ["A:1"], attempts=3, backoff_seconds=1.0, sleep=_no_sleep
            )
        )
    assert len(service.calls) == 3 and _no_sleep.waited == [1.0, 2.0]


def test_bisecting_isolates_the_curie_the_service_rejects():
    service = _Service(poison={"A:7"})
    failed: list = []
    batch = [f"A:{i}" for i in range(16)]
    result = asyncio.run(
        annotate_resilient(
            service,
            batch,
            attempts=2,
            backoff_seconds=0.1,
            failed=failed,
            sleep=_no_sleep,
        )
    )
    assert failed == ["A:7"]
    assert set(result) == set(batch) - {"A:7"}
    assert all(result[c] == {"symbol": c} for c in result)
    # one failing half per level, each with its own retries, not thousands of calls
    assert len(service.calls) < 30


def test_a_down_service_aborts_instead_of_silently_skipping():
    service = _Service(down=True)
    failed: list = []
    with pytest.raises(AnnotatorUnavailable, match="unavailable"):
        asyncio.run(
            annotate_resilient(
                service,
                [f"A:{i}" for i in range(64)],
                attempts=2,
                backoff_seconds=0.1,
                failed=failed,
                sleep=_no_sleep,
            )
        )
    assert len(service.calls) < 40  # it does not grind through every CURIE


def test_cache_makes_a_rerun_fetch_only_what_is_missing(tmp_path):
    cache = tmp_path / "annotations.jsonl"
    curies = [f"A:{i}" for i in range(10)]
    first = _Service(poison={"A:3"})
    got: dict = {}
    queried = _run_annotation(
        curies, 4, got.update, attempts=2, cache_path=cache, annotate=first
    )
    assert queried == 10
    assert set(got) == set(curies) - {"A:3"}

    # The cache holds every answer the service gave, not the CURIE it rejected.
    known = AnnotationCache(cache).known
    assert set(known) == set(curies) - {"A:3"}

    second = _Service()
    got2: dict = {}
    _run_annotation(
        curies, 4, got2.update, attempts=2, cache_path=cache, annotate=second
    )
    assert got2 == got | {"A:3": {"symbol": "A:3"}}
    assert second.calls == [["A:3"]]  # only the one that was missing
    assert "A:3" in AnnotationCache(cache).known


def test_cache_remembers_nothing_available_too(tmp_path):
    cache = tmp_path / "annotations.jsonl"

    async def nothing(batch):
        return {c: {} for c in batch}  # the Annotator's "no annotation" answer

    got: dict = {}
    _run_annotation(["A:1"], 10, got.update, cache_path=cache, annotate=nothing)
    assert got == {}
    assert AnnotationCache(cache).known == {"A:1": None}
    calls = []

    async def counting(batch):
        calls.append(batch)
        return {}

    _run_annotation(["A:1"], 10, got.update, cache_path=cache, annotate=counting)
    assert calls == []  # not asked again
