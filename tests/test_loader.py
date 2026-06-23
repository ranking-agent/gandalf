"""Unit tests for gandalf.loader module."""

import os
import tempfile

import pytest

from gandalf.loader import build_graph_from_jsonl
from gandalf.normalize import (
    _extract_qualifiers,
    _extract_attributes,
    _get_qualifier_fields,
)
from gandalf.graph import CSRGraph

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
NODES_FILE = os.path.join(FIXTURES_DIR, "nodes.jsonl")
EDGES_FILE = os.path.join(FIXTURES_DIR, "edges.jsonl")


@pytest.fixture
def graph():
    """Build a graph from test fixtures."""
    return build_graph_from_jsonl(EDGES_FILE, NODES_FILE)


class TestBuildGraphFromJsonl:
    """Tests for build_graph_from_jsonl function."""

    def test_returns_csr_graph(self, graph):
        """Should return a CSRGraph instance."""
        assert isinstance(graph, CSRGraph)

    def test_loads_correct_number_of_nodes(self, graph):
        """Should load all unique nodes from edges."""
        # Nodes in edges: CHEBI:6801, MONDO:0005148, NCBIGene:5468, NCBIGene:3643,
        # HP:0001943, CHEBI:17234, GO:0006006, NCBIGene:2645, NCBIGene:7124 (TNF)
        assert graph.num_nodes == 11

    def test_loads_edges(self, graph):
        """Should load edges into the graph, including duplicate (s,o,p) edges."""
        # 20 edges in file (includes duplicate s,o,p with different qualifiers/sources)
        assert len(graph.fwd_targets) == 20

    def test_node_id_to_idx_mapping(self, graph):
        """Should create bidirectional node ID mapping."""
        # Check that all expected nodes are in the mapping
        expected_nodes = [
            "CHEBI:6801",
            "MONDO:0005148",
            "NCBIGene:5468",
            "NCBIGene:3643",
            "HP:0001943",
            "CHEBI:17234",
            "GO:0006006",
            "NCBIGene:2645",
            "NCBIGene:7124",  # TNF - added for qualifier constraint tests
        ]
        for node_id in expected_nodes:
            assert node_id in graph.node_id_to_idx
            idx = graph.node_id_to_idx[node_id]
            assert graph.idx_to_node_id[idx] == node_id

    def test_predicate_vocabulary(self, graph):
        """Should build predicate vocabulary from edges."""
        expected_predicates = [
            "biolink:treats",
            "biolink:affects",
            "biolink:gene_associated_with_condition",
            "biolink:has_phenotype",
            "biolink:interacts_with",
            "biolink:participates_in",
        ]
        for pred in expected_predicates:
            assert pred in graph.predicate_to_idx


class TestNodeProperties:
    """Tests for node properties loading."""

    def test_node_properties_loaded(self, graph):
        """Should load node properties from nodes.jsonl."""
        metformin_idx = graph.node_id_to_idx["CHEBI:6801"]
        assert metformin_idx in graph.node_properties

    def test_node_name_property(self, graph):
        """Should correctly load node name property."""
        metformin_idx = graph.node_id_to_idx["CHEBI:6801"]
        assert graph.get_node_property(metformin_idx, "name") == "Metformin"

    def test_node_category_property(self, graph):
        """Should correctly load node category property."""
        pparg_idx = graph.node_id_to_idx["NCBIGene:5468"]
        categories = graph.get_node_property(pparg_idx, "categories")
        assert "biolink:Gene" in categories

    def test_node_attributes_contain_information_content(self, graph):
        """Should store information_content as a TRAPI attribute."""
        diabetes_idx = graph.node_id_to_idx["MONDO:0005148"]
        attributes = graph.get_node_property(diabetes_idx, "attributes")
        ic_attrs = [
            a
            for a in attributes
            if a["original_attribute_name"] == "information_content"
        ]
        assert len(ic_attrs) == 1
        assert ic_attrs[0]["value"] == pytest.approx(78.2)
        assert ic_attrs[0]["original_attribute_name"] == "information_content"

    def test_node_attributes_contain_equivalent_identifiers(self, graph):
        """Should store equivalent_identifiers as a TRAPI attribute."""
        metformin_idx = graph.node_id_to_idx["CHEBI:6801"]
        attributes = graph.get_node_property(metformin_idx, "attributes")
        eq_attrs = [
            a
            for a in attributes
            if a["original_attribute_name"] == "equivalent_identifiers"
        ]
        assert len(eq_attrs) == 1
        assert "DRUGBANK:DB00331" in eq_attrs[0]["value"]
        assert eq_attrs[0]["original_attribute_name"] == "equivalent_identifiers"

    def test_node_id_not_in_properties(self, graph):
        """Node id should be omitted from properties (it's the key in the KG)."""
        metformin_idx = graph.node_id_to_idx["CHEBI:6801"]
        props = graph.get_all_node_properties(metformin_idx)
        assert "id" not in props
        # Also should not appear in attributes
        attributes = props.get("attributes", [])
        id_attrs = [a for a in attributes if a["original_attribute_name"] == "id"]
        assert len(id_attrs) == 0


class TestEdgeProperties:
    """Tests for edge properties loading."""

    def test_edge_properties_stored(self, graph):
        """Should store edge properties."""
        assert len(graph.edge_properties) > 0

    def test_edge_predicate_property(self, graph):
        """Should correctly store predicate in edge properties."""
        src_idx = graph.node_id_to_idx["CHEBI:6801"]
        dst_idx = graph.node_id_to_idx["MONDO:0005148"]
        predicate = graph.get_edge_property(
            src_idx, dst_idx, "biolink:treats", "predicate"
        )
        assert predicate == "biolink:treats"

    def test_edge_source_property(self, graph):
        """Should correctly store knowledge source with gandalf aggregator."""
        src_idx = graph.node_id_to_idx["CHEBI:6801"]
        dst_idx = graph.node_id_to_idx["MONDO:0005148"]
        sources = graph.get_edge_property(src_idx, dst_idx, "biolink:treats", "sources")
        assert sources is not None
        assert len(sources) >= 2

        # First source should be gandalf as aggregator
        assert sources[0]["resource_id"] == "infores:gandalf"
        assert sources[0]["resource_role"] == "aggregator_knowledge_source"
        assert sources[0]["upstream_resource_ids"] == ["infores:drugcentral"]

        # Second source is the original primary knowledge source
        assert sources[1]["resource_id"] == "infores:drugcentral"
        assert sources[1]["resource_role"] == "primary_knowledge_source"

    def test_edge_sources_all_have_upstream_resource_ids(self, graph):
        """Every source entry should have an upstream_resource_ids list."""
        src_idx = graph.node_id_to_idx["CHEBI:6801"]
        dst_idx = graph.node_id_to_idx["MONDO:0005148"]
        sources = graph.get_edge_property(src_idx, dst_idx, "biolink:treats", "sources")
        for source in sources:
            assert "upstream_resource_ids" in source
            assert isinstance(source["upstream_resource_ids"], list)

    def test_edge_publications_in_attributes(self, graph):
        """Publications should be stored as a TRAPI attribute on the edge."""
        src_idx = graph.node_id_to_idx["CHEBI:6801"]
        dst_idx = graph.node_id_to_idx["NCBIGene:5468"]
        props = graph.get_all_edge_properties(src_idx, dst_idx, "biolink:affects")
        pub_attrs = [
            a
            for a in props.get("attributes", [])
            if a["attribute_type_id"] == "biolink:publications"
        ]
        assert len(pub_attrs) == 1
        assert "PMID:23456789" in pub_attrs[0]["value"]
        assert pub_attrs[0]["original_attribute_name"] == "publications"


class TestGraphStructure:
    """Tests for CSR graph structure."""

    def test_neighbors_returns_correct_targets(self, graph):
        """Should return correct neighbor nodes."""
        metformin_idx = graph.node_id_to_idx["CHEBI:6801"]
        neighbors = graph.neighbors(metformin_idx)

        # Metformin should connect to: MONDO:0005148, NCBIGene:5468, CHEBI:17234,
        # NCBIGene:3643 (INSR), NCBIGene:2645 (GCK), NCBIGene:7124 (TNF)
        neighbor_ids = {graph.idx_to_node_id[int(n)] for n in neighbors}
        assert "MONDO:0005148" in neighbor_ids
        assert "NCBIGene:5468" in neighbor_ids
        assert "CHEBI:17234" in neighbor_ids
        assert "NCBIGene:3643" in neighbor_ids  # INSR - qualifier test edge
        assert "NCBIGene:2645" in neighbor_ids  # GCK - qualifier test edge
        assert "NCBIGene:7124" in neighbor_ids  # TNF - qualifier test edge
        assert "MONDO:0005015" in neighbor_ids

    def test_neighbors_with_predicate_filter(self, graph):
        """Should filter neighbors by predicate."""
        metformin_idx = graph.node_id_to_idx["CHEBI:6801"]
        neighbors = graph.neighbors(metformin_idx, predicate_filter="biolink:treats")

        neighbor_ids = {graph.idx_to_node_id[int(n)] for n in neighbors}
        assert "MONDO:0005148" in neighbor_ids
        assert "MONDO:0005015" in neighbor_ids
        # 3 treats edges total (2 to MONDO:0005148 with different sources, 1 to MONDO:0005015)
        assert len(neighbors) == 3

    def test_degree_calculation(self, graph):
        """Should correctly calculate node degree."""
        metformin_idx = graph.node_id_to_idx["CHEBI:6801"]
        degree = graph.degree(metformin_idx)
        # Metformin has 11 outgoing edges:
        # - treats MONDO:0005148 (drugcentral)
        # - treats MONDO:0005148 (chembl) — duplicate s,o,p, different source
        # - ameliorates MONDO:0005148
        # - preventative MONDO:0005148
        # - affects NCBIGene:5468 (PPARG)
        # - affects CHEBI:17234 (Glucose)
        # - affects NCBIGene:3643 (INSR) with qualifiers (activity/increased, from ctd)
        # - affects NCBIGene:3643 (INSR) with qualifiers (abundance/decreased, from hetio)
        # - affects NCBIGene:2645 (GCK) with qualifiers
        # - affects NCBIGene:7124 (TNF) with qualifiers
        # - treats MONDO:0005015
        assert degree == 11

    def test_degree_with_predicate_filter(self, graph):
        """Should correctly calculate filtered degree."""
        metformin_idx = graph.node_id_to_idx["CHEBI:6801"]
        degree = graph.degree(metformin_idx, predicate_filter="biolink:treats")
        # 3 treats edges: 2 to MONDO:0005148 (different sources) + 1 to MONDO:0005015
        assert degree == 3

    def test_get_edges_returns_tuples(self, graph):
        """Should return edges as (neighbor_idx, predicate) tuples."""
        metformin_idx = graph.node_id_to_idx["CHEBI:6801"]
        edges = graph.get_edges(metformin_idx)

        # Metformin has 11 outgoing edges (including duplicate s,o,p edges)
        assert len(edges) == 11
        for neighbor_idx, predicate in edges:
            assert isinstance(neighbor_idx, int)
            assert isinstance(predicate, str)
            assert predicate.startswith("biolink:")


class TestGraphMmapSaveLoad:
    """Tests for memory-mapped graph serialization."""

    def test_save_mmap_creates_directory(self, graph):
        """Should create directory with expected files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            graph.save_mmap(temp_dir)

            # Check all expected files exist
            expected_files = [
                "fwd_targets.npy",
                "fwd_predicates.npy",
                "fwd_offsets.npy",
                "rev_sources.npy",
                "rev_predicates.npy",
                "rev_offsets.npy",
                "rev_to_fwd.npy",
                "metadata.pkl",
                # Edge properties stored as split mmap-friendly components
                "edge_sources_idx.npy",
                "edge_quals_idx.npy",
                "edge_property_pools.pkl",
            ]
            for filename in expected_files:
                assert os.path.exists(
                    os.path.join(temp_dir, filename)
                ), f"Missing {filename}"

            # Node store LMDB should exist
            assert os.path.isdir(os.path.join(temp_dir, "node_store.lmdb"))

    def test_mmap_save_and_load_roundtrip(self, graph):
        """Should correctly save and load graph in mmap format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            graph.save_mmap(temp_dir)
            loaded_graph = CSRGraph.load_mmap(temp_dir)

            # Verify key attributes match
            assert loaded_graph.num_nodes == graph.num_nodes
            assert len(loaded_graph.fwd_targets) == len(graph.fwd_targets)
            assert loaded_graph.predicate_to_idx == graph.predicate_to_idx

            # Node ID mappings should round-trip via LMDB-backed store
            for node_id, node_idx in graph.node_id_to_idx.items():
                assert loaded_graph.get_node_idx(node_id) == node_idx
                assert loaded_graph.get_node_id(node_idx) == node_id

    def test_mmap_loaded_graph_neighbors_work(self, graph):
        """Should be able to query neighbors after mmap load."""
        with tempfile.TemporaryDirectory() as temp_dir:
            graph.save_mmap(temp_dir)
            loaded_graph = CSRGraph.load_mmap(temp_dir)

            metformin_idx = loaded_graph.get_node_idx("CHEBI:6801")
            neighbors = loaded_graph.neighbors(metformin_idx)
            # Metformin has 11 outgoing edges (including duplicate s,o,p edges)
            assert len(neighbors) == 11

    def test_mmap_loaded_graph_edge_properties(self, graph):
        """Should correctly load edge properties in mmap format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            graph.save_mmap(temp_dir)
            loaded_graph = CSRGraph.load_mmap(temp_dir)

            # Test edge property lookup
            src_idx = loaded_graph.get_node_idx("CHEBI:6801")
            dst_idx = loaded_graph.get_node_idx("MONDO:0005148")
            predicate = loaded_graph.get_edge_property(
                src_idx, dst_idx, "biolink:treats", "predicate"
            )
            assert predicate == "biolink:treats"

    def test_mmap_loaded_graph_node_properties(self, graph):
        """Should correctly load node properties in mmap format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            graph.save_mmap(temp_dir)
            loaded_graph = CSRGraph.load_mmap(temp_dir)

            metformin_idx = loaded_graph.get_node_idx("CHEBI:6801")
            name = loaded_graph.get_node_property(metformin_idx, "name")
            assert name == "Metformin"

    def test_mmap_loaded_graph_qualifiers(self, graph):
        """Should correctly load edge qualifiers in mmap format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            graph.save_mmap(temp_dir)
            loaded_graph = CSRGraph.load_mmap(temp_dir)

            # Check an edge with qualifiers
            metformin_idx = loaded_graph.get_node_idx("CHEBI:6801")
            insr_idx = loaded_graph.get_node_idx("NCBIGene:3643")

            qualifiers = loaded_graph.get_edge_property(
                metformin_idx, insr_idx, "biolink:affects", "qualifiers"
            )
            assert qualifiers is not None
            assert len(qualifiers) == 2

    def test_mmap_with_no_mmap_mode(self, graph):
        """Should load fully into memory when mmap_mode=None."""
        with tempfile.TemporaryDirectory() as temp_dir:
            graph.save_mmap(temp_dir)
            loaded_graph = CSRGraph.load_mmap(temp_dir, mmap_mode=None)

            # Verify it still works
            assert loaded_graph.num_nodes == graph.num_nodes
            metformin_idx = loaded_graph.get_node_idx("CHEBI:6801")
            neighbors = loaded_graph.neighbors(metformin_idx)
            assert len(neighbors) == 11


class TestDuplicateEdges:
    """Tests for edges sharing (subject, object, predicate) but with different qualifiers/sources."""

    def test_duplicate_treats_edges_both_stored(self, graph):
        """Two treats edges (CHEBI:6801 -> MONDO:0005148) with different sources should both exist."""
        metformin_idx = graph.node_id_to_idx["CHEBI:6801"]
        diabetes_idx = graph.node_id_to_idx["MONDO:0005148"]

        # get_all_edges_between returns ALL edges, including duplicates
        treats_edges = graph.get_all_edges_between(
            metformin_idx, diabetes_idx, predicate_filter=["biolink:treats"]
        )
        assert len(treats_edges) == 2

        # Each edge should have different sources
        source_ids = set()
        for _pred, props in treats_edges:
            primary = [
                s
                for s in props["sources"]
                if s["resource_role"] == "primary_knowledge_source"
            ]
            assert len(primary) == 1
            source_ids.add(primary[0]["resource_id"])

        assert source_ids == {"infores:drugcentral", "infores:chembl"}

    def test_duplicate_affects_edges_different_qualifiers(self, graph):
        """Two affects edges (CHEBI:6801 -> NCBIGene:3643) with different qualifiers should both exist."""
        metformin_idx = graph.node_id_to_idx["CHEBI:6801"]
        insr_idx = graph.node_id_to_idx["NCBIGene:3643"]

        affects_edges = graph.get_all_edges_between(
            metformin_idx, insr_idx, predicate_filter=["biolink:affects"]
        )
        assert len(affects_edges) == 2

        # Extract qualifier combinations
        qualifier_combos = set()
        for _pred, props in affects_edges:
            quals = props.get("qualifiers", [])
            combo = tuple(
                sorted((q["qualifier_type_id"], q["qualifier_value"]) for q in quals)
            )
            qualifier_combos.add(combo)

        assert len(qualifier_combos) == 2  # Two distinct qualifier combinations

    def test_forward_neighbors_return_all_duplicate_edges(self, graph):
        """neighbors_with_properties should return both duplicate edges."""
        metformin_idx = graph.node_id_to_idx["CHEBI:6801"]
        diabetes_idx = graph.node_id_to_idx["MONDO:0005148"]

        treats_results = [
            (target, pred, props, eidx)
            for target, pred, props, eidx in graph.neighbors_with_properties(
                metformin_idx
            )
            if target == diabetes_idx and pred == "biolink:treats"
        ]
        assert len(treats_results) == 2
        # Each should have a unique fwd_edge_idx
        assert treats_results[0][3] != treats_results[1][3]

    def test_reverse_neighbors_return_all_duplicate_edges(self, graph):
        """incoming_neighbors_with_properties should return both duplicate edges."""
        metformin_idx = graph.node_id_to_idx["CHEBI:6801"]
        diabetes_idx = graph.node_id_to_idx["MONDO:0005148"]

        treats_results = [
            (src, pred, props, eidx)
            for src, pred, props, eidx in graph.incoming_neighbors_with_properties(
                diabetes_idx
            )
            if src == metformin_idx and pred == "biolink:treats"
        ]
        assert len(treats_results) == 2
        # Each should have correct, distinct properties
        source_ids = set()
        for _, _, props, _ in treats_results:
            primary = [
                s
                for s in props["sources"]
                if s["resource_role"] == "primary_knowledge_source"
            ]
            source_ids.add(primary[0]["resource_id"])
        assert source_ids == {"infores:drugcentral", "infores:chembl"}

    def test_get_edge_properties_by_index(self, graph):
        """get_edge_properties_by_index should return correct props for each duplicate."""
        metformin_idx = graph.node_id_to_idx["CHEBI:6801"]

        edge_indices = []
        for target, pred, props, eidx in graph.neighbors_with_properties(metformin_idx):
            if (
                pred == "biolink:treats"
                and graph.idx_to_node_id[target] == "MONDO:0005148"
            ):
                edge_indices.append(eidx)

        assert len(edge_indices) == 2
        props_a = graph.get_edge_properties_by_index(edge_indices[0])
        props_b = graph.get_edge_properties_by_index(edge_indices[1])
        assert props_a["sources"] != props_b["sources"]

    def test_rev_to_fwd_roundtrip_mmap(self, graph):
        """rev_to_fwd should survive save/load mmap roundtrip and produce correct results."""
        with tempfile.TemporaryDirectory() as temp_dir:
            graph.save_mmap(temp_dir)
            loaded = CSRGraph.load_mmap(temp_dir)

            metformin_idx = loaded.get_node_idx("CHEBI:6801")
            diabetes_idx = loaded.get_node_idx("MONDO:0005148")

            treats_results = [
                (src, pred, props, eidx)
                for src, pred, props, eidx in loaded.incoming_neighbors_with_properties(
                    diabetes_idx
                )
                if src == metformin_idx and pred == "biolink:treats"
            ]
            assert len(treats_results) == 2
            source_ids = set()
            for _, _, props, _ in treats_results:
                primary = [
                    s
                    for s in props["sources"]
                    if s["resource_role"] == "primary_knowledge_source"
                ]
                source_ids.add(primary[0]["resource_id"])
            assert source_ids == {"infores:drugcentral", "infores:chembl"}


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_nonexistent_predicate_filter_returns_empty(self, graph):
        """Should return empty array for nonexistent predicate."""
        metformin_idx = graph.node_id_to_idx["CHEBI:6801"]
        neighbors = graph.neighbors(
            metformin_idx, predicate_filter="biolink:nonexistent"
        )
        assert len(neighbors) == 0

    def test_node_with_no_outgoing_edges(self, graph):
        """Should handle nodes with no outgoing edges."""
        # HP:0001943 (Hypoglycemia) has no outgoing edges in our test data
        hypoglycemia_idx = graph.node_id_to_idx["HP:0001943"]
        neighbors = graph.neighbors(hypoglycemia_idx)
        assert len(neighbors) == 0

    def test_get_node_idx_returns_none_for_unknown(self, graph):
        """Should return None for unknown node ID."""
        assert graph.get_node_idx("UNKNOWN:12345") is None

    def test_get_node_property_default_value(self, graph):
        """Should return default for missing property."""
        metformin_idx = graph.node_id_to_idx["CHEBI:6801"]
        value = graph.get_node_property(
            metformin_idx, "nonexistent_key", default="default"
        )
        assert value == "default"


class TestQualifierExtraction:
    """Tests that top-level qualifier fields are routed to qualifiers, not attributes.

    Regression test for issue #20: qualifier terms beyond the original hardcoded
    set (e.g. frequency_qualifier, subject_form_or_variant_qualifier) were ending
    up in edge ``attributes`` instead of ``qualifiers``.
    """

    EDGE = {
        "id": "edge1",
        "subject": "NCBIGene:1028",
        "predicate": "biolink:associated_with",
        "object": "MONDO:0004979",
        "primary_knowledge_source": "infores:test",
        "qualified_predicate": "biolink:causes",
        "object_aspect_qualifier": "activity",
        "frequency_qualifier": "HP:0040283",
        "subject_form_or_variant_qualifier": "genetic_variant_form",
        # Added in biolink 4.3.2 (the version tier 1 pins); must be a qualifier.
        "disease_context_qualifier": "Orphanet:397590",
        "p_value": 0.03,
        "publications": ["PMID:11111111"],
    }

    EXPECTED_QUALIFIERS = {
        "qualified_predicate",
        "object_aspect_qualifier",
        "frequency_qualifier",
        "subject_form_or_variant_qualifier",
        "disease_context_qualifier",
    }

    def test_qualifier_set_is_model_derived(self):
        """The qualifier-field set should come from the Biolink Model via BMT."""
        fields = _get_qualifier_fields()
        assert self.EXPECTED_QUALIFIERS <= fields

    def test_disease_context_qualifier_is_a_qualifier(self):
        """disease_context_qualifier is a qualifier in biolink 4.3.2 (tier 1 parity).

        It is NOT present in older biolink (e.g. 4.2.2, BMT's bare default), so
        this guards against the toolkit silently using an unpinned version.
        """
        assert "disease_context_qualifier" in _get_qualifier_fields()
        quals = _extract_qualifiers(self.EDGE)
        by_type = {q["qualifier_type_id"]: q["qualifier_value"] for q in quals}
        assert by_type.get("biolink:disease_context_qualifier") == "Orphanet:397590"
        # And it must not also appear as an attribute.
        attr_names = {
            a["original_attribute_name"] for a in _extract_attributes(self.EDGE)
        }
        assert "disease_context_qualifier" not in attr_names

    def test_qualifiers_extracted_with_biolink_prefix(self):
        """All qualifier fields become TRAPI qualifiers with biolink-prefixed type ids."""
        qualifiers = _extract_qualifiers(self.EDGE)
        by_type = {q["qualifier_type_id"]: q["qualifier_value"] for q in qualifiers}
        for field in self.EXPECTED_QUALIFIERS:
            assert f"biolink:{field}" in by_type
            assert by_type[f"biolink:{field}"] == self.EDGE[field]

    def test_list_valued_qualifier_coerced_to_json_string(self):
        """A non-string (list) qualifier value becomes a single JSON-string entry.

        Matches the tier 1 (BioPack/retriever) driver, which coerces non-string
        qualifier values via orjson rather than splitting or dropping them. TRAPI
        requires qualifier_value to be a scalar string.
        """
        edge = dict(self.EDGE, object_aspect_qualifier=["activity", "abundance"])
        qualifiers = _extract_qualifiers(edge)
        aspect = [
            q
            for q in qualifiers
            if q["qualifier_type_id"] == "biolink:object_aspect_qualifier"
        ]
        # Exactly one entry, with the JSON-encoded value (no split).
        assert len(aspect) == 1
        assert aspect[0]["qualifier_value"] == '["activity","abundance"]'
        # Every value is a string.
        assert all(isinstance(q["qualifier_value"], str) for q in qualifiers)

    def test_qualified_predicate_value_gets_biolink_prefix(self):
        """qualified_predicate values are normalized to carry a biolink: prefix."""
        # Unprefixed value gains the prefix.
        quals = _extract_qualifiers(dict(self.EDGE, qualified_predicate="causes"))
        qp = next(
            q for q in quals if q["qualifier_type_id"] == "biolink:qualified_predicate"
        )
        assert qp["qualifier_value"] == "biolink:causes"
        # Already-prefixed value is left intact (idempotent).
        quals = _extract_qualifiers(
            dict(self.EDGE, qualified_predicate="biolink:causes")
        )
        qp = next(
            q for q in quals if q["qualifier_type_id"] == "biolink:qualified_predicate"
        )
        assert qp["qualifier_value"] == "biolink:causes"

    def test_qualifiers_not_in_attributes(self):
        """Qualifier fields must NOT leak into edge attributes."""
        attributes = _extract_attributes(self.EDGE)
        attr_names = {a["original_attribute_name"] for a in attributes}
        assert self.EXPECTED_QUALIFIERS.isdisjoint(attr_names)

    def test_non_qualifier_fields_remain_attributes(self):
        """Genuine attribute fields should still be extracted as attributes."""
        attributes = _extract_attributes(self.EDGE)
        attr_names = {a["original_attribute_name"] for a in attributes}
        assert "p_value" in attr_names
        assert "publications" in attr_names


class TestMetaKgListValuedQualifier:
    """Regression test: a list-valued qualifier must not crash meta-KG build.

    Qualifier terms are routed to the hot-path qualifier store, and the meta
    knowledge graph builder iterates their values. A list-valued qualifier once
    raised ``TypeError: unhashable type: 'list'`` in ``build_meta_kg``; the
    loader now coerces such values to a JSON string (matching tier 1) so the
    meta-KG only ever sees scalar strings.
    """

    def _build_graph_with_list_qualifier(self, tmp_path):
        nodes = (
            '{"id": "NCBIGene:1028", "name": "G", "category": ["biolink:Gene"]}\n'
            '{"id": "MONDO:0004979", "name": "D", "category": ["biolink:Disease"]}\n'
        )
        edge = {
            "id": "e1",
            "subject": "NCBIGene:1028",
            "predicate": "biolink:associated_with",
            "object": "MONDO:0004979",
            "primary_knowledge_source": "infores:test",
            # List-valued qualifier value (the regression trigger)
            "object_aspect_qualifier": ["activity", "abundance"],
        }
        nodes_file = os.path.join(tmp_path, "nodes.jsonl")
        edges_file = os.path.join(tmp_path, "edges.jsonl")
        with open(nodes_file, "w") as f:
            f.write(nodes)
        with open(edges_file, "w") as f:
            import json

            f.write(json.dumps(edge) + "\n")
        return build_graph_from_jsonl(edges_file, nodes_file)

    def test_build_metadata_coerces_list_qualifier_to_json_string(self):
        with tempfile.TemporaryDirectory() as tmp_path:
            graph = self._build_graph_with_list_qualifier(tmp_path)
            # Must not raise TypeError on the list-valued qualifier.
            graph.build_metadata()

            qual_values = {
                qtype: q["applicable_values"]
                for edge in graph.meta_kg["edges"]
                for q in edge["qualifiers"]
                for qtype in [q["qualifier_type_id"]]
            }
            assert "biolink:object_aspect_qualifier" in qual_values
            # The list value is coerced to a single JSON-string applicable value.
            assert set(qual_values["biolink:object_aspect_qualifier"]) == {
                '["activity","abundance"]',
            }

    def test_normal_lookup_with_list_qualifier_does_not_crash(self, bmt):
        """A normal lookup over an edge with a list-valued qualifier must succeed.

        Regression: the per-edge dedup key in lookup hashes (type_id, value)
        pairs, so a list qualifier_value raised "unhashable type: 'list'".
        """
        from gandalf.search import lookup

        with tempfile.TemporaryDirectory() as tmp_path:
            nodes = (
                '{"id": "CHEBI:6801", "name": "M", "category": ["biolink:SmallMolecule"]}\n'
                '{"id": "NCBIGene:1028", "name": "G", "category": ["biolink:Gene"]}\n'
            )
            edge = {
                "id": "e1",
                "subject": "CHEBI:6801",
                "predicate": "biolink:affects",
                "object": "NCBIGene:1028",
                "primary_knowledge_source": "infores:test",
                "object_aspect_qualifier": ["activity", "abundance"],
            }
            nodes_file = os.path.join(tmp_path, "nodes.jsonl")
            edges_file = os.path.join(tmp_path, "edges.jsonl")
            with open(nodes_file, "w") as f:
                f.write(nodes)
            with open(edges_file, "w") as f:
                import json

                f.write(json.dumps(edge) + "\n")
            graph = build_graph_from_jsonl(edges_file, nodes_file)

            query = {
                "message": {
                    "query_graph": {
                        "nodes": {
                            "n0": {"ids": ["CHEBI:6801"]},
                            "n1": {"ids": ["NCBIGene:1028"]},
                        },
                        "edges": {
                            "e0": {
                                "subject": "n0",
                                "object": "n1",
                                "predicates": ["biolink:affects"],
                            },
                        },
                    },
                },
            }

            response = lookup(graph, query, bmt=bmt)
            results = response["message"]["results"]
            assert len(results) == 1

            # The returned edge carries a single qualifier whose value is the
            # JSON-encoded list string (scalar string, matching tier 1).
            kg_edges = response["message"]["knowledge_graph"]["edges"]
            aspect_vals = [
                q["qualifier_value"]
                for e in kg_edges.values()
                for q in e.get("qualifiers", [])
                if q["qualifier_type_id"] == "biolink:object_aspect_qualifier"
            ]
            assert aspect_vals == ['["activity","abundance"]']
