"""Sanity tests: verify CSR storage has no crossed wires.

Each test picks a concrete node, selects a known predicate from the fixture
data, and asserts that the edge properties stored in the CSR match the
expected values from edges.jsonl.  The goal is confidence that the forward
CSR, reverse CSR, edge property store, and LMDB cold-path all reference the
correct data for every edge — no off-by-one index bugs, no swapped source
configs, no qualifier/publication misattribution.
"""

import os

import pytest

from gandalf.loader import build_graph_from_jsonl

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
NODES_FILE = os.path.join(FIXTURES_DIR, "nodes.jsonl")
EDGES_FILE = os.path.join(FIXTURES_DIR, "edges.jsonl")


@pytest.fixture
def graph():
    """Build a graph from test fixtures."""
    return build_graph_from_jsonl(EDGES_FILE, NODES_FILE)


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------


def _primary_source_id(sources):
    """Extract the primary_knowledge_source resource_id from a sources list."""
    for s in sources:
        if s["resource_role"] == "primary_knowledge_source":
            return s["resource_id"]
    return None


def _aggregator_source(sources):
    """Extract the gandalf aggregator entry from a sources list."""
    for s in sources:
        if s["resource_id"] == "infores:gandalf":
            return s
    return None


def _qualifier_map(qualifiers):
    """Convert a qualifiers list to a {type_id: value} dict for easy assertion."""
    return {q["qualifier_type_id"]: q["qualifier_value"] for q in qualifiers}


# ---------------------------------------------------------------
# Forward CSR edge property sanity
# ---------------------------------------------------------------


class TestForwardEdgeSanity:
    """Pick specific nodes, follow a known outgoing predicate, and verify
    that the hot-path (sources, qualifiers) and cold-path (attributes)
    properties stored in the CSR are exactly what the fixture JSONL defines.
    """

    @staticmethod
    def _get_publications(props):
        """Extract publication values from the TRAPI attributes list."""
        for attr in props.get("attributes", []):
            if attr["attribute_type_id"] == "biolink:publications":
                return attr["value"]
        return []

    def test_metformin_treats_diabetes_sources(self, graph):
        """CHEBI:6801 --treats--> MONDO:0005148 should have two treat edges
        with distinct primary sources (drugcentral and chembl)."""
        met = graph.node_id_to_idx["CHEBI:6801"]
        t2d = graph.node_id_to_idx["MONDO:0005148"]

        edges = graph.get_all_edges_between(
            met, t2d, predicate_filter=["biolink:treats"]
        )
        assert len(edges) == 2

        primary_ids = {_primary_source_id(props["sources"]) for _, props in edges}
        assert primary_ids == {"infores:drugcentral", "infores:chembl"}

        # Both should have the gandalf aggregator prepended
        for _, props in edges:
            agg = _aggregator_source(props["sources"])
            assert agg is not None
            assert agg["resource_role"] == "aggregator_knowledge_source"

    def test_metformin_treats_diabetes_publications(self, graph):
        """The two treats edges should carry distinct PMIDs from the fixture."""
        met = graph.node_id_to_idx["CHEBI:6801"]
        t2d = graph.node_id_to_idx["MONDO:0005148"]

        results = [
            (target, pred, props, eidx)
            for target, pred, props, eidx in graph.neighbors_with_properties(met)
            if target == t2d and pred == "biolink:treats"
        ]
        assert len(results) == 2

        # Cold-path lookup by forward edge index
        pubs = set()
        for _, _, _, eidx in results:
            full_props = graph.get_edge_properties_by_index(eidx)
            for pmid in self._get_publications(full_props):
                pubs.add(pmid)

        assert "PMID:12345678" in pubs  # drugcentral edge
        assert "PMID:55555555" in pubs  # chembl edge

    def test_metformin_affects_pparg_no_qualifiers(self, graph):
        """CHEBI:6801 --affects--> NCBIGene:5468 should have empty qualifiers
        and ctd as primary source."""
        met = graph.node_id_to_idx["CHEBI:6801"]
        pparg = graph.node_id_to_idx["NCBIGene:5468"]

        sources = graph.get_edge_property(met, pparg, "biolink:affects", "sources")
        assert _primary_source_id(sources) == "infores:ctd"

        qualifiers = graph.get_edge_property(
            met, pparg, "biolink:affects", "qualifiers"
        )
        assert qualifiers == []

        props = graph.get_all_edge_properties(met, pparg, "biolink:affects")
        pubs = self._get_publications(props)
        assert "PMID:23456789" in pubs

    def test_metformin_affects_insr_qualified_edges(self, graph):
        """CHEBI:6801 --affects--> NCBIGene:3643 has two edges with different
        qualifier combos: (activity, increased) from ctd and (abundance,
        decreased) from hetio."""
        met = graph.node_id_to_idx["CHEBI:6801"]
        insr = graph.node_id_to_idx["NCBIGene:3643"]

        edges = graph.get_all_edges_between(
            met, insr, predicate_filter=["biolink:affects"]
        )
        assert len(edges) == 2

        seen = set()
        for _, props in edges:
            qmap = _qualifier_map(props["qualifiers"])
            primary = _primary_source_id(props["sources"])

            aspect = qmap.get("biolink:object_aspect_qualifier")
            direction = qmap.get("biolink:object_direction_qualifier")
            seen.add((aspect, direction, primary))

        assert ("activity", "increased", "infores:ctd") in seen
        assert ("abundance", "decreased", "infores:hetio") in seen

    def test_metformin_affects_gck_qualifier(self, graph):
        """CHEBI:6801 --affects--> NCBIGene:2645 (GCK) should have
        activity/decreased qualifiers from ctd."""
        met = graph.node_id_to_idx["CHEBI:6801"]
        gck = graph.node_id_to_idx["NCBIGene:2645"]

        edges = graph.get_all_edges_between(
            met, gck, predicate_filter=["biolink:affects"]
        )
        assert len(edges) == 1

        _, props = edges[0]
        qmap = _qualifier_map(props["qualifiers"])
        assert qmap["biolink:object_aspect_qualifier"] == "activity"
        assert qmap["biolink:object_direction_qualifier"] == "decreased"
        assert _primary_source_id(props["sources"]) == "infores:ctd"

    def test_metformin_affects_tnf_qualifier(self, graph):
        """CHEBI:6801 --affects--> NCBIGene:7124 (TNF) should have
        abundance/increased qualifiers from ctd."""
        met = graph.node_id_to_idx["CHEBI:6801"]
        tnf = graph.node_id_to_idx["NCBIGene:7124"]

        edges = graph.get_all_edges_between(
            met, tnf, predicate_filter=["biolink:affects"]
        )
        assert len(edges) == 1

        _, props = edges[0]
        qmap = _qualifier_map(props["qualifiers"])
        assert qmap["biolink:object_aspect_qualifier"] == "abundance"
        assert qmap["biolink:object_direction_qualifier"] == "increased"
        assert _primary_source_id(props["sources"]) == "infores:ctd"

    def test_pparg_interacts_with_insr(self, graph):
        """NCBIGene:5468 --interacts_with--> NCBIGene:3643 should have intact
        as source and PMID:56789012."""
        pparg = graph.node_id_to_idx["NCBIGene:5468"]
        insr = graph.node_id_to_idx["NCBIGene:3643"]

        sources = graph.get_edge_property(
            pparg, insr, "biolink:interacts_with", "sources"
        )
        assert _primary_source_id(sources) == "infores:intact"

        props = graph.get_all_edge_properties(pparg, insr, "biolink:interacts_with")
        pubs = self._get_publications(props)
        assert "PMID:56789012" in pubs

    def test_diabetes_has_phenotype_hypoglycemia(self, graph):
        """MONDO:0005148 --has_phenotype--> HP:0001943 should have hpo as
        source and no publications attribute."""
        t2d = graph.node_id_to_idx["MONDO:0005148"]
        hypo = graph.node_id_to_idx["HP:0001943"]

        sources = graph.get_edge_property(t2d, hypo, "biolink:has_phenotype", "sources")
        assert _primary_source_id(sources) == "infores:hpo"

        props = graph.get_all_edge_properties(t2d, hypo, "biolink:has_phenotype")
        pubs = self._get_publications(props)
        assert pubs == []

    def test_glucose_participates_in_pathway(self, graph):
        """CHEBI:17234 --participates_in--> GO:0006006 should have reactome
        as source."""
        glucose = graph.node_id_to_idx["CHEBI:17234"]
        pathway = graph.node_id_to_idx["GO:0006006"]

        sources = graph.get_edge_property(
            glucose, pathway, "biolink:participates_in", "sources"
        )
        assert _primary_source_id(sources) == "infores:reactome"

    def test_subclass_of_chain(self, graph):
        """MONDO:0005148 --subclass_of--> MONDO:0005015 --subclass_of-->
        MONDO:0004995 should both exist with mondo as source."""
        t2d = graph.node_id_to_idx["MONDO:0005148"]
        dm = graph.node_id_to_idx["MONDO:0005015"]
        cvd = graph.node_id_to_idx["MONDO:0004995"]

        sources_1 = graph.get_edge_property(t2d, dm, "biolink:subclass_of", "sources")
        assert _primary_source_id(sources_1) == "infores:mondo"

        sources_2 = graph.get_edge_property(dm, cvd, "biolink:subclass_of", "sources")
        assert _primary_source_id(sources_2) == "infores:mondo"


# ---------------------------------------------------------------
# Reverse CSR sanity — incoming edges carry the right properties
# ---------------------------------------------------------------


class TestReverseEdgeSanity:
    """For a target node, query incoming edges via the reverse CSR and verify
    that the properties (fetched through the rev_to_fwd mapping) match what
    the forward CSR stores for the same edge."""

    def test_diabetes_incoming_treats_from_metformin(self, graph):
        """Type 2 Diabetes should have 2 incoming treats edges from Metformin
        with drugcentral and chembl as primary sources."""
        met = graph.node_id_to_idx["CHEBI:6801"]
        t2d = graph.node_id_to_idx["MONDO:0005148"]

        incoming = [
            (src, pred, props, eidx)
            for src, pred, props, eidx in graph.incoming_neighbors_with_properties(t2d)
            if src == met and pred == "biolink:treats"
        ]
        assert len(incoming) == 2

        primary_ids = {
            _primary_source_id(props["sources"]) for _, _, props, _ in incoming
        }
        assert primary_ids == {"infores:drugcentral", "infores:chembl"}

    def test_insr_incoming_affects_qualified(self, graph):
        """NCBIGene:3643 (INSR) should have 2 incoming affects edges from
        Metformin with different qualifier combos, looked up via reverse CSR."""
        met = graph.node_id_to_idx["CHEBI:6801"]
        insr = graph.node_id_to_idx["NCBIGene:3643"]

        incoming = [
            (src, pred, props, eidx)
            for src, pred, props, eidx in graph.incoming_neighbors_with_properties(insr)
            if src == met and pred == "biolink:affects"
        ]
        assert len(incoming) == 2

        seen = set()
        for _, _, props, _ in incoming:
            qmap = _qualifier_map(props["qualifiers"])
            aspect = qmap.get("biolink:object_aspect_qualifier")
            direction = qmap.get("biolink:object_direction_qualifier")
            seen.add((aspect, direction))

        assert ("activity", "increased") in seen
        assert ("abundance", "decreased") in seen

    def test_pathway_incoming_from_glucose_and_gck(self, graph):
        """GO:0006006 should have incoming participates_in from both
        CHEBI:17234 (Glucose) and NCBIGene:2645 (GCK)."""
        pathway = graph.node_id_to_idx["GO:0006006"]

        incoming = graph.incoming_neighbors_with_properties(
            pathway, predicate_filter=["biolink:participates_in"]
        )
        source_ids = {graph.idx_to_node_id[src] for src, _, _, _ in incoming}
        assert source_ids == {"CHEBI:17234", "NCBIGene:2645"}

    def test_hypoglycemia_incoming_has_phenotype(self, graph):
        """HP:0001943 should have exactly one incoming has_phenotype edge from
        MONDO:0005148 with hpo source."""
        hypo = graph.node_id_to_idx["HP:0001943"]

        incoming = graph.incoming_neighbors_with_properties(
            hypo, predicate_filter=["biolink:has_phenotype"]
        )
        assert len(incoming) == 1
        src, pred, props, _ = incoming[0]
        assert graph.idx_to_node_id[src] == "MONDO:0005148"
        assert _primary_source_id(props["sources"]) == "infores:hpo"


# ---------------------------------------------------------------
# Forward ↔ Reverse consistency
# ---------------------------------------------------------------


class TestForwardReverseConsistency:
    """Verify that for every forward edge, the reverse CSR contains a matching
    entry with identical properties — proving there are no crossed wires
    between the two CSR structures."""

    def test_every_forward_edge_appears_in_reverse(self, graph):
        """Walk every forward edge and confirm a matching incoming edge exists
        at the target node."""
        for src_idx in range(graph.num_nodes):
            for target, pred, fwd_props, fwd_eidx in graph.neighbors_with_properties(
                src_idx
            ):
                # The target's incoming edges must include one from src_idx
                # with the same predicate and same forward edge index.
                incoming = graph.incoming_neighbors_with_properties(target)
                match = [
                    (s, p, rprops, ridx)
                    for s, p, rprops, ridx in incoming
                    if s == src_idx and p == pred and ridx == fwd_eidx
                ]
                assert len(match) == 1, (
                    f"Forward edge ({graph.idx_to_node_id[src_idx]} "
                    f"--{pred}--> {graph.idx_to_node_id[target]}, "
                    f"fwd_idx={fwd_eidx}) not found in reverse CSR"
                )

    def test_rev_to_fwd_yields_identical_properties(self, graph):
        """For every reverse-CSR position, the properties fetched via
        rev_to_fwd must equal the properties at the forward position."""
        for node_idx in range(graph.num_nodes):
            for (
                src,
                pred,
                rev_props,
                fwd_eidx,
            ) in graph.incoming_neighbors_with_properties(node_idx):
                fwd_props = graph.edge_properties._get_props(fwd_eidx)
                assert rev_props["sources"] == fwd_props["sources"], (
                    f"Source mismatch for rev edge "
                    f"{graph.idx_to_node_id[src]} --{pred}--> "
                    f"{graph.idx_to_node_id[node_idx]}"
                )
                assert rev_props["qualifiers"] == fwd_props["qualifiers"], (
                    f"Qualifier mismatch for rev edge "
                    f"{graph.idx_to_node_id[src]} --{pred}--> "
                    f"{graph.idx_to_node_id[node_idx]}"
                )


# ---------------------------------------------------------------
# Edge index consistency
# ---------------------------------------------------------------


class TestEdgeIndexConsistency:
    """Verify that get_edge_properties_by_index returns the same hot-path
    data that neighbors_with_properties returns inline."""

    def test_by_index_matches_inline_for_all_edges(self, graph):
        """For every edge, properties from by_index should match the inline
        props returned by neighbors_with_properties."""
        for src_idx in range(graph.num_nodes):
            for target, pred, inline_props, fwd_eidx in graph.neighbors_with_properties(
                src_idx
            ):
                by_index = graph.get_edge_properties_by_index(fwd_eidx)

                assert (
                    by_index["sources"] == inline_props["sources"]
                ), f"sources mismatch at fwd_idx={fwd_eidx}"
                assert (
                    by_index["qualifiers"] == inline_props["qualifiers"]
                ), f"qualifiers mismatch at fwd_idx={fwd_eidx}"
                assert (
                    by_index["predicate"] == pred
                ), f"predicate mismatch at fwd_idx={fwd_eidx}"

    def test_duplicate_edges_have_distinct_indices(self, graph):
        """Duplicate (src, dst, pred) edges must have distinct forward edge
        indices so their properties stay separated."""
        met = graph.node_id_to_idx["CHEBI:6801"]
        t2d = graph.node_id_to_idx["MONDO:0005148"]

        treats = [
            (target, pred, props, eidx)
            for target, pred, props, eidx in graph.neighbors_with_properties(met)
            if target == t2d and pred == "biolink:treats"
        ]
        assert len(treats) == 2
        assert treats[0][3] != treats[1][3], "Duplicate edges share the same fwd index"

        # Verify the two indices actually hold different source data
        props_a = graph.get_edge_properties_by_index(treats[0][3])
        props_b = graph.get_edge_properties_by_index(treats[1][3])
        assert props_a["sources"] != props_b["sources"]


# ---------------------------------------------------------------
# Node property sanity
# ---------------------------------------------------------------


class TestNodePropertySanity:
    """Spot-check that node properties are stored against the correct index."""

    @staticmethod
    def _get_attr_value(attributes, attribute_type_id):
        """Find a TRAPI attribute by type and return its value."""
        for attr in attributes:
            if attr["original_attribute_name"] == attribute_type_id:
                return attr["value"]
        return None

    def test_metformin_properties(self, graph):
        idx = graph.node_id_to_idx["CHEBI:6801"]
        assert graph.get_node_property(idx, "name") == "Metformin"
        cats = graph.get_node_property(idx, "categories")
        assert "biolink:SmallMolecule" in cats
        assert "biolink:Drug" in cats
        attrs = graph.get_node_property(idx, "attributes")
        eqids = self._get_attr_value(attrs, "equivalent_identifiers")
        assert "DRUGBANK:DB00331" in eqids
        ic = self._get_attr_value(attrs, "information_content")
        assert ic == pytest.approx(85.5)

    def test_pparg_properties(self, graph):
        idx = graph.node_id_to_idx["NCBIGene:5468"]
        assert graph.get_node_property(idx, "name") == "PPARG"
        cats = graph.get_node_property(idx, "categories")
        assert "biolink:Gene" in cats
        attrs = graph.get_node_property(idx, "attributes")
        ic = self._get_attr_value(attrs, "information_content")
        assert ic == pytest.approx(92.3)

    def test_diabetes_properties(self, graph):
        idx = graph.node_id_to_idx["MONDO:0005148"]
        assert graph.get_node_property(idx, "name") == "Type 2 Diabetes"
        assert "biolink:Disease" in graph.get_node_property(idx, "categories")
        attrs = graph.get_node_property(idx, "attributes")
        ic = self._get_attr_value(attrs, "information_content")
        assert ic == pytest.approx(78.2)

    def test_node_id_roundtrips(self, graph):
        """Every node should survive idx → id → idx roundtrip."""
        for node_id, idx in graph.node_id_to_idx.items():
            assert graph.idx_to_node_id[idx] == node_id
            assert graph.get_node_idx(node_id) == idx
            assert graph.get_node_id(idx) == node_id
