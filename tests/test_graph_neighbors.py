"""Predicate-filter behavior of the CSR neighbor iterators.

These tests pin down the contract that ``neighbors_with_properties``,
``incoming_neighbors_with_properties``, and ``neighbors_filtered_by_targets``
share: when ``predicate_filter`` is supplied they only return edges whose
predicate is in the set, and they short-circuit to an empty result when the
filter contains no predicate that actually exists in this graph.
"""

from tests.search_fixtures import graph  # noqa: F401


def _idx(graph, node_id):
    idx = graph.get_node_idx(node_id)
    assert idx is not None, f"missing node {node_id}"
    return idx


class TestNeighborsWithProperties:
    def test_filter_keeps_only_requested_predicate(self, graph):
        metformin = _idx(graph, "CHEBI:6801")
        results = graph.neighbors_with_properties(
            metformin, predicate_filter={"biolink:treats"}
        )
        preds = {pred for _, pred, _, _ in results}
        assert preds == {"biolink:treats"}
        # Should hit MONDO:0005148 and MONDO:0005015 via treats
        target_ids = {graph.get_node_id(t) for t, _, _, _ in results}
        assert "MONDO:0005148" in target_ids
        assert "MONDO:0005015" in target_ids

    def test_unknown_predicate_returns_empty(self, graph):
        metformin = _idx(graph, "CHEBI:6801")
        results = graph.neighbors_with_properties(
            metformin, predicate_filter={"biolink:no_such_predicate"}
        )
        assert results == []

    def test_filter_none_keeps_everything(self, graph):
        metformin = _idx(graph, "CHEBI:6801")
        unfiltered = graph.neighbors_with_properties(metformin)
        with_filter = graph.neighbors_with_properties(
            metformin,
            predicate_filter={
                "biolink:treats",
                "biolink:affects",
                "biolink:ameliorates_condition",
                "biolink:preventative_for_condition",
            },
        )
        assert len(unfiltered) == len(with_filter)


class TestIncomingNeighborsWithProperties:
    def test_filter_keeps_only_requested_predicate(self, graph):
        t2d = _idx(graph, "MONDO:0005148")
        results = graph.incoming_neighbors_with_properties(
            t2d, predicate_filter={"biolink:gene_associated_with_condition"}
        )
        preds = {pred for _, pred, _, _ in results}
        assert preds == {"biolink:gene_associated_with_condition"}
        # Three gene edges to T2D in fixtures: PPARG, INSR, GCK
        src_ids = {graph.get_node_id(s) for s, _, _, _ in results}
        assert {"NCBIGene:5468", "NCBIGene:3643", "NCBIGene:2645"} <= src_ids

    def test_unknown_predicate_returns_empty(self, graph):
        t2d = _idx(graph, "MONDO:0005148")
        results = graph.incoming_neighbors_with_properties(
            t2d, predicate_filter={"biolink:no_such_predicate"}
        )
        assert results == []


class TestNeighborsFilteredByTargets:
    def test_target_and_predicate_filter_combined(self, graph):
        metformin = _idx(graph, "CHEBI:6801")
        t2d = _idx(graph, "MONDO:0005148")
        results = graph.neighbors_filtered_by_targets(
            metformin, {t2d}, predicate_filter={"biolink:treats"}
        )
        # Two treats edges from Metformin -> T2D (drugcentral + chembl)
        assert len(results) == 2
        for tgt, pred, _, _ in results:
            assert tgt == t2d
            assert pred == "biolink:treats"

    def test_target_filter_skips_non_target_edges(self, graph):
        metformin = _idx(graph, "CHEBI:6801")
        ppar_g = _idx(graph, "NCBIGene:5468")
        results = graph.neighbors_filtered_by_targets(
            metformin, {ppar_g}, predicate_filter={"biolink:affects"}
        )
        assert all(t == ppar_g for t, _, _, _ in results)
        assert all(pred == "biolink:affects" for _, pred, _, _ in results)

    def test_unknown_predicate_returns_empty(self, graph):
        metformin = _idx(graph, "CHEBI:6801")
        t2d = _idx(graph, "MONDO:0005148")
        results = graph.neighbors_filtered_by_targets(
            metformin, {t2d}, predicate_filter={"biolink:no_such_predicate"}
        )
        assert results == []
