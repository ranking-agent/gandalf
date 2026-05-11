"""Tests for the degree-aware qedge planner."""

from gandalf.query_planner import (
    N,
    get_next_qedge,
    get_num_ids,
)
from tests.search_fixtures import graph  # noqa: F401


class TestGetNumIds:
    def test_unpinned_qnode_uses_N(self):
        qgraph = {
            "nodes": {
                "n0": {"ids": ["CHEBI:6801"]},
                "n1": {"categories": ["biolink:Gene"]},
            },
            "edges": {},
        }
        sizes = get_num_ids(qgraph)
        assert sizes["n0"] == 1
        assert sizes["n1"] == N

    def test_pinned_qnode_uses_degree_when_graph_given(self, graph):
        # Metformin has many outgoing edges in the fixture; degree > 1
        qgraph = {
            "nodes": {"n0": {"ids": ["CHEBI:6801"]}},
            "edges": {},
        }
        sizes_count = get_num_ids(qgraph)
        sizes_degree = get_num_ids(qgraph, graph=graph)
        assert sizes_count["n0"] == 1
        assert sizes_degree["n0"] > 1

    def test_unknown_id_is_skipped(self, graph):
        qgraph = {
            "nodes": {"n0": {"ids": ["NOT:areal_id"]}},
            "edges": {},
        }
        # Unknown id contributes 0 degree -> fall back to floor of 1.
        assert get_num_ids(qgraph, graph=graph)["n0"] == 1

    def test_mcq_member_ids_count_when_no_graph(self):
        qgraph = {
            "nodes": {
                "n0": {
                    "set_interpretation": "MANY",
                    "member_ids": ["CHEBI:6801", "CHEBI:17234"],
                },
            },
            "edges": {},
        }
        assert get_num_ids(qgraph)["n0"] == 2

    def test_idempotent_when_already_normalised(self):
        # If qnode["ids"] is already an int (pre-normalised by get_next_qedge),
        # get_num_ids should return it unchanged.
        qgraph = {
            "nodes": {"n0": {"ids": 7}},
            "edges": {},
        }
        assert get_num_ids(qgraph)["n0"] == 7
        # Graph-aware path also trusts the precomputed int.
        assert get_num_ids(qgraph, graph=None)["n0"] == 7


class TestPinnedSizeCap:
    def test_pinned_capped_below_N(self, graph):
        # Simulate a pinned qnode whose sum_degree exceeds N: ask for the
        # size with a fake high-N constant by passing many ids. Even with
        # a synthetic 2 * N member_ids count the result must remain < N
        # so the planner never confuses pinned with unpinned.
        qgraph = {
            "nodes": {
                "n0": {
                    "set_interpretation": "MANY",
                    "member_ids": ["X"] * (2 * N),
                }
            },
            "edges": {},
        }
        size = get_num_ids(qgraph)["n0"]
        assert size < N
        assert size == N - 1

    def test_high_degree_pinned_does_not_match_unpinned(self, graph):
        # Pinned-with-high-degree must remain strictly cheaper than truly
        # unpinned, otherwise get_next_qedge can pick a both-unpinned edge.
        # The fixture's CHEBI:6801 has the highest fwd degree.
        qgraph = {
            "nodes": {
                "n_pinned": {"ids": ["CHEBI:6801"]},
                "n_unpinned": {"categories": ["biolink:Gene"]},
            },
            "edges": {},
        }
        sizes = get_num_ids(qgraph, graph=graph)
        assert sizes["n_pinned"] < sizes["n_unpinned"]


class TestGetNextQedge:
    def test_picks_more_pinned_edge_first(self):
        # e0 has both endpoints pinned (id counts 1, 1).
        # e1 has only n0 pinned. Both-pinned edge should win.
        qgraph = {
            "nodes": {
                "n0": {"ids": ["CHEBI:6801"]},
                "n1": {"ids": ["MONDO:0005148"]},
                "n2": {"categories": ["biolink:Gene"]},
            },
            "edges": {
                "e0": {"subject": "n0", "object": "n1"},
                "e1": {"subject": "n0", "object": "n2"},
            },
        }
        qedge_id, _ = get_next_qedge(qgraph)
        assert qedge_id == "e0"

    def test_degree_aware_breaks_tie_between_equal_id_counts(self, graph):
        # Two qedges have the same pinned id counts but different degree
        # sums on the pinned side. The planner should prefer the smaller-
        # degree edge (cheaper traversal) over the higher-degree one.
        # In the fixture, MONDO:0005148 has higher reverse degree than
        # CHEBI:17234 (3 gene_associated_with_condition edges in vs. 1
        # participates_in edge in).
        qgraph = {
            "nodes": {
                "n_low": {"ids": ["CHEBI:17234"]},
                "n_high": {"ids": ["MONDO:0005148"]},
                "u0": {"categories": ["biolink:Gene"]},
                "u1": {"categories": ["biolink:Gene"]},
            },
            "edges": {
                "e_low": {"subject": "n_low", "object": "u0"},
                "e_high": {"subject": "n_high", "object": "u1"},
            },
        }
        # Without the graph, id counts tie 1-1, planner falls back to
        # whichever max() returns first -- not what we're testing here.
        # With the graph: the lower-degree edge has the cheaper effort
        # estimate and therefore the higher priority.
        qedge_id, _ = get_next_qedge(qgraph, graph)
        assert qedge_id == "e_low"

    def test_never_picks_both_unpinned_edge_when_pinned_edge_exists(self, graph):
        # 3-node chain with one pinned endpoint at each end and an
        # unpinned-unpinned edge bridging them. The planner must pick
        # one of the pinned-side edges first regardless of degree.
        qgraph = {
            "nodes": {
                "n0": {"ids": ["CHEBI:6801"]},  # high degree (~11)
                "n1": {"categories": ["biolink:Gene"]},
                "n2": {"categories": ["biolink:Gene"]},
                "n3": {"ids": ["MONDO:0005148"]},  # also fairly high degree
            },
            "edges": {
                "e_pinned_left": {"subject": "n0", "object": "n1"},
                "e_middle_unpinned": {"subject": "n1", "object": "n2"},
                "e_pinned_right": {"subject": "n2", "object": "n3"},
            },
        }
        qedge_id, _ = get_next_qedge(qgraph, graph)
        assert qedge_id != "e_middle_unpinned"

    def test_with_graph_does_not_change_obviously_correct_ordering(self, graph):
        # Single-pinned-vs-unpinned-only should always pick the pinned edge
        # whether or not we pass the graph.
        qgraph = {
            "nodes": {
                "n0": {"ids": ["CHEBI:6801"]},
                "u0": {"categories": ["biolink:Disease"]},
                "u1": {"categories": ["biolink:Gene"]},
            },
            "edges": {
                "e_pinned": {"subject": "n0", "object": "u0"},
                "e_unpinned": {"subject": "u0", "object": "u1"},
            },
        }
        qid_a, _ = get_next_qedge(qgraph)
        qid_b, _ = get_next_qedge(qgraph, graph)
        assert qid_a == qid_b == "e_pinned"
