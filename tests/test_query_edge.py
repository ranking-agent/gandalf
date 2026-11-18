"""``query_edge`` against a brute-force reading of what it should return.

The traversal reads many nodes' edges at once, in steps, and filters them as
arrays.  The oracle here does none of that: it walks every stored edge of the
graph and asks whether it matches, one edge at a time.  The two must agree on
every edge, direction and order, whatever the step size.
"""

import numpy as np
import pytest

from gandalf.graph import CSRGraph
from gandalf.search import query_edge as query_edge_module
from gandalf.search.edge_constraints import EdgeConstraints
from gandalf.search.edge_matches import EdgeMatches
from gandalf.search.query_edge import query_edge, query_subclass_edge
from tests.search_fixtures import graph  # noqa: F401

METFORMIN = "CHEBI:6801"
GLUCOSE = "CHEBI:17234"
PPARG = "NCBIGene:5468"
INSR = "NCBIGene:3643"
GCK = "NCBIGene:2645"
T2D = "MONDO:0005148"
DIABETES = "MONDO:0005015"


@pytest.fixture(params=["in_memory", "mapped"])
def any_graph(request, graph, tmp_path):
    """The fixture graph as built, and as saved and mapped back.

    Only the mapped graph has the node category index, so the two take
    different routes through the category check.
    """
    if request.param == "in_memory":
        assert graph.node_categories is None
        return graph
    graph.save_mmap(tmp_path / "graph")
    mapped = CSRGraph.load_mmap(tmp_path / "graph")
    assert mapped.node_categories is not None
    return mapped


def _stored_edges(graph):
    """Every edge of the graph as ``(subj, predicate, obj, fwd_edge_idx)``."""
    return [
        (subj, graph.id_to_predicate[int(graph.fwd_predicates[pos])], obj, pos)
        for subj in range(graph.num_nodes)
        for pos in range(int(graph.fwd_offsets[subj]), int(graph.fwd_offsets[subj + 1]))
        for obj in [int(graph.fwd_targets[pos])]
    ]


def _oracle(
    graph,
    start_idxes,
    end_idxes,
    start_categories=(),
    end_categories=(),
    allowed_predicates=(),
    constraints=EdgeConstraints(),
    inverse_predicates=None,
    pair_targets=None,
):
    """The matches, as a set of tuples, found by testing every stored edge."""

    def end_ok(node, pinned, categories, partner=None, partner_is_start=False):
        if pinned is not None:
            if node not in pinned:
                return False
        elif categories:
            node_categories = graph.get_all_node_properties(node).get("categories", [])
            if not any(category in node_categories for category in categories):
                return False
        return True

    def pair_ok(start, end):
        if pair_targets is None or start_idxes is None or end_idxes is None:
            return True
        return end in pair_targets.get(start, set())

    matches = set()
    for subj, predicate, obj, fwd in _stored_edges(graph):
        props = graph.edge_properties._get_props(fwd)
        if constraints and not constraints.permits(graph, props, fwd):
            continue
        if (
            (not allowed_predicates or predicate in allowed_predicates)
            and end_ok(subj, start_idxes, start_categories)
            and end_ok(obj, end_idxes, end_categories)
            and pair_ok(subj, obj)
        ):
            matches.add((subj, predicate, obj, False, fwd))
        if (
            inverse_predicates is not None
            and (not inverse_predicates or predicate in inverse_predicates)
            and end_ok(obj, start_idxes, start_categories)
            and end_ok(subj, end_idxes, end_categories)
            and pair_ok(obj, subj)
        ):
            matches.add((subj, predicate, obj, True, fwd))
    return matches


def _idxes(graph, ids):
    return None if ids is None else [graph.get_node_idx(i) for i in ids]


CASES = [
    # Forward from a pinned subject
    dict(start=[METFORMIN], end=None, allowed=["biolink:affects"]),
    dict(start=[METFORMIN], end=None, allowed=[]),
    dict(
        start=[METFORMIN],
        end=None,
        allowed=["biolink:affects"],
        end_cats=["biolink:Gene"],
    ),
    dict(
        start=[METFORMIN, GLUCOSE, PPARG],
        end=None,
        allowed=[],
        end_cats=["biolink:Disease"],
    ),
    # Backward from a pinned object
    dict(start=None, end=[T2D], allowed=["biolink:treats"]),
    dict(
        start=None, end=[T2D, INSR, DIABETES], allowed=[], start_cats=["biolink:Gene"]
    ),
    # Symmetric and inverse matching, restricted and wildcard
    dict(
        start=[INSR],
        end=None,
        allowed=["biolink:interacts_with"],
        inverse=["biolink:interacts_with"],
    ),
    dict(
        start=None,
        end=[INSR, PPARG],
        allowed=["biolink:interacts_with"],
        inverse=["biolink:interacts_with"],
    ),
    dict(start=[T2D], end=None, allowed=[], inverse=["biolink:treats"]),
    dict(start=[T2D, INSR, METFORMIN, GCK], end=None, allowed=[], inverse=[]),
    dict(
        start=None,
        end=[T2D, INSR, METFORMIN, GCK],
        allowed=[],
        inverse=[],
        start_cats=["biolink:Gene", "biolink:Disease"],
    ),
    # A node pinned twice must not double its matches
    dict(start=[METFORMIN, T2D, METFORMIN], end=None, allowed=[], inverse=[]),
    # Both ends pinned
    dict(start=[METFORMIN], end=[T2D, INSR, GCK], allowed=[]),
    dict(
        start=[METFORMIN, PPARG, INSR],
        end=[T2D, INSR, PPARG, METFORMIN],
        allowed=[],
        inverse=[],
    ),
    dict(
        start=[T2D, INSR],
        end=[METFORMIN, PPARG],
        allowed=[],
        inverse=["biolink:treats", "biolink:interacts_with"],
    ),
    dict(start=[METFORMIN], end=[], allowed=[]),
    dict(start=[], end=None, allowed=[]),
    # Pairs
    dict(
        start=[METFORMIN, PPARG],
        end=[T2D, INSR],
        allowed=[],
        inverse=[],
        pairs={METFORMIN: [T2D], PPARG: [INSR]},
    ),
    dict(
        start=[METFORMIN, PPARG, INSR],
        end=[T2D, INSR, PPARG],
        allowed=[],
        inverse=[],
        pairs={INSR: [PPARG], METFORMIN: [GCK]},
    ),
    # Edge constraints
    dict(
        start=[METFORMIN],
        end=None,
        allowed=["biolink:affects"],
        constraints={"qualifiers": [{"biolink:object_aspect_qualifier": "activity"}]},
    ),
    dict(
        start=[METFORMIN],
        end=None,
        allowed=[],
        constraints={"sources": {"behavior": "DENY", "values": ["infores:ctd"]}},
    ),
    dict(
        start=None,
        end=[T2D],
        allowed=[],
        constraints={
            "sources": {
                "behavior": "ALLOW",
                "values": ["infores:chembl", "infores:disgenet"],
            },
            "knowledge_level": {"behavior": "ALLOW", "values": ["knowledge_assertion"]},
            "agent_type": {"behavior": "DENY", "values": ["text_mining_agent"]},
        },
    ),
    dict(
        start=None,
        end=[T2D],
        allowed=[],
        constraints={
            "knowledge_level": {"behavior": "DENY", "values": ["knowledge_assertion"]}
        },
    ),
    dict(
        start=[METFORMIN],
        end=None,
        allowed=[],
        inverse=[],
        constraints={
            "attributes": [
                {
                    "id": "biolink:publications",
                    "name": "p",
                    "operator": "==",
                    "value": "PMID:23456789",
                }
            ]
        },
    ),
]


@pytest.mark.parametrize("step_edges", [1, 3, 262_144])
@pytest.mark.parametrize("case", CASES, ids=range(len(CASES)))
def test_matches_agree_with_brute_force(any_graph, monkeypatch, case, step_edges):
    graph = any_graph
    monkeypatch.setattr(query_edge_module, "TRAVERSAL_CHUNK_EDGES", step_edges)
    constraints = EdgeConstraints.parse({"constraints": case.get("constraints")})
    pairs = case.get("pairs")
    pair_targets = pairs and {
        graph.get_node_idx(subj): {graph.get_node_idx(obj) for obj in objs}
        for subj, objs in pairs.items()
    }
    args = (
        graph,
        _idxes(graph, case["start"]),
        _idxes(graph, case["end"]),
        case.get("start_cats", []),
        case.get("end_cats", []),
        case["allowed"],
        constraints,
    )
    kwargs = dict(inverse_predicates=case.get("inverse"), pair_targets=pair_targets)

    matches = query_edge(*args, **kwargs)

    assert isinstance(matches, EdgeMatches)
    found = matches.to_tuples()
    assert len(found) == len(set(found)), "a match was reported twice"
    assert set(found) == _oracle(
        graph,
        args[1],
        args[2],
        start_categories=args[3],
        end_categories=args[4],
        allowed_predicates=args[5],
        constraints=constraints,
        **kwargs,
    )


@pytest.mark.parametrize("case", CASES, ids=range(len(CASES)))
def test_step_size_changes_nothing(graph, monkeypatch, case):
    """Matches come in the same order however the pinned nodes are batched."""
    constraints = EdgeConstraints.parse({"constraints": case.get("constraints")})

    def run(step_edges):
        monkeypatch.setattr(query_edge_module, "TRAVERSAL_CHUNK_EDGES", step_edges)
        return query_edge(
            graph,
            _idxes(graph, case["start"]),
            _idxes(graph, case["end"]),
            case.get("start_cats", []),
            case.get("end_cats", []),
            case["allowed"],
            constraints,
            inverse_predicates=case.get("inverse"),
        ).to_tuples()

    assert run(1) == run(4) == run(262_144)


def test_matches_come_node_by_node_direct_before_inverse(graph):
    """Each pinned node's direct matches, then its inverse ones, in pinned order."""
    pinned = _idxes(graph, [INSR, T2D, PPARG])
    matches = query_edge(
        graph, pinned, None, [], [], [], EdgeConstraints(), inverse_predicates=[]
    )
    owners = [
        (pinned.index(obj if via_inverse else subj), via_inverse)
        for subj, _pred, obj, via_inverse, _fwd in matches.to_tuples()
    ]
    assert owners == sorted(owners)
    assert {owner for owner, _ in owners} == {0, 1, 2}


def test_pinned_node_with_attribute_constraints(graph):
    """Constraints on a pinned node filter it (and do not raise)."""
    t2d = graph.get_node_idx(T2D)
    ic_value = next(
        attribute["value"]
        for attribute in graph.get_all_node_properties(t2d)["attributes"]
        if attribute["attribute_type_id"] == "biolink:information_content"
    )

    def run(threshold):
        return query_edge(
            graph,
            None,
            [t2d],
            [],
            [],
            ["biolink:treats"],
            EdgeConstraints(),
            end_node_constraints=[
                {
                    "id": "biolink:information_content",
                    "name": "information content",
                    "operator": ">",
                    "value": threshold,
                }
            ],
        )

    assert len(run(ic_value - 1)) == 2
    assert len(run(ic_value + 1)) == 0


def test_subclass_edge_matches(graph):
    """A superclass matches itself (no real edge) and its descendants to *depth*."""
    t2d, diabetes, top = _idxes(graph, [T2D, DIABETES, "MONDO:0004995"])

    def run(depth):
        matches = query_subclass_edge(graph, None, [top], depth)
        assert set(matches.predicates[c] for c in matches.pred.tolist()) == {
            "biolink:subclass_of"
        }
        assert not matches.via_inverse.any()
        return list(zip(matches.subj.tolist(), matches.obj.tolist()))

    assert run(0) == [(top, top)]
    assert run(1) == [(top, top), (diabetes, top)]
    assert run(2) == [(top, top), (diabetes, top), (t2d, top)]
    self_match = query_subclass_edge(graph, None, np.array([top]), 0)
    assert self_match.fwd_edge_idx.tolist() == [-1]
    assert len(query_subclass_edge(graph, None, None, 2)) == 0
