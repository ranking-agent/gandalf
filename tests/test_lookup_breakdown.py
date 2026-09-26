"""scripts/benchmarks/lookup_breakdown.py times a copy of _build_response
compiled from its source.  That copy must build exactly what the real one
builds, and a whole breakdown must run -- so a change to _build_response
that the script cannot follow fails here rather than on a benchmark run.
"""

import importlib.util
import json
import os

import orjson
import pytest

from tests.search_fixtures import graph  # noqa: F401

from gandalf.search import lookup

_SCRIPT = os.path.join(
    os.path.dirname(__file__), os.pardir, "scripts", "benchmarks", "lookup_breakdown.py"
)

METFORMIN = "CHEBI:6801"
T2D = "MONDO:0005148"
DIABETES = "MONDO:0005015"
HYPOGLYCEMIA = "HP:0001943"


def _query(nodes: dict, edges: dict) -> dict:
    return {"message": {"query_graph": {"nodes": nodes, "edges": edges}}}


def _edge(subject: str, obj: str, predicate: str) -> dict:
    return {"subject": subject, "object": obj, "predicates": [predicate]}


QUERIES = {
    # Single-path and multi-path results over two edge columns.
    "two_hop_mixed": _query(
        {
            "n0": {"ids": [METFORMIN]},
            "n1": {"categories": ["biolink:Gene"]},
            "n2": {"ids": [T2D]},
        },
        {
            "e0": _edge("n0", "n1", "biolink:affects"),
            "e1": _edge("n1", "n2", "biolink:gene_associated_with_condition"),
        },
    ),
    # Subclass inference: a direct edge and an inferred one.
    "direct_and_inferred": _query(
        {"n0": {"ids": [METFORMIN]}, "n1": {"ids": [DIABETES]}},
        {"e0": _edge("n0", "n1", "biolink:treats")},
    ),
    # A single-path result through a subclass edge.
    "inferred_only": _query(
        {"n0": {"ids": [DIABETES]}, "n1": {"ids": [HYPOGLYCEMIA]}},
        {"e0": _edge("n0", "n1", "biolink:has_phenotype")},
    ),
}


@pytest.fixture(scope="module")
def breakdown():
    spec = importlib.util.spec_from_file_location("lookup_breakdown", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _serialized(response: dict) -> tuple:
    message = response["message"]
    return (
        orjson.dumps(message),
        list(message["knowledge_graph"]["nodes"]),
        list(message["knowledge_graph"]["edges"]),
        list(message.get("auxiliary_graphs", {})),
    )


@pytest.mark.parametrize("dehydrated", [False, True])
@pytest.mark.parametrize("name", list(QUERIES))
def test_timed_copy_builds_the_same_response(
    breakdown, monkeypatch, graph, bmt, name, dehydrated
):
    query = QUERIES[name]
    reference = lookup(graph, query, bmt=bmt, subclass=True, dehydrated=dehydrated)
    build = breakdown.InstrumentedBuild()
    with build.installed() as timed_build:
        monkeypatch.setattr(breakdown.L, "_build_response", timed_build)
        timed = lookup(graph, query, bmt=bmt, subclass=True, dehydrated=dehydrated)
    assert _serialized(timed) == _serialized(reference)

    # The per-result loop ran once, and its first statement once per result.
    results = len(reference["message"]["results"])
    assert results
    loop = next(i for i, s in enumerate(build.stmts) if s["main_loop"])
    first = next(i for i, s in enumerate(build.stmts) if s["parent"] == loop)
    assert build.cnt[loop] == 1
    assert build.cnt[first] == results


def test_breakdown_of_one_query(breakdown, graph, bmt):
    cal = breakdown.calibrate(n=10_000, rounds=2)
    query = {"name": "two_hop_mixed", "body": QUERIES["two_hop_mixed"]}
    rec = breakdown.breakdown(
        graph,
        bmt,
        query,
        breakdown.InstrumentedBuild(),
        cal,
        warmup=0,
        repeat=1,
        sample=50,
        trace=True,
        log=lambda message: None,
    )

    results = len(lookup(graph, query["body"], bmt=bmt)["message"]["results"])
    assert rec["results"] == results
    t = rec["time"]
    assert t["row_fetch_groups"] == results
    assert next(s for s in t["statements"] if s["main_loop"])["count"] == 1
    assert all(s["seconds"] >= 0 for s in t["statements"])
    assert t["helpers"]["_full_edge"]["calls"] > 0
    bindings = rec["memory"]["bindings"]
    assert bindings["distinct_ids"] <= bindings["single_id_objects"]
    assert bindings["single_id"] <= bindings["bindings"]
    assert rec["memory"]["tracemalloc"]["sites"]

    lines: list[str] = []
    breakdown.print_report(rec, out=lines.append)
    assert any("per-result loop" in line for line in lines)
    json.dumps(breakdown._json_ready(rec))
