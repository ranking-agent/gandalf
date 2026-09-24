#!/usr/bin/env python3
"""Phase 0 check for the single-path fast path in ``_build_response``.

For each query this reports:

* the result shape: how many results come from a single path, how many carry
  an inferred subclass edge (a subclass match that is not an identity), and
  how many the fast path therefore handles;
* whether the response is byte-identical with the fast path off and on
  (the orjson-serialized message, plus the key order of the knowledge
  graph's nodes and edges and of the auxiliary graphs);
* ``_build_response`` wall-clock time, off vs on, with runs alternating so
  drift affects both sides alike.

Usage::

    python scripts/benchmarks/fast_path_ab.py --graph GRAPH_DIR \\
        --queries QUERIES.json [--only xl_ xxl_] [--repeat 3] [--out out.json]

Queries under ``--min-results`` (default 1000) are checked for identity but
not timed.  Exits non-zero if any response differs.
"""

import argparse
import gc
import importlib
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import orjson

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_lookup import load_queries, lookup_kwargs  # noqa: E402

# ``gandalf.search`` re-exports a ``lookup`` function under the module's name.
L = importlib.import_module("gandalf.search.lookup")

_orig_build = L._build_response
_orig_group_rows = L._group_rows
_last: dict = {}


def _group_rows(keys):
    groups = _orig_group_rows(keys)
    _last["groups"] = groups
    return groups


def _shape(path_data, query_graph, groups) -> dict:
    """Group sizes and inferred-edge share of one response."""
    nodes = path_data.paths_nodes
    inferred = np.zeros(len(nodes), dtype=bool)
    for col, qedge_id in path_data.col_to_qedge.items():
        qedge = query_graph["edges"][qedge_id]
        if qedge.get("_subclass"):
            subj = path_data.qnode_to_col[qedge["subject"]]
            obj = path_data.qnode_to_col[qedge["object"]]
            inferred |= nodes[:, subj] != nodes[:, obj]
    sizes = np.fromiter((len(g) for g in groups), dtype=np.int64, count=len(groups))
    group_inferred = np.fromiter(
        (inferred[g].any() for g in groups), dtype=bool, count=len(groups)
    )
    n = max(len(groups), 1)
    return {
        "paths": int(sizes.sum()),
        "single_path": float((sizes == 1).sum() / n),
        "inferred": float(group_inferred.sum() / n),
        "fast_eligible": float(((sizes == 1) & ~group_inferred).sum() / n),
    }


def _build(graph, response, path_data, query_graph, *args, **kwargs):
    L._group_rows = _group_rows
    try:
        t0 = time.perf_counter()
        out = _orig_build(graph, response, path_data, query_graph, *args, **kwargs)
        _last["build_s"] = time.perf_counter() - t0
    finally:
        L._group_rows = _orig_group_rows
    if _last.get("want_shape"):
        _last["shape"] = _shape(path_data, query_graph, _last["groups"])
    _last.pop("groups", None)
    return out


L._build_response = _build


def _default(obj):
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    raise TypeError(type(obj))


def _canonical(response) -> tuple:
    msg = response["message"]
    kg = msg.get("knowledge_graph") or {}
    return (
        orjson.dumps(msg, default=_default),
        list(kg.get("nodes") or {}),
        list(kg.get("edges") or {}),
        list(msg.get("auxiliary_graphs") or {}),
    )


def _run(graph, bmt, body, fast: bool, shape: bool = False):
    L._SINGLE_PATH_FAST = fast
    _last["want_shape"] = shape
    _last.pop("build_s", None)
    gc.collect()
    response = L.lookup(graph, body, bmt=bmt, **lookup_kwargs(body))
    return response, _last.get("build_s", 0.0)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--graph", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--only", nargs="*", help="keep queries whose name has any")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--min-results", type=int, default=1000)
    parser.add_argument("--out", help="write the rows as JSON here")
    args = parser.parse_args(argv)

    from gandalf.biolink import make_toolkit
    from gandalf.graph import CSRGraph

    bmt = make_toolkit()
    graph = CSRGraph.load_mmap(Path(args.graph))
    queries = load_queries(Path(args.queries))
    if args.only:
        queries = [q for q in queries if any(s in q["name"] for s in args.only)]

    rows, all_same = [], True
    for query in queries:
        body = query["body"]
        ref, _ = _run(graph, bmt, body, fast=False, shape=True)
        shape = _last.pop("shape", None)
        new, _ = _run(graph, bmt, body, fast=True)
        results = len(ref["message"]["results"])
        labels = ["message", "kg node order", "kg edge order", "aux graph order"]
        diffs = [
            label
            for label, a, b in zip(labels, _canonical(ref), _canonical(new))
            if a != b
        ]
        del ref, new
        all_same &= not diffs
        row = {"name": query["name"], "results": results, "diffs": diffs}
        if shape:
            row.update(shape)
        line = f"{query['name']:40s} {results:>10,} results  " + (
            "same" if not diffs else "DIFF: " + ", ".join(diffs)
        )
        if shape:
            line += (
                f"  single-path {shape['single_path']:4.0%}"
                f"  inferred {shape['inferred']:4.0%}"
                f"  fast {shape['fast_eligible']:4.0%}"
            )
        if results >= args.min_results:
            off, on = [], []
            for _ in range(args.repeat):
                off.append(_run(graph, bmt, body, fast=False)[1])
                on.append(_run(graph, bmt, body, fast=True)[1])
            row["build_off_s"] = statistics.median(off)
            row["build_on_s"] = statistics.median(on)
            line += (
                f"  build {row['build_off_s']:.2f}s -> {row['build_on_s']:.2f}s"
                f" ({row['build_off_s'] / row['build_on_s']:.2f}x)"
            )
        print(line, flush=True)
        rows.append(row)

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1))
    print("all responses identical" if all_same else "SOME RESPONSES DIFFER")
    return 0 if all_same else 1


if __name__ == "__main__":
    sys.exit(main())
