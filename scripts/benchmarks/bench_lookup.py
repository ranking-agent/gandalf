#!/usr/bin/env python3
"""In-process ``lookup()`` benchmark, for tracking performance across commits.

Runs a set of TRAPI queries directly against a loaded graph (no server, no
HTTP, no serialization), times each one, breaks the time down by pipeline
stage, and fingerprints the results so a speed-up that changes the answer
is caught rather than celebrated.

Typical workflow::

    # 1. Baseline, on the commit before a change
    python scripts/benchmarks/bench_lookup.py run \\
        --graph /data/graph_mmap --queries my_big_queries.json \\
        --out bench_results/baseline.json

    # 2. After the change, compare against it
    python scripts/benchmarks/bench_lookup.py run \\
        --graph /data/graph_mmap --queries my_big_queries.json \\
        --out bench_results/after.json --compare bench_results/baseline.json

    # Or compare two saved runs later
    python scripts/benchmarks/bench_lookup.py compare \\
        bench_results/baseline.json bench_results/after.json

Without a real graph, ``--synthetic medium`` builds (once, then caches) a
deterministic synthetic graph and uses its built-in query set; see
``synthetic_graph.py``.

The query file is either one TRAPI request body or a JSON list of them --
the same JSON you would POST to ``/query``.  An optional top-level ``name``
key labels a query in the report; otherwise it is ``<file>#<index>``.
Parameters are read from the body's ``parameters`` exactly as the server
reads them (``subclass``, ``subclass_depth``, ``dehydrated``,
``filter_config``).

Measurement:
    * ``--warmup`` untimed runs first, so the OS page cache and lazily built
      structures are warm.  Use ``--warmup 0`` to see cold behaviour.
    * ``--repeat`` timed runs with the profiler **off**; the report shows the
      median and min.  The profiler itself costs time, so it never runs
      during a timed run.
    * One extra run with ``profile=True`` supplies the per-stage breakdown
      and LMDB call totals.
    * With ``--memory``, one more run under ``tracemalloc`` records the peak
      Python + NumPy allocation during the query.  It is opt-in because
      tracing slows the run several-fold.  (Process RSS is no use here:
      memory freed by one run stays with the process and hides the next
      run's peak.)

The BMT toolkit is built once up front and shared, as the server does, so
BMT initialization never counts toward a query.
"""

import argparse
import gc
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Optional

import orjson

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_HERE))

#: Where ``--synthetic`` graphs are cached between runs.
DEFAULT_CACHE_DIR = _REPO / "bench_results" / "cache"

#: Stages reported as columns, as paths into the profiler tree.  A path
#: matches every node with that name sequence, and their durations are
#: summed (e.g. one ``qedge`` stage per query edge).
STAGE_COLUMNS = [
    ("subclass_rewrite", ("subclass_rewrite",)),
    ("edge_queries", ("qedge",)),
    ("reconstruct", ("reconstruct",)),
    ("  joins", ("reconstruct", "join")),
    ("  node_cache", ("reconstruct", "node_cache_build")),
    ("build_response", ("build_response",)),
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_queries(path: Path) -> list[dict]:
    """Read a query file into ``[{"name": ..., "body": ...}, ...]``."""
    with open(path) as fh:
        data = json.load(fh)
    bodies = data if isinstance(data, list) else [data]
    return [_named(body, f"{path.name}#{i}") for i, body in enumerate(bodies)]


def _named(body: dict, default_name: str) -> dict:
    """Split the optional ``name`` label (and generator metadata) off a body.

    ``generate_queries.py`` records how each query was made under a
    ``generated`` key; like ``name``, it is not part of the request.
    """
    body = dict(body)
    name = body.pop("name", None) or default_name
    body.pop("generated", None)
    return {"name": name, "body": body}


def resolve_graph(args) -> tuple[Path, list[dict]]:
    """Find (or build) the graph directory and the queries to run on it."""
    if args.synthetic:
        from synthetic_graph import cached_graph_dir, synthetic_queries

        graph_dir = cached_graph_dir(
            args.synthetic, args.seed, Path(args.cache_dir), args.rebuild
        )
        queries = (
            load_queries(Path(args.queries))
            if args.queries
            else [_named(q, q["name"]) for q in synthetic_queries()]
        )
        return graph_dir, queries

    if not args.graph or not args.queries:
        sys.exit("run: pass --graph and --queries, or --synthetic SCALE")
    return Path(args.graph), load_queries(Path(args.queries))


# ---------------------------------------------------------------------------
# Measuring
# ---------------------------------------------------------------------------


def lookup_kwargs(body: dict) -> dict:
    """Map a request's ``parameters`` onto ``lookup()`` arguments, as the server does."""
    params = body.get("parameters") or {}
    return {
        "subclass": params.get("subclass", True),
        "subclass_depth": params.get("subclass_depth", 1),
        "dehydrated": params.get("dehydrated"),
        "filter_config": params.get("filter_config"),
    }


def run_once(graph, body: dict, bmt, profile: bool = False) -> tuple[float, dict]:
    """Run one lookup and return ``(wall_ms, response)``.

    Collects garbage first (outside the timer) so one run's leftovers are not
    billed to the next; ``lookup`` itself disables GC while it runs.
    """
    from gandalf.search.lookup import lookup

    gc.collect()
    t0 = time.perf_counter()
    response = lookup(graph, body, bmt=bmt, profile=profile, **lookup_kwargs(body))
    return (time.perf_counter() - t0) * 1000.0, response


def _serialize_default(obj):
    """orjson ``default`` matching the server's (it serializes sets as lists)."""
    if isinstance(obj, set):
        return list(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def serialize(response: dict) -> tuple[float, int]:
    """Serialize a response as the server does; return ``(ms, bytes)``."""
    t0 = time.perf_counter()
    data = orjson.dumps(
        response, default=_serialize_default, option=orjson.OPT_SERIALIZE_NUMPY
    )
    return (time.perf_counter() - t0) * 1000.0, len(data)


def profile_tree(response: dict) -> Optional[dict]:
    """Pull the profiler's ``ProfileSummary`` tree out of the response logs."""
    for entry in response.get("logs") or []:
        if entry.get("code") == "ProfileSummary":
            return json.loads(entry["message"])
    return None


def stage_ms(tree: dict, path: tuple) -> float:
    """Sum the durations of every stage matching *path* below the root."""

    def walk(node: dict, remaining: tuple) -> float:
        total = 0.0
        for child in node.get("children", []):
            if child["name"] != remaining[0]:
                continue
            if len(remaining) == 1:
                total += child.get("duration_ms") or 0.0
            else:
                total += walk(child, remaining[1:])
        return total

    return walk(tree, path)


def fingerprint(response: dict) -> str:
    """A content hash of the results that ignores ordering and generated IDs.

    Knowledge-graph edge IDs are not stable (edges without an original ID
    get a random one, and inferred edges are named after their parts), so
    each bound edge is described by its content instead: subject, predicate,
    object, sources, qualifiers, and -- for inferred edges -- the
    descriptions of the edges in its support graph.  Results are then
    sorted, so two responses fingerprint the same iff they bind the same
    nodes to the same edges.
    """
    message = response["message"]
    kg_edges = message["knowledge_graph"]["edges"]
    aux_graphs = message.get("auxiliary_graphs") or {}

    def describe(edge_id: str) -> tuple:
        edge = kg_edges[edge_id]
        support: tuple = ()
        for attr in edge.get("attributes") or []:
            if attr.get("attribute_type_id") == "biolink:support_graphs":
                support = tuple(
                    sorted(
                        describe(e)
                        for aux_id in attr["value"]
                        for e in aux_graphs[aux_id]["edges"]
                    )
                )
        return (
            edge["subject"],
            edge["predicate"],
            edge["object"],
            tuple(sorted(s["resource_id"] for s in edge.get("sources") or [])),
            tuple(
                sorted(
                    (q["qualifier_type_id"], q["qualifier_value"])
                    for q in edge.get("qualifiers") or []
                )
            ),
            support,
        )

    canon = []
    for result in message["results"]:
        nodes = tuple(
            sorted(
                (qnode, tuple(sorted(binding["ids"])))
                for qnode, binding in result["node_bindings"].items()
            )
        )
        edges = tuple(
            sorted(
                (qedge, tuple(sorted(describe(e) for e in binding["ids"])))
                for analysis in result.get("analyses") or []
                for qedge, binding in analysis["edge_bindings"].items()
            )
        )
        canon.append((nodes, edges))
    canon.sort()
    return hashlib.sha256(json.dumps(canon).encode()).hexdigest()[:16]


def peak_alloc_mb(graph, body: dict, bmt) -> float:
    """Peak traced allocation (MB) over one lookup, measured with tracemalloc."""
    gc.collect()
    tracemalloc.start()
    try:
        run_once(graph, body, bmt)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak / 2**20


def _print_run(label: str, ms: float) -> None:
    """Report one finished run inline, so a slow query shows progress."""
    print(f" {label}{_fmt_ms(ms)}", end="", flush=True)


def bench_query(
    graph,
    query: dict,
    bmt,
    warmup: int,
    repeat: int,
    memory: bool = False,
    on_run=_print_run,
) -> dict:
    """Benchmark one query: warmup, timed runs, one profiled run, and
    optionally one memory-traced run.

    That is ``warmup + repeat + 1`` runs (``+ 1`` more, several times
    slower, with *memory*); ``on_run(label, ms)`` is called as each finishes.
    """
    body = query["body"]
    # Only one response is ever alive at a time: a run starts only after the
    # previous run's response is released.  Holding the last one while the
    # next is built doubled peak memory, which on multi-million-result
    # queries pushed the machine into swap and made each run slower than
    # the one before it.
    for _ in range(warmup):
        ms = run_once(graph, body, bmt)[0]
        on_run("warmup ", ms)

    runs_ms = []
    response: dict = {}
    for _ in range(repeat):
        response = {}
        ms, response = run_once(graph, body, bmt)
        runs_ms.append(ms)
        on_run("", ms)

    # Serializing is the rest of what the server does with a response; time
    # it too, since a change can move cost between the two.
    serialize_ms, response_bytes = serialize(response)
    on_run("serialize ", serialize_ms)

    message = response["message"]
    record: dict[str, Any] = {
        "name": query["name"],
        "runs_ms": runs_ms,
        "median_ms": statistics.median(runs_ms),
        "min_ms": min(runs_ms),
        "results": len(message["results"]),
        "kg_nodes": len(message["knowledge_graph"]["nodes"]),
        "kg_edges": len(message["knowledge_graph"]["edges"]),
        "fingerprint": fingerprint(response),
        "serialize_ms": serialize_ms,
        "response_bytes": response_bytes,
    }
    del response, message

    profiled_ms, profiled = run_once(graph, body, bmt, profile=True)
    on_run("profiled ", profiled_ms)
    tree = profile_tree(profiled) or {}
    del profiled
    record["profiled_ms"] = profiled_ms
    record["peak_alloc_mb"] = None
    if memory:
        t0 = time.perf_counter()
        record["peak_alloc_mb"] = peak_alloc_mb(graph, body, bmt)
        on_run("memory ", (time.perf_counter() - t0) * 1000.0)
    record["stages_ms"] = {label: stage_ms(tree, path) for label, path in STAGE_COLUMNS}
    record["num_paths"] = tree.get("metrics", {}).get("num_paths")
    lmdb = tree.get("lmdb") or {}
    record["lmdb"] = {
        "calls": lmdb.get("calls", 0),
        "keys": lmdb.get("total_keys", 0),
        "ms": lmdb.get("total_ms", 0.0),
    }
    return record


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=_REPO, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def environment(graph_dir: Path, graph, label: str) -> dict:
    """Describe what was measured, so saved runs can be told apart later."""
    return {
        "label": label,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_commit": _git("rev-parse", "--short", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "python": platform.python_version(),
        "host": platform.node(),
        "cpu_count": os.cpu_count(),
        "graph_dir": str(graph_dir),
        "graph_nodes": int(graph.num_nodes),
        "graph_edges": int(len(graph.fwd_targets)),
    }


def _fmt_ms(ms: Optional[float]) -> str:
    if ms is None:
        return "-"
    return f"{ms / 1000:.2f}s" if ms >= 1000 else f"{ms:.1f}ms"


def _fmt_mb(mb: Optional[float]) -> str:
    return "-" if mb is None else f"{mb:.0f}MB"


def _with_serialize(q: dict) -> Optional[float]:
    """Median lookup time plus serialization, or None for runs without it."""
    serialize_ms = q.get("serialize_ms")
    return None if serialize_ms is None else q["median_ms"] + serialize_ms


def print_run(report: dict) -> None:
    """Print one run's per-query table and stage breakdown."""
    env = report["environment"]
    dirty = " (dirty)" if env["git_dirty"] else ""
    print(
        f"\n== {env['label']}  @ {env['git_commit']}{dirty}  "
        f"graph: {env['graph_nodes']:,} nodes / {env['graph_edges']:,} edges"
    )
    header = (
        f"{'query':<38}{'median':>10}{'min':>10}{'+serialize':>12}"
        f"{'results':>10}{'paths':>11}{'peak mem':>10}  fingerprint"
    )
    print(header)
    print("-" * len(header))
    for q in report["queries"]:
        paths = q.get("num_paths")
        print(
            f"{q['name']:<38}{_fmt_ms(q['median_ms']):>10}{_fmt_ms(q['min_ms']):>10}"
            f"{_fmt_ms(_with_serialize(q)):>12}"
            f"{q['results']:>10,}{(paths if paths is not None else 0):>11,}"
            f"{_fmt_mb(q.get('peak_alloc_mb')):>10}  {q['fingerprint']}"
        )
    total = sum(q["median_ms"] for q in report["queries"])
    totals = [_with_serialize(q) for q in report["queries"]]
    total_ser = (
        None if any(t is None for t in totals) else sum(t or 0.0 for t in totals)
    )
    print(
        f"{'TOTAL (sum of medians)':<38}{_fmt_ms(total):>10}{'':>10}"
        f"{_fmt_ms(total_ser):>12}"
    )

    print("\nStage breakdown (profiled run):")
    labels = [label for label, _ in STAGE_COLUMNS]
    print(f"{'query':<38}" + "".join(f"{label.strip():>15}" for label in labels))
    for q in report["queries"]:
        print(
            f"{q['name']:<38}"
            + "".join(f"{_fmt_ms(q['stages_ms'][label]):>15}" for label in labels)
        )


def _ratio(before: Optional[float], after: Optional[float]) -> str:
    """``before / after`` as a speedup, or ``-`` when either is missing."""
    if before is None or not after:
        return "-"
    return f"{before / after:.2f}x"


def print_compare(base: dict, new: dict) -> bool:
    """Print a before/after table.  Returns False if any result changed."""
    base_env, new_env = base["environment"], new["environment"]
    print(
        f"\n== compare: {base_env['label']} @ {base_env['git_commit']}  ->  "
        f"{new_env['label']} @ {new_env['git_commit']}"
    )
    if (base_env["graph_nodes"], base_env["graph_edges"]) != (
        new_env["graph_nodes"],
        new_env["graph_edges"],
    ):
        print("  WARNING: the two runs used different graphs; timings not comparable")

    base_by_name = {q["name"]: q for q in base["queries"]}
    header = (
        f"{'query':<38}{'before':>10}{'after':>10}{'speedup':>9}"
        f"{'+ser before':>13}{'+ser after':>12}{'speedup':>9}"
        f"{'mem before':>12}{'mem after':>11}  results"
    )
    print(header)
    print("-" * len(header))
    all_same = True
    tot_before = tot_after = 0.0
    ser_before: Optional[float] = 0.0
    ser_after: Optional[float] = 0.0
    for q in new["queries"]:
        b = base_by_name.get(q["name"])
        if b is None:
            print(f"{q['name']:<38}{'-':>10}{_fmt_ms(q['median_ms']):>10}      new")
            continue
        tot_before += b["median_ms"]
        tot_after += q["median_ms"]
        b_ser, q_ser = _with_serialize(b), _with_serialize(q)
        ser_before = None if ser_before is None or b_ser is None else ser_before + b_ser
        ser_after = None if ser_after is None or q_ser is None else ser_after + q_ser
        same = b["fingerprint"] == q["fingerprint"]
        all_same &= same
        verdict = "same" if same else f"CHANGED ({b['results']:,} -> {q['results']:,})"
        print(
            f"{q['name']:<38}{_fmt_ms(b['median_ms']):>10}{_fmt_ms(q['median_ms']):>10}"
            f"{b['median_ms'] / q['median_ms']:>8.2f}x"
            f"{_fmt_ms(b_ser):>13}{_fmt_ms(q_ser):>12}{_ratio(b_ser, q_ser):>9}"
            f"{_fmt_mb(b.get('peak_alloc_mb')):>12}{_fmt_mb(q.get('peak_alloc_mb')):>11}"
            f"  {verdict}"
        )
    if tot_after:
        print(
            f"{'TOTAL (sum of medians)':<38}{_fmt_ms(tot_before):>10}"
            f"{_fmt_ms(tot_after):>10}{tot_before / tot_after:>8.2f}x"
            f"{_fmt_ms(ser_before):>13}{_fmt_ms(ser_after):>12}"
            f"{_ratio(ser_before, ser_after):>9}"
        )

    print("\nStage deltas (profiled run, before -> after):")
    labels = [label for label, _ in STAGE_COLUMNS]
    print(f"{'query':<38}" + "".join(f"{label.strip():>22}" for label in labels))
    for q in new["queries"]:
        b = base_by_name.get(q["name"])
        if b is None:
            continue
        cells = []
        for label in labels:
            before, after = b["stages_ms"][label], q["stages_ms"][label]
            cells.append(f"{_fmt_ms(before)}->{_fmt_ms(after)}")
        print(f"{q['name']:<38}" + "".join(f"{c:>22}" for c in cells))

    if not all_same:
        print("\nWARNING: results changed for at least one query (see 'results').")
    return all_same


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def cmd_run(args) -> int:
    from gandalf.biolink import make_toolkit
    from gandalf.graph import CSRGraph

    graph_dir, queries = resolve_graph(args)
    if args.only:
        queries = [q for q in queries if any(s in q["name"] for s in args.only)]
        if not queries:
            sys.exit(f"no query name contains any of {args.only}")

    print("Initializing BMT ...")
    bmt = make_toolkit()
    print(f"Loading graph from {graph_dir} ...")
    t0 = time.perf_counter()
    graph = CSRGraph.load_mmap(graph_dir)
    print(f"  loaded in {time.perf_counter() - t0:.1f}s")

    label = args.label or _git("rev-parse", "--short", "HEAD") or "run"
    report = {"environment": environment(graph_dir, graph, label), "queries": []}
    for query in queries:
        print(f"  {query['name']}:", end="", flush=True)
        record = bench_query(
            graph,
            query,
            bmt,
            args.warmup,
            args.repeat,
            memory=args.memory,
        )
        print(
            f"  -> median {_fmt_ms(record['median_ms'])}"
            f"  ({record['results']:,} results)"
        )
        report["queries"].append(record)

    print_run(report)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        print(f"\nSaved to {out}")
    if args.compare:
        base = json.loads(Path(args.compare).read_text())
        if not print_compare(base, report):
            return 1
    return 0


def cmd_compare(args) -> int:
    base = json.loads(Path(args.baseline).read_text())
    new = json.loads(Path(args.candidate).read_text())
    return 0 if print_compare(base, new) else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="benchmark queries against a graph")
    run.add_argument("--graph", help="graph directory (as written by gandalf-build)")
    run.add_argument("--queries", help="TRAPI request body, or a JSON list of them")
    run.add_argument(
        "--synthetic",
        choices=["tiny", "small", "medium"],
        help="build/reuse a synthetic graph of this size instead of --graph",
    )
    run.add_argument("--seed", type=int, default=42, help="synthetic graph seed")
    run.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    run.add_argument("--rebuild", action="store_true", help="rebuild synthetic graph")
    run.add_argument("--only", nargs="+", help="run queries whose name contains any")
    run.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="untimed runs per query (each query runs warmup + repeat + 1 times)",
    )
    run.add_argument("--repeat", type=int, default=3, help="timed runs per query")
    run.add_argument(
        "--memory",
        action="store_true",
        help="also measure peak allocation with tracemalloc (one slow extra run)",
    )
    run.add_argument("--label", help="name for this run (default: git commit)")
    run.add_argument("--out", help="save the report as JSON here")
    run.add_argument("--compare", help="saved report to compare this run against")
    run.set_defaults(func=cmd_run)

    compare = sub.add_parser("compare", help="compare two saved reports")
    compare.add_argument("baseline")
    compare.add_argument("candidate")
    compare.set_defaults(func=cmd_compare)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
