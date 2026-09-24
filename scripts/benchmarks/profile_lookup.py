#!/usr/bin/env python3
"""Profile one query's ``lookup()`` in depth, to decide what to optimize next.

Where ``bench_lookup.py`` answers "how long does each stage take", this
answers "why": it runs one query from a query file and prints

1. **Shape statistics** -- results, paths, distinct knowledge-graph edges
   versus edge bindings (how often the same edge is bound by many
   results), inferred subclass edges and auxiliary graphs.  These decide
   between restructuring options for response building, e.g. building
   each edge once per response only pays off when edges are reused a lot.
2. **A function profile** (cProfile), top functions by own time.
3. **A line profile of response building** (``--lines``; needs
   ``pip install line_profiler``), top lines of ``_build_response`` by time.

Usage::

    python scripts/benchmarks/profile_lookup.py \\
        --graph translatorkg/gandalf_mmap \\
        --queries bench_results/real_queries.json \\
        --query xl_3hop_pathfinder_exact_3 --lines

The query runs once with the stage profiler (which also warms the page
cache and supplies the shape statistics), then once under cProfile, then --
with ``--lines`` -- once under the line profiler.  Both
profilers slow the run several-fold, so read their numbers as shares of
the total, not as absolute times.  Paste the whole output back.
"""

import argparse
import cProfile
import importlib
import io
import pstats
import sys
import time
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_HERE))

from bench_lookup import load_queries, lookup_kwargs  # noqa: E402


def shape_stats(response: dict) -> list[str]:
    """Describe the response's size and how much its edges are shared."""
    message = response["message"]
    results = message["results"]
    kg_edges = message["knowledge_graph"]["edges"]
    aux_graphs = message.get("auxiliary_graphs") or {}

    bindings = 0
    edges_per_result = []
    bound_by: Counter = Counter()
    for result in results:
        n = 0
        for analysis in result.get("analyses") or []:
            for binding in analysis["edge_bindings"].values():
                n += len(binding["ids"])
                bound_by.update(binding["ids"])
        bindings += n
        edges_per_result.append(n)

    inferred = sum(
        1
        for edge in kg_edges.values()
        if any(
            a.get("attribute_type_id") == "biolink:support_graphs"
            for a in edge.get("attributes") or []
        )
    )
    support_edges = sum(len(g["edges"]) for g in aux_graphs.values())
    edges_per_result.sort()
    n_results = max(len(results), 1)

    def pct(p: float) -> int:
        return (
            edges_per_result[min(int(p * n_results), n_results - 1)] if results else 0
        )

    reuse = bindings / max(len(bound_by), 1)
    most_bound = bound_by.most_common(1)[0][1] if bound_by else 0
    return [
        f"results:                      {len(results):,}",
        f"KG nodes / edges:             {len(message['knowledge_graph']['nodes']):,}"
        f" / {len(kg_edges):,}",
        f"edge bindings (all results):  {bindings:,}",
        f"distinct bound edges:         {len(bound_by):,}",
        f"bindings per distinct edge:   {reuse:.1f}  (most-bound edge: {most_bound:,})",
        f"edge bindings per result:     median {pct(0.5)}, p90 {pct(0.9)},"
        f" max {edges_per_result[-1] if results else 0}",
        f"inferred (subclass) edges:    {inferred:,}",
        f"aux graphs / support edges:   {len(aux_graphs):,} / {support_edges:,}",
    ]


def stage_lines(response: dict) -> list[str]:
    """Top-level stage durations from the profiler tree in the logs."""
    import json

    for entry in response.get("logs") or []:
        if entry.get("code") == "ProfileSummary":
            tree = json.loads(entry["message"])
            lines = []
            for child in tree.get("children", []):
                name = child["name"]
                qedge_id = child.get("fields", {}).get("qedge_id")
                if qedge_id:
                    name = f"{name} {qedge_id}"
                lines.append(f"  {name:<18}{child['duration_ms'] / 1000:>8.2f}s")
                for sub in child.get("children", []):
                    lines.append(
                        f"    {sub['name']:<16}{sub['duration_ms'] / 1000:>8.2f}s"
                    )
            return lines
    return ["  (no profile in response)"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--graph", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--query", required=True, help="name of the query to run")
    parser.add_argument(
        "--lines",
        action="store_true",
        help="also line-profile _build_response (pip install line_profiler)",
    )
    parser.add_argument("--top", type=int, default=25, help="rows per profile")
    args = parser.parse_args(argv)

    from gandalf.biolink import make_toolkit
    from gandalf.graph import CSRGraph

    # ``gandalf.search.lookup`` the attribute is the function; get the module.
    lookup_module = importlib.import_module("gandalf.search.lookup")
    lookup = lookup_module.lookup

    queries = {q["name"]: q["body"] for q in load_queries(Path(args.queries))}
    if args.query not in queries:
        sys.exit(f"no query named {args.query!r}; have: {', '.join(queries)}")
    body = queries[args.query]
    kwargs = lookup_kwargs(body)

    print("Initializing BMT ...", flush=True)
    bmt = make_toolkit()
    print(f"Loading graph from {args.graph} ...", flush=True)
    graph = CSRGraph.load_mmap(args.graph)

    print("Stage-profiled run ...", flush=True)
    t0 = time.perf_counter()
    response = lookup(graph, body, bmt=bmt, profile=True, **kwargs)
    wall = time.perf_counter() - t0
    print(f"\n== {args.query}: {wall:.2f}s (first run, stage profiler on)\n")
    print("Stages:")
    print("\n".join(stage_lines(response)))
    print("\nShape:")
    print("\n".join("  " + line for line in shape_stats(response)))
    del response

    print("\ncProfile run ...", flush=True)
    profiler = cProfile.Profile()
    profiler.enable()
    lookup(graph, body, bmt=bmt, **kwargs)
    profiler.disable()
    out = io.StringIO()
    stats = pstats.Stats(profiler, stream=out)
    stats.sort_stats("tottime").print_stats(args.top)
    text = out.getvalue()
    print("\nFunctions by own time (cProfile):")
    print(text[text.index("   ncalls") :] if "   ncalls" in text else text)

    if args.lines:
        try:
            from line_profiler import LineProfiler
        except ImportError:
            print("--lines needs line_profiler: pip install line_profiler")
            return 1
        print("Line-profile run of _build_response ...", flush=True)
        lp = LineProfiler(lookup_module._build_response)
        lp.runcall(lookup, graph, body, bmt=bmt, **kwargs)
        print(f"\nTop {args.top} lines of _build_response:")
        print(top_lines(lp, args.top))
    return 0


def top_lines(lp, top: int) -> str:
    """The *top* most expensive lines from a LineProfiler, with their source."""
    import linecache

    rows = []
    total = 0.0
    for (filename, _, func), timings in lp.get_stats().timings.items():
        if func != "_build_response":
            continue
        for lineno, hits, t in timings:
            seconds = t * lp.get_stats().unit
            total += seconds
            rows.append((seconds, lineno, hits, linecache.getline(filename, lineno)))
    rows.sort(reverse=True)
    lines = [f"  {'line':>5} {'hits':>11} {'time':>9} {'%':>6}  source"]
    for seconds, lineno, hits, source in rows[:top]:
        share = 100.0 * seconds / total if total else 0.0
        lines.append(
            f"  {lineno:>5} {hits:>11,} {seconds:>8.2f}s {share:>5.1f}%  "
            f"{source.strip()[:80]}"
        )
    lines.append(f"  total in _build_response: {total:.2f}s")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
