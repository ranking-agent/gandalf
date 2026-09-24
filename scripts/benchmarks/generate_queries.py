#!/usr/bin/env python3
"""Generate a benchmark query set spanning result sizes, for any gandalf graph.

Queries are built from paths that really exist in the graph, then run once
(a *probe*) and sorted into size tiers by how many results they return, so
the output is a query set with a spread of sizes measured on that graph
rather than guessed::

    # For a real graph
    python scripts/benchmarks/generate_queries.py \\
        --graph /data/graph_mmap --out bench_results/real_queries.json

    # For a synthetic graph (built and cached as by bench_lookup.py)
    python scripts/benchmarks/generate_queries.py \\
        --synthetic medium --out bench_results/medium_queries.json

    # Then benchmark it
    python scripts/benchmarks/bench_lookup.py run \\
        --graph /data/graph_mmap --queries bench_results/real_queries.json

How a candidate query is made:

1. **Anchor.**  A node is drawn with its degree (in + out) spread
   log-uniformly between 1 and the graph's largest degree, so leaves,
   ordinary nodes and hubs are all represented.
2. **Walk.**  A random walk of 1-3 edges from the anchor, following stored
   edges in either direction and skipping ``subclass_of``.
3. **Shape.**  The walk becomes a query graph whose edges keep the walked
   edges' stored direction and predicate:

   * ``1hop``: the anchor pinned, its neighbour free;
   * ``2hop``: the anchor pinned, two free nodes beyond it;
   * ``3hop_chain``: the anchor pinned, three free nodes beyond it (the
     shape that reaches the largest tiers);
   * ``3hop_pathfinder``: both ends of a 3-edge walk pinned, two free
     intermediates (the Pathfinder shape).

4. **Breadth.**  How much the free nodes are constrained:

   * ``exact``: each free node keeps its walked node's first category;
   * ``broad``: free nodes carry no category;
   * ``related_to``: no categories, and every predicate ``biolink:related_to``.

Because each query contains the walk it came from, none is empty by
construction; hub anchors and broader filters make it larger.

Each candidate is then run once, with ``--probe-timeout`` as its time budget,
and assigned to the tier its result count falls in (see ``TIERS``).  A
candidate that times out is discarded.  When the only tiers still short are
the large ones, sampling shifts toward hub anchors and broader filters (and
the reverse for small tiers), so rare sizes are found without a huge number
of probes.  Generation stops when every tier has ``--per-tier`` queries, or
``--max-probes`` / ``--time-budget`` is reached.

The timeout is checked at pipeline stage boundaries, so one huge probe can
overrun it; ``--max-probes`` and ``--time-budget`` bound the whole run.

Each output query carries a ``name`` (e.g. ``l_2hop_broad_1``) and a
``generated`` record (tier, shape, probe result count and time); the
benchmark runner strips both before calling ``lookup``.  The same ``--seed``
on the same graph and code proposes the same candidates.
"""

import argparse
import json
import math
import multiprocessing
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import psutil

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_HERE))

#: Size tiers by result count: ``(name, smallest count, largest count)``,
#: half-open on the right.
TIERS = [
    ("xs", 1, 100),
    ("s", 100, 1_000),
    ("m", 1_000, 10_000),
    ("l", 10_000, 100_000),
    ("xl", 100_000, 1_000_000),
    ("xxl", 1_000_000, math.inf),
]

#: Walk length for each query shape.
SHAPES = {"1hop": 1, "2hop": 2, "3hop_chain": 3, "3hop_pathfinder": 3}

BREADTHS = ["exact", "broad", "related_to"]

SUBCLASS_OF = "biolink:subclass_of"
RELATED_TO = "biolink:related_to"


def tier_of(num_results: int) -> Optional[str]:
    """The tier a result count falls in, or None for zero results.

    >>> [tier_of(n) for n in (0, 1, 99, 100, 5_000, 2_000_000)]
    [None, 'xs', 'xs', 's', 'm', 'xxl']
    """
    for name, low, high in TIERS:
        if low <= num_results < high:
            return name
    return None


# ---------------------------------------------------------------------------
# Sampling from the graph
# ---------------------------------------------------------------------------


@dataclass
class GraphSampler:
    """Draws anchors and random walks from a loaded ``CSRGraph``."""

    graph: object
    rng: random.Random
    degree: np.ndarray = field(init=False)
    by_degree: np.ndarray = field(init=False)
    sorted_degree: np.ndarray = field(init=False)
    subclass_pred_id: Optional[int] = field(init=False)

    def __post_init__(self) -> None:
        g = self.graph
        self.degree = np.diff(np.asarray(g.fwd_offsets)) + np.diff(
            np.asarray(g.rev_offsets)
        )
        connected = np.flatnonzero(self.degree > 0)
        order = np.argsort(self.degree[connected], kind="stable")
        self.by_degree = connected[order]
        self.sorted_degree = self.degree[self.by_degree]
        self.subclass_pred_id = g.predicate_to_idx.get(SUBCLASS_OF)

    def anchor(self, mode: str) -> int:
        """Draw an anchor node, its degree log-uniform over a range set by *mode*.

        ``mode`` is ``"any"`` (the whole degree range), ``"big"`` (the top
        tenth of it, on a log scale) or ``"small"`` (the bottom half).
        """
        log_max = math.log(max(int(self.sorted_degree[-1]), 1))
        lo, hi = {"any": (0.0, 1.0), "big": (0.9, 1.0), "small": (0.0, 0.5)}[mode]
        target = math.exp(self.rng.uniform(lo * log_max, hi * log_max))
        # Nodes whose degree is within a factor of 1.5 of the target
        left = int(np.searchsorted(self.sorted_degree, target / 1.5, side="left"))
        right = int(np.searchsorted(self.sorted_degree, target * 1.5, side="right"))
        if right <= left:
            left = min(left, len(self.by_degree) - 1)
            right = left + 1
        return int(self.by_degree[self.rng.randrange(left, right)])

    def incident_edge(self, node: int) -> Optional[tuple[int, int, int]]:
        """A random stored edge touching *node*, as ``(subject, pred_id, object)``.

        Skips ``subclass_of`` edges (a query rewrite of its own); gives up
        after a few tries on a node that has little else.
        """
        g = self.graph
        out_start, out_end = int(g.fwd_offsets[node]), int(g.fwd_offsets[node + 1])
        in_start, in_end = int(g.rev_offsets[node]), int(g.rev_offsets[node + 1])
        n_out, n_in = out_end - out_start, in_end - in_start
        if n_out + n_in == 0:
            return None
        for _ in range(8):
            k = self.rng.randrange(n_out + n_in)
            if k < n_out:
                pos = out_start + k
                edge = (node, int(g.fwd_predicates[pos]), int(g.fwd_targets[pos]))
            else:
                pos = in_start + k - n_out
                edge = (int(g.rev_sources[pos]), int(g.rev_predicates[pos]), node)
            if edge[1] != self.subclass_pred_id:
                return edge
        return None

    def walk(self, start: int, length: int) -> Optional[tuple[list, list]]:
        """A self-avoiding random walk of *length* edges from *start*.

        Returns:
            ``(nodes, edges)``: the node indices in walk order, and each edge
            as ``(subject position, pred_id, object position)`` into
            ``nodes`` -- or None if the walk got stuck.
        """
        nodes = [start]
        edges = []
        for _ in range(length):
            current = nodes[-1]
            for _ in range(8):
                edge = self.incident_edge(current)
                if edge is None:
                    return None
                subj, pred, obj = edge
                other = obj if subj == current else subj
                if other not in nodes:
                    break
            else:
                return None
            nodes.append(other)
            here, there = len(nodes) - 2, len(nodes) - 1
            edges.append(
                (here, pred, there) if subj == current else (there, pred, here)
            )
        return nodes, edges

    def first_category(self, node: int) -> Optional[str]:
        categories = self.graph.get_node_property(node, "categories", []) or []
        return categories[0] if categories else None


# ---------------------------------------------------------------------------
# Building queries
# ---------------------------------------------------------------------------


def build_query(
    sampler: GraphSampler, nodes: list, edges: list, shape: str, breadth: str
) -> dict:
    """Turn a walk into a TRAPI request body of the given shape and breadth."""
    g = sampler.graph
    pinned = {0, len(nodes) - 1} if shape == "3hop_pathfinder" else {0}

    qnodes = {}
    for pos, node in enumerate(nodes):
        qnode: dict = {}
        if pos in pinned:
            qnode["ids"] = [g.get_node_id(node)]
            category = sampler.first_category(node)
            if category:
                qnode["categories"] = [category]
        elif breadth == "exact":
            category = sampler.first_category(node)
            if category:
                qnode["categories"] = [category]
        qnodes[f"n{pos}"] = qnode

    qedges = {}
    for i, (subj_pos, pred_id, obj_pos) in enumerate(edges):
        predicate = (
            RELATED_TO if breadth == "related_to" else g.id_to_predicate[pred_id]
        )
        qedges[f"e{i}"] = {
            "subject": f"n{subj_pos}",
            "object": f"n{obj_pos}",
            "predicates": [predicate],
        }
    return {"message": {"query_graph": {"nodes": qnodes, "edges": qedges}}}


def _probe_worker(conn, graph_dir: str, path_cap: int) -> None:
    """Probe-process main loop: load the graph once, then run queries on request.

    Receives ``(body, timeout_s)``; replies ``(outcome, result count, ms)``
    where outcome is ``"ok"``, ``"timeout"`` (the query's own deadline
    expired) or ``"over_cap"`` (a join hit ``path_cap`` rows, so the count
    would be a truncated one).  ``None`` ends the loop.
    """
    from gandalf.biolink import make_toolkit
    from gandalf.graph import CSRGraph
    from gandalf.search import reconstruct
    from gandalf.search.lookup import lookup
    from gandalf.trapi import Deadline

    bmt = make_toolkit()
    graph = CSRGraph.load_mmap(graph_dir)
    # Bound every join; a probe that reaches the bound is reported, not counted.
    reconstruct.MAX_PATH_LIMIT = path_cap
    conn.send("ready")

    while True:
        request = conn.recv()
        if request is None:
            return
        body, timeout_s = request
        t0 = time.perf_counter()
        response = lookup(graph, body, bmt=bmt, deadline=Deadline(timeout_s))
        ms = (time.perf_counter() - t0) * 1000.0
        if response.get("status") == "Timeout":
            conn.send(("timeout", None, ms))
        elif any(
            "Truncating" in entry.get("message", "")
            for entry in response.get("logs") or []
        ):
            conn.send(("over_cap", None, ms))
        else:
            conn.send(("ok", len(response["message"]["results"]), ms))


class ProbeWorker:
    """Runs probes in a child process that can be killed without losing the run.

    A probe's own deadline is only checked between pipeline stages, and a
    runaway candidate can exhaust memory within one stage.  Isolating probes
    lets the parent enforce a hard wall-clock limit and a memory ceiling
    (polling the child's resident size, which works on every platform),
    killing and restarting the child when either is crossed.  A restart
    costs one graph load.
    """

    def __init__(self, graph_dir: Path, path_cap: int, mem_limit_bytes: int):
        self._graph_dir = str(graph_dir)
        self._path_cap = path_cap
        self._mem_limit = mem_limit_bytes
        self._ctx = multiprocessing.get_context("spawn")
        self.restarts = 0
        self._start()

    def _start(self) -> None:
        self._conn, child_conn = self._ctx.Pipe()
        self._proc = self._ctx.Process(
            target=_probe_worker,
            args=(child_conn, self._graph_dir, self._path_cap),
            daemon=True,
        )
        self._proc.start()
        child_conn.close()
        if self._conn.recv() != "ready":
            raise RuntimeError("probe worker failed to start")

    def _restart(self) -> None:
        self._proc.kill()
        self._proc.join()
        self.restarts += 1
        self._start()

    def probe(self, body: dict, timeout_s: float) -> tuple[str, Optional[int], float]:
        """Run *body* once in the worker.

        Returns:
            ``(outcome, result count or None, ms)``.  Besides the worker's own
            outcomes, ``"killed_time"`` / ``"killed_memory"`` mean the parent
            stopped a probe that overran its hard limits, and ``"crashed"``
            that the worker died (e.g. the OS killed it).
        """
        # The query's deadline first; the hard limit only when that fails.
        hard_limit = timeout_s * 1.5 + 10.0
        self._conn.send((body, timeout_s))
        child = psutil.Process(self._proc.pid)
        t0 = time.perf_counter()
        while not self._conn.poll(0.25):
            elapsed = time.perf_counter() - t0
            outcome = None
            if not self._proc.is_alive():
                outcome = "crashed"
            elif elapsed > hard_limit:
                outcome = "killed_time"
            else:
                try:
                    if child.memory_info().rss > self._mem_limit:
                        outcome = "killed_memory"
                except psutil.NoSuchProcess:
                    outcome = "crashed"
            if outcome is not None:
                self._restart()
                return outcome, None, elapsed * 1000.0
        try:
            return self._conn.recv()
        except EOFError:
            elapsed = time.perf_counter() - t0
            self._restart()
            return "crashed", None, elapsed * 1000.0

    def close(self) -> None:
        try:
            self._conn.send(None)
        except (BrokenPipeError, OSError):
            pass
        self._proc.join(timeout=10)
        if self._proc.is_alive():
            self._proc.kill()


# ---------------------------------------------------------------------------
# The generation loop
# ---------------------------------------------------------------------------


@dataclass
class TierFill:
    """Accepted queries per tier, with a per-shape cap for variety."""

    per_tier: int
    queries: dict = field(default_factory=lambda: {name: [] for name, _, _ in TIERS})

    @property
    def shape_cap(self) -> int:
        # With the cap below per_tier, every full tier mixes shapes.
        return max(1, math.ceil(self.per_tier / 2))

    def accepts(self, tier: str, shape: str) -> bool:
        held = self.queries[tier]
        return (
            len(held) < self.per_tier
            and sum(q["generated"]["shape"] == shape for q in held) < self.shape_cap
        )

    def missing(self) -> list[int]:
        """Indices into ``TIERS`` of the tiers not yet full."""
        return [
            i
            for i, (name, _, _) in enumerate(TIERS)
            if len(self.queries[name]) < self.per_tier
        ]

    def done(self) -> bool:
        return not self.missing()


def pick_mode(fill: TierFill) -> tuple[str, list, list]:
    """Choose where to sample next from the tiers still short.

    Returns ``(anchor mode, shape weights, breadth weights)``, weights in
    ``SHAPES`` and ``BREADTHS`` order.  Only large tiers missing steers
    toward hub anchors, open-ended shapes and broad filters (a pathfinder
    query, pinned at both ends, stays small whatever its anchor); only small
    tiers missing, toward low-degree anchors and exact filters; otherwise the
    whole range.
    """
    missing = fill.missing()
    small = [i for i in missing if i <= 2]
    large = [i for i in missing if i >= 3]
    if large and not small:
        return "big", [0.1, 0.5, 0.35, 0.05], [0.2, 0.4, 0.4]
    if small and not large:
        return "small", [0.3, 0.3, 0.1, 0.3], [0.8, 0.2, 0.0]
    return "any", [0.3, 0.3, 0.15, 0.25], [0.5, 0.3, 0.2]


def generate(graph, worker: ProbeWorker, args) -> tuple[list, dict]:
    """Propose and probe candidates until the tiers fill or a budget runs out.

    Returns:
        ``(queries, stats)``: the accepted queries in tier order, and counts
        of what happened to every candidate.
    """
    rng = random.Random(args.seed)
    sampler = GraphSampler(graph, rng)
    fill = TierFill(args.per_tier)
    seen: set = set()
    stats = {
        "probes": 0,
        "discarded": {},
        "empty": 0,
        "tier_full": 0,
        "stuck": 0,
    }
    t_start = time.perf_counter()

    while not fill.done():
        if stats["probes"] >= args.max_probes:
            print(f"Stopping: reached --max-probes {args.max_probes}")
            break
        if time.perf_counter() - t_start > args.time_budget:
            print(f"Stopping: reached --time-budget {args.time_budget:.0f}s")
            break

        mode, shape_weights, breadth_weights = pick_mode(fill)
        shape = rng.choices(list(SHAPES), weights=shape_weights)[0]
        breadth = rng.choices(BREADTHS, weights=breadth_weights)[0]
        walked = sampler.walk(sampler.anchor(mode), SHAPES[shape])
        if walked is None:
            stats["stuck"] += 1
            continue
        body = build_query(sampler, *walked, shape, breadth)
        key = json.dumps(body, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)

        stats["probes"] += 1
        status, count, ms = worker.probe(body, args.probe_timeout)
        if status != "ok":
            stats["discarded"][status] = stats["discarded"].get(status, 0) + 1
            outcome = status
        else:
            tier = tier_of(count)
            if tier is None:
                stats["empty"] += 1
                outcome = "empty"
            elif fill.accepts(tier, shape):
                n = len(fill.queries[tier]) + 1
                body["name"] = f"{tier}_{shape}_{breadth}_{n}"
                body["generated"] = {
                    "tier": tier,
                    "shape": shape,
                    "breadth": breadth,
                    "probe_results": count,
                    "probe_ms": round(ms, 1),
                    "seed": args.seed,
                }
                fill.queries[tier].append(body)
                outcome = f"-> {body['name']}"
            else:
                stats["tier_full"] += 1
                outcome = f"{tier} (full)"
        print(
            f"  probe {stats['probes']:>3}: {shape:<16}{breadth:<11}"
            f"{'-' if count is None else f'{count:,}':>11} results "
            f"{ms / 1000:>7.2f}s  {outcome}",
            flush=True,
        )

    queries = [q for name, _, _ in TIERS for q in fill.queries[name]]
    stats["elapsed_s"] = round(time.perf_counter() - t_start, 1)
    stats["per_tier"] = {name: len(fill.queries[name]) for name, _, _ in TIERS}
    return queries, stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--graph", help="graph directory (as written by gandalf-build)")
    parser.add_argument(
        "--synthetic",
        choices=["tiny", "small", "medium"],
        help="use (building if needed) a synthetic graph instead of --graph",
    )
    parser.add_argument(
        "--graph-seed", type=int, default=42, help="synthetic graph seed"
    )
    parser.add_argument("--cache-dir", default=str(_REPO / "bench_results" / "cache"))
    parser.add_argument("--out", required=True, help="write the query list here")
    parser.add_argument("--per-tier", type=int, default=3, help="queries per tier")
    parser.add_argument(
        "--probe-timeout",
        type=float,
        default=120.0,
        help="seconds allowed per probe; slower candidates are discarded",
    )
    parser.add_argument(
        "--path-cap",
        type=int,
        default=5_000_000,
        help="discard a probe whose joins exceed this many intermediate paths",
    )
    parser.add_argument(
        "--probe-mem-gb",
        type=float,
        default=psutil.virtual_memory().total / 2**30 / 2,
        help="kill a probe whose process exceeds this much RSS (default: half of RAM)",
    )
    parser.add_argument("--max-probes", type=int, default=300)
    parser.add_argument(
        "--time-budget", type=float, default=3600.0, help="seconds for the whole run"
    )
    parser.add_argument("--seed", type=int, default=0, help="query sampling seed")
    args = parser.parse_args(argv)

    from gandalf.graph import CSRGraph

    if args.synthetic:
        from synthetic_graph import cached_graph_dir

        graph_dir = cached_graph_dir(
            args.synthetic, args.graph_seed, Path(args.cache_dir)
        )
    elif args.graph:
        graph_dir = Path(args.graph)
    else:
        parser.error("pass --graph or --synthetic")

    print(f"Loading graph from {graph_dir} ...")
    graph = CSRGraph.load_mmap(graph_dir)
    print("Starting probe worker (loads BMT and the graph) ...")
    worker = ProbeWorker(graph_dir, args.path_cap, int(args.probe_mem_gb * 2**30))
    print(
        f"Probing candidates (tiers: {', '.join(n for n, _, _ in TIERS)}; "
        f"{args.per_tier} each; limits: {args.probe_timeout:.0f}s, "
        f"{args.probe_mem_gb:.1f}GB, {args.path_cap:,} paths) ..."
    )
    try:
        queries, stats = generate(graph, worker, args)
    finally:
        worker.close()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(queries, indent=2))

    print(f"\nWrote {len(queries)} queries to {out}")
    discarded = ", ".join(f"{n} {k}" for k, n in stats["discarded"].items())
    print(
        f"  {stats['probes']} probes in {stats['elapsed_s']}s: "
        f"{stats['tier_full']} landed in a full tier, {stats['empty']} empty"
        + (f", discarded: {discarded}" if discarded else "")
        + (f" ({worker.restarts} worker restarts)" if worker.restarts else "")
    )
    for name, low, high in TIERS:
        span = f"{low:,}+" if high == math.inf else f"{low:,}-{high - 1:,}"
        have = stats["per_tier"][name]
        note = "" if have >= args.per_tier else "  (short)"
        print(f"  {name:<4}{span:>18} results: {have}/{args.per_tier}{note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
