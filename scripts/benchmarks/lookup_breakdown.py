#!/usr/bin/env python3
"""Where a lookup's time and memory go, down to the statements of
``_build_response``.

For each query this reports:

* **Time.**  The pipeline stages (edge queries, path reconstruction,
  response building), then response building statement by statement: every
  top-level statement of ``_build_response``, and every statement of its
  per-result loop, gets a wall-clock timer; fetching each group's rows is
  timed inside the loop; and the costlier helpers (``_group_rows``, the
  LMDB prefetch, ``_full_edge`` and the edge-properties read inside it) are
  timed and credited to the statement that called them.  The timers go into
  a copy of the function compiled from its own source, so ``lookup.py`` is
  not changed; their cost is calibrated and subtracted, and the total is
  checked against an untimed run.  Everything is measured inside a real
  lookup: replaying a step on its own, with warm caches and none of the
  lookup's memory churn, can make it look several times cheaper.  After the
  lookup: serializing each part of the message, the garbage collection the
  caller pays for right after it, a full collection, and freeing the
  response.
* **Memory.**  Process RSS before, at its peak during, and after the lookup
  (needs ``psutil``, from the server extra); the path arrays; the per-result
  group arrays; the edge data prefetched from LMDB; each part of the
  response as Python objects; and how many binding objects the results hold
  against how many distinct IDs they bind.  With ``--tracemalloc``, also the
  peak of Python's own allocations and the source lines that allocated what
  is still alive when response building ends.  That run is slow and needs
  a lot of memory, so use it with ``--only`` on one query at a time.

Usage::

    python scripts/benchmarks/lookup_breakdown.py --graph GRAPH_DIR \\
        --queries QUERIES.json [--only xl_ xxl_] [--out breakdown.json]

Each query runs ``--warmup`` + ``--repeat`` untimed lookups (the reference
timings), one timed lookup, then the analysis passes, plus one more lookup
with ``--tracemalloc``.  Sizes marked ``~`` are estimated from a sample of
``--sample`` items.
"""

import __future__
import argparse
import ast
import bisect
import gc
import importlib
import inspect
import json
import linecache
import random
import statistics
import sys
import textwrap
import threading
import time
import tracemalloc
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Optional

import numpy as np
import orjson

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_lookup import (  # noqa: E402
    _serialize_default,
    environment,
    load_queries,
    lookup_kwargs,
    profile_tree,
    stage_ms,
)

try:
    import psutil
except ImportError:  # psutil comes with the server extra
    psutil = None

# ``gandalf.search`` re-exports a ``lookup`` function under the module's name.
L = importlib.import_module("gandalf.search.lookup")

# The real functions, whatever they are patched to while measuring.
BUILD = L._build_response
ITER_GROUP_ROWS = L._iter_group_rows

#: Helpers of ``_build_response`` to time, where they exist.  Only ones that
#: take a few microseconds or more: a wrapper costs about one, so timing a
#: cheaper helper (``_append_edge_binding``, ``prune_edge``, ...) would
#: mostly measure the wrapper.  Their time stays in the calling statement.
HELPERS = ("_group_rows", "_full_edge", "_minimal_edge")

#: Plain names for statements of ``_build_response``, matched on their code:
#: ``(level, substring of the statement's code, name)``.  An unmatched
#: statement is shown as its code.
NAMES = (
    ("top", "_group_rows(", "group paths into results"),
    ("top", "get_edge_ids_batch(", "prefetch edge data and IDs"),
    ("top", "node_idx_by_id = ", "index nodes by ID"),
    ("top", "pop('_edge_id'", "strip internal markers from KG edges"),
    ("loop", "deadline.check(", "deadline check"),
    ("loop", "'node_bindings': {}", "new result dict"),
    ("loop", "for col in range(pa_num_node_cols)", "node bindings"),
    ("loop", "if single_path_fast", "single-path fast path"),
    ("loop", "edge_seen_keys[qedge_id]", "general loop: collect and dedup edges"),
    ("loop", "edge_bindings_by_qedge = defaultdict", "general loop: set up"),
    ("loop", "edge_seen_keys: dict", "general loop: set up"),
    (
        "loop",
        "for edge_id, edges in edge_bindings_by_qedge",
        "general loop: KG edges and bindings",
    ),
    ("loop", ".append(result)", "append the result"),
)

_REPO = Path(__file__).resolve().parents[2]
_MISSING = object()


# ---------------------------------------------------------------------------
# Timing _build_response statement by statement
# ---------------------------------------------------------------------------


def _relocate(node: ast.AST, like: ast.stmt, recurse: bool = True) -> None:
    """Give *node* (and, with *recurse*, everything in it) *like*'s position."""
    for sub in ast.walk(node) if recurse else (node,):
        if "lineno" in sub._attributes:
            sub.lineno, sub.col_offset = like.lineno, like.col_offset
            sub.end_lineno = like.end_lineno
            sub.end_col_offset = like.end_col_offset


def _timed(stmt: ast.stmt, index: int, clock: str) -> list[ast.stmt]:
    """*stmt* inside a timer that adds its duration, and one execution, to
    slot *index* of the accumulators -- also when it leaves by ``continue``,
    ``return`` or an exception."""
    start, wrapper = ast.parse(
        f"{clock} = _lb_pc()\n"
        "try:\n"
        "    pass\n"
        "finally:\n"
        f"    _lb_acc[{index}] += _lb_pc() - {clock}\n"
        f"    _lb_cnt[{index}] += 1\n"
    ).body
    assert isinstance(wrapper, ast.Try)
    wrapper.body = [stmt]
    _relocate(start, stmt)
    _relocate(wrapper, stmt, recurse=False)
    for sub in wrapper.finalbody:
        _relocate(sub, stmt)
    return [start, wrapper]


def _aliases(like: ast.stmt) -> ast.stmt:
    """Bind the timers' clock and accumulators to fast local names."""
    stmt = ast.parse("_lb_pc, _lb_acc, _lb_cnt = _LB_PC, _LB_ACC, _LB_CNT").body[0]
    _relocate(stmt, like)
    return stmt


def _header(stmt: ast.stmt) -> str:
    """One line of code naming *stmt*: a compound statement's header."""
    if isinstance(stmt, ast.If):
        text = f"if {ast.unparse(stmt.test)}"
    elif isinstance(stmt, ast.For):
        text = f"for {ast.unparse(stmt.target)} in {ast.unparse(stmt.iter)}"
    elif isinstance(stmt, ast.While):
        text = f"while {ast.unparse(stmt.test)}"
    elif isinstance(stmt, (ast.With, ast.Try)):
        text = type(stmt).__name__.lower()
    else:
        text = ast.unparse(stmt)
    return " ".join(text.split())


class InstrumentedBuild:
    """A copy of ``_build_response`` with a timer around each statement.

    Every top-level statement, and every statement in the body of the
    per-result loop (the top-level ``for`` with the largest body), is timed.
    The copy is compiled from the function's source with its real line
    numbers, into the module's globals, so it runs the same code.
    """

    def __init__(self, func=BUILD):
        lines, first_line = inspect.getsourcelines(func)
        module = ast.parse(textwrap.dedent("".join(lines)))
        ast.increment_lineno(module, first_line - 1)
        fdef = module.body[0]
        assert isinstance(fdef, ast.FunctionDef)
        body = list(fdef.body)
        docstring = []
        first = body[0] if body else None
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            docstring = [body.pop(0)]
        loops = [stmt for stmt in body if isinstance(stmt, ast.For)]
        main_loop = max(loops, key=lambda stmt: len(stmt.body), default=None)

        self.stmts: list[dict] = []
        new_body: list[ast.stmt] = []
        for stmt in body:
            index = self._add(stmt, "top", None, stmt is main_loop)
            if stmt is main_loop:
                inner: list[ast.stmt] = []
                for sub in stmt.body:
                    inner += _timed(sub, self._add(sub, "loop", index), "_lb_t1")
                stmt.body = inner
            new_body += _timed(stmt, index, "_lb_t0")
        fdef.body = docstring + [_aliases(body[0])] + new_body
        ast.fix_missing_locations(module)
        code = compile(
            module,
            inspect.getsourcefile(func) or "<_build_response>",
            "exec",
            flags=__future__.annotations.compiler_flag,
            dont_inherit=True,
        )
        self.globals = func.__globals__
        namespace: dict = {}
        exec(code, self.globals, namespace)
        self.func = namespace[fdef.name]
        self.code = self.func.__code__
        self.acc = [0.0] * len(self.stmts)
        self.cnt = [0] * len(self.stmts)

        # Each source line's innermost timed statement, for crediting a
        # helper call to the statement that made it.
        self.line_to_stmt: dict[int, int] = {}
        for level in ("top", "loop"):
            for index, stmt in enumerate(self.stmts):
                if stmt["level"] == level:
                    for line in range(stmt["line"], stmt["end_line"] + 1):
                        self.line_to_stmt[line] = index

    def _add(self, stmt, level: str, parent: Optional[int], main_loop=False) -> int:
        code = _header(stmt)
        full = ast.unparse(stmt)
        name = "per-result loop" if main_loop else code
        for name_level, needle, plain in NAMES:
            if not main_loop and name_level == level and needle in full:
                name = plain
                break
        self.stmts.append(
            {
                "line": stmt.lineno,
                "end_line": stmt.end_lineno,
                "level": level,
                "parent": parent,
                "main_loop": main_loop,
                "code": code,
                "name": name,
            }
        )
        return len(self.stmts) - 1

    @contextmanager
    def installed(self):
        """Zero the accumulators and give the copy its clock and slots."""
        for index in range(len(self.acc)):
            self.acc[index], self.cnt[index] = 0.0, 0
        names = {"_LB_PC": time.perf_counter, "_LB_ACC": self.acc, "_LB_CNT": self.cnt}
        self.globals.update(names)
        try:
            yield self.func
        finally:
            for name in names:
                self.globals.pop(name, None)


class HelperTracker:
    """Times wrapped helper functions, crediting each call to the timed
    statement of an ``InstrumentedBuild`` that made it, or to the wrapped
    helper it was made from."""

    def __init__(self, code=None, line_to_stmt: Optional[dict] = None):
        self.code = code
        # The calling statement is found from the caller's instruction offset
        # (``f_lasti``, cheap), looked up once per call site: ``f_lineno``
        # scans the code's line table from the start on every read, which
        # costs tens of microseconds in a function as large as
        # _build_response.
        self.starts: list[int] = []
        self.stmt_at: list[Optional[int]] = []
        if code is not None:
            for start, _end, line in code.co_lines():
                self.starts.append(start)
                self.stmt_at.append((line_to_stmt or {}).get(line))
        self.stack: list[str] = []
        self.stats: dict[str, dict] = {}

    @classmethod
    def for_build(cls, build: "InstrumentedBuild") -> "HelperTracker":
        return cls(build.code, build.line_to_stmt)

    def wrap(self, name: str, fn):
        clock, getframe, find = time.perf_counter, sys._getframe, bisect.bisect_right
        stack, code, starts, stmt_at = self.stack, self.code, self.starts, self.stmt_at
        rec = self.stats.setdefault(name, {"seconds": 0.0, "calls": 0, "by_caller": {}})
        by_caller = rec["by_caller"]
        call_sites: dict[int, Any] = {}

        def timed(*args, **kwargs):
            if stack:
                caller: Any = stack[-1]
            else:
                frame = getframe(1)
                if frame.f_code is code:
                    offset = frame.f_lasti
                    caller = call_sites.get(offset, _MISSING)
                    if caller is _MISSING:
                        caller = stmt_at[find(starts, offset) - 1]
                        call_sites[offset] = caller
                else:
                    caller = None
            stack.append(name)
            start = clock()
            try:
                return fn(*args, **kwargs)
            finally:
                elapsed = clock() - start
                stack.pop()
                rec["seconds"] += elapsed
                rec["calls"] += 1
                slot = by_caller.get(caller)
                if slot is None:
                    by_caller[caller] = [elapsed, 1]
                else:
                    slot[0] += elapsed
                    slot[1] += 1

        return timed


def _timed_rows(rows, rec: list):
    """Yield what *rows* yields, adding the time each item took to produce
    to ``rec[0]`` and one per item to ``rec[1]``."""
    clock = time.perf_counter
    while True:
        start = clock()
        try:
            item = next(rows)
        except StopIteration:
            rec[0] += clock() - start
            return
        rec[0] += clock() - start
        rec[1] += 1
        yield item


def _best(fn, *args, rounds: int) -> float:
    times = []
    for _ in range(rounds):
        start = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - start)
    return min(times)


def calibrate(n: int = 200_000, rounds: int = 5) -> dict:
    """Seconds each kind of timer adds per use: ``*_total`` to the code
    around it, ``*_inside`` to the time it measures itself."""
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        # A statement timer, around ``pass``.
        source = "def run(n):\n    for _ in range(n):\n        pass\n"
        plain: dict = {}
        exec(source, plain)
        module = ast.parse(source)
        fdef = module.body[0]
        assert isinstance(fdef, ast.FunctionDef)
        loop = fdef.body[0]
        assert isinstance(loop, ast.For)
        loop.body = _timed(loop.body[0], 0, "_lb_t1")
        fdef.body.insert(0, _aliases(loop))
        ast.fix_missing_locations(module)
        acc, cnt = [0.0], [0]
        timed: dict = {"_LB_PC": time.perf_counter, "_LB_ACC": acc, "_LB_CNT": cnt}
        exec(compile(module, "<calibration>", "exec"), timed)
        empty = _best(plain["run"], n, rounds=rounds)
        totals, insides = [], []
        for _ in range(rounds):
            acc[0], cnt[0] = 0.0, 0
            start = time.perf_counter()
            timed["run"](n)
            totals.append(time.perf_counter() - start)
            insides.append(acc[0] / n)

        # A helper wrapper, called from timed code (crediting the call by
        # its caller's offset) and from inside another wrapped helper.
        def noop():
            return None

        def call(count, fn):
            for _ in range(count):
                fn()

        lines = {line: 0 for _, _, line in call.__code__.co_lines()}
        tracker = HelperTracker(call.__code__, lines)
        top, inner = tracker.wrap("top", noop), tracker.wrap("inner", noop)
        outer = tracker.wrap("outer", call)
        plain_calls = _best(call, n, noop, rounds=rounds)
        noop_cost = (plain_calls - empty) / n
        helper: dict[str, list] = {"top": [[], []], "inner": [[], []]}
        for _ in range(rounds):
            for name, run in (
                ("top", lambda: call(n, top)),
                ("inner", lambda: outer(n, inner)),
            ):
                tracker.stats[name]["seconds"] = 0.0
                start = time.perf_counter()
                run()
                helper[name][0].append(time.perf_counter() - start)
                helper[name][1].append(tracker.stats[name]["seconds"] / n)

        # The generator wrapper that times fetching each group's rows.
        def rows(count):
            yield from range(count)

        def drain(it):
            for _ in it:
                pass

        plain_rows = _best(lambda: drain(rows(n)), rounds=rounds)
        row_totals, row_insides = [], []
        for _ in range(rounds):
            rec = [0.0, 0]
            start = time.perf_counter()
            drain(_timed_rows(rows(n), rec))
            row_totals.append(time.perf_counter() - start)
            row_insides.append(rec[0] / n)
        row_cost = (plain_rows - empty) / n
    finally:
        if was_enabled:
            gc.enable()
    return {
        "statement_total": max((min(totals) - empty) / n, 0.0),
        "statement_inside": min(insides),
        "helper_total": max((min(helper["top"][0]) - plain_calls) / n, 0.0),
        "helper_inside": max(min(helper["top"][1]) - noop_cost, 0.0),
        "nested_total": max((min(helper["inner"][0]) - plain_calls) / n, 0.0),
        "nested_inside": max(min(helper["inner"][1]) - noop_cost, 0.0),
        "rows_total": max((min(row_totals) - plain_rows) / n, 0.0),
        "rows_inside": max(min(row_insides) - row_cost, 0.0),
    }


def _corrected(build: InstrumentedBuild, tracker: HelperTracker, rows: list, cal: dict):
    """Statement, helper and row-fetching times with the timers' own cost
    taken out.

    Returns ``(statement seconds, helpers, rows seconds, overhead)``, where
    *helpers* maps each helper to its corrected total and per-caller times
    and *overhead* is everything the timers added to the build.
    """
    stmts = build.stmts
    stats = tracker.stats
    calls = {name: rec["calls"] for name, rec in stats.items()}
    nested = {name: 0 for name in stats}  # calls made from inside each helper
    for rec in stats.values():
        for caller, (_, count) in rec["by_caller"].items():
            if isinstance(caller, str):
                nested[caller] += count

    def per_call(name: str, caller) -> float:
        """What wrapping costs one call of *name*, its nested calls included."""
        own = cal["nested_total"] if isinstance(caller, str) else cal["helper_total"]
        inner = nested[name] * cal["nested_total"] / calls[name] if calls[name] else 0.0
        return own + inner

    helper_overhead = [0.0] * len(stmts)
    for name, rec in stats.items():
        for caller, (_, count) in rec["by_caller"].items():
            if isinstance(caller, int):
                helper_overhead[caller] += count * per_call(name, caller)

    seconds = []
    for index, stmt in enumerate(stmts):
        value = (
            build.acc[index]
            - build.cnt[index] * cal["statement_inside"]
            - helper_overhead[index]
        )
        if stmt["main_loop"]:
            value -= (rows[1] + 1) * cal["rows_total"]
            for child, sub in enumerate(stmts):
                if sub["parent"] == index:
                    value -= (
                        build.cnt[child] * cal["statement_total"]
                        + helper_overhead[child]
                    )
        seconds.append(max(value, 0.0))

    helpers = {}
    for name, rec in stats.items():
        inside = []
        for caller, (secs, count) in rec["by_caller"].items():
            own = (
                cal["nested_inside"]
                if isinstance(caller, str)
                else cal["helper_inside"]
            )
            inner = (
                nested[name] * cal["nested_total"] / calls[name] if calls[name] else 0.0
            )
            inside.append(
                {
                    "caller": caller,
                    "seconds": max(secs - count * (own + inner), 0.0),
                    "calls": count,
                }
            )
        helpers[name] = {
            "seconds": sum(c["seconds"] for c in inside),
            "calls": rec["calls"],
            "by_caller": inside,
        }
    rows_seconds = max(rows[0] - (rows[1] + 1) * cal["rows_inside"], 0.0)
    # Only what the timers added inside _build_response: a wrapped helper
    # can also be called from elsewhere in the lookup.
    overhead = (
        sum(build.cnt) * cal["statement_total"]
        + sum(helper_overhead)
        + (rows[1] + 1) * cal["rows_total"]
    )
    return seconds, helpers, rows_seconds, overhead


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


@contextmanager
def _patched(patches):
    """Set ``obj.name = value`` for each ``(obj, name, value)``, then undo."""
    saved = []
    try:
        for obj, name, value in patches:
            saved.append((obj, name, vars(obj).get(name, _MISSING)))
            setattr(obj, name, value)
        yield
    finally:
        for obj, name, old in reversed(saved):
            if old is _MISSING:
                delattr(obj, name)
            else:
                setattr(obj, name, old)


class RssSampler:
    """The process's resident set size, and its peak over a block of code
    (sampled every *interval* seconds by a background thread)."""

    def __init__(self, interval: float = 0.01):
        self.process = psutil.Process()
        self.interval = interval

    def now(self) -> int:
        return self.process.memory_info().rss

    @contextmanager
    def peak(self):
        box = {"peak": self.now()}
        stop = threading.Event()

        def sample():
            while not stop.wait(self.interval):
                box["peak"] = max(box["peak"], self.now())

        thread = threading.Thread(target=sample, daemon=True)
        thread.start()
        try:
            yield box
        finally:
            stop.set()
            thread.join()
            box["peak"] = max(box["peak"], self.now())


def _serialize_parts(response: dict) -> dict:
    """Serialize each part of the message as the server would; time and size."""
    message = response["message"]
    kg = message.get("knowledge_graph") or {}
    parts = {
        "results": message.get("results") or [],
        "kg_nodes": kg.get("nodes") or {},
        "kg_edges": kg.get("edges") or {},
        "aux_graphs": message.get("auxiliary_graphs") or {},
    }
    out = {}
    for name, part in parts.items():
        start = time.perf_counter()
        data = orjson.dumps(
            part, default=_serialize_default, option=orjson.OPT_SERIALIZE_NUMPY
        )
        out[name] = {"seconds": time.perf_counter() - start, "bytes": len(data)}
        del data
    return out


def _clean_run(graph, bmt, body, kwargs, rss: Optional[RssSampler], after: bool):
    """One lookup with nothing but a timer around ``_build_response``; with
    *after*, also serialize, collect garbage and free the response."""
    timing: dict = {}

    def timed_build(*args, **kw):
        start = time.perf_counter()
        try:
            return BUILD(*args, **kw)
        finally:
            timing["build"] = time.perf_counter() - start

    gc.collect()
    was_enabled = gc.isenabled()
    # lookup() turns GC off while it runs and back on when it returns,
    # without collecting, so the next allocation after it triggers a
    # young-generation collection over everything the lookup created: a cost
    # the caller pays.  Entering with GC off makes lookup() leave it off, so
    # that collection can be timed below instead of happening unseen.
    gc.disable()
    try:
        run: dict = {"rss_before": rss.now() if rss else None}
        with _patched([(L, "_build_response", timed_build)]):
            with rss.peak() if rss else nullcontext({}) as peak:
                start = time.perf_counter()
                response = L.lookup(graph, body, bmt=bmt, **kwargs)
                run["lookup"] = time.perf_counter() - start
        run["build"] = timing.get("build")
        if rss:
            run["rss_peak"] = peak["peak"]
            run["rss_after"] = rss.now()
        if after:
            run["serialize"] = _serialize_parts(response)
            start = time.perf_counter()
            gc.collect(0)
            run["gc_young"] = time.perf_counter() - start
            start = time.perf_counter()
            gc.collect()
            run["gc_full"] = time.perf_counter() - start
            start = time.perf_counter()
            del response
            run["free"] = time.perf_counter() - start
            if rss:
                run["rss_after_free"] = rss.now()
    finally:
        if was_enabled:
            gc.enable()
    return run


def _instrumented_run(graph, bmt, body, kwargs, build: InstrumentedBuild):
    """One profiled lookup through the timed copy of ``_build_response``,
    capturing what the analysis needs."""
    tracker = HelperTracker.for_build(build)
    rows = [0.0, 0]
    cap: dict = {"rows": rows}
    signature = inspect.signature(BUILD)

    def timed_build(*args, **kw):
        bound = signature.bind(*args, **kw)
        cap["path_data"] = bound.arguments["path_data"]
        cap["query_graph"] = bound.arguments["query_graph"]
        start = time.perf_counter()
        try:
            return build.func(*args, **kw)
        finally:
            cap["build_raw"] = time.perf_counter() - start

    def iter_group_rows(arrays, groups, *rest, **kw):
        cap["groups"] = groups
        return _timed_rows(ITER_GROUP_ROWS(arrays, groups, *rest, **kw), rows)

    patches: list[tuple] = [
        (L, "_build_response", timed_build),
        (L, "_iter_group_rows", iter_group_rows),
    ]
    for name in HELPERS:
        if hasattr(L, name):
            patches.append((L, name, tracker.wrap(name, getattr(L, name))))
    # The edge IDs batch, and the costly part of building an Edge dict
    # (credited to _full_edge).
    parts = (
        (graph, "get_edge_ids_batch", "edge IDs"),
        (graph, "get_edge_properties_by_index", "edge properties and attributes"),
    )
    for obj, attr, name in parts:
        if hasattr(obj, attr):
            patches.append((obj, attr, tracker.wrap(name, getattr(obj, attr))))
    store = getattr(graph, "lmdb_store", None)
    if store is not None:
        get_batch = store.get_batch

        def keep_detail(*args, **kw):
            cap["edge_detail"] = detail = get_batch(*args, **kw)
            return detail

        patches.append(
            (store, "get_batch", tracker.wrap("LMDB get_batch", keep_detail))
        )

    gc.collect()
    with build.installed(), _patched(patches):
        response = L.lookup(graph, body, bmt=bmt, profile=True, **kwargs)
    return response, tracker, cap


def _tracemalloc_run(graph, bmt, body, kwargs, top: int = 12) -> dict:
    """Python's own allocations over one lookup, and the source lines that
    allocated what is still alive when response building ends."""
    held: dict = {}

    def build(*args, **kw):
        out = BUILD(*args, **kw)
        held["live"] = tracemalloc.get_traced_memory()[0]
        held["snapshot"] = tracemalloc.take_snapshot()
        return out

    gc.collect()
    tracemalloc.start(1)
    try:
        base = tracemalloc.get_traced_memory()[0]
        with _patched([(L, "_build_response", build)]):
            response = L.lookup(graph, body, bmt=bmt, **kwargs)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        data = orjson.dumps(
            response, default=_serialize_default, option=orjson.OPT_SERIALIZE_NUMPY
        )
        serialize_peak = tracemalloc.get_traced_memory()[1] - current
        del data, response
    finally:
        tracemalloc.stop()
    sites = []
    snapshot = held.pop("snapshot", None)
    if snapshot is not None:
        for stat in snapshot.statistics("lineno")[:top]:
            frame = stat.traceback[0]
            path = Path(frame.filename)
            try:
                shown = str(path.resolve().relative_to(_REPO))
            except ValueError:
                shown = path.name
            sites.append(
                {
                    "where": f"{shown}:{frame.lineno}",
                    "code": linecache.getline(frame.filename, frame.lineno).strip(),
                    "bytes": stat.size,
                    "count": stat.count,
                }
            )
        del snapshot
    return {
        "peak_during_lookup": peak - base,
        "live_at_end_of_build": held.get("live", base) - base,
        "live_after_lookup": current - base,
        "serialize_peak": serialize_peak,
        "sites": sites,
    }


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def _shape(path_data, query_graph, groups) -> dict:
    """How many results come from one path, and how many carry an inferred
    subclass edge (a subclass match that is not an identity)."""
    nodes = path_data.paths_nodes
    inferred = np.zeros(len(nodes), dtype=bool)
    subclass_qedges = 0
    for qedge_id in path_data.col_to_qedge.values():
        qedge = query_graph["edges"][qedge_id]
        if qedge.get("_subclass"):
            subclass_qedges += 1
            subj = path_data.qnode_to_col[qedge["subject"]]
            obj = path_data.qnode_to_col[qedge["object"]]
            inferred |= nodes[:, subj] != nodes[:, obj]
    n = len(groups)
    if n == 0:
        return {
            "subclass_qedges": subclass_qedges,
            "single_path": 0.0,
            "inferred": 0.0,
            "fast_path": 0.0,
        }
    sizes = np.fromiter(map(len, groups), dtype=np.int64, count=n)
    rows = np.concatenate(groups)
    group_of_row = np.repeat(np.arange(n), sizes)
    group_inferred = np.bincount(group_of_row, weights=inferred[rows], minlength=n) > 0
    return {
        "subclass_qedges": subclass_qedges,
        "single_path": float((sizes == 1).mean()),
        "inferred": float(group_inferred.mean()),
        "fast_path": float(((sizes == 1) & ~group_inferred).mean()),
    }


def _deep_size(items, sample: int, rng, skip_strings=False, exclude=()) -> dict:
    """Bytes of Python objects reachable from each item, estimated from a
    sample: each object counts once, and objects in *exclude* (by ``id``)
    not at all."""
    n = len(items)
    if n == 0:
        return {"bytes": 0, "per_item": 0, "objects_per_item": 0, "sampled": 0}
    picks = range(n) if n <= sample else rng.sample(range(n), sample)
    seen = set(exclude)
    total = objects = 0
    for pick in picks:
        stack = [items[pick]]
        while stack:
            obj = stack.pop()
            key = id(obj)
            if key in seen:
                continue
            seen.add(key)
            if skip_strings and isinstance(obj, str):
                continue
            total += sys.getsizeof(obj)
            objects += 1
            if isinstance(obj, dict):
                stack.extend(obj.keys())
                stack.extend(obj.values())
            elif isinstance(obj, (list, tuple, set, frozenset)):
                stack.extend(obj)
    k = len(picks)
    return {
        "bytes": total * n / k,
        "per_item": total / k,
        "objects_per_item": objects / k,
        "sampled": k if k < n else 0,
    }


def _binding_stats(results: list) -> dict:
    """Binding objects in the results against the distinct IDs they bind.

    A binding of exactly ``{"ids": [x]}`` could be one object per ID shared
    by every result that binds ``x``; this counts how many separate objects
    there are now and what sharing them would save.
    """
    first: dict = {}
    total = single = separate = 0
    each = 0
    for result in results:
        groups = [result["node_bindings"].values()]
        groups += [a["edge_bindings"].values() for a in result.get("analyses") or ()]
        for bindings in groups:
            for binding in bindings:
                total += 1
                ids = binding.get("ids")
                if len(binding) == 1 and ids is not None and len(ids) == 1:
                    single += 1
                    seen = first.get(ids[0])
                    if seen is None:
                        first[ids[0]] = binding
                        separate += 1
                        if not each:
                            each = sys.getsizeof(binding) + sys.getsizeof(ids)
                    elif seen is not binding:
                        separate += 1
    return {
        "bindings": total,
        "single_id": single,
        "single_id_objects": separate,
        "distinct_ids": len(first),
        "bytes_each": each,
        "shareable_bytes": (separate - len(first)) * each,
    }


def _groups_memory(groups: list) -> dict:
    """The per-result row arrays: their objects, the list, and the arrays
    they are views of."""
    bases = {id(g.base): g.base.nbytes for g in groups if g.base is not None}
    return {
        "count": len(groups),
        "bytes": sys.getsizeof(groups)
        + sum(map(sys.getsizeof, groups))
        + sum(bases.values()),
    }


def breakdown(
    graph,
    bmt,
    query: dict,
    build: InstrumentedBuild,
    cal: dict,
    warmup: int = 1,
    repeat: int = 1,
    sample: int = 20_000,
    trace: bool = False,
    log=print,
) -> dict:
    """Measure one query; see the module docstring."""
    body = query["body"]
    kwargs = lookup_kwargs(body)
    qgraph = body["message"]["query_graph"]
    rss = RssSampler() if psutil else None
    rng = random.Random(0)
    rec: dict[str, Any] = {
        "name": query["name"],
        "qnodes": len(qgraph.get("nodes") or {}),
        "qedges": len(qgraph.get("edges") or {}),
    }

    for i in range(warmup):
        log(f"  warmup {i + 1}/{warmup}")
        _clean_run(graph, bmt, body, kwargs, None, after=False)
    runs = []
    for i in range(repeat):
        log(f"  clean run {i + 1}/{repeat}")
        runs.append(_clean_run(graph, bmt, body, kwargs, rss, after=i == repeat - 1))
    last = runs[-1]
    time_rec: dict[str, Any] = {
        "lookup": statistics.median(r["lookup"] for r in runs),
        "build": (
            statistics.median(r["build"] for r in runs)
            if all(r["build"] is not None for r in runs)
            else None
        ),
        "runs": len(runs),
        "serialize": last["serialize"],
        "gc_young": last["gc_young"],
        "gc_full": last["gc_full"],
        "free": last["free"],
    }
    rec["time"] = time_rec
    rec["memory"] = {
        "rss": (
            {
                "before": last["rss_before"],
                "peak": last["rss_peak"],
                "after": last["rss_after"],
                "after_free": last["rss_after_free"],
            }
            if rss
            else None
        )
    }

    log("  instrumented run")
    response, tracker, cap = _instrumented_run(graph, bmt, body, kwargs, build)
    results = response["message"]["results"]
    rec["results"] = len(results)
    tree = profile_tree(response) or {}
    time_rec["stages"] = {
        "subclass_rewrite": stage_ms(tree, ("subclass_rewrite",)) / 1000,
        "edge_queries": stage_ms(tree, ("qedge",)) / 1000,
        "reconstruct": stage_ms(tree, ("reconstruct",)) / 1000,
        "joins": stage_ms(tree, ("reconstruct", "join")) / 1000,
        "node_cache": stage_ms(tree, ("reconstruct", "node_cache_build")) / 1000,
    }
    if "path_data" not in cap:
        log("  response building did not run")
        return rec

    log("  analysing")
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        path_data, groups = cap["path_data"], cap.get("groups", [])
        rec["paths"] = int(path_data.paths_nodes.shape[0])
        rec["mode"] = "dehydrated" if path_data.lightweight else "full"
        rec["shape"] = _shape(path_data, cap["query_graph"], groups)

        seconds, helpers, rows_seconds, overhead = _corrected(
            build, tracker, cap["rows"], cal
        )
        corrected_build = cap["build_raw"] - overhead
        scale = (
            time_rec["build"] / corrected_build
            if time_rec["build"] and corrected_build > 0
            else 1.0
        )
        time_rec["instrumented_build"] = cap["build_raw"]
        time_rec["instrumented_build_corrected"] = corrected_build
        time_rec["scale"] = scale
        stmts = []
        for index, stmt in enumerate(build.stmts):
            stmts.append(
                {
                    **stmt,
                    "seconds": seconds[index] * scale,
                    "count": build.cnt[index],
                    "helpers": [
                        {
                            "name": name,
                            "seconds": c["seconds"] * scale,
                            "calls": c["calls"],
                        }
                        for name, h in helpers.items()
                        for c in h["by_caller"]
                        if c["caller"] == index
                    ],
                }
            )
        time_rec["statements"] = stmts
        loop = next((i for i, s in enumerate(stmts) if s["main_loop"]), None)
        if loop is not None:
            time_rec["loop_overhead"] = max(
                stmts[loop]["seconds"]
                - sum(s["seconds"] for s in stmts if s["parent"] == loop),
                0.0,
            )
        time_rec["row_fetch"] = rows_seconds * scale
        time_rec["row_fetch_groups"] = cap["rows"][1]
        time_rec["helpers"] = {
            name: {
                "seconds": h["seconds"] * scale,
                "calls": h["calls"],
                # the wrapped helpers this one called, and their share
                "parts": [
                    {"name": part, "seconds": c["seconds"] * scale, "calls": c["calls"]}
                    for part, ph in helpers.items()
                    for c in ph["by_caller"]
                    if c["caller"] == name
                ],
            }
            for name, h in helpers.items()
        }

        message = response["message"]
        kg = message["knowledge_graph"]
        aux = message.get("auxiliary_graphs") or {}
        node_bindings = sum(
            len(b["ids"]) for r in results for b in r["node_bindings"].values()
        )
        edge_bindings = sum(
            len(b["ids"])
            for r in results
            for a in r.get("analyses") or ()
            for b in a["edge_bindings"].values()
        )
        rec["kg"] = {
            "nodes": len(kg["nodes"]),
            "edges": len(kg["edges"]),
            "aux_graphs": len(aux),
        }
        rec["bindings_per_result"] = (
            (node_bindings + edge_bindings) / len(results) if results else 0.0
        )

        memory = rec["memory"]
        memory["path_arrays"] = sum(
            a.nbytes
            for a in (
                path_data.paths_nodes,
                path_data.paths_preds,
                path_data.paths_via_inverse,
                path_data.paths_fwd_edge_idx,
            )
        )
        memory["groups"] = _groups_memory(groups)
        memory["node_cache_tables"] = sys.getsizeof(
            path_data.node_cache
        ) + sys.getsizeof(path_data.node_id_cache)
        detail = cap.get("edge_detail") or {}
        memory["edge_detail_prefetch"] = _deep_size(list(detail.values()), sample, rng)
        memory["edge_detail_prefetch"]["bytes"] += sys.getsizeof(detail)
        memory["id_strings"] = {
            "count": len(kg["nodes"]) + len(kg["edges"]),
            "bytes": sum(map(sys.getsizeof, kg["nodes"]))
            + sum(map(sys.getsizeof, kg["edges"])),
        }
        node_ids = {id(key) for key in kg["nodes"]}
        memory["results"] = _deep_size(results, sample, rng, skip_strings=True)
        memory["results"]["bytes"] += sys.getsizeof(results)
        memory["kg_nodes"] = _deep_size(list(kg["nodes"].values()), sample, rng)
        memory["kg_nodes"]["bytes"] += sys.getsizeof(kg["nodes"])
        memory["kg_edges"] = _deep_size(
            list(kg["edges"].values()), sample, rng, exclude=node_ids
        )
        memory["kg_edges"]["bytes"] += sys.getsizeof(kg["edges"])
        memory["aux_graphs"] = _deep_size(list(aux.values()), sample, rng)
        memory["bindings"] = _binding_stats(results)
        del node_ids, detail
    finally:
        del response, results, cap
        if was_enabled:
            gc.enable()
    gc.collect()

    if trace:
        log("  tracemalloc run")
        rec["memory"]["tracemalloc"] = _tracemalloc_run(graph, bmt, body, kwargs)
    return rec


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _size(nbytes: Optional[float], approx: bool = False) -> str:
    if nbytes is None:
        return "-"
    mark = "~" if approx else ""
    if abs(nbytes) >= 2**30:
        return f"{mark}{nbytes / 2**30:.2f} GB"
    return f"{mark}{nbytes / 2**20:.0f} MB"


def _line(label: str, seconds=None, total=None, count=None, extra: str = "") -> str:
    secs = "" if seconds is None else f"{seconds:.2f}"
    share = f"{100 * seconds / total:.0f}%" if seconds is not None and total else ""
    shown_count = f"{count:,}" if count is not None else ""
    per = ""
    if seconds is not None and count and count > 1:
        per = f"{1e6 * seconds / count:.1f}us"
    text = f"{label[:52]:<52}{secs:>8}{share:>6}{shown_count:>12}{per:>10}"
    return f"{text}  {extra}".rstrip()


def print_report(rec: dict, out=print, fold: float = 0.005) -> None:
    """Print one query's breakdown; statements under *fold* of response
    building are summed into one line per level."""
    t = rec["time"]
    out(f"== {rec['name']} ==")
    if "paths" not in rec:
        out(f"{rec.get('results', 0):,} results; response building did not run\n")
        return
    shape = rec["shape"]
    kg = rec["kg"]
    out(
        f"{rec['results']:,} results from {rec['paths']:,} paths | "
        f"{rec['qnodes']} qnodes, {rec['qedges']} qedges "
        f"(+{shape['subclass_qedges']} subclass) | {rec['mode']} responses"
    )
    out(
        f"single-path {shape['single_path']:.0%} | fast path {shape['fast_path']:.0%}"
        f" | inferred edges {shape['inferred']:.0%} | "
        f"{rec['bindings_per_result']:.1f} bindings per result"
    )
    out(
        f"knowledge graph: {kg['nodes']:,} nodes, {kg['edges']:,} edges, "
        f"{kg['aux_graphs']:,} aux graphs"
    )
    out("")
    total = t["lookup"]
    runs = f", median of {t['runs']}" if t["runs"] > 1 else ""
    out(f"{'TIME':<52}{'seconds':>8}{'share':>6}{'count':>12}{'per item':>10}")
    out(_line(f"lookup (untimed run{runs})", total, total))
    stages = t["stages"]
    out(_line("  subclass rewrite", stages["subclass_rewrite"], total))
    out(_line("  edge queries", stages["edge_queries"], total))
    out(_line("  path reconstruction", stages["reconstruct"], total))
    out(_line("    joins", stages["joins"], total))
    out(_line("    node cache", stages["node_cache"], total))
    build = t["build"] or 0.0
    out(_line("  response building", build, total))

    stmts = t["statements"]

    def show(index: int, indent: str) -> None:
        stmt = stmts[index]
        label = f"{indent}L{stmt['line']:<5} {stmt['name']}"
        out(_line(label, stmt["seconds"], total, stmt["count"]))
        for helper in sorted(stmt["helpers"], key=lambda h: -h["seconds"]):
            out(
                _line(
                    f"{indent}        {helper['name']}",
                    helper["seconds"],
                    total,
                    helper["calls"],
                )
            )

    def show_level(parent: Optional[int], indent: str) -> None:
        folded, folded_n = 0.0, 0
        for index, stmt in enumerate(stmts):
            if stmt["parent"] != parent:
                continue
            if stmt["main_loop"]:
                show(index, indent)
                overhead = t.get("loop_overhead", 0.0)
                fetch = min(t.get("row_fetch", 0.0), overhead)
                out(_line(f"{indent}        loop overhead", overhead, total))
                out(
                    _line(
                        f"{indent}          fetching each group's rows",
                        fetch,
                        total,
                        t.get("row_fetch_groups"),
                    )
                )
                out(
                    _line(
                        f"{indent}          the rest (enumerate, unpacking)",
                        overhead - fetch,
                        total,
                    )
                )
                show_level(index, indent + "  ")
            elif stmt["seconds"] >= fold * build or stmt["helpers"]:
                show(index, indent)
            else:
                folded += stmt["seconds"]
                folded_n += 1
        if folded_n:
            out(_line(f"{indent}other ({folded_n} statements)", folded, total))

    show_level(None, "    ")
    upstream = (
        stages["subclass_rewrite"] + stages["edge_queries"] + stages["reconstruct"]
    )
    out(
        _line(
            "  other (setup, post-processing)",
            max(total - upstream - build, 0.0),
            total,
        )
    )
    out(
        f"  (timers: instrumented build {t['instrumented_build']:.2f}s, "
        f"{t['instrumented_build_corrected']:.2f}s with their cost removed, "
        f"untimed {build:.2f}s; statement times scaled by {t['scale']:.2f})"
    )
    if abs(t["scale"] - 1) > 0.15:
        out("  (warning: the timers distort this query by more than 15%)")
    for name in ("_full_edge", "_minimal_edge"):
        helper = t["helpers"].get(name)
        if not helper or not helper["calls"] or not helper["parts"]:
            continue
        calls = helper["calls"]
        parts = [
            f"{part['name']} {1e6 * part['seconds'] / calls:.1f}us"
            for part in sorted(helper["parts"], key=lambda part: -part["seconds"])
        ]
        rest = helper["seconds"] - sum(part["seconds"] for part in helper["parts"])
        out(
            f"  {name}: {1e6 * helper['seconds'] / calls:.1f}us per edge = "
            + " + ".join(parts)
            + f" + its own code {1e6 * max(rest, 0.0) / calls:.1f}us"
        )
    out("")
    out(f"{'AFTER THE LOOKUP':<52}{'seconds':>8}{'':>6}{'size':>12}")
    labels = {
        "results": "serialize results",
        "kg_nodes": "serialize knowledge graph nodes",
        "kg_edges": "serialize knowledge graph edges",
        "aux_graphs": "serialize auxiliary graphs",
    }
    for part, label in labels.items():
        ser = t["serialize"][part]
        out(f"{'  ' + label:<52}{ser['seconds']:>8.2f}{'':>6}{_size(ser['bytes']):>12}")
    out(f"{'  first GC collection after the lookup (young)':<52}{t['gc_young']:>8.2f}")
    out(f"{'  a full GC collection with the response alive':<52}{t['gc_full']:>8.2f}")
    out(f"{'  free the response':<52}{t['free']:>8.2f}")
    out("")

    m = rec["memory"]
    out("MEMORY")
    rss = m.get("rss")
    if rss:
        out(
            f"  process RSS: {_size(rss['before'])} before, "
            f"{_size(rss['peak'])} peak (+{_size(rss['peak'] - rss['before'])}), "
            f"{_size(rss['after'])} after, {_size(rss['after_free'])} after freeing"
        )
    else:
        out("  process RSS: install psutil to measure it")

    def mem(label: str, nbytes, approx=False, note: str = "") -> None:
        out(f"{label:<52}{_size(nbytes, approx):>10}  {note}".rstrip())

    mem("  path arrays (numpy)", m["path_arrays"])
    mem(
        "  result groups (one array per result)",
        m["groups"]["bytes"],
        note=f"{m['groups']['count']:,} arrays",
    )
    mem("  node cache tables", m["node_cache_tables"])
    prefetch = m["edge_detail_prefetch"]
    mem(
        "  edge data prefetched from LMDB (transient)",
        prefetch["bytes"],
        bool(prefetch["sampled"]),
    )
    out("  response, as Python objects")
    ids = m["id_strings"]
    mem(
        "    node and edge IDs (strings)",
        ids["bytes"],
        note=f"{ids['count']:,} strings",
    )
    for key, label, unit in (
        ("results", "results (dicts and lists)", "result"),
        ("kg_nodes", "knowledge graph nodes", "node"),
        ("kg_edges", "knowledge graph edges", "edge"),
        ("aux_graphs", "auxiliary graphs", "graph"),
    ):
        part = m[key]
        note = (
            f"{part['per_item']:,.0f} B and {part['objects_per_item']:.1f} objects per {unit}"
            if part["per_item"]
            else ""
        )
        mem(f"    {label}", part["bytes"], bool(part["sampled"]), note)
    b = m["bindings"]
    out(
        f"  bindings: {b['single_id_objects']:,} separate {{\"ids\": [x]}} objects "
        f"for {b['distinct_ids']:,} distinct IDs ({b['bindings']:,} bindings); "
        f"one object per ID would save {_size(b['shareable_bytes'])}"
    )
    trace = m.get("tracemalloc")
    if trace:
        out(
            f"  tracemalloc: peak {_size(trace['peak_during_lookup'])} during the lookup, "
            f"{_size(trace['live_at_end_of_build'])} alive when response building ends, "
            f"{_size(trace['live_after_lookup'])} after the lookup; serializing "
            f"peaks {_size(trace['serialize_peak'])} above that"
        )
        out("  alive when response building ends, by the line that allocated it:")
        for site in trace["sites"]:
            out(
                f"    {_size(site['bytes']):>10} {site['count']:>11,}  "
                f"{site['where']:<34} {site['code'][:60]}"
            )
    if any(m[k].get("sampled") for k in ("results", "kg_nodes", "kg_edges")):
        out("  (~ estimated from a sample of the items)")
    out("")


def _json_ready(obj):
    """Make keys JSON-safe (statement callers are ints or None)."""
    if isinstance(obj, dict):
        return {str(k): _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_ready(v) for v in obj]
    return obj


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--graph", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--only", nargs="*", help="keep queries whose name has any")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=1, help="untimed runs")
    parser.add_argument("--sample", type=int, default=20_000)
    parser.add_argument(
        "--tracemalloc",
        action="store_true",
        help="also trace Python allocations (slow; use with --only)",
    )
    parser.add_argument("--out", help="write the measurements as JSON here")
    args = parser.parse_args(argv)
    args.repeat = max(args.repeat, 1)

    from gandalf.biolink import make_toolkit
    from gandalf.graph import CSRGraph

    queries = load_queries(Path(args.queries))
    if args.only:
        queries = [q for q in queries if any(s in q["name"] for s in args.only)]
        if not queries:
            sys.exit(f"no query name contains any of {args.only}")

    def log(message: str) -> None:
        print(message, file=sys.stderr, flush=True)

    log("Initializing BMT ...")
    bmt = make_toolkit()
    log(f"Loading graph from {args.graph} ...")
    graph = CSRGraph.load_mmap(Path(args.graph))
    env = environment(Path(args.graph), graph, "breakdown")
    build = InstrumentedBuild()
    cal = calibrate()
    print(
        f"commit {env['git_commit']}{' (dirty)' if env['git_dirty'] else ''} | "
        f"graph {env['graph_nodes']:,} nodes, {env['graph_edges']:,} edges | "
        f"timer cost {1e9 * cal['statement_total']:.0f} ns per statement, "
        f"{1e9 * cal['helper_total']:.0f} ns per helper call\n"
    )

    records = []
    for query in queries:
        log(f"{query['name']}:")
        rec = breakdown(
            graph,
            bmt,
            query,
            build,
            cal,
            warmup=args.warmup,
            repeat=args.repeat,
            sample=args.sample,
            trace=args.tracemalloc,
            log=log,
        )
        print_report(rec)
        records.append(rec)

    if args.out:
        report = {"environment": env, "calibration": cal, "queries": records}
        Path(args.out).write_text(json.dumps(_json_ready(report), indent=1))
        log(f"Saved to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
