# Benchmarks

| Script | Measures |
|---|---|
| `bench_lookup.py` | `lookup()` in-process: per-query wall time, per-stage breakdown, peak allocation, and a results fingerprint. Use this to track the effect of a code change. |
| `fast_path_ab.py` | Whether `_build_response`'s single-path fast path leaves a response byte-identical (including key order) on a given graph and query set, how many results take it, and how much faster it builds them. |
| `lookup_breakdown.py` | Where one lookup's time and memory go: pipeline stages, then `_build_response` statement by statement, serialization, garbage collection, and the size of each part of the response. Use this to decide what to optimize next. |
| `generate_queries.py` | Builds a query set for any graph, spread across result-size tiers from under 100 results to over a million. |
| `profile_query.py` | One query against a running server, end to end (includes serialization and transport). |
| `run_benchmark.py` | The shared query set against deployed Gandalf / Retriever instances. |

## Tracking a change with `bench_lookup.py`

Take a baseline on the commit **before** the change, then run again after it
and compare:

```bash
# Before
python scripts/benchmarks/bench_lookup.py run \
    --graph /data/graph_mmap --queries big_queries.json \
    --label baseline --out bench_results/baseline.json

# After
python scripts/benchmarks/bench_lookup.py run \
    --graph /data/graph_mmap --queries big_queries.json \
    --label after --out bench_results/after.json \
    --compare bench_results/baseline.json
```

`compare` re-prints the comparison from any two saved runs:

```bash
python scripts/benchmarks/bench_lookup.py compare \
    bench_results/baseline.json bench_results/after.json
```

The query file holds either one TRAPI request body or a JSON list of them,
the same JSON you would POST to `/query`. An optional top-level `"name"` key
labels a query in the report. `parameters` (`subclass`, `subclass_depth`,
`dehydrated`, `filter_config`) are applied exactly as the server applies them.

### What each run does

For every query:

1. `--warmup` untimed runs (default 1) so the page cache is warm.
2. `--repeat` timed runs (default 3) with the profiler off. The report
   shows the median and the minimum.
3. One run with `profile=True` for the stage breakdown (edge queries, joins,
   node cache, response building) and LMDB call totals.
4. With `--memory`, one more run under `tracemalloc` for peak allocation.
   This run is several times slower, so it is opt-in.

The **fingerprint** is a hash of the results that ignores ordering and
generated edge IDs. `compare` flags any query whose fingerprint changed and
exits non-zero, so an optimization that alters the answer can't pass as a
speed-up.

So each query runs `warmup + repeat + 1` times, 5 by default (6 with
`--memory`, and the memory run is several times slower). Each run prints its
time as it finishes. A query's generation-time probe (`generated.probe_ms`)
is one run on the code that generated it. The same query on older, slower
code, like a pre-optimization baseline, can take many times longer.

On a very large query, `--repeat 1 --warmup 0` cuts that to 2 runs, at the
cost of noisier numbers. Use the same flags for the before and after runs.

### Generating a query set

`generate_queries.py` builds queries from paths that exist in the graph,
runs each candidate once, and keeps it in the size tier its result count
falls in:

| tier | results |
|---|---|
| xs | 1–99 |
| s | 100–999 |
| m | 1,000–9,999 |
| l | 10,000–99,999 |
| xl | 100,000–999,999 |
| xxl | 1,000,000+ |

```bash
python scripts/benchmarks/generate_queries.py \
    --graph /data/graph_mmap --out bench_results/real_queries.json
```

It mixes four shapes (1-hop, 2-hop, 3-hop chain, and 3-hop Pathfinder
pinned at both ends) with three filter levels. The levels are exact
categories on the free nodes, no categories, and `biolink:related_to`
everywhere. Anchor nodes range from leaves to the biggest hubs. Once only
the large tiers (or only the small ones) are still short, it steers
sampling toward them.

Candidates run in a separate worker process, so a runaway one can't take
the run down. A candidate is discarded when it:

* times out (`--probe-timeout`, default 120s),
* exceeds `--probe-mem-gb` of memory (default: half of RAM), or
* builds more than `--path-cap` intermediate paths (default 5M).

A killed worker restarts, which costs one graph load. The run stops when
every tier has `--per-tier` queries (default 3), or at `--max-probes` or
`--time-budget`. The summary lists any tiers left short.

Generate once per graph and commit to the file: the point is to run the
same queries before and after a change. Don't compare the same file across
different graphs. A tier is a result count measured on one graph.

### Without a real graph

`--synthetic {tiny,small,medium}` builds a deterministic synthetic graph, caches
it under `bench_results/cache/`, and runs a built-in query set covering
forward, backward, and both-pinned traversal, inverse and symmetric predicates,
qualifier constraints, subclass expansion, multi-hop joins, dehydrated
responses, and node filters. See `synthetic_graph.py`.

```bash
python scripts/benchmarks/bench_lookup.py run --synthetic medium \
    --out bench_results/medium.json
```

`medium` (~57k nodes, ~1M edges) builds in about a minute. Its largest
query returns ~245k results.

Timings depend on the machine, so compare only runs from the same host.
Each report records the commit, host, and graph size.

## Finding where the time and memory go with `lookup_breakdown.py`

```bash
python scripts/benchmarks/lookup_breakdown.py \
    --graph /data/graph_mmap --queries big_queries.json \
    --only xl_ xxl_ --out bench_results/breakdown.json
```

For each query it prints three blocks:

* **TIME.**  The pipeline stages, then `_build_response` broken down by
  statement: each top-level statement, each statement of the per-result
  loop (with its execution count and time per execution), fetching each
  group's rows, and the costlier helpers credited to the statement that
  called them (`_group_rows`, the LMDB prefetch, `_full_edge` and the
  edge-properties read inside it).  The timers go into a copy of
  `_build_response` compiled from its source, so `lookup.py` is not
  modified.  Their cost is calibrated and subtracted, and the report shows
  the timed build next to an untimed one.  Statement times are scaled to
  the untimed run, and the report warns when the two differ by more than
  15%.
* **AFTER THE LOOKUP.**  Serializing each part of the message, and two
  garbage-collection costs.  `lookup()` pauses GC while it runs and resumes
  it without collecting, so a caller's next allocation triggers a
  collection over everything the lookup created: "first GC collection after
  the lookup" is that cost for a library caller.  The server does not pay
  it: its query handlers hold the pause (`gc_utils.gc_disabled`) until the
  response has been serialized and freed.  Then a full collection, and
  freeing the response.
* **MEMORY.**  Process RSS before, at its peak during, and after the
  lookup (needs `psutil`, from the server extra); the path arrays; the
  per-result group arrays; the edge data read from LMDB (most of it ends up
  inside the KG edges, so don't add the two); each part of the response as Python objects, estimated from a sample (`~`); and how
  many `{"ids": [x]}` binding objects the results hold against how many
  distinct IDs they bind.

Everything is measured inside a real lookup, never replayed on its own:
replaying a step with warm caches and none of the lookup's memory churn can
make it look several times cheaper than it is.

`--tracemalloc` adds a run under `tracemalloc`: the peak of Python's own
allocations and the source lines that allocated what is still alive when
response building ends.  It is slow and needs a lot of memory, so run it
with `--only` on one query at a time.

Each query runs `--warmup` (default 1) + `--repeat` (default 1) untimed
lookups, one timed lookup, and the analysis passes: about three times as
long as one lookup, plus the tracemalloc run if asked for.
