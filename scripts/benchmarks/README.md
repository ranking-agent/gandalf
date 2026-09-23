# Benchmarks

| Script | Measures |
|---|---|
| `bench_lookup.py` | `lookup()` in-process: per-query wall time, per-stage breakdown, peak allocation, and a results fingerprint. Use this to track the effect of a code change. |
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

On a very large query, `--repeat 1 --warmup 0` keeps the run short, at the
cost of noisier numbers.

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
