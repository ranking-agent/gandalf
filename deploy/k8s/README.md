# Kubernetes deployment, queue mode

Reference manifests for running Gandalf as an autoscaled pool of workers
behind a small API tier.  Adapt them into your Helm chart; the values that
matter are called out inline.

```
                    ┌──────────────┐   XADD    ┌─────────┐  XREADGROUP  ┌────────────────┐
  clients ───────▶  │ gandalf-api  │ ────────▶ │  Redis  │ ◀──────────  │ gandalf-worker │ ×N (KEDA)
  /query            │ (Deployment) │ ◀──────── │ Streams │ ──────────▶  │  (Deployment)  │
  /asyncquery       └──────────────┘  BLPOP    └─────────┘  result/ack  └────────────────┘
                                                                              │ POST callback
                                                                              ▼
                                                                         async clients
```

| File | What it is |
|---|---|
| `api.yaml` | The HTTP tier: validates requests, enqueues jobs, waits for `/query` results, serves the light endpoints. Two replicas; scale on CPU or request rate if ever needed. |
| `worker.yaml` | The query executors: one job at a time each, graph mounted read-only, Prometheus on port 9100. No Service; nothing calls a worker. |
| `keda.yaml` | The `ScaledObject` that adds workers when the stream's lag (jobs not yet taken by any worker) grows. |

Both tiers run the same image; the worker overrides the command with
`python -m gandalf.worker`.

## What to size

- **Worker memory.** A worker runs one query at a time, so its limit is one
  query's peak anonymous RSS (around 11 GB on the largest Translator
  queries) plus page-cache headroom for the mapped graph.  The graph itself
  costs no private memory (it is memory-mapped and shared across the pods on
  a node); what a worker keeps between queries is retained heap, which it
  returns to the OS after each job (`GANDALF_HEAP_TRIM_THRESHOLD_MB`).  The
  cgroup charges mapped file pages to the pod; too tight a limit thrashes
  the mapping before it OOM-kills.
- **CPU.** One per worker.  A query is single-threaded Python, so a second
  core goes unused; more throughput is more workers, not bigger ones.
- **`terminationGracePeriodSeconds`** on the worker must cover one query:
  SIGTERM makes the worker finish its current job before exiting, and a
  KEDA scale-down or a rollout sends SIGTERM.
- **`queue_claim_idle_seconds` vs. `queue_keepalive_seconds`.** A running
  worker refreshes its job every `keepalive` seconds; a job idle for longer
  than `claim_idle` is taken over by another worker as abandoned.  Keep
  `claim_idle` several keepalives long.
- **Result sizes in Redis.** `/query` results are stored zstd-compressed
  until the API collects them (seconds, normally; `result_ttl_seconds` at
  most).  Size Redis for a few in-flight results, compressed.  Async
  results never enter Redis.
- **Graph delivery.** The worker pods mount the graph from the shared
  read-only volume the loader writes, so a new worker is ready in seconds
  and pods on one node share the page cache for the mapped files.

## Watching it

`GET /status` on the `gandalf` Service is the live view: waiting and running
jobs, every worker and what it is doing, recent jobs with latencies, dead
letters, and this pod's request counts.  It is built from Redis and the API
itself, so it needs no access to the cluster, Prometheus or Jaeger.  Port-
forward or expose the Service and open it in a browser; `/status.json` is
the same data for scripts.

## Alerts

With a Slack incoming webhook in the `gandalf-slack` Secret (key
`webhook-url`), both tiers post events to its channel: workers joining and
leaving (scaling, rollouts), a worker lost without a clean exit (an OOM
kill, with what it was running), jobs retried or dead-lettered, failed
queries, the backlog crossing a threshold and clearing, and a job stuck
too long.  Without the Secret the same events still appear in the
"Alerts" section of `/status`.  The kinds and thresholds are settings;
see "Alerts and Slack" in the README.

## Scaling signal

KEDA's `redis-streams` scaler reads the consumer group's lag
(`XINFO GROUPS`, Redis 7.0+): entries not yet delivered to any worker.
With one job per worker, a lag of 1 means a query is waiting for a worker
that does not exist, so the threshold (`lagCount`) is 1 and KEDA adds a
worker per waiting job up to `maxReplicaCount`.  `cooldownPeriod` keeps
workers around after a burst; a new worker costs a graph open, not a
download, so it can be short.

The same numbers are on the API's `/metrics` as `gandalf_queue_lag` and
`gandalf_queue_pending`, with the worker's `gandalf_jobs_total`,
`gandalf_job_duration_seconds`, `gandalf_job_queue_wait_seconds`,
`gandalf_job_result_bytes` and `gandalf_process_rss_anon_bytes` on port
9100, for dashboards and alerts.
