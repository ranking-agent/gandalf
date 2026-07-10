# Autoscaling async queries (Redis queue + KEDA)

## The problem this solves

`/asyncquery` used to run the lookup as an in-process FastAPI `BackgroundTask`.
When Gandalf is overloaded, those tasks pile up behind the worker thread pool;
if a worker recycles, is OOM-killed, or the request times out, the queued work
just disappears and the client's callback never fires. There is also no signal
anywhere that says "the backlog is growing," so nothing can autoscale on it.

The Redis-backed path fixes both:

- **Durability.** Jobs live on a Redis Stream. Overflow waits in Redis instead
  of volatile process memory, and survives an API- or worker-pod restart.
- **A real scaling signal.** The consumer-group backlog *is* the overload
  metric. KEDA's `redis-streams` scaler reads it straight from Redis — no
  Prometheus, no metrics adapter, nothing outside Gandalf's own chart.

## Architecture

```
  POST /asyncquery ──▶ API pod validates, XADD ──▶ Redis Stream ──▶ worker pool ──▶ callback
                        (fixed, small tier)         (durable backlog)   (KEDA-scaled)
                                                          │
                                                  KEDA redis-streams
                                                  scaler → HPA on the
                                                  worker Deployment
```

Two tiers, split on purpose:

| Tier | Deployment | Scales on | Why |
|------|-----------|-----------|-----|
| API | fixed, small | — | Enqueue is cheap; a few replicas absorb bursts. |
| Worker | KEDA-managed | stream lag | The compute (graph lookup) is the expensive, scalable part. |

`python -m gandalf.worker` is the worker entrypoint. Each worker loads the graph
once, then loops: reclaim jobs abandoned by dead peers → read new jobs → run the
lookup → POST the result to the callback → `XACK`.

## Delivery semantics

Redis Streams give **at-least-once** delivery:

- `XREADGROUP` moves a job into the group's Pending Entries List (PEL); it is
  removed only by an explicit `XACK`.
- If a worker dies *after* reading but *before* acking (OOM, eviction, SIGKILL),
  the job stays pending. A live worker reclaims it via `XAUTOCLAIM` once it has
  been idle longer than `GANDALF_QUEUE_RECLAIM_IDLE_MS` (default 10 min).
- A genuine **poison job** — one that keeps killing its worker — is
  dead-lettered after `_MAX_DELIVERIES` (5) reclaims so it can't take the pool
  down in a retry loop.
- Application-level failures (lookup raises, callback unreachable) are logged and
  the job is acked — same best-effort contract as the old background task. Only
  worker *crashes* trigger a retry.

## Configuration

All settings live in `gandalf/config.py` (env prefix `GANDALF_`):

| Env var | Default | Meaning |
|---------|---------|---------|
| `GANDALF_QUEUE_ENABLED` | `false` | Route `/asyncquery` to Redis instead of an in-process task. |
| `GANDALF_REDIS_URL` | `""` | e.g. `redis://redis:6379/0`. Required when enabled. |
| `GANDALF_QUEUE_STREAM` | `gandalf:asyncquery` | Stream key. Must match the KEDA trigger. |
| `GANDALF_QUEUE_GROUP` | `gandalf-workers` | Consumer group. Must match the KEDA trigger. |
| `GANDALF_QUEUE_CONSUMER` | hostname+pid | Per-worker consumer name; leave unset. |
| `GANDALF_QUEUE_MAX_LEN` | `100000` | Approx `XADD MAXLEN` cap. |
| `GANDALF_QUEUE_BLOCK_MS` | `5000` | Blocking read timeout per poll. |
| `GANDALF_QUEUE_BATCH` | `1` | Jobs claimed per read. |
| `GANDALF_QUEUE_RECLAIM_IDLE_MS` | `600000` | Idle time before a pending job is reclaimed. Set above your longest lookup. |

With the queue **disabled**, `/asyncquery` behaves exactly as before (in-process
background task), so this is a safe, flag-gated rollout.

`GET /queue_status` reports `{length, pending}` for eyeballing the backlog;
it's not needed for scaling (KEDA talks to Redis directly).

## Deploying

Reference manifests are in [`deploy/keda/`](../deploy/keda/): a prototype Redis,
the worker Deployment, and the KEDA `ScaledObject`. See that directory's README
for the wiring.

## Caveats (read before relying on this)

- **Cold start.** A new worker loads the whole graph before it runs anything —
  potentially tens of seconds. Autoscaling smooths *sustained* load; it will not
  rescue a spike that overflows in seconds. Keep `minReplicaCount` headroom and
  a generous `cooldownPeriod`.
- **Memory scales linearly with replicas.** No cross-pod copy-on-write; each
  worker is a full graph copy. That bounds `maxReplicaCount` — size it against
  available node memory, not just desired throughput.
- **Sync `/query` is out of scope.** This only backs `/asyncquery`. A client on
  the synchronous endpoint holds the connection; scale that with API replicas /
  CPU HPA instead.
- **Redis is now on the critical path** for async queries. The prototype Redis
  is single-instance and non-persistent; use managed/HA Redis with AOF in
  production if the backlog must survive a Redis failure.
