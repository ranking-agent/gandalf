# Autoscaling Gandalf async queries with KEDA + Redis

These are **reference manifests** for the Redis-backed async-query queue. They
show how the pieces wire together; adapt the names/namespaces/resources to the
Helm chart Gandalf owns. Nothing here depends on Prometheus or any cluster-wide
metrics stack — KEDA reads the Redis stream directly.

## The shape

```
                    enqueue (XADD)                 XREADGROUP / XACK
  client ──POST /asyncquery──▶  API pods ──▶  Redis Stream  ──▶  worker pods ──▶ callback URL
                                (small, fixed)   (gandalf:asyncquery)   (KEDA-scaled)
                                                       │
                                                       ▼
                                          KEDA redis-streams scaler
                                          (scales the worker Deployment
                                           on consumer-group lag)
```

- **API pods** stay a small, fixed-size Deployment. `/asyncquery` validates the
  request and `XADD`s a job — cheap and fast, so a handful of replicas absorb
  bursts without dropping anything.
- **Worker pods** (`python -m gandalf.worker`) are the autoscaled tier. Each
  loads the graph and drains jobs. KEDA grows/shrinks this Deployment based on
  how many jobs are waiting.
- **Redis** holds the durable backlog. Overflow now waits in Redis instead of an
  in-process queue that evaporates on restart.

## Files

| File | What it is |
|------|------------|
| `redis.yaml` | A single-instance Redis for the prototype. **Not HA** — use a managed Redis or a replicated chart in production. |
| `gandalf-worker-deployment.yaml` | The worker Deployment KEDA scales. |
| `keda-scaledobject.yaml` | The KEDA `ScaledObject` (the autoscaling policy). |

## Turning it on

Set these on **both** the API and worker Deployments:

```
GANDALF_QUEUE_ENABLED=true
GANDALF_REDIS_URL=redis://redis:6379/0
```

Everything else (`GANDALF_QUEUE_STREAM`, `GANDALF_QUEUE_GROUP`, reclaim/idle
timings) has defaults in `gandalf/config.py` and only needs overriding if you
change the stream/group names — in which case the `ScaledObject` trigger must
match.

## Why redis-streams / lag

The `redis-streams` scaler with `lagCount` counts entries not yet delivered to
any consumer — i.e. the real backlog a client is waiting on. `XACK` on
completion (and Gandalf's `XDEL`) removes finished jobs so they stop counting.
`pendingEntriesCount` is an alternative trigger if you'd rather scale on
delivered-but-unacked depth instead.

## Read before you rely on this

- **Cold start.** A new worker must load the whole graph before it processes
  anything — potentially tens of seconds. Autoscaling smooths *sustained* load;
  it will not rescue a spike that overflows in seconds. Keep `minReplicaCount`
  headroom and set `cooldownPeriod` so workers don't thrash.
- **Memory scales with replicas.** There is no cross-pod copy-on-write; every
  worker is a full graph copy. N workers ≈ N × graph RAM. This bounds how far
  you can scale — size the graph and node memory before raising `maxReplicaCount`.
- **Sync `/query` is unaffected.** This queue only backs `/asyncquery`. A client
  on the synchronous endpoint still holds the connection; that path needs a
  different lever (replica count / CPU HPA on the API tier).
