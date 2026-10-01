"""The status snapshot behind ``/status.json`` and the ``/status`` page.

An application-level view of the deployment, built from what the processes
themselves report into Redis -- the job stream, the worker registry and the
job history -- plus this pod's own HTTP counters.  It needs no access to
Kubernetes, Prometheus or Jaeger, which is the point: anyone who can reach
the API can see what Gandalf is doing.

Without a queue the snapshot still describes this process (graph, memory,
request counts); the queue, worker and job sections are then absent.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Any, Optional

from gandalf import __version__
from gandalf.config import settings
from gandalf.metrics import metrics_registry, rss_anon_kb, rss_kb
from gandalf.jobs import JobHistory, JobQueue, WorkerRegistry
from gandalf.notify import Notifier

#: Outcomes a job can end with, in the order the page lists them.
OUTCOMES = ("ok", "timeout", "expired", "error", "poisoned")

#: Time windows the aggregates are computed over, in seconds.
WINDOWS = {"5m": 300, "1h": 3600, "24h": 86400}


def percentile(values: list[float], fraction: float) -> Optional[float]:
    """The nearest-rank percentile of *values*, or None for no values.

    Examples:
        >>> percentile([1, 2, 3, 4], 0.5)
        2
        >>> percentile([5, 1, 3], 0.95)
        5
        >>> percentile([], 0.5) is None
        True
    """
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, int(round(fraction * len(ordered))))
    return ordered[min(rank, len(ordered)) - 1]


def aggregate(records: list[dict], window_s: float, now: float) -> dict:
    """Counts and latencies over the records finished within *window_s*.

    Examples:
        >>> recs = [
        ...     {"finished_at": 100, "outcome": "ok", "duration_s": 1.0, "wait_s": 0.1, "bytes": 10},
        ...     {"finished_at": 100, "outcome": "error", "duration_s": 3.0, "wait_s": 0.5, "bytes": 0},
        ...     {"finished_at": 10, "outcome": "ok", "duration_s": 9.0, "wait_s": 9.0, "bytes": 1},
        ... ]
        >>> agg = aggregate(recs, 60, now=120)
        >>> agg["jobs"], agg["outcomes"]["ok"], agg["outcomes"]["error"]
        (2, 1, 1)
        >>> agg["duration_p95_s"], agg["wait_p95_s"], agg["bytes"]
        (3.0, 0.5, 10)
    """
    since = now - window_s
    within = [r for r in records if r.get("finished_at", 0) >= since]
    outcomes = {o: 0 for o in OUTCOMES}
    for r in within:
        outcomes[r.get("outcome", "error")] = (
            outcomes.get(r.get("outcome", "error"), 0) + 1
        )
    durations = [float(r["duration_s"]) for r in within if "duration_s" in r]
    waits = [float(r["wait_s"]) for r in within if "wait_s" in r]
    failed = outcomes["error"] + outcomes["poisoned"]
    return {
        "window_s": window_s,
        "jobs": len(within),
        "outcomes": outcomes,
        "failed": failed,
        "per_minute": len(within) / (window_s / 60.0),
        "duration_p50_s": percentile(durations, 0.5),
        "duration_p95_s": percentile(durations, 0.95),
        "duration_max_s": max(durations) if durations else None,
        "wait_p50_s": percentile(waits, 0.5),
        "wait_p95_s": percentile(waits, 0.95),
        "bytes": sum(int(r.get("bytes", 0)) for r in within),
    }


def per_minute(records: list[dict], minutes: int, now: float) -> list[dict]:
    """Jobs finished in each of the last *minutes* minutes, oldest first.

    Each bucket is ``{"minute": epoch, "ok", "timed_out", "failed"}``:
    the three classes the page's chart stacks (``timeout`` and ``expired``
    are both the budget running out; ``error`` and ``poisoned`` both a
    failure).

    Examples:
        >>> buckets = per_minute([{"finished_at": 95, "outcome": "expired"}], 2, now=130)
        >>> [(b["minute"], b["timed_out"]) for b in buckets]
        [(60, 1), (120, 0)]
    """
    start = int(now // 60) * 60 - 60 * (minutes - 1)
    buckets = [
        {"minute": start + 60 * i, "ok": 0, "timed_out": 0, "failed": 0}
        for i in range(minutes)
    ]
    for r in records:
        index = int((r.get("finished_at", 0) - start) // 60)
        if 0 <= index < minutes:
            outcome = r.get("outcome")
            key = (
                "ok"
                if outcome == "ok"
                else "timed_out" if outcome in ("timeout", "expired") else "failed"
            )
            buckets[index][key] += 1
    return buckets


def http_counts() -> list[dict]:
    """This pod's request counts by route and status, from its Prometheus registry."""
    rows = []
    for metric in metrics_registry().collect():
        if metric.name != "gandalf_http_requests":
            continue
        for sample in metric.samples:
            if sample.name != "gandalf_http_requests_total":
                continue
            rows.append(
                {
                    "method": sample.labels["method"],
                    "route": sample.labels["route"],
                    "status": sample.labels["status"],
                    "count": int(sample.value),
                }
            )
    rows.sort(key=lambda r: (-r["count"], r["route"], r["status"]))
    return rows


def snapshot(
    graph: Any,
    queue: Optional[JobQueue],
    workers: Optional[WorkerRegistry],
    history: Optional[JobHistory],
    notifier: Optional[Notifier] = None,
    *,
    recent: int = 50,
    now: Optional[float] = None,
) -> dict:
    """Everything the status page shows, as one JSON-able dict."""
    now = time.time() if now is None else now
    data: dict = {
        "generated_at": datetime.fromtimestamp(now, timezone.utc).isoformat(),
        "mode": "queue" if queue is not None else "in-process",
        "server": _server_section(graph),
        "http": http_counts(),
    }
    if queue is None:
        return data

    stats = queue.stats()
    pending = queue.pending_entries()
    data["queue"] = {
        "stream": queue.stream,
        "group": queue.group,
        "lag": stats.lag,
        "pending": stats.pending,
        "length": queue.length(),
        "oldest_pending_s": max((p["idle_s"] for p in pending), default=0.0),
        "pending_entries": pending,
        "dead_letters": queue.dead_letter_count(),
        "recent_dead_letters": queue.dead_letters(),
    }
    if workers is not None:
        reports = workers.workers()
        for report in reports:
            report["age_s"] = now - float(report.get("last_seen", now))
            if report.get("state") == "running" and report.get("job_started_at"):
                report["job_elapsed_s"] = now - float(report["job_started_at"])
        data["workers"] = {
            "alive": len(reports),
            "busy": sum(1 for r in reports if r.get("state") == "running"),
            "reports": reports,
        }
    if history is not None:
        records = history.recent(max(recent, settings.history_maxlen))
        data["jobs"] = {
            "recent": records[:recent],
            "windows": {
                name: aggregate(records, seconds, now)
                for name, seconds in WINDOWS.items()
            },
            "per_minute": per_minute(records, 60, now),
            "history_size": len(records),
        }
    if notifier is not None:
        data["alerts"] = {
            "slack": notifier.slack_enabled,
            "kinds": sorted(notifier.kinds),
            "recent": notifier.recent(recent),
        }
    return data


def _server_section(graph: Any) -> dict:
    section: dict = {
        "version": __version__,
        "infores": settings.infores,
        "biolink_version": settings.biolink_version,
        "pid": os.getpid(),
        "rss_kb": rss_kb(),
        "rss_anon_kb": rss_anon_kb(),
        "graph_loaded": graph is not None,
    }
    if graph is not None:
        section["graph"] = {
            "path": settings.graph_path,
            "nodes": int(graph.num_nodes),
            "edges": int(len(graph.fwd_targets)),
        }
        metadata = getattr(graph, "graph_metadata", None)
        if isinstance(metadata, dict):
            section["graph"]["metadata"] = metadata
    return section
