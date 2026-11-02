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

import math
import os
import time
from datetime import datetime, timezone
from typing import Any, Optional

from gandalf import __version__
from gandalf.config import settings
from gandalf.metrics import metrics_registry, rss_anon_kb, rss_kb
from gandalf.jobs import (
    JobHistory,
    JobQueue,
    RedisHealth,
    WorkerRegistry,
    latency_bucket_value,
)
from gandalf.notify import Notifier

#: Outcomes a job can end with, in the order the page lists them.
OUTCOMES = ("ok", "timeout", "expired", "error", "poisoned", "refused")

#: Time windows the aggregates are computed over, in seconds.
WINDOWS = {"5m": 300, "1h": 3600, "24h": 86400}


def histogram_percentile(
    hist: dict[int, int], fraction: float, top: Optional[float] = None
) -> Optional[float]:
    """The nearest-rank percentile of a latency histogram, or None if empty.

    The answer is the middle of the bucket the rank falls in (within about
    6% of the true value; see :func:`gandalf.jobs.latency_bucket`), never
    above *top*, the largest value seen.

    Examples:
        >>> from gandalf.jobs import latency_bucket
        >>> hist = {latency_bucket(1.0): 90, latency_bucket(10.0): 10}
        >>> round(histogram_percentile(hist, 0.5), 2), round(histogram_percentile(hist, 0.95), 1)
        (1.06, 10.6)
        >>> histogram_percentile(hist, 0.95, top=10.0)
        10.0
        >>> histogram_percentile({}, 0.5) is None
        True
    """
    total = sum(hist.values())
    if not total:
        return None
    rank = max(1, math.ceil(fraction * total))
    seen = 0
    for index in sorted(hist):
        seen += hist[index]
        if seen >= rank:
            value = latency_bucket_value(index)
            return min(value, top) if top is not None else value
    return None  # unreachable: the ranks add up to total


def _merge_hist(target: dict[int, int], hist: dict[int, int]) -> None:
    for index, count in hist.items():
        target[index] = target.get(index, 0) + count


def _max(values: list[Optional[float]]) -> Optional[float]:
    present = [v for v in values if v is not None]
    return max(present) if present else None


def aggregate_counters(buckets: list[dict], window_s: float, now: float) -> dict:
    """Counts and latencies over the counter buckets within *window_s*.

    *buckets* are :meth:`JobHistory.minutes` or ``hours`` entries.  The
    window is whole buckets: those starting within it, so the oldest,
    partly-outside bucket is left out and the rate is over the time the
    counted buckets span.  Latency percentiles come from histograms, so are
    close, not exact.

    Examples:
        >>> m = {"jobs": 2, "outcomes": {"ok": 1, "timeout": 1}, "bytes": 5,
        ...      "duration_hist": {0: 2}, "wait_hist": {-20: 2},
        ...      "duration_max_s": 1.0, "wait_max_s": 0.1}
        >>> agg = aggregate_counters([dict(m, start=0), dict(m, start=60)], 150, now=150)
        >>> agg["jobs"], agg["outcomes"]["timeout"], agg["bytes"], agg["per_minute"]
        (4, 2, 10, 1.6)
        >>> agg["duration_max_s"], round(agg["duration_p95_s"], 2)
        (1.0, 1.0)
        >>> aggregate_counters([dict(m, start=0), dict(m, start=60)], 120, now=150)["jobs"]
        2
    """
    since = now - window_s
    within = [m for m in buckets if since <= m["start"] <= now]
    outcomes = {o: 0 for o in OUTCOMES}
    durations: dict[int, int] = {}
    waits: dict[int, int] = {}
    for m in within:
        for outcome, count in m["outcomes"].items():
            outcomes[outcome] = outcomes.get(outcome, 0) + count
        _merge_hist(durations, m["duration_hist"])
        _merge_hist(waits, m["wait_hist"])
    jobs = sum(m["jobs"] for m in within)
    duration_max = _max([m["duration_max_s"] for m in within])
    wait_max = _max([m["wait_max_s"] for m in within])
    span_s = now - within[0]["start"] if within else window_s
    return {
        "window_s": window_s,
        "jobs": jobs,
        "outcomes": outcomes,
        "failed": outcomes["error"] + outcomes["poisoned"] + outcomes["refused"],
        "per_minute": jobs / (max(span_s, 1.0) / 60.0),
        "duration_p50_s": histogram_percentile(durations, 0.5, duration_max),
        "duration_p95_s": histogram_percentile(durations, 0.95, duration_max),
        "duration_max_s": duration_max,
        "wait_p50_s": histogram_percentile(waits, 0.5, wait_max),
        "wait_p95_s": histogram_percentile(waits, 0.95, wait_max),
        "bytes": sum(m["bytes"] for m in within),
    }


def _chart_class(outcome: Optional[str]) -> str:
    """Which of the chart's three stacks an outcome is drawn in."""
    if outcome == "ok":
        return "ok"
    return "timed_out" if outcome in ("timeout", "expired") else "failed"


def minute_bucket(minute: dict) -> dict:
    """One minute of the page's chart from that minute's counters.

    Each is ``{"minute": epoch, "ok", "timed_out", "failed"}``, the three
    classes the chart stacks (``timeout`` and ``expired`` are both the budget
    running out; ``error``, ``poisoned`` and ``refused`` all a failure), and
    the queue wait of those jobs -- ``wait_p50_s``, ``wait_p95_s``,
    ``wait_max_s`` -- which are None in a minute that finished none.

    Examples:
        >>> b = minute_bucket({"start": 60, "jobs": 3, "outcomes": {"ok": 2, "refused": 1},
        ...                    "wait_hist": {0: 3}, "wait_max_s": 1.0})
        >>> b["ok"], b["timed_out"], b["failed"], b["wait_max_s"]
        (2, 0, 1, 1.0)
    """
    bucket: dict = {"minute": minute["start"], "ok": 0, "timed_out": 0, "failed": 0}
    for outcome, count in minute["outcomes"].items():
        bucket[_chart_class(outcome)] += count
    top = minute["wait_max_s"]
    bucket["wait_p50_s"] = histogram_percentile(minute["wait_hist"], 0.5, top)
    bucket["wait_p95_s"] = histogram_percentile(minute["wait_hist"], 0.95, top)
    bucket["wait_max_s"] = top
    return bucket


def sync_waiting() -> int:
    """``/query`` requests this pod is waiting on right now (all its processes).

    The ``gandalf_sync_waiting`` gauge, summed over the pod's gunicorn
    workers; each pod counts its own, so this is not the deployment's.
    """
    for metric in metrics_registry().collect():
        if metric.name == "gandalf_sync_waiting":
            return int(sum(sample.value for sample in metric.samples))
    return 0


def http_counts() -> list[dict]:
    """This pod's request counts by route and status, from its Prometheus registry."""
    rows: list[dict[str, Any]] = []
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
        "oldest_waiting_s": queue.oldest_waiting_s(),
        "sync_waiting": sync_waiting(),
        "pending_entries": pending,
        "dead_letters": queue.dead_letter_count(),
        "recent_dead_letters": queue.dead_letters(),
    }
    data["redis"] = RedisHealth.read(queue.client).as_dict()
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
        data["jobs"] = job_statistics(history, recent, now)
    if notifier is not None:
        data["alerts"] = {
            "slack": notifier.slack_enabled,
            "kinds": sorted(notifier.kinds),
            "recent": notifier.recent(recent),
        }
    return data


def job_statistics(history: JobHistory, recent: int, now: float) -> dict:
    """The page's job section: the recent table, the windows and the chart.

    The table is the newest *recent* jobs from the history stream.  The
    windows and the chart come from the per-minute and per-hour counters,
    which count every job however many there were: the windows up to an
    hour, and the chart, from minutes; the day from whole hours plus the
    minutes of the partial hour before them.
    """
    this_minute = int(now // 60) * 60
    # An hour's window starts inside the minute 60 back, so 61 cover it.
    minutes = history.minutes(this_minute - 60 * 60, 61)
    windows = {}
    for name, seconds in WINDOWS.items():
        since = now - seconds
        if seconds <= 3600:
            buckets = minutes
        else:
            first_hour = math.ceil(since / 3600) * 3600
            first_minute = math.ceil(since / 60) * 60
            hours = (int(now // 3600) * 3600 - first_hour) // 3600 + 1
            buckets = history.minutes(
                first_minute, (first_hour - first_minute) // 60
            ) + history.hours(first_hour, hours)
        windows[name] = aggregate_counters(buckets, seconds, now)
    records = history.recent(recent)
    return {
        "recent": records,
        "windows": windows,
        "per_minute": [minute_bucket(m) for m in minutes[1:]],
        "history_size": len(records),
    }


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
