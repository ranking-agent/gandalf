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


def percentile(values: list[float], fraction: float) -> Optional[float]:
    """The nearest-rank percentile of *values*, or None for no values.

    The rank is rounded up, as nearest-rank has it; ``round()`` would round
    a half to even and take the median of five values from the second.

    Examples:
        >>> percentile([1, 2, 3, 4], 0.5)
        2
        >>> percentile([5, 1, 3], 0.95)
        5
        >>> percentile([1, 2, 3, 4, 5], 0.5)
        3
        >>> percentile([], 0.5) is None
        True
    """
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(fraction * len(ordered)))
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
    failed = outcomes["error"] + outcomes["poisoned"] + outcomes["refused"]
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
        "source": "sample",
    }


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
    """:func:`aggregate`'s figures from counter buckets (every job counted).

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
        >>> agg["duration_max_s"], agg["source"]
        (1.0, 'counters')
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
        "source": "counters",
    }


def _chart_class(outcome: Optional[str]) -> str:
    """Which of the chart's three stacks an outcome is drawn in."""
    if outcome == "ok":
        return "ok"
    return "timed_out" if outcome in ("timeout", "expired") else "failed"


def minute_bucket(minute: dict) -> dict:
    """A :func:`per_minute` bucket from one minute's counters.

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


def per_minute(records: list[dict], minutes: int, now: float) -> list[dict]:
    """Jobs finished in each of the last *minutes* minutes, oldest first.

    Each bucket is ``{"minute": epoch, "ok", "timed_out", "failed"}``:
    the three classes the page's chart stacks (``timeout`` and ``expired``
    are both the budget running out; ``error``, ``poisoned`` and
    ``refused`` all a failure).  It also has the queue wait of those jobs --
    ``wait_p50_s``, ``wait_p95_s``, ``wait_max_s`` -- which are None in a
    minute that finished none.

    Examples:
        >>> buckets = per_minute([{"finished_at": 95, "outcome": "expired"}], 2, now=130)
        >>> [(b["minute"], b["timed_out"]) for b in buckets]
        [(60, 1), (120, 0)]
        >>> recs = [{"finished_at": 125, "outcome": "ok", "wait_s": w} for w in (1, 2, 9)]
        >>> b = per_minute(recs, 1, now=130)[0]
        >>> b["wait_p50_s"], b["wait_p95_s"], b["wait_max_s"]
        (2.0, 9.0, 9.0)
    """
    start = int(now // 60) * 60 - 60 * (minutes - 1)
    buckets: list[dict] = [
        {"minute": start + 60 * i, "ok": 0, "timed_out": 0, "failed": 0}
        for i in range(minutes)
    ]
    waits: list[list[float]] = [[] for _ in range(minutes)]
    for r in records:
        index = int((r.get("finished_at", 0) - start) // 60)
        if 0 <= index < minutes:
            buckets[index][_chart_class(r.get("outcome"))] += 1
            if "wait_s" in r:
                waits[index].append(float(r["wait_s"]))
    for bucket, minute_waits in zip(buckets, waits):
        bucket["wait_p50_s"] = percentile(minute_waits, 0.5)
        bucket["wait_p95_s"] = percentile(minute_waits, 0.95)
        bucket["wait_max_s"] = max(minute_waits) if minute_waits else None
    return buckets


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

    The history stream keeps only the newest ``history_maxlen`` jobs.  A
    window or chart minute that sample reaches back past is computed from it
    exactly; one it does not (more jobs than that finished since) comes from
    the per-minute counters, which count every job.  Should the counters
    hold fewer jobs than the sample (they began after it, or Redis lost
    them), the sample is used: it is a floor either way.
    """
    wanted = max(recent, history.maxlen)
    records = history.recent(wanted)
    # Fewer than asked for: nothing has been trimmed, the sample is all.
    complete = len(records) < wanted
    oldest = float(records[-1].get("finished_at", now)) if records else now

    def covers(since: float) -> bool:
        return complete or oldest < since

    windows = {
        name: aggregate(records, seconds, now) for name, seconds in WINDOWS.items()
    }
    chart = per_minute(records, 60, now)
    # Minutes for the chart and every window up to an hour; hours past that.
    counters: dict[int, list[dict]] = {}

    def buckets(size: int) -> list[dict]:
        if size not in counters:
            count = (3600 if size == 60 else max(WINDOWS.values())) // size + 1
            first = int(now // size) * size - size * (count - 1)
            counters[size] = (
                history.minutes(first, count)
                if size == 60
                else history.hours(first, count)
            )
        return counters[size]

    for name, seconds in WINDOWS.items():
        if covers(now - seconds):
            continue
        since = now - seconds
        if seconds <= 3600:
            counted_buckets = buckets(60)
        else:
            # Whole hours, and the minutes of the partial hour before them.
            first_hour = math.ceil(since / 3600) * 3600
            first_minute = math.ceil(since / 60) * 60
            counted_buckets = history.minutes(
                first_minute, (first_hour - first_minute) // 60
            ) + [h for h in buckets(3600) if h["start"] >= first_hour]
        counted = aggregate_counters(counted_buckets, seconds, now)
        if counted["jobs"] >= windows[name]["jobs"]:
            windows[name] = counted
    if not covers(chart[0]["minute"]):
        by_minute = {m["start"]: m for m in buckets(60)}
        for i, bucket in enumerate(chart):
            counted_minute = by_minute.get(bucket["minute"])
            if covers(bucket["minute"]) or counted_minute is None:
                continue
            sampled = bucket["ok"] + bucket["timed_out"] + bucket["failed"]
            if counted_minute["jobs"] >= sampled:
                chart[i] = minute_bucket(counted_minute)
    return {
        "recent": records[:recent],
        "windows": windows,
        "per_minute": chart,
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
