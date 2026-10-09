"""Events worth telling a person about: the alert log, Slack, and the monitor.

An :class:`Event` is something that happened in the deployment -- a worker
started or was lost, a job was retried or dead-lettered, the backlog crossed
a threshold.  :class:`Notifier.emit` records it on a capped Redis stream (the
"Alerts" section of ``/status``) and, when ``GANDALF_SLACK_WEBHOOK_URL`` is
set and the event's kind is enabled, posts it to Slack.

Who emits what, so each event fires once however many processes run:

* **Discrete events** are emitted by the process that sees them: the
  worker, for its own start and stop, a job it took over from a dead
  worker, a job it dead-lettered, a job that raised.
* **Threshold events** come from the :class:`Monitor`, a thread in every API
  process of which exactly one is active at a time (a Redis lock elects it;
  if that process dies the lock expires and another takes over).  It notices
  workers that disappeared without a clean exit (an OOM kill), a backlog
  over the threshold for longer than a moment (and its clearing), and a
  job stuck longer than a query should take.

Noisy kinds (``job_failed``, ``job_retried``) are throttled per kind through
Redis; a message after a quiet period says how many were suppressed.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import httpx
import orjson
import redis

from gandalf.config import settings
from gandalf.jobs import JobQueue, WorkerRegistry, _decode, _field

logger = logging.getLogger(__name__)

#: Every kind the system can emit, with its severity: info, warning, critical.
KINDS: dict[str, str] = {
    "worker_started": "info",
    "worker_stopped": "info",
    "worker_recycled": "info",
    "worker_restarted": "critical",
    "worker_lost": "critical",
    "job_retried": "warning",
    "job_dead_lettered": "critical",
    "job_failed": "warning",
    "queue_backlog": "warning",
    "queue_backlog_cleared": "info",
    "queue_stuck": "warning",
}

#: Kinds that can fire in bursts and go through the throttle: job kinds per
#: kind, worker kinds per kind and worker (a flapping worker sends one
#: message per interval; a scale-up of several workers announces each).
THROTTLED_KINDS = frozenset(
    {
        "job_failed",
        "job_retried",
        "worker_started",
        "worker_stopped",
        "worker_recycled",
        "worker_restarted",
    }
)

_EMOJI = {
    "info": ":information_source:",
    "warning": ":warning:",
    "critical": ":rotating_light:",
}

#: Redis keys shared by the notifier and the monitor.
LOG_STREAM = "gandalf:alerts"
KNOWN_WORKERS = "gandalf:notify:known_workers"
LEAVING_PREFIX = "gandalf:notify:leaving"
MONITOR_LOCK = "gandalf:notify:monitor"
BACKLOG_FLAG = "gandalf:notify:backlog"
BACKLOG_SINCE = "gandalf:notify:backlog_since"
STUCK_FLAG = "gandalf:notify:stuck"
THROTTLE_PREFIX = "gandalf:notify:throttle"


@dataclass(frozen=True)
class Event:
    """One thing that happened.

    Attributes:
        kind: A key of :data:`KINDS`.
        title: One line, the headline of the Slack message.
        detail: A sentence or two with the specifics.
        fields: Small structured facts (worker, job, counts) shown with it.
        at: When it happened, epoch seconds.
    """

    kind: str
    title: str
    detail: str = ""
    fields: dict = field(default_factory=dict)
    at: float = field(default_factory=time.time)

    @property
    def severity(self) -> str:
        """``info``, ``warning`` or ``critical``, from the kind."""
        return KINDS[self.kind]

    @property
    def throttle_key(self) -> str:
        """What the throttle counts this event under.

        Examples:
            >>> Event("job_failed", "x").throttle_key
            'job_failed'
            >>> Event("worker_started", "x", fields={"worker": "w1"}).throttle_key
            'worker_started:w1'
        """
        if self.kind.startswith("worker_") and self.fields.get("worker"):
            return f"{self.kind}:{self.fields['worker']}"
        return self.kind


def enabled_kinds(spec: str) -> frozenset[str]:
    """The kinds a comma-separated setting names, ``all`` meaning every kind.

    Examples:
        >>> sorted(enabled_kinds("worker_lost, job_failed"))
        ['job_failed', 'worker_lost']
        >>> enabled_kinds("all") == frozenset(KINDS)
        True
        >>> enabled_kinds("")
        frozenset()
    """
    names = {name.strip() for name in spec.split(",") if name.strip()}
    if "all" in names:
        return frozenset(KINDS)
    unknown = names - set(KINDS)
    if unknown:
        raise ValueError(
            f"unknown Slack event kinds {sorted(unknown)}; known: {sorted(KINDS)}"
        )
    return frozenset(names)


def slack_payload(
    event: Event,
    *,
    environment: str = "",
    status_url: str = "",
    suppressed: int = 0,
) -> dict:
    """The incoming-webhook body for *event*: a fallback text plus one block.

    Examples:
        >>> payload = slack_payload(Event("worker_lost", "Worker lost", "it died",
        ...     fields={"worker": "w1"}), environment="prod", status_url="http://g/status")
        >>> payload["text"]
        ':rotating_light: [prod] Worker lost: it died'
        >>> "*worker:* w1" in payload["blocks"][0]["text"]["text"]
        True
    """
    prefix = f"[{environment}] " if environment else ""
    emoji = _EMOJI[event.severity]
    lines = [f"{emoji} *{prefix}{event.title}*"]
    if event.detail:
        lines.append(event.detail)
    if event.fields:
        lines.append("   ".join(f"*{k}:* {v}" for k, v in event.fields.items()))
    if suppressed:
        lines.append(
            f"_{suppressed} similar event{'s' if suppressed != 1 else ''} since the last message_"
        )
    if status_url:
        lines.append(f"<{status_url}|Open the status page>")
    text = f"{emoji} {prefix}{event.title}" + (
        f": {event.detail}" if event.detail else ""
    )
    return {
        "text": text,
        "blocks": [
            {"type": "section", "text": {"type": "mrkdwn", "text": "\n".join(lines)}}
        ],
    }


class SlackWebhook:
    """Posts payloads to a Slack incoming webhook from a background thread.

    Sending never blocks the caller (a worker between jobs, the monitor) and
    a failure is logged, not raised: an alert that cannot be delivered must
    not take the thing it is alerting about down with it.
    """

    def __init__(self, url: str, *, timeout_seconds: float = 5.0):
        self.url = url
        self.timeout = timeout_seconds
        self._pending: list[dict] = []
        self._wake = threading.Condition()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="gandalf-slack"
        )
        self._thread.start()
        self.sent = 0
        self.failed = 0

    def post(self, payload: dict) -> None:
        """Queue *payload* for delivery."""
        with self._wake:
            self._pending.append(payload)
            self._wake.notify()

    def flush(self, timeout: float = 10.0) -> bool:
        """Wait until everything queued has been attempted; for tests and exit."""
        deadline = time.monotonic() + timeout
        with self._wake:
            while self._pending and time.monotonic() < deadline:
                self._wake.wait(0.05)
            return not self._pending

    def _loop(self) -> None:
        while True:
            with self._wake:
                while not self._pending:
                    self._wake.wait()
                payload = self._pending[0]
            self._deliver(payload)
            with self._wake:
                self._pending.pop(0)
                self._wake.notify_all()

    def _deliver(self, payload: dict) -> None:
        try:
            response = httpx.post(self.url, json=payload, timeout=self.timeout)
            response.raise_for_status()
        except Exception as exc:
            self.failed += 1
            logger.warning("Slack webhook failed: %s", exc)
            return
        self.sent += 1


class Notifier:
    """Records events and forwards the enabled kinds to Slack.

    Args:
        client: Redis, for the alert log and the throttle; None keeps only
            Slack (nothing is then throttled).
        webhook: Where to post, or None to log only.
        kinds: The kinds Slack receives.
        environment: A label put in front of every Slack title.
        status_url: Linked from every Slack message, if set.
        throttle_seconds: Minimum gap between Slack messages of one
            throttled kind.
        log_maxlen: Alert-log length.
    """

    def __init__(
        self,
        client: Optional[redis.Redis],
        *,
        webhook: Optional[SlackWebhook] = None,
        kinds: frozenset[str] = frozenset(),
        environment: str = "",
        status_url: str = "",
        throttle_seconds: float = 300.0,
        log_maxlen: int = 500,
    ):
        self._r = client
        self.webhook = webhook
        self.kinds = kinds
        self.environment = environment
        self.status_url = status_url
        self.throttle_seconds = throttle_seconds
        self.log_maxlen = log_maxlen

    @classmethod
    def from_settings(cls, client: Optional[redis.Redis]) -> "Notifier":
        """A notifier configured from the ``GANDALF_SLACK_*`` settings."""
        webhook = (
            SlackWebhook(settings.slack_webhook_url)
            if settings.slack_webhook_url
            else None
        )
        status_url = settings.slack_status_url or (
            f"{settings.server_url.rstrip('/')}/status" if settings.server_url else ""
        )
        return cls(
            client,
            webhook=webhook,
            kinds=enabled_kinds(settings.slack_events) if webhook else frozenset(),
            environment=settings.slack_environment,
            status_url=status_url,
            throttle_seconds=settings.slack_throttle_seconds,
            log_maxlen=settings.alert_log_maxlen,
        )

    @property
    def slack_enabled(self) -> bool:
        """Whether a webhook is configured."""
        return self.webhook is not None

    def emit(self, event: Event) -> bool:
        """Log *event* and send it to Slack if its kind is enabled and not throttled.

        Returns:
            Whether a Slack message was queued.
        """
        logger.info("alert %s: %s %s", event.kind, event.title, event.detail)
        self._log(event)
        if self.webhook is None or event.kind not in self.kinds:
            return False
        suppressed = self._throttle(event)
        if suppressed is None:
            return False
        self.webhook.post(
            slack_payload(
                event,
                environment=self.environment,
                status_url=self.status_url,
                suppressed=suppressed,
            )
        )
        return True

    def _log(self, event: Event) -> None:
        if self._r is None:
            return
        self._r.xadd(
            LOG_STREAM,
            {
                "kind": event.kind,
                "severity": event.severity,
                "title": event.title,
                "detail": event.detail,
                "fields": _field(event.fields),
                "at": repr(event.at),
            },
            maxlen=self.log_maxlen,
            approximate=True,
        )

    def _throttle(self, event: Event) -> Optional[int]:
        """None if one like *event* was sent too recently, else the suppressed count."""
        if (
            self._r is None
            or event.kind not in THROTTLED_KINDS
            or self.throttle_seconds <= 0
        ):
            return 0
        key = f"{THROTTLE_PREFIX}:{event.throttle_key}"
        counter = f"{key}:suppressed"
        if self._r.set(key, "1", nx=True, px=int(self.throttle_seconds * 1000)):
            suppressed = self._r.getdel(counter)
            return int(suppressed) if suppressed else 0
        self._r.incr(counter)
        return None

    def recent(self, count: int = 50) -> list[dict]:
        """The newest alerts, newest first, for the status page."""
        if self._r is None:
            return []
        alerts = []
        for entry_id, fields in self._r.xrevrange(LOG_STREAM, count=count):
            alert = _decode(fields)
            alert["at"] = float(alert["at"])
            alert["fields"] = (
                orjson.loads(alert["fields"]) if alert.get("fields") else {}
            )
            alert["entry_id"] = entry_id.decode()
            alerts.append(alert)
        return alerts


def mark_worker_leaving(client: redis.Redis, name: str) -> None:
    """Tell the monitor *name* is exiting on purpose, so it is not reported lost."""
    pipe = client.pipeline(transaction=True)
    pipe.set(f"{LEAVING_PREFIX}:{name}", "1", ex=120)
    pipe.hdel(KNOWN_WORKERS, name)
    pipe.execute()


class Monitor:
    """The threshold watcher: one active instance across every API process.

    Args:
        client: Redis.
        queue: The job stream.
        registry: The worker registry.
        notifier: Where events go.
        interval_seconds: How often the active monitor looks.
        lag_per_worker: Backlog (stream lag) per live worker at or above
            which the queue counts as backed up; with no live worker the
            threshold is this, as for one.
        backlog_seconds: How long the backlog must stay at or above the
            threshold, on every look, before ``queue_backlog`` fires.
        stuck_seconds: A job delivered and unacknowledged this long is stuck.
    """

    def __init__(
        self,
        client: redis.Redis,
        queue: JobQueue,
        registry: WorkerRegistry,
        notifier: Notifier,
        *,
        interval_seconds: float = 15.0,
        lag_per_worker: int = 50,
        backlog_seconds: float = 60.0,
        stuck_seconds: float = 1800.0,
    ):
        self._r = client
        self.queue = queue
        self.registry = registry
        self.notifier = notifier
        self.interval = interval_seconds
        self.lag_per_worker = lag_per_worker
        self.backlog_seconds = backlog_seconds
        self.stuck_seconds = stuck_seconds
        self.identity = uuid.uuid4().hex
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Run :meth:`tick` every interval on a daemon thread."""
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="gandalf-monitor"
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval + 1)

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            # A Redis outage must not kill the thread; the next tick retries.
            try:
                self.tick()
            except redis.RedisError as exc:
                logger.warning("Monitor tick skipped: %s", exc)

    def is_leader(self) -> bool:
        """Take or renew the lock; False means another process holds it."""
        ttl = int(self.interval * 3 * 1000)
        if self._r.set(MONITOR_LOCK, self.identity, nx=True, px=ttl):
            return True
        if self._r.get(MONITOR_LOCK) == self.identity.encode():
            self._r.pexpire(MONITOR_LOCK, ttl)
            return True
        return False

    def tick(self) -> list[Event]:
        """One look at the deployment; returns the events it emitted."""
        if not self.is_leader():
            return []
        now = time.time()
        events = self._check_workers(now)
        events += self._check_backlog(now)
        events += self._check_stuck()
        for event in events:
            self.notifier.emit(event)
        return events

    def _check_workers(self, now: float) -> list[Event]:
        """Workers in the known set that no longer report: lost, unless leaving."""
        reports = {r["name"]: r for r in self.registry.workers()}
        known = {
            k.decode(): orjson.loads(v)
            for k, v in self._r.hgetall(KNOWN_WORKERS).items()
        }
        events = []
        for name, last in known.items():
            if name in reports:
                continue
            self._r.hdel(KNOWN_WORKERS, name)
            if self._r.exists(f"{LEAVING_PREFIX}:{name}"):
                continue
            fields = {"worker": name, "jobs done": last.get("jobs_done", 0)}
            if last.get("state") == "running":
                query = last.get("job_query") or {}
                fields["was running"] = (
                    f"{last.get('job_id', '')[:8]} "
                    f"({query.get('nodes', '?')} nodes / {query.get('edges', '?')} edges, "
                    f"ids {', '.join(query.get('ids') or []) or 'none'}) "
                    f"for {now - float(last.get('job_started_at') or now):.0f}s"
                )
                fields["anon RSS"] = f"{int(last.get('rss_anon_kb', 0)) // 1024} MB"
            events.append(
                Event(
                    "worker_lost",
                    "Worker lost",
                    f"Worker {name} stopped reporting without a clean exit. "
                    "A kernel OOM kill is the usual cause; its job will be retried "
                    "by another worker.",
                    fields,
                )
            )
        if reports:
            self._r.hset(
                KNOWN_WORKERS,
                mapping={
                    name: orjson.dumps(
                        {
                            k: r.get(k)
                            for k in (
                                "state",
                                "job_id",
                                "job_query",
                                "job_started_at",
                                "jobs_done",
                                "rss_anon_kb",
                            )
                        }
                    )
                    for name, r in reports.items()
                },
            )
        return events

    def lag_threshold(self, workers: int) -> int:
        """The backlog that counts as backed up with *workers* live workers.

        Examples:
            >>> m = Monitor.__new__(Monitor); m.lag_per_worker = 50
            >>> m.lag_threshold(3), m.lag_threshold(0)
            (150, 50)
        """
        return self.lag_per_worker * max(workers, 1)

    def _check_backlog(self, now: float) -> list[Event]:
        """``queue_backlog`` once the lag has stayed over the threshold long enough.

        The start of the current over-threshold spell is kept in Redis, so a
        new leader carries it on; one look under the threshold resets it.
        """
        stats = self.queue.stats()
        alerting = bool(self._r.exists(BACKLOG_FLAG))
        alive = self._r.zcard(f"{self.registry.prefix}:index")
        threshold = self.lag_threshold(alive)
        if stats.lag < threshold:
            self._r.delete(BACKLOG_SINCE)
        elif not alerting:
            self._r.set(BACKLOG_SINCE, repr(now), nx=True)
            since = float(self._r.get(BACKLOG_SINCE) or now)
            if now - since >= self.backlog_seconds:
                self._r.set(BACKLOG_FLAG, "1")
                self._r.delete(BACKLOG_SINCE)
                return [
                    Event(
                        "queue_backlog",
                        "Queue backlog",
                        f"{stats.lag} jobs are waiting for a worker, at or over the "
                        f"threshold ({threshold}: {self.lag_per_worker} per worker) "
                        f"for {now - since:.0f}s; {stats.pending} running on "
                        f"{alive} workers.",
                        {
                            "waiting": stats.lag,
                            "running": stats.pending,
                            "workers": alive,
                            "threshold": threshold,
                        },
                    )
                ]
        if stats.lag == 0 and alerting:
            self._r.delete(BACKLOG_FLAG)
            return [
                Event(
                    "queue_backlog_cleared",
                    "Queue backlog cleared",
                    f"Nothing is waiting; {stats.pending} running on {alive} workers.",
                    {"running": stats.pending, "workers": alive},
                )
            ]
        return []

    def _check_stuck(self) -> list[Event]:
        pending = self.queue.pending_entries()
        oldest = max(pending, key=lambda p: p["idle_s"], default=None)
        flagged = self._r.get(STUCK_FLAG)
        if oldest is not None and oldest["idle_s"] >= self.stuck_seconds:
            if flagged == oldest["entry_id"].encode():
                return []
            self._r.set(STUCK_FLAG, oldest["entry_id"])
            return [
                Event(
                    "queue_stuck",
                    "Job stuck",
                    f"A job has been with worker {oldest['consumer']} for "
                    f"{oldest['idle_s'] / 60:.0f} minutes without finishing "
                    f"(threshold {self.stuck_seconds / 60:.0f}).",
                    {
                        "entry": oldest["entry_id"],
                        "worker": oldest["consumer"],
                        "deliveries": oldest["deliveries"],
                    },
                )
            ]
        if flagged is not None:
            self._r.delete(STUCK_FLAG)
        return []


def utc(at: float) -> str:
    """An event time as ISO 8601 UTC, for logs and the page."""
    return datetime.fromtimestamp(at, timezone.utc).isoformat()
