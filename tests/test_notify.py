"""Events, Slack payloads and the webhook sender, without Redis.

The alert log, throttling, the monitor and the worker's own events need a
Redis and live in ``tests/test_queue_redis.py``.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from gandalf.notify import (
    KINDS,
    Event,
    Notifier,
    SlackWebhook,
    enabled_kinds,
    slack_payload,
)


class _Hook(BaseHTTPRequestHandler):
    received: list = []
    status = 200

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("content-length", "0"))
        self.__class__.received.append(json.loads(self.rfile.read(length)))
        self.send_response(self.__class__.status)
        self.end_headers()

    def log_message(self, fmt, *args):
        pass


@pytest.fixture
def webhook_server():
    _Hook.received = []
    _Hook.status = 200
    server = HTTPServer(("127.0.0.1", 0), _Hook)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/hook", _Hook
    finally:
        server.shutdown()


def test_every_kind_has_a_severity():
    assert set(KINDS.values()) <= {"info", "warning", "critical"}
    assert Event("worker_lost", "x").severity == "critical"
    assert Event("worker_started", "x").severity == "info"


@pytest.mark.parametrize(
    "spec, expected",
    [
        ("", frozenset()),
        ("worker_lost", frozenset({"worker_lost"})),
        (" job_failed ,queue_backlog, ", frozenset({"job_failed", "queue_backlog"})),
        ("all", frozenset(KINDS)),
    ],
)
def test_enabled_kinds(spec, expected):
    assert enabled_kinds(spec) == expected


def test_enabled_kinds_rejects_a_typo():
    with pytest.raises(ValueError, match="worker_lsot"):
        enabled_kinds("worker_lsot")


def test_slack_payload_carries_everything():
    event = Event(
        "queue_backlog",
        "Queue backlog",
        "7 jobs are waiting.",
        {"waiting": 7, "workers": 2},
    )
    payload = slack_payload(
        event, environment="prod", status_url="https://g/status", suppressed=3
    )
    assert payload["text"] == ":warning: [prod] Queue backlog: 7 jobs are waiting."
    text = payload["blocks"][0]["text"]["text"]
    assert text.startswith(":warning: *[prod] Queue backlog*\n7 jobs are waiting.")
    assert "*waiting:* 7   *workers:* 2" in text
    assert "_3 similar events since the last message_" in text
    assert "<https://g/status|Open the status page>" in text


def test_slack_payload_minimal():
    payload = slack_payload(Event("worker_started", "Worker started"))
    assert payload["text"] == ":information_source: Worker started"
    assert (
        payload["blocks"][0]["text"]["text"] == ":information_source: *Worker started*"
    )


def test_webhook_posts_in_the_background(webhook_server):
    url, hook = webhook_server
    webhook = SlackWebhook(url)
    webhook.post({"text": "one"})
    webhook.post({"text": "two"})
    assert webhook.flush(5.0)
    assert [r["text"] for r in hook.received] == ["one", "two"]
    assert webhook.sent == 2 and webhook.failed == 0


def test_webhook_failure_is_counted_not_raised(webhook_server):
    url, hook = webhook_server
    hook.status = 500
    webhook = SlackWebhook(url)
    webhook.post({"text": "x"})
    assert webhook.flush(5.0)
    assert webhook.failed == 1 and webhook.sent == 0


def test_webhook_unreachable_is_counted_not_raised():
    webhook = SlackWebhook("http://127.0.0.1:1/hook", timeout_seconds=0.5)
    webhook.post({"text": "x"})
    assert webhook.flush(5.0)
    assert webhook.failed == 1


def test_notifier_sends_only_enabled_kinds(webhook_server):
    url, hook = webhook_server
    notifier = Notifier(
        None,
        webhook=SlackWebhook(url),
        kinds=frozenset({"worker_lost"}),
        environment="dev",
    )
    assert notifier.emit(Event("worker_started", "Worker started")) is False
    assert notifier.emit(Event("worker_lost", "Worker lost", "gone")) is True
    assert notifier.webhook.flush(5.0)
    assert len(hook.received) == 1
    assert hook.received[0]["text"] == ":rotating_light: [dev] Worker lost: gone"


def test_notifier_without_slack_or_redis_is_quiet():
    notifier = Notifier(None)
    assert notifier.slack_enabled is False
    assert notifier.emit(Event("job_failed", "Job failed")) is False
    assert notifier.recent() == []
