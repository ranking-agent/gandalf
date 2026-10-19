"""Delivering an /asyncquery result to its callback, with retries."""

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from gandalf.execute import post_callback


class _Receiver(BaseHTTPRequestHandler):
    statuses: list = []  # what to answer, in order; the last repeats
    received: list = []

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("content-length", "0"))
        self.__class__.received.append(self.rfile.read(length))
        status = self.statuses[min(len(self.received) - 1, len(self.statuses) - 1)]
        self.send_response(status)
        self.end_headers()

    def log_message(self, fmt, *args):
        pass


@pytest.fixture
def receiver():
    _Receiver.received = []
    server = HTTPServer(("127.0.0.1", 0), _Receiver)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/cb", _Receiver
    finally:
        server.shutdown()


def _no_sleep(seconds):
    _no_sleep.waited.append(seconds)


@pytest.fixture(autouse=True)
def _reset():
    _no_sleep.waited = []


def test_retries_a_5xx_then_delivers(receiver):
    url, r = receiver
    r.statuses = [500, 503, 200]
    assert post_callback(
        url, b'{"ok":1}', attempts=3, backoff_seconds=2.0, sleep=_no_sleep
    )
    assert len(r.received) == 3 and r.received[-1] == b'{"ok":1}'
    assert _no_sleep.waited == [2.0, 4.0]


def test_a_4xx_is_final(receiver):
    url, r = receiver
    r.statuses = [404]
    assert (
        post_callback(url, b"{}", attempts=3, backoff_seconds=1.0, sleep=_no_sleep)
        is False
    )
    assert len(r.received) == 1 and _no_sleep.waited == []


def test_gives_up_after_the_attempts(receiver):
    url, r = receiver
    r.statuses = [500]
    assert (
        post_callback(url, b"{}", attempts=3, backoff_seconds=1.0, sleep=_no_sleep)
        is False
    )
    assert len(r.received) == 3 and _no_sleep.waited == [1.0, 2.0]


def test_retries_a_connection_failure():
    assert (
        post_callback(
            "http://127.0.0.1:1/cb",
            b"{}",
            attempts=2,
            backoff_seconds=0.5,
            timeout_seconds=1.0,
            sleep=_no_sleep,
        )
        is False
    )
    assert _no_sleep.waited == [0.5]


def test_defaults_come_from_settings(receiver, monkeypatch):
    from gandalf.config import settings

    url, r = receiver
    r.statuses = [500, 200]
    monkeypatch.setattr(settings, "callback_attempts", 2)
    monkeypatch.setattr(settings, "callback_backoff_seconds", 0.0)
    assert post_callback(url, b"{}", sleep=_no_sleep)
    assert len(r.received) == 2 and _no_sleep.waited == [0.0]
