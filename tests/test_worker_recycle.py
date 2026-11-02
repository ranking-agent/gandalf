"""Recycling a queue worker: staggered limits, and exec in place.

A worker that exited at its job limit was restarted by Kubernetes, which
counts every exit as a crash and backs off (10s doubling to 5 minutes) when
the container keeps exiting, and workers started together all hit one limit
together, so the pool could sit at zero.  These pin the fixes.
"""

from __future__ import annotations

import sys

import pytest

import gandalf.worker as worker_module
from gandalf.worker import jittered_max_jobs, restart_in_place


def test_zero_max_jobs_never_recycles():
    assert jittered_max_jobs(0, 500) == 0
    assert jittered_max_jobs(-1, 500) == 0


def test_jitter_stays_in_range_and_spreads():
    limits = {jittered_max_jobs(2000, 200) for _ in range(200)}
    assert min(limits) >= 2000 and max(limits) <= 2200
    assert len(limits) > 20  # not one shared limit


def test_no_jitter_is_the_plain_limit():
    assert jittered_max_jobs(2000, 0) == 2000
    assert jittered_max_jobs(2000, -5) == 2000


class _Execd(Exception):
    pass


def test_restart_in_place_execs_a_fresh_worker(monkeypatch, tmp_path):
    calls = []

    def fake_execv(path, argv):
        calls.append((path, argv))
        raise _Execd

    monkeypatch.setattr(worker_module.os, "execv", fake_execv)
    heartbeat = tmp_path / "heartbeat"
    with pytest.raises(_Execd):
        restart_in_place(str(heartbeat))
    assert calls == [(sys.executable, [sys.executable, "-m", "gandalf.worker"])]
    # Touched, so the liveness probe gives the new process its full window.
    assert heartbeat.exists()
