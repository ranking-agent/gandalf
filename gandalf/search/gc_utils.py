"""Garbage collection monitoring and control utilities."""

import gc
import logging
import threading
import time
from contextlib import contextmanager
from typing import Optional


class GCMonitor:
    """Monitor garbage collection events and log timing information."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Args:
        logger: Logger to report slow collections to. Pass a query's own
            logger so the report lands in that query's TRAPI logs; defaults
            to this module's logger.
        """
        self.gc_events: list[dict] = []
        self._start_time = None
        self._logger = logger if logger is not None else logging.getLogger(__name__)

    def _gc_callback(self, phase, info):
        """Callback invoked by gc module on collection events."""
        if phase == "start":
            self._start_time = time.perf_counter()
        elif phase == "stop" and self._start_time is not None:
            duration = time.perf_counter() - self._start_time
            generation = info.get("generation", "?")
            collected = info.get("collected", 0)
            self.gc_events.append(
                {
                    "generation": generation,
                    "duration": duration,
                    "collected": collected,
                }
            )
            if duration > 0.1:  # Only log slow GC (>100ms)
                self._logger.debug(
                    "  [GC] Gen %s: %.2fs, collected %s objects",
                    generation,
                    duration,
                    collected,
                )
            self._start_time = None

    def start(self):
        """Start monitoring GC events."""
        self.gc_events = []
        gc.callbacks.append(self._gc_callback)

    def stop(self):
        """Stop monitoring and return summary."""
        if self._gc_callback in gc.callbacks:
            gc.callbacks.remove(self._gc_callback)
        return self.gc_events

    def summary(self):
        """Return summary of GC activity."""
        if not self.gc_events:
            return None
        total_time = sum(e["duration"] for e in self.gc_events)
        total_collected = sum(e["collected"] for e in self.gc_events)
        return {
            "total_collections": len(self.gc_events),
            "total_time": total_time,
            "total_collected": total_collected,
        }


# Holders of the GC pause (see pause_gc), and whether GC was on when the
# first of them arrived.
_pause_lock = threading.Lock()
_pause_holders = 0
_pause_restores_gc = False


def pause_gc() -> None:
    """Turn the cyclic GC off until every ``pause_gc`` has its ``resume_gc``.

    A lookup builds millions of container objects; with GC on, collections
    would keep scanning them while they are built.  The pause is counted, so
    concurrent queries (the server runs them on a thread pool) share it: GC
    comes back on only when the last holder resumes, and only if it was on
    when the first one paused.  Deciding per caller instead would let the
    first query to finish turn GC back on under a query still running.
    """
    global _pause_holders, _pause_restores_gc
    with _pause_lock:
        if _pause_holders == 0:
            _pause_restores_gc = gc.isenabled()
            gc.disable()
        _pause_holders += 1


def resume_gc() -> None:
    """Release one ``pause_gc``; the last release turns GC back on."""
    global _pause_holders
    with _pause_lock:
        if _pause_holders == 0:
            raise RuntimeError("resume_gc() without a matching pause_gc()")
        _pause_holders -= 1
        if _pause_holders == 0 and _pause_restores_gc:
            gc.enable()


@contextmanager
def gc_disabled():
    """Pause the cyclic GC for the duration of the block (see ``pause_gc``).

    >>> import gc
    >>> gc.isenabled()
    True
    >>> with gc_disabled():
    ...     with gc_disabled():
    ...         pass
    ...     gc.isenabled()
    False
    >>> gc.isenabled()
    True
    """
    pause_gc()
    try:
        yield
    finally:
        resume_gc()
