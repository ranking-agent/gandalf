"""Garbage collection monitoring and control utilities."""

import gc
import logging
import time
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class GCMonitor:
    """Monitor garbage collection events and log timing information."""

    def __init__(self):
        self.gc_events = []
        self._start_time = None

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
                logger.debug(
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


@contextmanager
def gc_disabled():
    """Context manager to temporarily disable GC."""
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()
