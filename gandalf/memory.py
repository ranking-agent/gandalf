"""Process memory: what is private, and giving freed heap back to the OS.

Dependency-free so the graph loader can use it; ``gandalf.metrics``
re-exports these for the server and the worker.

The number that matters is **anonymous** RSS.  The graph is memory-mapped,
so its pages are file-backed, shared by every process on the node that maps
the same files, and reclaimable; anonymous pages are the process's own and
are what an OOM kill is about.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging

logger = logging.getLogger(__name__)


def rss_kb() -> int:
    """Current resident set size in KB, anonymous and file-backed together."""
    return _proc_status_kb(b"VmRSS:")


def rss_anon_kb() -> int:
    """Anonymous (private, non-file-backed) RSS in KB.

    Reads RssAnon from /proc/self/status on Linux -- the precise metric for
    OOM risk, since file-backed pages (the mapped graph, LMDB, .so files)
    are reclaimable but anonymous pages are not.  Returns -1 elsewhere.
    """
    return _proc_status_kb(b"RssAnon:")


def _proc_status_kb(field: bytes) -> int:
    try:
        with open("/proc/self/status", "rb") as f:
            for line in f:
                if line.startswith(field):
                    return int(line.split()[1])
    except OSError:
        pass
    return -1


_libc = None
_malloc_trim_missing = False


def trim_heap() -> bool:
    """Give the memory freed after a large query back to the operating system.

    glibc keeps freed heap pages around for reuse, so a process that built a
    multi-hundred-megabyte response stays that big after it is gone.
    ``malloc_trim(0)`` returns the free pages; measured on a 373 MB
    response, anonymous RSS fell from 560 MB to 200 MB in about 10 ms.

    Returns:
        Whether a trim happened (False on a libc without ``malloc_trim``).
    """
    global _libc, _malloc_trim_missing
    if _malloc_trim_missing:
        return False
    if _libc is None:
        name = ctypes.util.find_library("c")
        try:
            _libc = ctypes.CDLL(name) if name else ctypes.CDLL(None)
            _libc.malloc_trim.argtypes = [ctypes.c_size_t]
            _libc.malloc_trim.restype = ctypes.c_int
        except (OSError, AttributeError):
            _malloc_trim_missing = True
            return False
    _libc.malloc_trim(0)
    return True


def trim_heap_if_large(threshold_mb: int) -> bool:
    """Trim when anonymous RSS is above *threshold_mb* (0 disables).

    Reading the size costs microseconds; the trim itself scales with the
    heap, so a small process is left alone.
    """
    if threshold_mb <= 0 or rss_anon_kb() < threshold_mb * 1024:
        return False
    return trim_heap()


class MemoryLedger:
    """Anonymous memory taken by each stage of a load, for the log and the page.

    Examples:
        >>> ledger = MemoryLedger()
        >>> junk = bytearray(50 * 1024 * 1024)
        >>> ledger.mark("junk")
        >>> ledger.stages["junk"] >= 45 * 1024
        True
        >>> ledger.total_kb() >= 45 * 1024
        True
    """

    def __init__(self) -> None:
        self.stages: dict[str, int] = {}
        self._last = rss_anon_kb()
        self.start_kb = self._last

    def mark(self, stage: str) -> None:
        """Attribute the anonymous memory taken since the last mark to *stage*."""
        now = rss_anon_kb()
        self.stages[stage] = self.stages.get(stage, 0) + (now - self._last)
        self._last = now

    def total_kb(self) -> int:
        """Anonymous memory taken since the ledger started, in KB."""
        return rss_anon_kb() - self.start_kb

    def summary(self, min_mb: int = 16) -> str:
        """One line naming the stages that took at least *min_mb*."""
        big = sorted(
            ((kb, name) for name, kb in self.stages.items() if kb >= min_mb * 1024),
            reverse=True,
        )
        parts = ", ".join(f"{name} +{kb // 1024} MB" for kb, name in big)
        return f"+{self.total_kb() // 1024} MB private" + (
            f" ({parts})" if parts else ""
        )
