"""Per-request profiler for gandalf lookups.

Captures named, nested stage timings (e.g. BMT init, per-edge query, path
reconstruction, response building, LMDB enrichment) and emits the data as
TRAPI ``LogEntry`` dicts on ``message.logs`` so any TRAPI client can consume
it without a separate UI.

The profiler is opt-in. ``current_profiler()`` returns a ``NullProfiler``
when no profiling context is active, so call sites can be unconditional and
pay only an attribute lookup + no-op generator when profiling is off.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Iterator, Optional, Union

logger = logging.getLogger(__name__)


_BATCH_SIZES_CAP = 1024


class NullProfiler:
    """No-op profiler. Every method is a cheap no-op or empty result."""

    enabled = False

    @contextmanager
    def stage(self, name: str, **fields: Any) -> Iterator["NullProfiler"]:
        yield self

    def event(self, name: str, **fields: Any) -> None:
        return None

    def add_metric(self, key: str, value: Any) -> None:
        return None

    def incr(self, key: str, n: int = 1) -> None:
        return None

    @contextmanager
    def lmdb_call(self, kind: str, n_keys: int) -> Iterator[None]:
        yield

    def to_log_entries(self) -> list[dict]:
        return []

    def to_dict(self) -> dict:
        return {}


class Profiler:
    """Active profiler. Builds a tree of timed stages with metrics + events."""

    enabled = True

    def __init__(self, root_name: str = "lookup") -> None:
        self._t_origin = time.perf_counter()
        self._root: dict = {
            "name": root_name,
            "t_start_ms": 0.0,
            "duration_ms": None,
            "fields": {},
            "metrics": {},
            "events": [],
            "children": [],
        }
        self._stack: list[dict] = [self._root]
        self._lmdb_totals: dict[str, Any] = {
            "calls": 0,
            "total_keys": 0,
            "total_ms": 0.0,
            "batch_sizes": [],
            "by_kind": {},
        }

    @contextmanager
    def stage(self, name: str, **fields: Any) -> Iterator["Profiler"]:
        node: dict = {
            "name": name,
            "t_start_ms": (time.perf_counter() - self._t_origin) * 1000.0,
            "duration_ms": None,
            "fields": dict(fields),
            "metrics": {},
            "events": [],
            "children": [],
        }
        self._stack[-1]["children"].append(node)
        self._stack.append(node)
        t0 = time.perf_counter()
        try:
            yield self
        finally:
            node["duration_ms"] = (time.perf_counter() - t0) * 1000.0
            self._stack.pop()

    def event(self, name: str, **fields: Any) -> None:
        self._stack[-1]["events"].append(
            {
                "name": name,
                "t_offset_ms": (time.perf_counter() - self._t_origin) * 1000.0,
                **fields,
            }
        )

    def add_metric(self, key: str, value: Any) -> None:
        self._stack[-1]["metrics"][key] = value

    def incr(self, key: str, n: int = 1) -> None:
        m = self._stack[-1]["metrics"]
        m[key] = m.get(key, 0) + n

    @contextmanager
    def lmdb_call(self, kind: str, n_keys: int) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            tot = self._lmdb_totals
            tot["calls"] += 1
            tot["total_keys"] += n_keys
            tot["total_ms"] += dt_ms
            if len(tot["batch_sizes"]) < _BATCH_SIZES_CAP:
                tot["batch_sizes"].append(n_keys)
            kind_tot = tot["by_kind"].setdefault(
                kind, {"calls": 0, "total_keys": 0, "total_ms": 0.0}
            )
            kind_tot["calls"] += 1
            kind_tot["total_keys"] += n_keys
            kind_tot["total_ms"] += dt_ms

    def finalize(self) -> None:
        """Close the root node and attach LMDB totals. Safe to call twice."""
        if self._root["duration_ms"] is None:
            self._root["duration_ms"] = (
                time.perf_counter() - self._t_origin
            ) * 1000.0
        self._root["lmdb"] = dict(self._lmdb_totals)

    def to_dict(self) -> dict:
        self.finalize()
        return self._root

    def to_log_entries(self) -> list[dict]:
        """Render the profile as TRAPI LogEntry dicts.

        Emits one ``DEBUG`` entry per stage with ``code: ProfileStage`` and
        a final ``INFO`` summary entry with ``code: ProfileSummary`` whose
        ``message`` is a JSON blob of the full tree.
        """
        self.finalize()
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        entries: list[dict] = []

        def _walk(node: dict, path: tuple[str, ...]) -> None:
            cur_path = path + (node["name"],)
            label = ":".join(cur_path)
            dur = node.get("duration_ms")
            metric_bits = []
            for k, v in node.get("metrics", {}).items():
                if isinstance(v, (int, float, str, bool)) or v is None:
                    metric_bits.append(f"{k}={v}")
            field_bits = []
            for k, v in node.get("fields", {}).items():
                if isinstance(v, (int, float, str, bool)) or v is None:
                    field_bits.append(f"{k}={v}")
            extras = []
            if field_bits:
                extras.append(", ".join(field_bits))
            if metric_bits:
                extras.append(", ".join(metric_bits))
            extras_str = f" ({'; '.join(extras)})" if extras else ""
            dur_str = f"{dur:.1f}ms" if isinstance(dur, (int, float)) else "?ms"
            entries.append(
                {
                    "timestamp": ts,
                    "level": "DEBUG",
                    "code": "ProfileStage",
                    "message": f"{label} {dur_str}{extras_str}",
                }
            )
            for child in node.get("children", []):
                _walk(child, cur_path)

        for child in self._root.get("children", []):
            _walk(child, (self._root["name"],))

        try:
            summary_blob = json.dumps(self._root, default=str)
        except Exception:
            summary_blob = json.dumps({"error": "profile_serialization_failed"})
        entries.append(
            {
                "timestamp": ts,
                "level": "INFO",
                "code": "ProfileSummary",
                "message": summary_blob,
            }
        )
        return entries


# A single shared NullProfiler is fine — it has no per-request state.
_NULL_PROFILER = NullProfiler()

profiler_var: ContextVar[Union[Profiler, NullProfiler]] = ContextVar(
    "gandalf_profiler", default=_NULL_PROFILER
)


def current_profiler() -> Union[Profiler, NullProfiler]:
    """Return the profiler bound to the current async/sync context."""
    return profiler_var.get()


@contextmanager
def set_profiler(prof: Union[Profiler, NullProfiler]) -> Iterator[Union[Profiler, NullProfiler]]:
    """Bind *prof* as the active profiler for the duration of the block."""
    token = profiler_var.set(prof)
    try:
        yield prof
    finally:
        profiler_var.reset(token)


def install_lmdb_hook(store: Any) -> Optional[tuple]:
    """Wrap ``store.get`` and ``store.get_batch`` to record into the active
    profiler's ``lmdb_call``.

    Returns ``(orig_get, orig_get_batch)`` so the caller can restore the
    original methods in a ``finally`` block. Returns ``None`` if the hook
    could not be installed (no store, or already wrapped by another profile
    in flight).

    The wrap is per-instance: it does not affect other ``LMDBPropertyStore``
    instances. The shared singleton is briefly observable to other concurrent
    requests; we guard against double-wrapping with a sentinel attribute.
    """
    if store is None:
        return None
    if getattr(store, "_gandalf_profiled", False):
        # Another profiler is already wrapping this store. Skip to avoid
        # double-counting; the caller can note this in the profile.
        return None

    orig_get = store.get
    orig_get_batch = store.get_batch
    prof = current_profiler()

    def get(edge_idx):
        with prof.lmdb_call("get", 1):
            return orig_get(edge_idx)

    def get_batch(edge_indices):
        idxs = list(edge_indices)
        with prof.lmdb_call("get_batch", len(idxs)):
            return orig_get_batch(idxs)

    store.get = get
    store.get_batch = get_batch
    store._gandalf_profiled = True
    return (orig_get, orig_get_batch)


def restore_lmdb_hook(store: Any, originals: Optional[tuple]) -> None:
    """Restore methods saved by :func:`install_lmdb_hook`."""
    if store is None or originals is None:
        return
    orig_get, orig_get_batch = originals
    store.get = orig_get
    store.get_batch = orig_get_batch
    if hasattr(store, "_gandalf_profiled"):
        try:
            delattr(store, "_gandalf_profiled")
        except AttributeError:
            pass
