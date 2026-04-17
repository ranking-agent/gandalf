"""Ingest per-node and per-edge publication counts.

Input contract (JSONL, one object per line):

  Per-node file:  {"node_id": "<original id>", "count": <uint32>}
  Per-edge file:  {"edge_id": "<edge id from graph build>", "count": <uint32>}

Fail-loud rules (no silent skips):
  * Any unknown node_id / edge_id raises.
  * Duplicate node_id / edge_id in the input raises.
  * An input that doesn't cover every node (or every edge) raises.
  * ``count`` must be a non-negative integer that fits in uint32.

On success the script writes ``node_pub_counts.npy`` (uint32[N]) or
``edge_pub_counts.npy`` (uint32[E]) alongside the graph and records the
field in ``scoring_manifest.json``.
"""

from __future__ import annotations

import json
import logging
import struct
from pathlib import Path
from typing import Dict, Iterator, Tuple

import lmdb
import numpy as np

from gandalf.metadata.manifest import Manifest, load_manifest, save_manifest
from gandalf.metadata.store import FIELD_EDGE_PUB_COUNTS, FIELD_NODE_PUB_COUNTS
from gandalf.node_store import NodeStore

logger = logging.getLogger(__name__)


_UINT32_MAX = np.iinfo(np.uint32).max


class PubCountIngestError(ValueError):
    """Raised on any validation failure during pub-count ingest."""


def _iter_jsonl(path: Path) -> Iterator[Tuple[int, dict]]:
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise PubCountIngestError(
                    f"{path}:{lineno}: invalid JSON ({e})"
                ) from e
            if not isinstance(obj, dict):
                raise PubCountIngestError(
                    f"{path}:{lineno}: expected a JSON object, got {type(obj).__name__}"
                )
            yield lineno, obj


def _validate_count(value, path: Path, lineno: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PubCountIngestError(
            f"{path}:{lineno}: 'count' must be an integer, got {type(value).__name__}"
        )
    if value < 0 or value > _UINT32_MAX:
        raise PubCountIngestError(
            f"{path}:{lineno}: 'count' {value} out of uint32 range [0, {_UINT32_MAX}]"
        )
    return value


def _require_key(obj: dict, key: str, path: Path, lineno: int):
    if key not in obj:
        raise PubCountIngestError(f"{path}:{lineno}: missing required key {key!r}")
    return obj[key]


def _build_edge_id_to_idx(edge_ids_lmdb_path: Path) -> Dict[str, int]:
    """Build a transient edge_id (string) -> edge_idx (int) lookup dict.

    Scans the ``edge_ids.lmdb`` written by the graph build.  Raises if
    duplicate edge ID strings are found (indicates a broken graph build, but
    we check defensively since this map must be single-valued).
    """
    env = lmdb.open(
        str(edge_ids_lmdb_path),
        readonly=True,
        max_dbs=0,
        map_size=256 * 1024 * 1024 * 1024,
        readahead=False,
        lock=False,
    )
    mapping: Dict[str, int] = {}
    try:
        with env.begin(buffers=True) as txn:
            cursor = txn.cursor()
            for key_buf, val_buf in cursor:
                edge_idx = struct.unpack(">I", bytes(key_buf))[0]
                edge_id = bytes(val_buf).decode("utf-8")
                if edge_id in mapping:
                    raise PubCountIngestError(
                        f"Duplicate edge_id {edge_id!r} in {edge_ids_lmdb_path} "
                        f"(indices {mapping[edge_id]} and {edge_idx})"
                    )
                mapping[edge_id] = edge_idx
    finally:
        env.close()
    return mapping


def _infer_num_edges(graph_dir: Path) -> int:
    """Read the edge count from fwd_targets.npy header without mmap overhead."""
    arr = np.load(graph_dir / "fwd_targets.npy", mmap_mode="r")
    try:
        return int(arr.shape[0])
    finally:
        del arr


def _infer_num_nodes(graph_dir: Path) -> int:
    """Read num_nodes from the graph metadata pickle."""
    import pickle

    with open(graph_dir / "metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    return int(metadata["num_nodes"])


def ingest_node_pub_counts(graph_dir: Path, input_path: Path) -> Path:
    """Ingest per-node publication counts into ``<graph_dir>/node_pub_counts.npy``.

    Returns the path to the written array.
    """
    graph_dir = Path(graph_dir)
    input_path = Path(input_path)
    logger.info("Ingesting node pub counts from %s", input_path)

    num_nodes = _infer_num_nodes(graph_dir)
    counts = np.zeros(num_nodes, dtype=np.uint32)
    filled = np.zeros(num_nodes, dtype=bool)

    node_store = NodeStore(graph_dir / "node_store.lmdb", readonly=True)
    try:
        for lineno, obj in _iter_jsonl(input_path):
            node_id = _require_key(obj, "node_id", input_path, lineno)
            count = _validate_count(
                _require_key(obj, "count", input_path, lineno), input_path, lineno
            )
            if not isinstance(node_id, str):
                raise PubCountIngestError(
                    f"{input_path}:{lineno}: 'node_id' must be a string"
                )
            idx = node_store.get_node_idx(node_id)
            if idx is None:
                raise PubCountIngestError(
                    f"{input_path}:{lineno}: unknown node_id {node_id!r}"
                )
            if filled[idx]:
                raise PubCountIngestError(
                    f"{input_path}:{lineno}: duplicate node_id {node_id!r}"
                )
            counts[idx] = count
            filled[idx] = True
    finally:
        node_store.close()

    missing = int(num_nodes - filled.sum())
    if missing:
        raise PubCountIngestError(
            f"Input {input_path} does not cover every node: "
            f"{missing} of {num_nodes} nodes missing a pub count entry"
        )

    out_path = graph_dir / "node_pub_counts.npy"
    np.save(out_path, counts)

    manifest = load_manifest(graph_dir)
    manifest.set_field(
        FIELD_NODE_PUB_COUNTS,
        {
            "file": out_path.name,
            "dtype": "uint32",
            "shape": [num_nodes],
            "source": str(input_path),
            "total": int(counts.sum()),
            "max": int(counts.max(initial=0)),
            "nonzero": int((counts > 0).sum()),
        },
    )
    save_manifest(graph_dir, manifest)
    logger.info(
        "Wrote %s (%d nodes, total=%d, max=%d)",
        out_path,
        num_nodes,
        int(counts.sum()),
        int(counts.max(initial=0)),
    )
    return out_path


def ingest_edge_pub_counts(graph_dir: Path, input_path: Path) -> Path:
    """Ingest per-edge publication counts into ``<graph_dir>/edge_pub_counts.npy``.

    Returns the path to the written array.
    """
    graph_dir = Path(graph_dir)
    input_path = Path(input_path)
    logger.info("Ingesting edge pub counts from %s", input_path)

    num_edges = _infer_num_edges(graph_dir)

    edge_ids_lmdb = graph_dir / "edge_ids.lmdb"
    if not edge_ids_lmdb.exists():
        raise PubCountIngestError(
            f"edge_pub_counts ingest requires {edge_ids_lmdb}; "
            "rebuild the graph with edge IDs enabled."
        )
    logger.info("Building transient edge_id -> edge_idx map (%d edges)...", num_edges)
    edge_id_to_idx = _build_edge_id_to_idx(edge_ids_lmdb)
    if len(edge_id_to_idx) != num_edges:
        raise PubCountIngestError(
            f"{edge_ids_lmdb} has {len(edge_id_to_idx)} entries but graph has "
            f"{num_edges} edges; edge ID store is out of sync with the CSR."
        )

    counts = np.zeros(num_edges, dtype=np.uint32)
    filled = np.zeros(num_edges, dtype=bool)

    for lineno, obj in _iter_jsonl(input_path):
        edge_id = _require_key(obj, "edge_id", input_path, lineno)
        count = _validate_count(
            _require_key(obj, "count", input_path, lineno), input_path, lineno
        )
        if not isinstance(edge_id, str):
            raise PubCountIngestError(
                f"{input_path}:{lineno}: 'edge_id' must be a string"
            )
        idx = edge_id_to_idx.get(edge_id)
        if idx is None:
            raise PubCountIngestError(
                f"{input_path}:{lineno}: unknown edge_id {edge_id!r}"
            )
        if filled[idx]:
            raise PubCountIngestError(
                f"{input_path}:{lineno}: duplicate edge_id {edge_id!r}"
            )
        counts[idx] = count
        filled[idx] = True

    missing = int(num_edges - filled.sum())
    if missing:
        raise PubCountIngestError(
            f"Input {input_path} does not cover every edge: "
            f"{missing} of {num_edges} edges missing a pub count entry"
        )

    out_path = graph_dir / "edge_pub_counts.npy"
    np.save(out_path, counts)

    manifest = load_manifest(graph_dir)
    manifest.set_field(
        FIELD_EDGE_PUB_COUNTS,
        {
            "file": out_path.name,
            "dtype": "uint32",
            "shape": [num_edges],
            "source": str(input_path),
            "total": int(counts.sum()),
            "max": int(counts.max(initial=0)),
            "nonzero": int((counts > 0).sum()),
        },
    )
    save_manifest(graph_dir, manifest)
    logger.info(
        "Wrote %s (%d edges, total=%d, max=%d)",
        out_path,
        num_edges,
        int(counts.sum()),
        int(counts.max(initial=0)),
    )
    return out_path
