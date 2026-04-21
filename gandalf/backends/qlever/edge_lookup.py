"""QLever-specific edge ID to CSR index lookup."""

import logging
import shutil
import struct
from pathlib import Path
from typing import TYPE_CHECKING

import lmdb

DEFAULT_READ_MAP_SIZE = 256 * 1024 * 1024 * 1024
INITIAL_WRITE_MAP_SIZE = 4 * 1024 * 1024 * 1024

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from gandalf.graph import CSRGraph


def _encode_idx(idx: int) -> bytes:
    return struct.pack(">I", idx)


def _decode_idx(payload: bytes) -> int:
    return struct.unpack(">I", payload)[0]


def _put_with_resize(env, txn, key: bytes, val: bytes, pending):
    try:
        txn.put(key, val)
        pending.append((key, val))
        return txn
    except lmdb.MapFullError:
        txn.abort()
        new_size = env.info()["map_size"] * 2
        env.set_mapsize(new_size)
        logger.warning(
            "    EdgeIdLookup LMDB: map full, resized to %.0f GB",
            new_size / (1024**3),
        )
        txn = env.begin(write=True)
        for pending_key, pending_val in pending:
            txn.put(pending_key, pending_val)
        txn.put(key, val)
        pending.append((key, val))
        return txn


def synthetic_qlever_edge_id(fwd_edge_idx: int) -> str:
    return f"urn:gandalf:qlever-edge:{int(fwd_edge_idx)}"


def resolved_qlever_edge_id(edge_id: str | None, fwd_edge_idx: int) -> str:
    if edge_id is None:
        return synthetic_qlever_edge_id(fwd_edge_idx)
    return edge_id if isinstance(edge_id, str) else str(edge_id)


class EdgeIdLookup:
    """Read-only lookup from edge ID string to forward CSR edge index."""

    def __init__(self, path: str | Path, readonly: bool = True):
        self.path = Path(path)
        self._env = lmdb.open(
            str(self.path),
            readonly=readonly,
            max_dbs=0,
            map_size=DEFAULT_READ_MAP_SIZE if readonly else INITIAL_WRITE_MAP_SIZE,
            readahead=False,
            lock=not readonly,
        )

    def get(self, edge_id: str) -> int | None:
        with self._env.begin(buffers=True) as txn:
            payload = txn.get(edge_id.encode("utf-8"))
            if payload is None:
                return None
            return _decode_idx(bytes(payload))

    def get_many(self, edge_ids: list[str]) -> tuple[dict[str, int], list[str]]:
        results: dict[str, int] = {}
        missing: list[str] = []
        with self._env.begin(buffers=True) as txn:
            for edge_id in edge_ids:
                payload = txn.get(edge_id.encode("utf-8"))
                if payload is None:
                    missing.append(edge_id)
                    continue
                results[edge_id] = _decode_idx(bytes(payload))
        return results, missing

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None

    def __enter__(self) -> "EdgeIdLookup":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> bool:
        self.close()
        return False

    def __del__(self) -> None:
        self.close()

    @classmethod
    def build_from_edge_ids_lmdb(
        cls,
        edge_ids_path: str | Path,
        output_path: str | Path,
        commit_every: int = 50_000,
    ) -> "EdgeIdLookup":
        edge_ids_path = Path(edge_ids_path)
        output_path = Path(output_path)

        if output_path.exists():
            shutil.rmtree(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        source_env = lmdb.open(
            str(edge_ids_path),
            readonly=True,
            max_dbs=0,
            map_size=DEFAULT_READ_MAP_SIZE,
            readahead=False,
            lock=False,
        )
        target_env = lmdb.open(
            str(output_path),
            readonly=False,
            max_dbs=0,
            map_size=INITIAL_WRITE_MAP_SIZE,
            readahead=False,
        )

        pending = []
        txn = target_env.begin(write=True)
        count = 0
        try:
            with source_env.begin(buffers=True) as source_txn:
                cursor = source_txn.cursor()
                for key, value in cursor:
                    edge_id = bytes(value)
                    txn = _put_with_resize(
                        target_env,
                        txn,
                        edge_id,
                        bytes(key),
                        pending,
                    )
                    count += 1
                    if count % commit_every == 0:
                        txn.commit()
                        pending.clear()
                        txn = target_env.begin(write=True)
            txn.commit()
        except BaseException:
            txn.abort()
            raise
        finally:
            source_env.close()
            target_env.close()

        return cls(output_path, readonly=True)

    @classmethod
    def build_from_graph(
        cls,
        graph: "CSRGraph",
        output_path: str | Path,
        commit_every: int = 50_000,
    ) -> "EdgeIdLookup":
        output_path = Path(output_path)

        if output_path.exists():
            shutil.rmtree(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        target_env = lmdb.open(
            str(output_path),
            readonly=False,
            max_dbs=0,
            map_size=INITIAL_WRITE_MAP_SIZE,
            readahead=False,
        )

        num_edges = len(graph.fwd_targets)
        pending = []
        txn = target_env.begin(write=True)
        count = 0

        try:
            if getattr(graph, "edge_ids", None) is not None:
                for edge_idx, edge_id in enumerate(graph.edge_ids):
                    lookup_id = resolved_qlever_edge_id(edge_id, edge_idx)
                    txn = _put_with_resize(
                        target_env,
                        txn,
                        lookup_id.encode("utf-8"),
                        _encode_idx(edge_idx),
                        pending,
                    )
                    count += 1
                    if count % commit_every == 0:
                        txn.commit()
                        pending.clear()
                        txn = target_env.begin(write=True)
            elif getattr(graph, "_edge_ids_env", None) is not None:
                seen = bytearray(num_edges)
                with graph._edge_ids_env.begin(buffers=True) as source_txn:
                    cursor = source_txn.cursor()
                    for key, value in cursor:
                        edge_idx = _decode_idx(bytes(key))
                        seen[edge_idx] = 1
                        txn = _put_with_resize(
                            target_env,
                            txn,
                            bytes(value),
                            _encode_idx(edge_idx),
                            pending,
                        )
                        count += 1
                        if count % commit_every == 0:
                            txn.commit()
                            pending.clear()
                            txn = target_env.begin(write=True)

                for edge_idx in range(num_edges):
                    if seen[edge_idx]:
                        continue
                    txn = _put_with_resize(
                        target_env,
                        txn,
                        synthetic_qlever_edge_id(edge_idx).encode("utf-8"),
                        _encode_idx(edge_idx),
                        pending,
                    )
                    count += 1
                    if count % commit_every == 0:
                        txn.commit()
                        pending.clear()
                        txn = target_env.begin(write=True)
            else:
                for edge_idx in range(num_edges):
                    txn = _put_with_resize(
                        target_env,
                        txn,
                        synthetic_qlever_edge_id(edge_idx).encode("utf-8"),
                        _encode_idx(edge_idx),
                        pending,
                    )
                    count += 1
                    if count % commit_every == 0:
                        txn.commit()
                        pending.clear()
                        txn = target_env.begin(write=True)

            txn.commit()
        except BaseException:
            txn.abort()
            raise
        finally:
            target_env.close()

        return cls(output_path, readonly=True)


__all__ = [
    "EdgeIdLookup",
    "resolved_qlever_edge_id",
    "synthetic_qlever_edge_id",
]
