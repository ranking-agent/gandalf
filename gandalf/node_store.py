"""LMDB-backed node property and ID mapping storage.

Replaces the in-memory Python dicts (node_id_to_idx, idx_to_node_id,
node_properties) that previously consumed ~5-10 GB of RAM for large graphs.

Uses a single LMDB environment with three named sub-databases:
- id_to_idx: node_id (UTF-8 string) -> node_idx (4-byte big-endian uint32)
- idx_to_id: node_idx (4-byte big-endian uint32) -> node_id (UTF-8 string)
- properties: node_idx (4-byte big-endian uint32) -> msgpack({name, categories, attributes})

LMDB is memory-mapped by the OS — multiple worker processes reading the same
file share physical memory pages automatically, with no per-process duplication.
"""

import logging
import shutil
import struct
from pathlib import Path
from typing import Iterator, Optional, Tuple

import lmdb
import msgpack

logger = logging.getLogger(__name__)

_DEFAULT_READ_MAP_SIZE = 256 * 1024 * 1024 * 1024  # 256 GB (virtual only)
_INITIAL_WRITE_MAP_SIZE = 4 * 1024 * 1024 * 1024  # 4 GB


def _put_with_resize(env, txn, db, key, val, pending):
    """Put key/value, auto-resizing the map on MapFullError.

    After a failed put(), LMDB invalidates the transaction — it must be
    aborted, not committed.  We abort, double the map, replay all writes
    since the last commit (*pending*), and retry.

    Callers must clear *pending* after every successful commit.

    Returns the (possibly new) write transaction.
    """
    try:
        txn.put(key, val, db=db)
        pending.append((db, key, val))
        return txn
    except lmdb.MapFullError:
        txn.abort()
        new_size = env.info()["map_size"] * 2
        env.set_mapsize(new_size)
        logger.warning(
            "    NodeStore LMDB: map full, resized to %.0f GB", new_size / (1024**3)
        )
        txn = env.begin(write=True)
        for pdb, pk, pv in pending:
            txn.put(pk, pv, db=pdb)
        txn.put(key, val, db=db)
        pending.append((db, key, val))
        return txn


def _encode_idx(idx: int) -> bytes:
    return struct.pack(">I", idx)


def _decode_idx(key: bytes) -> int:
    result: int = struct.unpack(">I", key)[0]
    return result


class NodeStore:
    """Disk-backed node ID mappings and property storage using LMDB.

    Provides dict-like access patterns with zero RAM footprint (beyond
    OS page cache for hot pages).
    """

    def __init__(self, path, readonly=True):
        self._path = Path(path)
        self._env = lmdb.open(
            str(self._path),
            readonly=readonly,
            max_dbs=3,
            map_size=_DEFAULT_READ_MAP_SIZE if readonly else _INITIAL_WRITE_MAP_SIZE,
            readahead=False,
            lock=not readonly,
        )
        self._db_id_to_idx = self._env.open_db(b"id_to_idx", create=not readonly)
        self._db_idx_to_id = self._env.open_db(b"idx_to_id", create=not readonly)
        self._db_properties = self._env.open_db(b"properties", create=not readonly)

    def get_node_idx(self, node_id: str) -> Optional[int]:
        """Look up integer index for a node ID string."""
        with self._env.begin(db=self._db_id_to_idx, buffers=True) as txn:
            val = txn.get(node_id.encode("utf-8"))
            if val is None:
                return None
            return _decode_idx(bytes(val))

    def get_node_id(self, node_idx: int) -> Optional[str]:
        """Look up original node ID string for an integer index."""
        with self._env.begin(db=self._db_idx_to_id, buffers=True) as txn:
            val = txn.get(_encode_idx(node_idx))
            if val is None:
                return None
            return bytes(val).decode("utf-8")

    def get_properties(self, node_idx: int) -> dict:
        """Get all properties for a node as a dict."""
        with self._env.begin(db=self._db_properties, buffers=True) as txn:
            val = txn.get(_encode_idx(node_idx))
            if val is None:
                return {}
            result: dict = msgpack.unpackb(val, raw=False)
            return result

    def get_property(self, node_idx: int, key: str, default=None):
        """Get a specific property for a node."""
        props = self.get_properties(node_idx)
        return props.get(key, default)

    def iter_id_to_idx(self) -> Iterator[Tuple[str, int]]:
        """Iterate all (node_id, node_idx) pairs. For build-time use."""
        with self._env.begin(db=self._db_id_to_idx, buffers=True) as txn:
            cursor = txn.cursor()
            for key_buf, val_buf in cursor:
                node_id = bytes(key_buf).decode("utf-8")
                node_idx = _decode_idx(bytes(val_buf))
                yield node_id, node_idx

    def iter_properties(self) -> Iterator[Tuple[int, dict]]:
        """Iterate all (node_idx, properties) pairs. For build-time use."""
        with self._env.begin(db=self._db_properties, buffers=True) as txn:
            cursor = txn.cursor()
            for key_buf, val_buf in cursor:
                node_idx = _decode_idx(bytes(key_buf))
                props = msgpack.unpackb(val_buf, raw=False)
                yield node_idx, props

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    @staticmethod
    def build(
        db_path, node_id_to_idx: dict, node_properties: dict, commit_every=50_000
    ):
        """Build a NodeStore LMDB from in-memory dicts.

        Args:
            db_path: Path for the LMDB directory.
            node_id_to_idx: Dict mapping node ID strings to integer indices.
            node_properties: Dict mapping node_idx to property dicts.
            commit_every: Commit transaction every N entries.

        Returns:
            NodeStore opened in read-only mode.
        """
        db_path = Path(db_path)
        if db_path.exists():
            shutil.rmtree(db_path)
        db_path.mkdir(parents=True, exist_ok=True)

        env = lmdb.open(
            str(db_path),
            map_size=_INITIAL_WRITE_MAP_SIZE,
            readonly=False,
            max_dbs=3,
            readahead=False,
        )
        db_id_to_idx = env.open_db(b"id_to_idx", create=True)
        db_idx_to_id = env.open_db(b"idx_to_id", create=True)
        db_properties = env.open_db(b"properties", create=True)

        # Write id_to_idx and idx_to_id mappings
        logger.debug(
            "  NodeStore: writing %s ID mappings...", f"{len(node_id_to_idx):,}"
        )
        pending: list[tuple[object, bytes, bytes]] = []
        txn = env.begin(write=True)
        count = 0
        try:
            for node_id, node_idx in node_id_to_idx.items():
                key_str = node_id.encode("utf-8")
                key_idx = _encode_idx(node_idx)
                txn = _put_with_resize(
                    env, txn, db_id_to_idx, key_str, key_idx, pending
                )
                txn = _put_with_resize(
                    env, txn, db_idx_to_id, key_idx, key_str, pending
                )
                count += 1
                if count % commit_every == 0:
                    txn.commit()
                    pending.clear()
                    txn = env.begin(write=True)
            txn.commit()
            pending.clear()
        except BaseException:
            txn.abort()
            raise

        # Write node properties
        logger.debug(
            "  NodeStore: writing %s node properties...", f"{len(node_properties):,}"
        )
        pending = []
        txn = env.begin(write=True)
        count = 0
        try:
            for node_idx, props in node_properties.items():
                key = _encode_idx(node_idx)
                val = msgpack.packb(props, use_bin_type=True)
                txn = _put_with_resize(env, txn, db_properties, key, val, pending)
                count += 1
                if count % commit_every == 0:
                    txn.commit()
                    pending.clear()
                    txn = env.begin(write=True)
            txn.commit()
            pending.clear()
        except BaseException:
            txn.abort()
            raise

        env.close()
        logger.debug("  NodeStore: built %s", db_path)
        return NodeStore(db_path, readonly=True)
