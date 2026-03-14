"""LMDB-backed edge property storage for attributes (including publications).

These are the "cold path" properties — only accessed during response enrichment
for the small set of result edges. Qualifiers and sources (hot path, needed during
traversal filtering) are kept in-memory via the dedup store in graph.py.

LMDB is a memory-mapped B-tree. Multiple worker processes share the same physical
memory pages via the OS, identical to how numpy mmap works. No per-process
duplication.
"""

import logging
import shutil
import struct
from pathlib import Path

import lmdb
import msgpack

logger = logging.getLogger(__name__)


# Read-only map size — large enough to cover any pre-built database.
# Only virtual address space; no physical allocation until pages are touched.
_DEFAULT_READ_MAP_SIZE = 256 * 1024 * 1024 * 1024   # 256 GB

# Initial write map size — intentionally small so the data.mdb file
# starts small on disk.  _put_with_resize() doubles it on MapFullError,
# so the file grows only as real data demands.
_INITIAL_WRITE_MAP_SIZE = 4 * 1024 * 1024 * 1024    # 4 GB


def _put_with_resize(env, txn, key, val, pending):
    """Put key/value into LMDB, auto-resizing the map if full.

    After a failed put(), LMDB marks the transaction as invalid — it must
    be aborted, not committed.  So we abort, double the map, and replay
    all writes since the last commit (tracked in *pending*).

    Callers must clear *pending* after every successful commit.

    Returns the (possibly new) write transaction.
    """
    try:
        txn.put(key, val)
        pending.append((key, val))
        return txn
    except lmdb.MapFullError:
        txn.abort()
        new_size = env.info()["map_size"] * 2
        env.set_mapsize(new_size)
        logger.warning("    LMDB: map full, resized to %.0f GB", new_size / (1024**3))
        txn = env.begin(write=True)
        for k, v in pending:
            txn.put(k, v)
        txn.put(key, val)
        pending.append((key, val))
        return txn


def _encode_key(edge_idx: int) -> bytes:
    """Encode edge index as 4-byte big-endian for correct LMDB sort order."""
    return struct.pack(">I", edge_idx)


def _decode_key(key: bytes) -> int:
    """Decode 4-byte big-endian key back to edge index."""
    return struct.unpack(">I", key)[0]


class LMDBPropertyStore:
    """Disk-backed edge property storage using LMDB.

    Stores attributes per edge as msgpack blobs.
    Keys are edge indices (matching CSR array positions) encoded as 4-byte
    big-endian integers for correct sort order.

    Memory-mapped by the OS — multiple worker processes reading the same
    file share physical memory pages automatically.
    """

    def __init__(self, path, readonly=True):
        self._path = Path(path)
        self._env = lmdb.open(
            str(self._path),
            readonly=readonly,
            max_dbs=0,
            map_size=_DEFAULT_READ_MAP_SIZE,
            readahead=False,  # We do point + small range reads
            lock=not readonly,  # No lock file needed for read-only
        )

    def get(self, edge_idx):
        """Get all detail properties for a single edge.

        Returns dict with an 'attributes' key (and any other stored keys),
        or empty dict if edge not found.
        """
        key = _encode_key(edge_idx)
        with self._env.begin(buffers=True) as txn:
            val = txn.get(key)
            if val is None:
                return {}
            return msgpack.unpackb(val, raw=False)

    def get_batch(self, edge_indices):
        """Get detail properties for multiple edges.

        Used during response enrichment for the result set (typically
        tens to low hundreds of edges).

        Returns dict mapping edge_idx -> properties dict.
        """
        results = {}
        with self._env.begin(buffers=True) as txn:
            for idx in edge_indices:
                key = _encode_key(idx)
                val = txn.get(key)
                if val is not None:
                    results[idx] = msgpack.unpackb(val, raw=False)
        return results

    def close(self):
        """Close the LMDB environment."""
        if self._env is not None:
            self._env.close()
            self._env = None

    def __del__(self):
        self.close()

    @staticmethod
    def build(db_path, edge_iterator, num_edges, commit_every=50_000):
        """Build an LMDB store by streaming edge properties.

        Args:
            db_path: Path for the LMDB directory.
            edge_iterator: Yields (edge_idx, props_dict) tuples where
                props_dict has an 'attributes' key (TRAPI Attribute list).
                Must yield in edge_idx order (0, 1, 2, ...).
            num_edges: Total number of edges (for progress reporting).
            commit_every: Commit transaction every N edges to limit memory.

        Returns:
            LMDBPropertyStore opened in read-only mode.
        """
        db_path = Path(db_path)
        if db_path.exists():
            shutil.rmtree(db_path)
        db_path.mkdir(parents=True, exist_ok=True)

        env = lmdb.open(
            str(db_path),
            map_size=_INITIAL_WRITE_MAP_SIZE,
            readonly=False,
            max_dbs=0,
            readahead=False,
        )

        txn = env.begin(write=True)
        pending = []
        count = 0
        try:
            for edge_idx, props in edge_iterator:
                key = _encode_key(edge_idx)
                val = msgpack.packb(props, use_bin_type=True)
                txn = _put_with_resize(env, txn, key, val, pending)
                count += 1

                if count % commit_every == 0:
                    txn.commit()
                    pending.clear()
                    if count % 1_000_000 == 0:
                        logger.debug("    LMDB: wrote %s/%s edges...", f"{count:,}", f"{num_edges:,}")
                    txn = env.begin(write=True)

            txn.commit()
        except BaseException:
            txn.abort()
            raise
        finally:
            env.close()

        logger.debug("    LMDB: wrote %s edges to %s", f"{count:,}", db_path)
        return LMDBPropertyStore(db_path, readonly=True)

    @staticmethod
    def build_sorted(db_path, temp_db_path, sort_permutation, num_edges,
                     commit_every=50_000):
        """Rewrite a temp LMDB in CSR-sorted order to produce the final store.

        Reads from temp_db_path using sort_permutation to reorder, writes
        to db_path with sequential keys 0..num_edges-1.

        This is the expensive build-time operation that ensures query-time
        keys match CSR edge indices with zero indirection.

        Args:
            db_path: Path for the final LMDB directory.
            temp_db_path: Path to temporary LMDB (keyed by original line index).
            sort_permutation: numpy array where sort_permutation[csr_pos] = original_line_idx.
            num_edges: Total number of edges.
            commit_every: Commit transaction every N edges.

        Returns:
            LMDBPropertyStore opened in read-only mode.
        """
        db_path = Path(db_path)
        if db_path.exists():
            shutil.rmtree(db_path)
        db_path.mkdir(parents=True, exist_ok=True)

        temp_env = lmdb.open(
            str(temp_db_path),
            readonly=True,
            lock=False,
            map_size=_DEFAULT_READ_MAP_SIZE,
            readahead=False,
        )
        final_env = lmdb.open(
            str(db_path),
            map_size=_INITIAL_WRITE_MAP_SIZE,
            readonly=False,
            max_dbs=0,
            readahead=False,
        )

        logger.debug("    LMDB: rewriting %s edges in CSR-sorted order...", f"{num_edges:,}")

        temp_txn = temp_env.begin(buffers=True)
        final_txn = final_env.begin(write=True)
        pending = []

        try:
            for csr_pos in range(num_edges):
                original_idx = int(sort_permutation[csr_pos])
                temp_key = _encode_key(original_idx)
                val = temp_txn.get(temp_key)

                final_key = _encode_key(csr_pos)
                final_txn = _put_with_resize(
                    final_env, final_txn, final_key, bytes(val), pending
                )

                if (csr_pos + 1) % commit_every == 0:
                    final_txn.commit()
                    pending.clear()
                    if (csr_pos + 1) % 1_000_000 == 0:
                        logger.debug("      %s/%s edges rewritten...", f"{csr_pos + 1:,}", f"{num_edges:,}")
                    final_txn = final_env.begin(write=True)

            final_txn.commit()
        except BaseException:
            final_txn.abort()
            raise
        finally:
            temp_txn.abort()
            temp_env.close()
            final_env.close()

        logger.debug("    LMDB: final store written to %s", db_path)
        return LMDBPropertyStore(db_path, readonly=True)
