"""LMDB-backed traversal-metadata store for plugin-owned per-node data.

Plugins store derived path-traversal metadata here. The store is mmap'd
LMDB so:

- Reads page in only what's actually touched; cold pages stay on disk.
- Memory is OS-page-cache, not Python heap. Multiple worker processes
  share the same physical pages.
- Persists to disk alongside the rest of the graph; not recomputed on load.

The store is always **separate from the TRAPI attribute list** and is
**never** read during enrichment / response building. A grep for
``traversal_metadata`` shows every site that touches it.

## Key layout (single LMDB env, single unnamed DB)

Each record's key is::

    <namespace_utf8> + b"\\x00" + <subkey_bytes>

where ``namespace`` is the plugin's registered enricher name (must be
ASCII; must not contain ``\\x00``), and ``subkey_bytes`` is either a
4-byte big-endian uint32 ``node_idx`` or ``b""`` for plugin-global data.

Values are msgpack-encoded. Plugins may stash any msgpack-encodable
Python object; for raw numeric arrays, encode with ``arr.tobytes()`` and
remember the dtype/shape inside the same plugin.

## Lifecycle

- Build time (``build_graph_from_jsonl``): a writable store is opened
  lazily at a temp directory the first time an enricher calls ``put``.
  If no enricher writes, no LMDB env is opened.
- Save time (``CSRGraph.save_mmap``): the writable env is copied to
  ``<directory>/traversal_metadata.lmdb`` (or skipped if empty).
- Load time (``CSRGraph.load_mmap``): a readonly mmap'd env is opened
  at ``<directory>/traversal_metadata.lmdb`` if the directory exists,
  otherwise an empty writable temp env (so newly-registered plugins can
  still backfill via ``run_enrichers``).
"""

import logging
import shutil
import struct
import tempfile
from pathlib import Path
from typing import Any, Iterator, Optional, Tuple, Union

import lmdb
import msgpack

logger = logging.getLogger(__name__)

# Same conventions as gandalf.node_store / gandalf.lmdb_store.
_DEFAULT_READ_MAP_SIZE = 256 * 1024 * 1024 * 1024  # 256 GB virtual
_INITIAL_WRITE_MAP_SIZE = 4 * 1024 * 1024 * 1024  # 4 GB initial; auto-resizes

_NS_SEP = b"\x00"


def _encode_subkey(sub_key: Optional[int]) -> bytes:
    if sub_key is None:
        return b""
    return struct.pack(">I", sub_key)


def _build_key(namespace: str, sub_key: Optional[int]) -> bytes:
    if _NS_SEP in namespace.encode("utf-8"):
        raise ValueError(f"Namespace must not contain a NUL byte: {namespace!r}")
    return namespace.encode("utf-8") + _NS_SEP + _encode_subkey(sub_key)


def _ns_prefix(namespace: str) -> bytes:
    return namespace.encode("utf-8") + _NS_SEP


def _put_with_resize(env, txn, key, val, pending):
    """Mirror of node_store._put_with_resize."""
    try:
        txn.put(key, val)
        pending.append((key, val))
        return txn
    except lmdb.MapFullError:
        txn.abort()
        new_size = env.info()["map_size"] * 2
        env.set_mapsize(new_size)
        logger.warning(
            "    TraversalMetadata LMDB: map full, resized to %.0f GB",
            new_size / (1024**3),
        )
        txn = env.begin(write=True)
        for pk, pv in pending:
            txn.put(pk, pv)
        txn.put(key, val)
        pending.append((key, val))
        return txn


class TraversalMetadataStore:
    """LMDB-backed plugin metadata. See module docstring for layout."""

    def __init__(self, path: Optional[Union[str, Path]] = None, readonly: bool = True):
        """Open or lazy-create a store.

        ``path=None`` with ``readonly=False`` defers opening until the first
        write; the store is then anchored at a fresh temp directory that
        will be cleaned up on ``close()`` unless the caller persists it via
        ``save_to``.
        """
        self._path: Optional[Path] = Path(path) if path is not None else None
        self._readonly = readonly
        self._env: Optional[lmdb.Environment] = None
        self._owns_temp_dir = False

        if self._path is not None and self._readonly:
            # Eager open: callers expect reads to work immediately.
            self._open()

    # -- lifecycle -----------------------------------------------------

    def _open(self) -> None:
        if self._env is not None:
            return
        if self._path is None:
            # Lazy temp dir for the writable case.
            self._path = Path(tempfile.mkdtemp(prefix="gandalf_traversal_"))
            self._owns_temp_dir = True
        else:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        self._env = lmdb.open(
            str(self._path),
            readonly=self._readonly,
            map_size=(
                _DEFAULT_READ_MAP_SIZE if self._readonly else _INITIAL_WRITE_MAP_SIZE
            ),
            readahead=False,
            subdir=True,
            lock=not self._readonly,
            create=not self._readonly,
        )

    @classmethod
    def open_readonly(cls, path: Union[str, Path]) -> "TraversalMetadataStore":
        """Open an existing store as readonly (memory-mapped)."""
        return cls(path=path, readonly=True)

    @classmethod
    def open_writable(
        cls, path: Optional[Union[str, Path]] = None
    ) -> "TraversalMetadataStore":
        """Open or lazy-create a writable store.

        ``path=None`` defers opening until the first write into a temp
        directory that is cleaned up on ``close()``.
        """
        return cls(path=path, readonly=False)

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None
        if self._owns_temp_dir and self._path is not None and self._path.exists():
            shutil.rmtree(self._path, ignore_errors=True)
            self._owns_temp_dir = False
            self._path = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    @property
    def is_readonly(self) -> bool:
        return self._readonly

    @property
    def path(self) -> Optional[Path]:
        """Backing directory; None if no env has been opened yet."""
        return self._path

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    # -- write ---------------------------------------------------------

    def put(self, namespace: str, sub_key: Optional[int], value: Any) -> None:
        """Write ``value`` under ``(namespace, sub_key)``.

        ``sub_key=None`` stores a plugin-global value. ``sub_key`` as an int
        stores a per-node value. Values are msgpack-encoded; pass
        ``arr.tobytes()`` for raw numeric arrays.
        """
        if self._readonly:
            raise RuntimeError("TraversalMetadataStore is readonly")
        self._open()
        key = _build_key(namespace, sub_key)
        val = msgpack.packb(value, use_bin_type=True)
        # Single-write path: small overhead per call. Plugins iterating
        # millions of nodes should use ``put_many`` for batching.
        with self._env.begin(write=True) as txn:
            txn.put(key, val)

    def put_many(
        self,
        namespace: str,
        items: Iterator[Tuple[Optional[int], Any]],
        commit_every: int = 50_000,
    ) -> int:
        """Bulk-write a stream of ``(sub_key, value)`` pairs. Returns count."""
        if self._readonly:
            raise RuntimeError("TraversalMetadataStore is readonly")
        self._open()
        ns_bytes = namespace.encode("utf-8")
        if _NS_SEP in ns_bytes:
            raise ValueError(f"Namespace must not contain a NUL byte: {namespace!r}")

        count = 0
        pending: list[tuple[bytes, bytes]] = []
        txn = self._env.begin(write=True)
        try:
            for sub_key, value in items:
                key = ns_bytes + _NS_SEP + _encode_subkey(sub_key)
                val = msgpack.packb(value, use_bin_type=True)
                txn = _put_with_resize(self._env, txn, key, val, pending)
                count += 1
                if count % commit_every == 0:
                    txn.commit()
                    pending.clear()
                    txn = self._env.begin(write=True)
            txn.commit()
            pending.clear()
        except Exception:
            txn.abort()
            raise
        return count

    # -- read ----------------------------------------------------------

    def get(self, namespace: str, sub_key: Optional[int], default: Any = None) -> Any:
        """Read a value. Returns ``default`` if the key is missing."""
        if self._env is None:
            return default
        key = _build_key(namespace, sub_key)
        with self._env.begin(buffers=True) as txn:
            val = txn.get(key)
            if val is None:
                return default
            return msgpack.unpackb(bytes(val), raw=False)

    def __contains__(self, namespace: str) -> bool:
        """True if any record exists in the namespace."""
        if self._env is None:
            return False
        prefix = _ns_prefix(namespace)
        with self._env.begin(buffers=True) as txn:
            cursor = txn.cursor()
            if not cursor.set_range(prefix):
                return False
            key = bytes(cursor.key())
            return key.startswith(prefix)

    def namespaces(self) -> set[str]:
        """All distinct namespaces present in the store."""
        if self._env is None:
            return set()
        result: set[str] = set()
        with self._env.begin(buffers=True) as txn:
            cursor = txn.cursor()
            cursor.first()
            while cursor.key():
                key = bytes(cursor.key())
                sep = key.find(_NS_SEP)
                if sep < 0:
                    break
                ns = key[:sep].decode("utf-8")
                result.add(ns)
                # Skip past every record in this namespace.
                next_prefix = key[: sep + 1] + b"\xff"
                if not cursor.set_range(next_prefix):
                    break
        return result

    def iter_namespace(
        self, namespace: str
    ) -> Iterator[Tuple[Optional[int], Any]]:
        """Yield ``(sub_key, value)`` for every record in ``namespace``."""
        if self._env is None:
            return
        prefix = _ns_prefix(namespace)
        with self._env.begin(buffers=True) as txn:
            cursor = txn.cursor()
            if not cursor.set_range(prefix):
                return
            for key_buf, val_buf in cursor:
                key = bytes(key_buf)
                if not key.startswith(prefix):
                    return
                sub_bytes = key[len(prefix):]
                if sub_bytes == b"":
                    sub_key: Optional[int] = None
                else:
                    sub_key = struct.unpack(">I", sub_bytes)[0]
                yield sub_key, msgpack.unpackb(bytes(val_buf), raw=False)

    # -- persistence ---------------------------------------------------

    def save_to(self, directory: Union[str, Path]) -> None:
        """Copy this store to ``<directory>/traversal_metadata.lmdb``.

        No-op when the store has never been opened (no plugin wrote
        anything). When the source path is already the target, also a
        no-op (the LMDB is already at its final home).
        """
        if self._env is None or self._path is None:
            return

        target = Path(directory) / "traversal_metadata.lmdb"
        target.parent.mkdir(parents=True, exist_ok=True)

        if self._path.resolve() == target.resolve():
            self._env.sync(force=True)
            return

        self._env.sync(force=True)
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(self._path, target)
