"""LMDB-backed CURIE <-> PMID index.

Two sub-databases live in one LMDB environment:

  * ``curie_to_pmid`` — dupsort + integerdup.  key = CURIE (UTF-8 bytes),
    values = the set of PMIDs (each stored as 4-byte big-endian uint32).
  * ``pmid_to_curie`` — dupsort.  key = PMID (4-byte big-endian uint32),
    values = the set of CURIE strings mentioned in that abstract.

Both directions are populated in the same streaming pass.  dupsort lets
us append (key, value) pairs freely — LMDB keeps them sorted and filters
duplicates when the DB is opened with ``MDB_DUPSORT`` (``dupsort=True``).

Query patterns:

  * For node counts: scan ``curie_to_pmid`` and count duplicates per key
    (or iterate pmids for a given CURIE with ``iter_pmids(curie)``).
  * For pair intersections: scan ``pmid_to_curie`` and enumerate CURIE
    pairs per PMID.

Memory-mapped, so multiple processes share pages via the OS page cache.
"""

from __future__ import annotations

import logging
import shutil
import struct
from pathlib import Path
from typing import Iterable, Iterator, Optional, Set, Tuple

import lmdb

logger = logging.getLogger(__name__)


_DEFAULT_READ_MAP_SIZE = 256 * 1024 * 1024 * 1024  # 256 GB virtual
_INITIAL_WRITE_MAP_SIZE = 4 * 1024 * 1024 * 1024  # 4 GB on disk start
_PMID_STRUCT = struct.Struct(">I")  # big-endian uint32


def _encode_pmid(pmid: int) -> bytes:
    return _PMID_STRUCT.pack(pmid)


def _decode_pmid(buf: bytes) -> int:
    return _PMID_STRUCT.unpack(buf)[0]


class PublicationsIndex:
    """CURIE <-> PMID bi-directional index backed by LMDB dupsort.

    The index is source-agnostic: any iterator of ``(pmid, curie)`` tuples
    can populate it.  ``pubtator.iter_pubtator_annotations`` is the first
    such source.
    """

    DB_CURIE_TO_PMID = b"curie_to_pmid"
    DB_PMID_TO_CURIE = b"pmid_to_curie"

    def __init__(self, path: Path, readonly: bool = True):
        self._path = Path(path)
        self._env = lmdb.open(
            str(self._path),
            readonly=readonly,
            max_dbs=2,
            map_size=_DEFAULT_READ_MAP_SIZE if readonly else _INITIAL_WRITE_MAP_SIZE,
            readahead=False,
            lock=not readonly,
        )
        # ``integerkey`` on the pmid_to_curie DB lets LMDB sort keys as
        # native ints instead of lexicographically.  We still pack big-endian
        # manually so that fallback lex ordering also yields correct order.
        self._db_curie_to_pmid = self._env.open_db(
            self.DB_CURIE_TO_PMID,
            create=not readonly,
            dupsort=True,
            integerdup=True,
            dupfixed=True,
        )
        self._db_pmid_to_curie = self._env.open_db(
            self.DB_PMID_TO_CURIE,
            create=not readonly,
            dupsort=True,
        )

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    def iter_pmids(self, curie: str) -> Iterator[int]:
        """Yield every PMID associated with ``curie`` (sorted ascending)."""
        key = curie.encode("utf-8")
        with self._env.begin(db=self._db_curie_to_pmid, buffers=True) as txn:
            cursor = txn.cursor()
            if not cursor.set_key(key):
                return
            for val_buf in cursor.iternext_dup():
                yield _decode_pmid(bytes(val_buf))

    def count_pmids(self, curie: str) -> int:
        """Number of unique PMIDs associated with ``curie``."""
        key = curie.encode("utf-8")
        with self._env.begin(db=self._db_curie_to_pmid, buffers=True) as txn:
            cursor = txn.cursor()
            if not cursor.set_key(key):
                return 0
            return cursor.count()

    def pmid_set(self, curie: str) -> Set[int]:
        """Materialize all PMIDs for ``curie`` as a Python set."""
        return set(self.iter_pmids(curie))

    def iter_curies_for_pmid(self, pmid: int) -> Iterator[str]:
        """Yield every CURIE mentioned by ``pmid``."""
        key = _encode_pmid(pmid)
        with self._env.begin(db=self._db_pmid_to_curie, buffers=True) as txn:
            cursor = txn.cursor()
            if not cursor.set_key(key):
                return
            for val_buf in cursor.iternext_dup():
                yield bytes(val_buf).decode("utf-8")

    def iter_pmid_groups(self) -> Iterator[Tuple[int, list]]:
        """Yield ``(pmid, [curie, ...])`` for every PMID in the index.

        Consumers that enumerate pairs (for the upcoming pair-intersection
        derivation) iterate this method.
        """
        with self._env.begin(db=self._db_pmid_to_curie, buffers=True) as txn:
            cursor = txn.cursor()
            if not cursor.first():
                return
            current_pmid: Optional[int] = None
            group: list = []
            for key_buf, val_buf in cursor:
                pmid = _decode_pmid(bytes(key_buf))
                if pmid != current_pmid:
                    if current_pmid is not None:
                        yield current_pmid, group
                    current_pmid = pmid
                    group = []
                group.append(bytes(val_buf).decode("utf-8"))
            if current_pmid is not None and group:
                yield current_pmid, group

    def stats(self) -> dict:
        """Lightweight counts for logging/diagnostics."""
        with self._env.begin() as txn:
            return {
                "curie_to_pmid_entries": txn.stat(self._db_curie_to_pmid)["entries"],
                "pmid_to_curie_entries": txn.stat(self._db_pmid_to_curie)["entries"],
            }

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    # ------------------------------------------------------------------
    # Builder
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        path: Path,
        annotations: Iterable[Tuple[int, str]],
        tracked_curies: Optional[Set[str]] = None,
        commit_every: int = 500_000,
        overwrite: bool = False,
    ) -> "PublicationsIndex":
        """Build a new index from an iterator of ``(pmid, curie)`` tuples.

        Args:
            path: Directory to hold the LMDB environment.
            annotations: Streaming source of (pmid, curie) tuples.
            tracked_curies: If set, only (pmid, curie) pairs where the
                CURIE is in this set are written.  Filters at write time so
                the index stays small.
            commit_every: Flush the LMDB write transaction every N inserts.
            overwrite: If True, delete any existing store at ``path``.
        """
        path = Path(path)
        if path.exists():
            if not overwrite:
                raise FileExistsError(
                    f"Publications index already exists at {path}; "
                    "pass overwrite=True to replace it."
                )
            shutil.rmtree(path)
        path.mkdir(parents=True)

        env = lmdb.open(
            str(path),
            readonly=False,
            max_dbs=2,
            map_size=_INITIAL_WRITE_MAP_SIZE,
            readahead=False,
        )
        db_c2p = env.open_db(
            cls.DB_CURIE_TO_PMID,
            create=True,
            dupsort=True,
            integerdup=True,
            dupfixed=True,
        )
        db_p2c = env.open_db(cls.DB_PMID_TO_CURIE, create=True, dupsort=True)

        logger.info("Building PublicationsIndex at %s", path)

        pending: list = []
        txn = env.begin(write=True)
        written = 0
        skipped_filtered = 0

        def _put(db, key, val):
            nonlocal txn
            try:
                txn.put(key, val, db=db, dupdata=True, overwrite=True)
                pending.append((db, key, val))
            except lmdb.MapFullError:
                txn.abort()
                new_size = env.info()["map_size"] * 2
                env.set_mapsize(new_size)
                logger.warning(
                    "  PublicationsIndex: LMDB map full, resized to %.0f GB",
                    new_size / (1024**3),
                )
                txn = env.begin(write=True)
                for pdb, pk, pv in pending:
                    txn.put(pk, pv, db=pdb, dupdata=True, overwrite=True)
                txn.put(key, val, db=db, dupdata=True, overwrite=True)
                pending.append((db, key, val))

        try:
            for pmid, curie in annotations:
                if tracked_curies is not None and curie not in tracked_curies:
                    skipped_filtered += 1
                    continue
                pmid_bytes = _encode_pmid(pmid)
                curie_bytes = curie.encode("utf-8")
                _put(db_c2p, curie_bytes, pmid_bytes)
                _put(db_p2c, pmid_bytes, curie_bytes)
                written += 1
                if written % commit_every == 0:
                    txn.commit()
                    pending.clear()
                    txn = env.begin(write=True)
                    if written % (commit_every * 20) == 0:
                        logger.info(
                            "  %s (pmid, curie) pairs written", f"{written:,}"
                        )
            txn.commit()
            pending.clear()
        except BaseException:
            txn.abort()
            raise
        finally:
            env.close()

        logger.info(
            "PublicationsIndex built: %s pairs written, %s skipped by filter",
            f"{written:,}",
            f"{skipped_filtered:,}",
        )
        return cls(path, readonly=True)
