"""Pair intersection CSR store and ingest.

Stores, for every pair of nodes (a, b) that share at least one
publication, how many publications they share.  Keyed by the graph's
integer node indices and normalized so ``a < b`` (undirected pairs).

On-disk layout, alongside the graph files:

  * ``pair_offsets.npy``   — ``uint64[N + 1]``.  Pairs for node ``a`` live at
    positions ``[pair_offsets[a] : pair_offsets[a + 1]]``.
  * ``pair_neighbors.npy`` — ``uint32[K]``.  The "other" node ``b`` for each
    pair, sorted ascending within each ``a`` row so lookups can binary-search.
  * ``pair_counts.npy``    — ``uint32[K]``.  Shared publication count.

Only pairs with count > 0 are stored — a sparse representation of an
otherwise N² matrix.  The same CSR shape mirrors the topology arrays,
so lookups are O(log degree) with mmap-shared pages across workers.

The ingest layer accepts a sorted stream of ``(a, b, count)`` tuples
with ``a < b``.  ``PairCountAccumulator`` is a small abstraction that
produces that sorted stream; the default implementation accumulates in
a Python dict and is fine for small-to-medium graphs.  A larger-scale
implementation (SQLite-backed, or external sort + merge-reduce) can be
dropped in behind the same interface without touching any downstream
code.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Iterator, Optional, Tuple

import numpy as np

from gandalf.config import settings
from gandalf.metadata.manifest import load_manifest, save_manifest

logger = logging.getLogger(__name__)


FIELD_PAIR_INTERSECTIONS = "pair_intersections"

_PAIR_OFFSETS_FILE = "pair_offsets.npy"
_PAIR_NEIGHBORS_FILE = "pair_neighbors.npy"
_PAIR_COUNTS_FILE = "pair_counts.npy"


class PairCountAccumulator:
    """Abstract accumulator for (a, b) pair counts.

    Subclasses record increments via ``add_pair`` and emit the final
    aggregated data via ``iter_sorted``.
    """

    def add_pair(self, a: int, b: int, count: int = 1) -> None:
        raise NotImplementedError

    def iter_sorted(self) -> Iterator[Tuple[int, int, int]]:
        """Yield ``(a, b, count)`` tuples with ``a < b``, sorted ascending by ``(a, b)``."""
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


class InMemoryPairCountAccumulator(PairCountAccumulator):
    """Python-dict backed accumulator.

    Trivially fast for small-to-medium graphs.  Unsuitable for the
    hundreds-of-millions-of-pairs regime — a SQLite-backed or
    external-sort accumulator should replace this when scale requires.
    """

    def __init__(self) -> None:
        self._counts: dict = {}

    def add_pair(self, a: int, b: int, count: int = 1) -> None:
        if a == b:
            return
        if a > b:
            a, b = b, a
        key = (a, b)
        self._counts[key] = self._counts.get(key, 0) + count

    def iter_sorted(self) -> Iterator[Tuple[int, int, int]]:
        for (a, b), count in sorted(self._counts.items()):
            yield a, b, count

    def __len__(self) -> int:
        return len(self._counts)


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


def _infer_num_nodes(graph_dir: Path) -> int:
    import pickle

    with open(graph_dir / "metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    return int(metadata["num_nodes"])


def write_pair_intersections(
    graph_dir: Path,
    num_nodes: int,
    num_pairs: int,
    sorted_pairs: Iterable[Tuple[int, int, int]],
    source: str,
) -> Path:
    """Write a sorted stream of (a, b, count) tuples to the CSR pair store.

    The caller must supply ``num_pairs`` (total unique pairs with count > 0)
    so we can pre-allocate the neighbor / count arrays without a second
    pass over the stream.

    Raises ``ValueError`` if the stream violates sort order or if ``a >= b``.
    """
    graph_dir = Path(graph_dir)

    offsets = np.zeros(num_nodes + 1, dtype=np.uint64)
    neighbors = np.empty(num_pairs, dtype=np.uint32)
    counts = np.empty(num_pairs, dtype=np.uint32)

    last_a = -1
    last_b = -1
    k = 0
    total_count = 0
    max_count = 0
    for a, b, count in sorted_pairs:
        if a >= b:
            raise ValueError(
                f"pair intersection stream must have a < b; got ({a}, {b})"
            )
        if a < 0 or b >= num_nodes:
            raise ValueError(
                f"pair intersection stream out of range for num_nodes={num_nodes}: "
                f"({a}, {b})"
            )
        if a < last_a or (a == last_a and b <= last_b):
            raise ValueError(
                f"pair intersection stream not strictly sorted: "
                f"previous=({last_a}, {last_b}), current=({a}, {b})"
            )
        if count <= 0:
            raise ValueError(
                f"pair intersection count must be > 0; got {count} at ({a}, {b})"
            )
        if k >= num_pairs:
            raise ValueError(
                f"pair intersection stream has more rows than declared num_pairs={num_pairs}"
            )

        # Close out any empty rows between last_a and a.
        if a != last_a:
            offsets[last_a + 1 : a + 1] = k
        neighbors[k] = b
        counts[k] = count
        k += 1
        total_count += int(count)
        if count > max_count:
            max_count = int(count)
        last_a = a
        last_b = b

    if k != num_pairs:
        raise ValueError(
            f"pair intersection stream yielded {k} rows but declared num_pairs={num_pairs}"
        )
    # Fill in offsets for any trailing empty rows.
    offsets[last_a + 1 :] = k

    np.save(graph_dir / _PAIR_OFFSETS_FILE, offsets)
    np.save(graph_dir / _PAIR_NEIGHBORS_FILE, neighbors)
    np.save(graph_dir / _PAIR_COUNTS_FILE, counts)

    manifest = load_manifest(graph_dir)
    manifest.set_field(
        FIELD_PAIR_INTERSECTIONS,
        {
            "files": {
                "offsets": _PAIR_OFFSETS_FILE,
                "neighbors": _PAIR_NEIGHBORS_FILE,
                "counts": _PAIR_COUNTS_FILE,
            },
            "num_nodes": num_nodes,
            "num_pairs": num_pairs,
            "dtypes": {"offsets": "uint64", "neighbors": "uint32", "counts": "uint32"},
            "source": source,
            "total_shared_publications": total_count,
            "max_count": max_count,
        },
    )
    save_manifest(graph_dir, manifest)
    logger.info(
        "Wrote pair intersections: %d pairs, total=%d, max=%d",
        num_pairs,
        total_count,
        max_count,
    )
    return graph_dir / _PAIR_OFFSETS_FILE


def ingest_pair_intersections_from_accumulator(
    graph_dir: Path,
    accumulator: PairCountAccumulator,
    source: str,
) -> Path:
    """Write an accumulator's contents to the pair intersection store."""
    num_nodes = _infer_num_nodes(graph_dir)
    num_pairs = len(accumulator)
    logger.info(
        "Ingesting pair intersections: %d pairs for %d nodes (%s)",
        num_pairs,
        num_nodes,
        source,
    )
    return write_pair_intersections(
        graph_dir=graph_dir,
        num_nodes=num_nodes,
        num_pairs=num_pairs,
        sorted_pairs=accumulator.iter_sorted(),
        source=source,
    )


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


def _mmap_npy(path: Path) -> np.ndarray:
    arr: np.ndarray = np.load(path, mmap_mode="r")
    if settings.load_mmaps_into_memory:
        arr = np.array(arr)
    return arr


class PairIntersections:
    """Read-only accessor for the pair intersection CSR store.

    Three mmap'd numpy arrays — lookup is O(log row_degree) via
    ``np.searchsorted`` within the row; row iteration is contiguous.
    """

    __slots__ = ("_offsets", "_neighbors", "_counts")

    def __init__(self, graph_dir: Path):
        graph_dir = Path(graph_dir)
        self._offsets = _mmap_npy(graph_dir / _PAIR_OFFSETS_FILE)
        self._neighbors = _mmap_npy(graph_dir / _PAIR_NEIGHBORS_FILE)
        self._counts = _mmap_npy(graph_dir / _PAIR_COUNTS_FILE)

    @property
    def num_nodes(self) -> int:
        return int(self._offsets.shape[0]) - 1

    @property
    def num_pairs(self) -> int:
        return int(self._neighbors.shape[0])

    def count(self, a: int, b: int) -> int:
        """Return the number of publications shared by nodes ``a`` and ``b``.

        Returns 0 when the pair has no shared publications or when either
        index is out of range.  Order of the arguments does not matter.
        """
        a = int(a)
        b = int(b)
        if a == b:
            return 0
        if a > b:
            a, b = b, a
        if a < 0 or b >= self.num_nodes:
            return 0
        start = int(self._offsets[a])
        end = int(self._offsets[a + 1])
        if start == end:
            return 0
        row = self._neighbors[start:end]
        idx = int(np.searchsorted(row, b))
        if idx < row.shape[0] and int(row[idx]) == b:
            return int(self._counts[start + idx])
        return 0

    def neighbors(self, a: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (neighbor_indices, counts) for pairs where ``a`` is the lower endpoint.

        Only returns pairs stored directly under ``a`` (i.e. pairs (a, b)
        with a < b).  To get *all* neighbors of ``a`` regardless of
        ordering, use ``all_neighbors`` which stitches both halves.
        """
        a = int(a)
        if a < 0 or a >= self.num_nodes:
            return (
                np.empty(0, dtype=self._neighbors.dtype),
                np.empty(0, dtype=self._counts.dtype),
            )
        start = int(self._offsets[a])
        end = int(self._offsets[a + 1])
        return self._neighbors[start:end], self._counts[start:end]

    def close(self) -> None:
        # np.memmap arrays hold no file handle we need to close explicitly;
        # rely on refcount + GC.  Method kept for API symmetry.
        pass
