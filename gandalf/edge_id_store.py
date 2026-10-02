"""Original edge IDs, stored as one memory-mapped blob of JSON strings.

Every full response looks up the original ID of each knowledge-graph edge,
millions of them on the largest queries.  They are stored in forward-CSR
order as two files:

- ``edge_ids.bin``: each edge's ID encoded as a JSON string (quotes
  included), back to back.  An edge with no ID has an empty span.
- ``edge_id_offsets.npy``: ``int64``, ``num_edges + 1``; edge *i*'s ID is
  ``blob[offsets[i]:offsets[i + 1]]``.

Both are memory-mapped, so gunicorn workers share one copy in the page
cache, and a batch lookup is two vectorized reads of the offsets plus one
slice per edge.  Storing the JSON lets the response writer copy an ID into
its output as it is; code that needs the Python ``str`` decodes the span.

The build also records, in ``metadata.pkl``, whether the IDs are unique
(``edge_ids_unique``), so a query need not check its own IDs for duplicates,
and whether any needed escaping (``edge_ids_escaped``); if none did, which is
the usual case, each ID decodes as the UTF-8 between its quotes.
"""

import mmap
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import numpy as np
import orjson

#: Recorded as ``edge_ids_format`` in ``metadata.pkl``; ``load_mmap`` refuses
#: a graph without it (its IDs are in ``edge_ids.lmdb``).
EDGE_IDS_FORMAT = "json-blob"

BLOB_FILE = "edge_ids.bin"
OFFSETS_FILE = "edge_id_offsets.npy"

Blob = Union[bytes, mmap.mmap]


def _encode_id(edge_id) -> bytes:
    """Encode one edge ID as it is stored: a JSON string, or empty for none.

    An ID that is not a string is stored as its ``str``, as the LMDB store
    did; an empty ID counts as no ID, since a response gives such an edge a
    fresh one either way.

    >>> _encode_id("infores:x/1")
    b'"infores:x/1"'
    >>> _encode_id(42)
    b'"42"'
    >>> _encode_id(None), _encode_id("")
    (b'', b'')
    """
    if edge_id is None:
        return b""
    text = edge_id if isinstance(edge_id, str) else str(edge_id)
    return orjson.dumps(text) if text else b""


def ids_unique(ids: Sequence[Optional[str]]) -> bool:
    """Whether the non-empty IDs in *ids* are all distinct.

    Compares 64-bit hashes in a NumPy array rather than building a set of
    tens of millions of strings, and checks only the strings whose hashes
    collide.

    >>> ids_unique(["a", "b", None, "", None])
    True
    >>> ids_unique(["a", "b", "a"])
    False
    """
    present = [i for i in ids if i]
    hashes = np.fromiter(map(hash, present), dtype=np.int64, count=len(present))
    order = np.argsort(hashes, kind="stable")
    ordered = hashes[order]
    clash = np.flatnonzero(ordered[1:] == ordered[:-1])
    if not len(clash):
        return True
    by_hash: dict[int, set] = {}
    for pos in np.union1d(clash, clash + 1).tolist():
        text = present[int(order[pos])]
        seen = by_hash.setdefault(int(ordered[pos]), set())
        if text in seen:
            return False
        seen.add(text)
    return True


def decode_id(span: bytes) -> str:
    r"""Decode one stored (non-empty) ID back to its ``str``.

    orjson escapes only ``"``, ``\`` and control characters, each with a
    backslash, so a span without one is the ID's UTF-8 between quotes, and
    slicing it is about twice as fast as parsing it.

    >>> decode_id(b'"infores:x/1"')
    'infores:x/1'
    >>> decode_id(orjson.dumps('say "hi" \\o/'))
    'say "hi" \\o/'
    """
    if b"\\" in span:
        text: str = orjson.loads(span)
        return text
    return span[1:-1].decode("utf-8")


class EdgeIdStore:
    """Edge IDs by forward-CSR position (see the module docstring).

    >>> store = EdgeIdStore.from_ids(["e1", None, 'say "hi"'])
    >>> store.get(0), store.get(1), store.get(2)
    ('e1', None, 'say "hi"')
    >>> store.get_json_batch(np.array([2, 0, 1]))
    [b'"say \\\\"hi\\\\""', b'"e1"', b'']
    >>> store.get_batch([0, 1, 2])
    {0: 'e1', 2: 'say "hi"'}
    >>> store.metadata()
    {'edge_ids_unique': True, 'edge_ids_escaped': True}
    """

    def __init__(
        self, blob: Blob, offsets: np.ndarray, unique: bool, escaped: bool = True
    ):
        self._blob = blob
        self.offsets = offsets
        #: No two edges share an ID.
        self.unique = unique
        #: Some ID's JSON has an escape; if not, IDs decode without parsing.
        self.escaped = escaped

    def __len__(self) -> int:
        return len(self.offsets) - 1

    @classmethod
    def from_ids(
        cls, ids: Sequence[Optional[str]], chunk: int = 1_000_000
    ) -> "EdgeIdStore":
        """Build from one ID (or None) per edge, in forward-CSR order.

        IDs are encoded *chunk* at a time, so a build holds at most that many
        encoded IDs as separate objects next to the blob.
        """
        offsets = np.zeros(len(ids) + 1, dtype=np.int64)
        pieces = []
        for start in range(0, len(ids), chunk):
            encoded = [_encode_id(i) for i in ids[start : start + chunk]]
            offsets[start + 1 : start + 1 + len(encoded)] = np.fromiter(
                map(len, encoded), dtype=np.int64, count=len(encoded)
            )
            pieces.append(b"".join(encoded))
        np.cumsum(offsets, out=offsets)
        escaped = any(b"\\" in piece for piece in pieces)
        blob = b"".join(pieces)
        return cls(blob, offsets, ids_unique(ids), escaped)

    def metadata(self) -> dict:
        """What :meth:`load` needs back, for the graph's ``metadata.pkl``."""
        return {"edge_ids_unique": self.unique, "edge_ids_escaped": self.escaped}

    def save(self, directory: Path) -> None:
        """Write ``edge_ids.bin`` and ``edge_id_offsets.npy`` to *directory*."""
        with open(directory / BLOB_FILE, "wb") as f:
            f.write(self._blob)
        np.save(directory / OFFSETS_FILE, self.offsets)

    @classmethod
    def load(
        cls, directory: Path, metadata: dict, in_memory: bool = False
    ) -> "EdgeIdStore":
        """Open the files :meth:`save` wrote, memory-mapped unless *in_memory*.

        A memory-mapped store is shared by every process that opens it.
        *metadata* holds what :meth:`metadata` returned when it was saved.
        """
        path = directory / BLOB_FILE
        blob: Blob
        if in_memory or path.stat().st_size == 0:
            # mmap cannot map an empty file (a graph whose edges have no IDs)
            blob = path.read_bytes()
        else:
            with open(path, "rb") as f:
                blob = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        offsets = np.load(directory / OFFSETS_FILE, mmap_mode="r")
        offsets = np.array(offsets) if in_memory else np.asarray(offsets)
        return cls(
            blob,
            offsets,
            metadata["edge_ids_unique"],
            metadata["edge_ids_escaped"],
        )

    def spans(self, indices: np.ndarray) -> tuple[list, list]:
        """Start and end offsets of each edge in *indices*, as Python lists."""
        indices = np.asarray(indices, dtype=np.int64)
        return self.offsets[indices].tolist(), self.offsets[indices + 1].tolist()

    def lengths(self, indices: np.ndarray) -> np.ndarray:
        """Encoded length of each edge's ID in *indices*; 0 means no ID."""
        indices = np.asarray(indices, dtype=np.int64)
        lengths: np.ndarray = self.offsets[indices + 1] - self.offsets[indices]
        return lengths

    def get_json_batch(self, indices: np.ndarray) -> list[bytes]:
        """Each edge's ID as its JSON string, ``b""`` for an edge with none."""
        blob = self._blob
        starts, ends = self.spans(indices)
        return [blob[s:e] for s, e in zip(starts, ends)]

    def decode(self, spans: list[bytes]) -> list[str]:
        """Decode IDs from :meth:`get_json_batch` (none of them empty)."""
        if self.escaped:
            return [decode_id(span) for span in spans]
        return [span[1:-1].decode("utf-8") for span in spans]

    def get(self, index: int) -> Optional[str]:
        """One edge's ID, or None."""
        start, end = int(self.offsets[index]), int(self.offsets[index + 1])
        return decode_id(self._blob[start:end]) if end > start else None

    def get_batch(self, indices: Iterable[int]) -> dict[int, str]:
        """Map each edge in *indices* that has an ID to it."""
        idx = np.fromiter(indices, dtype=np.int64)
        if self.escaped:
            return {
                i: decode_id(span)
                for i, span in zip(idx.tolist(), self.get_json_batch(idx))
                if span
            }
        blob = self._blob
        starts, ends = self.spans(idx)
        return {
            i: blob[s + 1 : e - 1].decode("utf-8")
            for i, s, e in zip(idx.tolist(), starts, ends)
            if e > s
        }

    def close(self) -> None:
        """Release the memory map, if any."""
        if isinstance(self._blob, mmap.mmap):
            self._blob.close()
