"""Load-side API for pre-computed scoring metadata.

``ScoringMetadata.load(graph_dir)`` reads the manifest and lazily memory-maps
each declared field.  Attached to ``CSRGraph`` as ``graph.scoring`` (or
``None`` when no manifest is present).

Adding a new field type means:
  1. An ingest script writes the file and records an entry in the manifest.
  2. This class exposes it via a new property that mmaps on first access.

No schema migration is required for existing graphs that don't have the
new field.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from gandalf.config import settings
from gandalf.metadata.manifest import Manifest, load_manifest

logger = logging.getLogger(__name__)


# Manifest field names.  Kept as module constants so ingest scripts and the
# loader agree on the key.
FIELD_NODE_PUB_COUNTS = "node_pub_counts"
FIELD_EDGE_PUB_COUNTS = "edge_pub_counts"


def _mmap_npy(path: Path) -> np.ndarray:
    """Mmap a .npy file, honoring ``settings.load_mmaps_into_memory``."""
    arr: np.ndarray = np.load(path, mmap_mode="r")
    if settings.load_mmaps_into_memory:
        arr = np.array(arr)
    return arr


class ScoringMetadata:
    """Read-only view of pre-computed scoring fields for a graph.

    Fields are exposed as attributes that return ``None`` when not present.
    Accessing a declared field mmaps the file on first access and caches the
    resulting numpy array.
    """

    def __init__(self, graph_dir: Path, manifest: Manifest):
        self._graph_dir = Path(graph_dir)
        self._manifest = manifest
        self._cache: dict = {}

    @classmethod
    def load(cls, graph_dir: Path) -> Optional["ScoringMetadata"]:
        """Load metadata for a graph directory.

        Returns ``None`` if no manifest is present (graph has no scoring
        metadata).  Does not open any field files until they are accessed.
        """
        graph_dir = Path(graph_dir)
        manifest = load_manifest(graph_dir)
        if not manifest.fields:
            return None
        return cls(graph_dir, manifest)

    @property
    def manifest(self) -> Manifest:
        return self._manifest

    def has(self, field_name: str) -> bool:
        return self._manifest.has_field(field_name)

    def _load_array_field(self, field_name: str) -> Optional[np.ndarray]:
        if field_name in self._cache:
            return self._cache[field_name]
        entry = self._manifest.get_field(field_name)
        if entry is None:
            return None
        file_name = entry.get("file")
        if not file_name:
            raise ValueError(
                f"Scoring manifest field {field_name!r} missing 'file' entry"
            )
        arr = _mmap_npy(self._graph_dir / file_name)
        self._cache[field_name] = arr
        return arr

    @property
    def node_pub_counts(self) -> Optional[np.ndarray]:
        """Per-node publication counts as uint32[N], or None if not ingested."""
        return self._load_array_field(FIELD_NODE_PUB_COUNTS)

    @property
    def edge_pub_counts(self) -> Optional[np.ndarray]:
        """Per-edge publication counts as uint32[E], or None if not ingested."""
        return self._load_array_field(FIELD_EDGE_PUB_COUNTS)
