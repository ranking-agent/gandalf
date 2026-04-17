"""Pre-computed scoring metadata for Gandalf graphs.

Holds data that is derived externally (outside the JSONL graph build) and
ingested afterward: publication counts, pairwise intersections, embeddings,
etc.  Each field is stored as its own mmap-friendly file alongside the graph,
indexed by the same uint32 node_idx / forward-CSR edge_idx used by the
topology arrays.  A single ``scoring_manifest.json`` records which fields are
present and their schema.
"""

from gandalf.metadata.manifest import (
    MANIFEST_FILENAME,
    Manifest,
    load_manifest,
    save_manifest,
)
from gandalf.metadata.store import ScoringMetadata

__all__ = [
    "MANIFEST_FILENAME",
    "Manifest",
    "ScoringMetadata",
    "load_manifest",
    "save_manifest",
]
