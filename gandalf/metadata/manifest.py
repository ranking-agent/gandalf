"""Scoring-metadata manifest: records which pre-computed fields exist.

The manifest is a single JSON file at ``<graph_dir>/scoring_manifest.json``.
Each ingest script adds or overwrites an entry describing one field (the
file it wrote, dtype, shape, when it was produced, and any field-specific
stats).  Load-side code reads the manifest to know what to mmap.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

MANIFEST_FILENAME = "scoring_manifest.json"
MANIFEST_VERSION = 1


class Manifest:
    """Mutable in-memory view of the scoring manifest."""

    def __init__(self, version: int = MANIFEST_VERSION, fields: Optional[Dict[str, Dict[str, Any]]] = None):
        self.version = version
        self.fields: Dict[str, Dict[str, Any]] = fields or {}

    def set_field(self, name: str, entry: Dict[str, Any]) -> None:
        entry = dict(entry)
        entry.setdefault("ingested_at", datetime.now(timezone.utc).isoformat())
        self.fields[name] = entry

    def get_field(self, name: str) -> Optional[Dict[str, Any]]:
        return self.fields.get(name)

    def has_field(self, name: str) -> bool:
        return name in self.fields

    def to_dict(self) -> Dict[str, Any]:
        return {"version": self.version, "fields": self.fields}


def load_manifest(graph_dir: Path) -> Manifest:
    """Load the manifest from ``graph_dir``.  Returns an empty manifest if absent."""
    path = Path(graph_dir) / MANIFEST_FILENAME
    if not path.exists():
        return Manifest()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    version = int(data.get("version", MANIFEST_VERSION))
    if version != MANIFEST_VERSION:
        raise ValueError(
            f"Unsupported scoring manifest version {version} at {path} "
            f"(expected {MANIFEST_VERSION})"
        )
    fields = data.get("fields", {})
    if not isinstance(fields, dict):
        raise ValueError(f"Malformed scoring manifest at {path}: 'fields' must be an object")
    return Manifest(version=version, fields=fields)


def save_manifest(graph_dir: Path, manifest: Manifest) -> None:
    """Atomically write the manifest to ``graph_dir``."""
    graph_dir = Path(graph_dir)
    graph_dir.mkdir(parents=True, exist_ok=True)
    path = graph_dir / MANIFEST_FILENAME
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, indent=2, sort_keys=True)
    tmp.replace(path)
