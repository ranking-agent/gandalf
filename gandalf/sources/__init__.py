"""Pluggable graph sources for the gandalf loader.

A graph source yields already-normalized node and edge records that the loader's
build core consumes. ``KGXJsonlSource`` reads KGX jsonl and normalizes;
``MongoSource`` reads pre-normalized documents from MongoDB.

``MongoSource`` is intentionally NOT imported here so that ``import
gandalf.sources`` never triggers the optional ``pymongo`` dependency. Import it
explicitly from ``gandalf.sources.mongo`` when needed.
"""

from gandalf.sources.base import (
    GraphSource,
    SourceValidationError,
    validate_normalized_edge,
    validate_normalized_node,
)
from gandalf.sources.kgx_jsonl import KGXJsonlSource

__all__ = [
    "GraphSource",
    "SourceValidationError",
    "validate_normalized_edge",
    "validate_normalized_node",
    "KGXJsonlSource",
]
