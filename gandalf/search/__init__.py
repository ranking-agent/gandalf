"""Search module for TRAPI query resolution.

This package implements complex graph traversal with predicate expansion,
qualifier matching, symmetric predicate handling, and subclass reasoning.
"""

from gandalf.search.expanders import PredicateExpander, QualifierExpander
from gandalf.search.gc_utils import GCMonitor, gc_disabled
from gandalf.search.lookup import lookup
from gandalf.search.path_arrays import PathArrays
from gandalf.search.path_finder import (
    do_one_hop,
    find_3hop_paths_filtered,
    find_3hop_paths_with_properties,
    find_mechanistic_paths,
)
from gandalf.search.qualifiers import edge_matches_qualifier_constraints
from gandalf.search.reconstruct import (
    LARGE_RESULT_PATH_THRESHOLD,
    MAX_PATH_LIMIT,
)

# Backwards-compatible alias: old code imports the underscore-prefixed name
_edge_matches_qualifier_constraints = edge_matches_qualifier_constraints

# Internal functions exposed for backwards compatibility with scripts
from gandalf.search.path_finder import _find_3hop_paths_directed_idx  # noqa: F401

__all__ = [
    # Main entry point
    "lookup",
    # Classes
    "PathArrays",
    "PredicateExpander",
    "QualifierExpander",
    "GCMonitor",
    # 3-hop path finding
    "find_3hop_paths_with_properties",
    "find_3hop_paths_filtered",
    "find_mechanistic_paths",
    "do_one_hop",
    # Utilities
    "gc_disabled",
    "edge_matches_qualifier_constraints",
    "_edge_matches_qualifier_constraints",
    # Constants
    "LARGE_RESULT_PATH_THRESHOLD",
    "MAX_PATH_LIMIT",
]
