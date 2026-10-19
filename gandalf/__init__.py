"""
Gandalf - Fast 3-hop path finding in large knowledge graphs
"""

from importlib.metadata import PackageNotFoundError, version as _dist_version

# Read from the installed distribution, so it is the version in
# pyproject.toml rather than a second copy that can go stale.
try:
    __version__ = _dist_version("gandalf-csr")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "0.0.0+unknown"

from gandalf import plugins  # noqa: F401  registers built-in node filters & enrichers
from gandalf.annotate import annotate_response
from gandalf.enrichment import enrich_knowledge_graph
from gandalf.diagnostics import (
    analyze_node_types,
    analyze_predicates,
    diagnose_path_explosion,
)
from gandalf.graph import CSRGraph, EdgePropertyStore, EdgePropertyStoreBuilder
from gandalf.lmdb_store import LMDBPropertyStore
from gandalf.loader import build_graph_from_jsonl, build_graph_from_mongo
from gandalf.node_annotations import (
    BIOTHINGS_ANNOTATIONS_ATTRIBUTE_TYPE_ID,
    annotate_node_properties,
    fetch_annotations,
)
from gandalf.normalize import normalize_edge, normalize_node
from gandalf.query_planner import get_next_qedge
from gandalf.search import (
    find_3hop_paths_filtered,
    find_3hop_paths_with_properties,
    find_mechanistic_paths,
    lookup,
)
from gandalf.validation import (
    ValidationError,
    ValidationResult,
    compare_trapi_messages,
    debug_missing_edge,
    diagnose_graph_edge_storage,
    find_edge_in_graph,
    validate_edge_exists,
    validate_edge_list,
    validate_node_exists,
    validate_trapi_response,
)

__all__ = [
    # Core classes
    "CSRGraph",
    "EdgePropertyStore",
    "EdgePropertyStoreBuilder",
    "LMDBPropertyStore",
    # Enrichment
    "enrich_knowledge_graph",
    # Annotation
    "annotate_response",
    # Loading
    "build_graph_from_jsonl",
    "build_graph_from_mongo",
    # Node annotation (build time)
    "annotate_node_properties",
    "fetch_annotations",
    "BIOTHINGS_ANNOTATIONS_ATTRIBUTE_TYPE_ID",
    # Normalization
    "normalize_edge",
    "normalize_node",
    # Search
    "find_3hop_paths_filtered",
    "find_3hop_paths_with_properties",
    "find_mechanistic_paths",
    "lookup",
    # Diagnostics
    "diagnose_path_explosion",
    "analyze_node_types",
    "analyze_predicates",
    # Query Planner
    "get_next_qedge",
    # Validation
    "ValidationError",
    "ValidationResult",
    "compare_trapi_messages",
    "validate_trapi_response",
    "validate_edge_list",
    "validate_node_exists",
    "validate_edge_exists",
    "find_edge_in_graph",
    "debug_missing_edge",
    "diagnose_graph_edge_storage",
]
