"""
Gandalf - Fast 3-hop path finding in large knowledge graphs
"""

__version__ = "0.1.0"

from gandalf.enrichment import enrich_knowledge_graph
from gandalf.diagnostics import (
    analyze_node_types,
    analyze_predicates,
    diagnose_path_explosion,
)
from gandalf.graph import CSRGraph, EdgePropertyStore, EdgePropertyStoreBuilder
from gandalf.lmdb_store import LMDBPropertyStore
from gandalf.loader import build_graph_from_jsonl
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
    # Loading
    "build_graph_from_jsonl",
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
