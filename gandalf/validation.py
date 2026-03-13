"""Validation utilities for verifying query results against the graph.

This module provides tools for development and testing to ensure that
query results are consistent with the actual graph data.
"""

from dataclasses import dataclass
from typing import Optional

from bmt.toolkit import Toolkit

from gandalf.graph import CSRGraph


# Module-level BMT instance for predicate lookups
_bmt: Optional[Toolkit] = None


def _get_bmt() -> Toolkit:
    """Get or create the BMT instance."""
    global _bmt
    if _bmt is None:
        _bmt = Toolkit()
    return _bmt


def _get_inverse_predicate(predicate: str) -> Optional[str]:
    """Get the inverse of a predicate if one exists."""
    bmt = _get_bmt()
    try:
        if bmt.has_inverse(predicate):
            return bmt.get_inverse_predicate(predicate, formatted=True)
    except Exception:
        pass
    return None


def _is_symmetric(predicate: str) -> bool:
    """Check if a predicate is symmetric."""
    bmt = _get_bmt()
    try:
        return bmt.is_symmetric(predicate)
    except Exception:
        return False


@dataclass
class ValidationError:
    """Represents a validation error found in a result."""
    error_type: str
    message: str
    path_index: Optional[int] = None
    edge_id: Optional[str] = None
    node_id: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of validating query results against the graph."""
    valid: bool
    total_paths: int
    valid_paths: int
    errors: list[ValidationError]

    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = [
            f"Validation {'PASSED' if self.valid else 'FAILED'}",
            f"  Total paths: {self.total_paths}",
            f"  Valid paths: {self.valid_paths}",
            f"  Invalid paths: {self.total_paths - self.valid_paths}",
        ]
        if self.errors:
            lines.append(f"  Errors ({len(self.errors)}):")
            for err in self.errors[:10]:  # Show first 10 errors
                lines.append(f"    - [{err.error_type}] {err.message}")
            if len(self.errors) > 10:
                lines.append(f"    ... and {len(self.errors) - 10} more errors")
        return "\n".join(lines)


def validate_node_exists(graph: CSRGraph, node_id: str) -> Optional[ValidationError]:
    """Check if a node exists in the graph."""
    idx = graph.get_node_idx(node_id)
    if idx is None:
        return ValidationError(
            error_type="NODE_NOT_FOUND",
            message=f"Node '{node_id}' not found in graph",
            node_id=node_id,
        )
    return None


def validate_edge_exists(
    graph: CSRGraph,
    subject_id: str,
    predicate: str,
    object_id: str,
    check_inverse: bool = True,
) -> Optional[ValidationError]:
    """
    Check if an edge exists in the graph, including inverse predicate checks.

    This function checks for:
    1. Direct match: subject --[predicate]--> object
    2. Symmetric match: object --[predicate]--> subject (if predicate is symmetric)
    3. Inverse match: object --[inverse(predicate)]--> subject

    Args:
        graph: The CSRGraph to validate against
        subject_id: Subject node ID
        predicate: Edge predicate
        object_id: Object node ID
        check_inverse: If True, also check for inverse predicates in reverse direction

    Returns:
        ValidationError if edge not found, None otherwise
    """
    subj_idx = graph.get_node_idx(subject_id)
    obj_idx = graph.get_node_idx(object_id)

    if subj_idx is None:
        return ValidationError(
            error_type="EDGE_SUBJECT_NOT_FOUND",
            message=f"Edge subject '{subject_id}' not found in graph",
            node_id=subject_id,
        )

    if obj_idx is None:
        return ValidationError(
            error_type="EDGE_OBJECT_NOT_FOUND",
            message=f"Edge object '{object_id}' not found in graph",
            node_id=object_id,
        )

    # Check forward direction: subject -> object with exact predicate
    edges = graph.get_all_edges_between(subj_idx, obj_idx)
    for edge_pred, _ in edges:
        if edge_pred == predicate:
            return None  # Found the edge

    if check_inverse:
        # Check reverse direction for symmetric predicates: object -> subject
        if _is_symmetric(predicate):
            reverse_edges = graph.get_all_edges_between(obj_idx, subj_idx)
            for edge_pred, _ in reverse_edges:
                if edge_pred == predicate:
                    return None  # Found symmetric edge in reverse

        # Check reverse direction for inverse predicate: object --[inverse]--> subject
        inverse_pred = _get_inverse_predicate(predicate)
        if inverse_pred:
            reverse_edges = graph.get_all_edges_between(obj_idx, subj_idx)
            for edge_pred, _ in reverse_edges:
                if edge_pred == inverse_pred:
                    return None  # Found inverse edge

        # Also check if the predicate itself is an inverse of something stored forward
        # e.g., result says "treated_by" but graph has "treats" in forward direction
        # This handles the case where the result predicate is the inverse
        for edge_pred, _ in edges:
            stored_inverse = _get_inverse_predicate(edge_pred)
            if stored_inverse == predicate:
                return None  # Result predicate is inverse of stored predicate

    return ValidationError(
        error_type="EDGE_NOT_FOUND",
        message=f"Edge '{subject_id}' --[{predicate}]--> '{object_id}' not found in graph",
    )


def validate_trapi_response(
    graph: CSRGraph,
    response: dict,
    check_inverse: bool = True,
    verbose: bool = False,
) -> ValidationResult:
    """
    Validate a TRAPI response against the graph.

    Checks that all nodes and edges in the knowledge graph and results
    actually exist in the source graph. For edges, this includes checking
    for inverse predicates (e.g., if result has "treats", also checks for
    "treated_by" in the reverse direction).

    Args:
        graph: The CSRGraph to validate against
        response: TRAPI response dict with message.knowledge_graph and message.results
        check_inverse: If True, also check for inverse predicates in reverse direction
        verbose: Print progress information

    Returns:
        ValidationResult with validation status and any errors found
    """
    errors = []
    message = response.get("message", {})
    kg = message.get("knowledge_graph", {})
    results = message.get("results", [])

    if verbose:
        print(f"Validating response with {len(kg.get('nodes', {}))} KG nodes, "
              f"{len(kg.get('edges', {}))} KG edges, {len(results)} results")

    # Validate knowledge graph nodes
    kg_nodes = kg.get("nodes", {})
    for node_id, node_data in kg_nodes.items():
        err = validate_node_exists(graph, node_id)
        if err:
            errors.append(err)

    if verbose:
        print(f"  Validated {len(kg_nodes)} KG nodes, {len([e for e in errors if e.error_type.startswith('NODE')])} errors")

    # Validate knowledge graph edges
    kg_edges = kg.get("edges", {})
    edge_errors_before = len(errors)
    for edge_id, edge_data in kg_edges.items():
        subject_id = edge_data.get("subject")
        predicate = edge_data.get("predicate")
        object_id = edge_data.get("object")

        if not all([subject_id, predicate, object_id]):
            errors.append(ValidationError(
                error_type="INVALID_EDGE_DATA",
                message=f"Edge '{edge_id}' missing required fields",
                edge_id=edge_id,
            ))
            continue

        err = validate_edge_exists(
            graph, subject_id, predicate, object_id,
            check_inverse=check_inverse
        )
        if err:
            err.edge_id = edge_id
            errors.append(err)

    if verbose:
        print(f"  Validated {len(kg_edges)} KG edges, {len(errors) - edge_errors_before} errors")

    # Count valid paths (results where all bindings are valid)
    valid_paths = 0
    for i, result in enumerate(results):
        path_valid = True

        # Check node bindings
        node_bindings = result.get("node_bindings", {})
        for qnode_id, bindings in node_bindings.items():
            for binding in bindings:
                node_id = binding.get("id")
                if node_id and node_id not in kg_nodes:
                    errors.append(ValidationError(
                        error_type="BINDING_NODE_NOT_IN_KG",
                        message=f"Result {i}: Node binding '{node_id}' not in knowledge graph",
                        path_index=i,
                        node_id=node_id,
                    ))
                    path_valid = False

        # Check edge bindings
        analyses = result.get("analyses", [])
        for analysis in analyses:
            edge_bindings = analysis.get("edge_bindings", {})
            for qedge_id, bindings in edge_bindings.items():
                for binding in bindings:
                    edge_id = binding.get("id")
                    if edge_id and edge_id not in kg_edges:
                        errors.append(ValidationError(
                            error_type="BINDING_EDGE_NOT_IN_KG",
                            message=f"Result {i}: Edge binding '{edge_id}' not in knowledge graph",
                            path_index=i,
                            edge_id=edge_id,
                        ))
                        path_valid = False

        if path_valid:
            valid_paths += 1

    if verbose:
        print(f"  Validated {len(results)} results, {valid_paths} valid")

    return ValidationResult(
        valid=len(errors) == 0,
        total_paths=len(results),
        valid_paths=valid_paths,
        errors=errors,
    )


def validate_edge_list(
    graph: CSRGraph,
    edges: list[tuple[int, str, int]],
    check_inverse: bool = True,
    verbose: bool = False,
) -> ValidationResult:
    """
    Validate a list of edges (as returned by _query_edge).

    Args:
        graph: The CSRGraph to validate against
        edges: List of (subject_idx, predicate, object_idx) tuples
        check_inverse: If True, also check for inverse predicates in reverse direction
        verbose: Print progress information

    Returns:
        ValidationResult with validation status and any errors found
    """
    errors = []
    valid_count = 0

    for i, (subj_idx, predicate, obj_idx) in enumerate(edges):
        # Get node IDs
        subj_id = graph.get_node_id(subj_idx)
        obj_id = graph.get_node_id(obj_idx)

        if subj_id is None:
            errors.append(ValidationError(
                error_type="INVALID_SUBJECT_IDX",
                message=f"Edge {i}: Subject index {subj_idx} has no node ID",
                path_index=i,
            ))
            continue

        if obj_id is None:
            errors.append(ValidationError(
                error_type="INVALID_OBJECT_IDX",
                message=f"Edge {i}: Object index {obj_idx} has no node ID",
                path_index=i,
            ))
            continue

        # Check if edge exists
        err = validate_edge_exists(
            graph, subj_id, predicate, obj_id,
            check_inverse=check_inverse
        )
        if err:
            err.path_index = i
            errors.append(err)
        else:
            valid_count += 1

    if verbose:
        print(f"Validated {len(edges)} edges: {valid_count} valid, {len(errors)} errors")

    return ValidationResult(
        valid=len(errors) == 0,
        total_paths=len(edges),
        valid_paths=valid_count,
        errors=errors,
    )


def diagnose_graph_edge_storage(
    graph: CSRGraph,
    node_id: str,
    max_neighbors: int = 5,
) -> str:
    """
    Diagnose edge storage for a node to check if predicates are being stored correctly.

    Args:
        graph: The CSRGraph to check
        node_id: Node ID to diagnose
        max_neighbors: Maximum neighbors to show

    Returns:
        Human-readable diagnostic report
    """
    lines = [f"Edge storage diagnosis for node '{node_id}':", ""]

    node_idx = graph.get_node_idx(node_id)
    if node_idx is None:
        lines.append(f"  Node not found in graph!")
        return "\n".join(lines)

    lines.append(f"  Node index: {node_idx}")

    # Check outgoing edges
    lines.append("")
    lines.append("Outgoing edges (neighbors_with_properties):")
    try:
        out_count = 0
        for neighbor_idx, predicate, props, _fwd_edge_idx in graph.neighbors_with_properties(node_idx):
            if out_count >= max_neighbors:
                lines.append(f"  ... (limited to {max_neighbors})")
                break
            neighbor_id = graph.get_node_id(neighbor_idx)
            lines.append(f"  -> {neighbor_id} (idx={neighbor_idx})")
            lines.append(f"     predicate: {predicate!r}")
            lines.append(f"     props keys: {list(props.keys()) if props else '(none)'}")
            out_count += 1
        if out_count == 0:
            lines.append("  (no outgoing edges)")
    except Exception as e:
        lines.append(f"  ERROR: {e}")

    # Check incoming edges
    lines.append("")
    lines.append("Incoming edges (incoming_neighbors_with_properties):")
    try:
        in_count = 0
        for neighbor_idx, predicate, props, _fwd_edge_idx in graph.incoming_neighbors_with_properties(node_idx):
            if in_count >= max_neighbors:
                lines.append(f"  ... (limited to {max_neighbors})")
                break
            neighbor_id = graph.get_node_id(neighbor_idx)
            lines.append(f"  <- {neighbor_id} (idx={neighbor_idx})")
            lines.append(f"     predicate: {predicate!r}")
            lines.append(f"     props keys: {list(props.keys()) if props else '(none)'}")
            in_count += 1
        if in_count == 0:
            lines.append("  (no incoming edges)")
    except Exception as e:
        lines.append(f"  ERROR: {e}")

    # Check raw neighbor indices
    lines.append("")
    lines.append("Raw neighbor check (neighbors vs neighbors_with_properties):")
    try:
        raw_neighbors = graph.neighbors(node_idx)
        lines.append(f"  neighbors() count: {len(raw_neighbors)}")
        with_props_count = sum(1 for _ in graph.neighbors_with_properties(node_idx))
        lines.append(f"  neighbors_with_properties() count: {with_props_count}")
        if len(raw_neighbors) != with_props_count:
            lines.append(f"  WARNING: Counts don't match!")
    except Exception as e:
        lines.append(f"  ERROR: {e}")

    return "\n".join(lines)


def find_edge_in_graph(
    graph: CSRGraph,
    subject_id: str,
    object_id: str,
) -> list[dict]:
    """
    Find all edges between two nodes in both directions.

    Useful for debugging when an expected edge is not found.

    Args:
        graph: The CSRGraph to search
        subject_id: First node ID
        object_id: Second node ID

    Returns:
        List of edge dicts with direction, predicate, and properties
    """
    results = []

    subj_idx = graph.get_node_idx(subject_id)
    obj_idx = graph.get_node_idx(object_id)

    if subj_idx is None or obj_idx is None:
        return results

    # Forward direction: subject -> object
    forward_edges = graph.get_all_edges_between(subj_idx, obj_idx)
    for predicate, props in forward_edges:
        results.append({
            "direction": "forward",
            "subject": subject_id,
            "predicate": predicate,
            "object": object_id,
            "properties": props,
        })

    # Reverse direction: object -> subject
    reverse_edges = graph.get_all_edges_between(obj_idx, subj_idx)
    for predicate, props in reverse_edges:
        results.append({
            "direction": "reverse",
            "subject": object_id,
            "predicate": predicate,
            "object": subject_id,
            "properties": props,
        })

    return results


def _result_node_fingerprint(
    result: dict,
) -> frozenset[tuple[str, str]]:
    """Create a hashable fingerprint from a result's node bindings."""
    pairs: list[tuple[str, str]] = []
    for qnode_id, bindings in result.get("node_bindings", {}).items():
        for binding in bindings:
            pairs.append((qnode_id, binding.get("id", "")))
    return frozenset(pairs)


def _get_qgraph_path_order(
    query_graph: dict,
) -> tuple[list[str], list[str]]:
    """Determine a linear ordering of qnode and qedge IDs from the query graph.

    Walks the query graph edges to produce ``(ordered_qnode_ids,
    ordered_qedge_ids)`` suitable for rendering a chain-like path.
    Falls back to sorted IDs when the graph is cyclic or empty.
    """
    qedges = query_graph.get("edges", {})
    if not qedges:
        return sorted(query_graph.get("nodes", {}).keys()), []

    # Map each subject qnode to its outgoing (qedge_id, object_qnode).
    subj_to_edge: dict[str, tuple[str, str]] = {}
    subj_set: set[str] = set()
    obj_set: set[str] = set()

    for qedge_id, qedge in qedges.items():
        subj = qedge["subject"]
        obj = qedge["object"]
        subj_to_edge[subj] = (qedge_id, obj)
        subj_set.add(subj)
        obj_set.add(obj)

    # Start node: appears as a subject but never as an object.
    start_nodes = subj_set - obj_set
    if not start_nodes:
        return sorted(query_graph.get("nodes", {}).keys()), sorted(qedges.keys())

    start = sorted(start_nodes)[0]

    ordered_qnodes: list[str] = [start]
    ordered_qedges: list[str] = []
    current = start
    visited: set[str] = {start}

    while current in subj_to_edge:
        qedge_id, next_node = subj_to_edge[current]
        ordered_qedges.append(qedge_id)
        if next_node in visited:
            break
        ordered_qnodes.append(next_node)
        visited.add(next_node)
        current = next_node

    return ordered_qnodes, ordered_qedges


def _format_result_path(
    result: dict,
    kg_edges: dict,
    qnode_order: list[str],
    qedge_order: list[str],
) -> str:
    """Format a TRAPI result as a human-readable path string.

    Produces output like::

        CHEBI:6801 --(biolink:affects [object_aspect_qualifier: activity])--> NCBIGene:5468
    """
    node_bindings = result.get("node_bindings", {})

    # Fall back to sorted qnode IDs when no ordering is available.
    if not qnode_order:
        qnode_order = sorted(node_bindings.keys())

    # Collect edge bindings from the first analysis block.
    edge_bindings: dict = {}
    analyses = result.get("analyses", [])
    if analyses:
        edge_bindings = analyses[0].get("edge_bindings", {})

    # Map qnode_id -> bound node ID.
    qnode_to_id: dict[str, str] = {}
    for qnode_id, bindings in node_bindings.items():
        if bindings:
            qnode_to_id[qnode_id] = bindings[0].get("id", "?")

    parts: list[str] = []
    for i, qnode_id in enumerate(qnode_order):
        parts.append(qnode_to_id.get(qnode_id, "?"))

        if i < len(qedge_order):
            qedge_id = qedge_order[i]
            edge_label = _format_edge_bindings(qedge_id, edge_bindings, kg_edges)
            parts.append(f" --({edge_label})--> ")

    return "".join(parts)


def _format_edge_bindings(
    qedge_id: str,
    edge_bindings: dict,
    kg_edges: dict,
) -> str:
    """Format all edge bindings for a single qedge as a compact label.

    Multiple distinct predicate/qualifier combinations are separated by
    ``" | "``.
    """
    bindings = edge_bindings.get(qedge_id, [])
    if not bindings:
        return "?"

    descriptions: list[str] = []
    seen: set[tuple[str, str]] = set()

    for binding in bindings:
        edge_id = binding.get("id")
        edge = kg_edges.get(edge_id, {}) if edge_id else {}

        predicate = edge.get("predicate", "?")
        qualifiers = edge.get("qualifiers") or []

        qual_parts: list[str] = []
        for q in qualifiers:
            q_type = q.get("qualifier_type_id", "").replace("biolink:", "")
            q_val = q.get("qualifier_value", "")
            qual_parts.append(f"{q_type}: {q_val}")

        quals_str = ", ".join(sorted(qual_parts))
        key = (predicate, quals_str)
        if key in seen:
            continue
        seen.add(key)

        if quals_str:
            descriptions.append(f"{predicate} [{quals_str}]")
        else:
            descriptions.append(predicate)

    return " | ".join(descriptions)


def compare_trapi_messages(
    message_a: dict,
    message_b: dict,
) -> list[str]:
    """Find results/paths in message_a that are not present in message_b.

    Each result is identified by its unique node binding combination (the set
    of ``(qnode_id, node_id)`` pairs from ``node_bindings``).  For every
    result in *message_a* whose node-binding fingerprint does not appear in
    *message_b*, a human-readable path string is returned showing the node
    IDs, edge predicates, and any edge qualifiers along the path.

    Args:
        message_a: TRAPI message dict (must contain ``query_graph``,
            ``knowledge_graph``, and ``results``).
        message_b: TRAPI message dict to compare against.

    Returns:
        List of human-readable path descriptions for results present in
        *message_a* but missing from *message_b*.
    """
    # Build node-binding fingerprints from message_b for O(1) lookup.
    b_fingerprints: set[frozenset[tuple[str, str]]] = set()
    for result in message_b.get("results", []):
        b_fingerprints.add(_result_node_fingerprint(result))

    kg_edges = message_a.get("knowledge_graph", {}).get("edges", {})
    query_graph = message_a.get("query_graph", {})
    qnode_order, qedge_order = _get_qgraph_path_order(query_graph)

    missing: list[str] = []
    for result in message_a.get("results", []):
        if _result_node_fingerprint(result) not in b_fingerprints:
            missing.append(
                _format_result_path(result, kg_edges, qnode_order, qedge_order)
            )

    return missing


def debug_missing_edge(
    graph: CSRGraph,
    subject_id: str,
    predicate: str,
    object_id: str,
) -> str:
    """
    Generate a debug report for a missing edge.

    Args:
        graph: The CSRGraph to search
        subject_id: Expected subject node ID
        predicate: Expected predicate
        object_id: Expected object node ID

    Returns:
        Human-readable debug report
    """
    lines = [
        f"Debug report for missing edge:",
        f"  Expected: {subject_id} --[{predicate}]--> {object_id}",
        "",
    ]

    # Show predicate info
    inverse_pred = _get_inverse_predicate(predicate)
    is_symmetric = _is_symmetric(predicate)
    lines.append("Predicate info:")
    lines.append(f"  Predicate: {predicate}")
    lines.append(f"  Is symmetric: {is_symmetric}")
    lines.append(f"  Inverse: {inverse_pred if inverse_pred else '(none)'}")

    # Check if nodes exist
    subj_idx = graph.get_node_idx(subject_id)
    obj_idx = graph.get_node_idx(object_id)

    lines.append("")
    lines.append("Node existence:")
    lines.append(f"  Subject '{subject_id}': EXISTS (idx={subj_idx})" if subj_idx is not None else f"  Subject '{subject_id}': NOT FOUND")
    lines.append(f"  Object '{object_id}': EXISTS (idx={obj_idx})" if obj_idx is not None else f"  Object '{object_id}': NOT FOUND")

    if subj_idx is None or obj_idx is None:
        return "\n".join(lines)

    # Find all edges between these nodes
    lines.append("")
    lines.append("Edges found between these nodes:")

    found_edges = find_edge_in_graph(graph, subject_id, object_id)
    if not found_edges:
        lines.append("  (none)")
    else:
        for edge in found_edges:
            edge_inverse = _get_inverse_predicate(edge['predicate'])
            inverse_note = f" (inverse: {edge_inverse})" if edge_inverse else ""
            lines.append(
                f"  [{edge['direction']}] {edge['subject']} --[{edge['predicate']}]--> {edge['object']}{inverse_note}"
            )

    # Show what we expected to find
    lines.append("")
    lines.append("Expected matches (any of these would validate):")
    lines.append(f"  1. Forward: {subject_id} --[{predicate}]--> {object_id}")
    if is_symmetric:
        lines.append(f"  2. Symmetric reverse: {object_id} --[{predicate}]--> {subject_id}")
    if inverse_pred:
        lines.append(f"  3. Inverse reverse: {object_id} --[{inverse_pred}]--> {subject_id}")

    # Check neighbors
    lines.append("")
    lines.append(f"Subject '{subject_id}' neighbors (first 10):")
    neighbors = graph.neighbors(subj_idx)
    for i, neighbor_idx in enumerate(neighbors[:10]):
        neighbor_id = graph.get_node_id(neighbor_idx)
        edges = graph.get_all_edges_between(subj_idx, neighbor_idx)
        preds = [p for p, _ in edges]
        lines.append(f"  -> {neighbor_id} via {preds}")
    if len(neighbors) > 10:
        lines.append(f"  ... and {len(neighbors) - 10} more")

    lines.append("")
    lines.append(f"Object '{object_id}' incoming neighbors (first 10):")
    incoming = graph.incoming_neighbors(obj_idx)
    for i, neighbor_idx in enumerate(incoming[:10]):
        neighbor_id = graph.get_node_id(neighbor_idx)
        edges = graph.get_all_edges_between(neighbor_idx, obj_idx)
        preds = [p for p, _ in edges]
        lines.append(f"  <- {neighbor_id} via {preds}")
    if len(incoming) > 10:
        lines.append(f"  ... and {len(incoming) - 10} more")

    return "\n".join(lines)
