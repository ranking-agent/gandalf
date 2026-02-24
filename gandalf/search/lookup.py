"""Main TRAPI query lookup implementation."""

import copy
import gc
import time
import uuid
from collections import defaultdict

from bmt.toolkit import Toolkit

from gandalf.query_planner import get_next_qedge, remove_orphaned
from gandalf.search.expanders import PredicateExpander, QualifierExpander
from gandalf.search.gc_utils import GCMonitor
from gandalf.search.query_edge import query_edge, query_subclass_edge
from gandalf.search.reconstruct import reconstruct_paths


def lookup(graph, query: dict, bmt=None, verbose=True, subclass=True, subclass_depth=1):
    """Take an arbitrary Translator query graph and return all matching paths.

    Args:
        graph: CSRGraph instance
        query: Full TRAPI request dict containing message.query_graph
        bmt: Biolink Model Toolkit instance (optional, will create if not provided)
        verbose: Print progress information
        subclass: If True, expand pinned nodes to include subclass descendants
        subclass_depth: Maximum number of subclass_of hops to traverse (default 1)

    Returns:
        TRAPI response dict with message containing results, knowledge_graph, etc.
    """
    t_start = time.perf_counter()

    # Start GC monitoring to track collection events
    gc_monitor = GCMonitor(verbose=verbose)
    gc_monitor.start()

    # Disable GC for the entire query to prevent expensive Gen 2 collections
    # during traversal.  The graph's long-lived numpy/CSR arrays cause Gen 2
    # scans to take 1-3s each while collecting 0 objects.  We re-enable and
    # run a single collection at the end of the query.
    gc_was_enabled_at_start = gc.isenabled()
    gc.disable()

    try:
        return _lookup_inner(graph, query, bmt, verbose, subclass, subclass_depth,
                             t_start, gc_monitor)
    finally:
        gc_monitor.stop()
        if gc_was_enabled_at_start:
            gc.enable()


def _lookup_inner(graph, query, bmt, verbose, subclass, subclass_depth,
                  t_start, gc_monitor):
    """Inner implementation of lookup with all the core logic."""
    if bmt is None:
        bmt = Toolkit()
        t_bmt = time.perf_counter()
        if verbose:
            print(f"BMT initialization: {t_bmt - t_start:.2f}s")
    elif verbose:
        print("Using provided BMT instance")

    # Create predicate expander for handling symmetric/inverse predicates at query time
    predicate_expander = PredicateExpander(bmt)

    # Create qualifier expander for handling qualifier value hierarchy at query time
    qualifier_expander = QualifierExpander(bmt)

    original_query_graph = query["message"]["query_graph"]
    query_graph = copy.deepcopy(original_query_graph)
    subqgraph = copy.deepcopy(query_graph)

    # Rewrite query graph for subclass expansion if requested
    if subclass and subqgraph["edges"]:
        if verbose:
            print(f"Rewriting query graph for subclass expansion (depth={subclass_depth})")
        _rewrite_for_subclass(subqgraph, subclass_depth=subclass_depth)
        # Use the rewritten graph as the query graph for the rest of the pipeline
        query_graph = copy.deepcopy(subqgraph)

    # Store results for each edge query
    # edge_id -> list of (subject_idx, predicate, object_idx) tuples
    edge_results = {}

    # Store inverse predicates for each edge (needed for path reconstruction)
    # edge_id -> set of inverse predicates
    edge_inverse_preds = {}

    # Track original query graph structure for path reconstruction
    original_edges = list(query_graph["edges"].keys())
    original_nodes = set(query_graph["nodes"].keys())

    if verbose:
        print(f"Query graph: {len(original_nodes)} nodes, {len(original_edges)} edges")

    # Process edges one at a time
    while len(subqgraph["edges"].keys()) > 0:
        # Get next edge to query
        next_edge_id, next_edge = get_next_qedge(subqgraph)

        if verbose:
            print(
                f"\nProcessing edge '{next_edge_id}': {next_edge['subject']} -> {next_edge['object']}"
            )

        # Get node constraints
        start_node = subqgraph["nodes"][next_edge["subject"]]
        end_node = subqgraph["nodes"][next_edge["object"]]

        # Get pinned node indices
        start_node_idxes = None
        if len(start_node.get("ids", [])) > 0:
            start_node_idxes = [
                graph.get_node_idx(node_id)
                for node_id in start_node["ids"]
                if graph.get_node_idx(node_id) is not None
            ]

        end_node_idxes = None
        if len(end_node.get("ids", [])) > 0:
            end_node_idxes = [
                graph.get_node_idx(node_id)
                for node_id in end_node["ids"]
                if graph.get_node_idx(node_id) is not None
            ]

        # Handle subclass edges with dedicated traversal
        if next_edge.get("_subclass"):
            subclass_edge_depth = next_edge.get("_subclass_depth", 1)
            edge_matches = query_subclass_edge(
                graph, start_node_idxes, end_node_idxes, subclass_edge_depth, verbose
            )
            edge_inverse_preds[next_edge_id] = set()
        else:
            # Get allowed predicates using reasoner-transpiler rules:
            # 1. Handle 'related_to' as "any predicate"
            # 2. Expand to descendants filtered to canonical OR symmetric only
            # 3. Also expand inverse predicates for bidirectional matching
            query_predicates = next_edge.get("predicates", [])
            forward_predicates, inverse_predicates = predicate_expander.expand_predicates(
                query_predicates
            )

            # Forward predicates are used for direct edge matching
            # Inverse predicates are used for reverse direction matching
            # Keep them separate to avoid confusion in reverse_pred_map construction
            allowed_predicates = forward_predicates

            if verbose and query_predicates:
                print(f"  Query predicates: {query_predicates}")
                print(f"  Expanded to {len(forward_predicates)} forward, {len(inverse_predicates) if inverse_predicates is not None else 0} inverse predicates")

            # Store inverse predicates for this edge (for path reconstruction)
            edge_inverse_preds[next_edge_id] = set(inverse_predicates) if inverse_predicates is not None else set()

            # Get qualifier constraints for this edge and expand to include descendant values
            qualifier_constraints = next_edge.get("qualifier_constraints", [])
            if qualifier_constraints:
                qualifier_constraints = qualifier_expander.expand_qualifier_constraints(
                    qualifier_constraints
                )

            # Query for matching edges
            edge_matches = query_edge(
                graph,
                start_node_idxes,
                end_node_idxes,
                start_node.get("categories", []),
                end_node.get("categories", []),
                allowed_predicates,
                qualifier_constraints,
                verbose,
                inverse_predicates=inverse_predicates,
            )

        # Store results for this edge
        edge_results[next_edge_id] = edge_matches

        if verbose:
            print(f"  Found {len(edge_matches):,} matching edges")

        if len(edge_matches) <= 0:
            print("Found no edge matches, returning 0 results.")
            original_edges = []
            break

        # Update subgraph with discovered nodes for next iteration
        discovered_subjects = set()
        discovered_objects = set()

        for subj_idx, pred, obj_idx, via_inverse, _fwd_edge_idx in edge_matches:
            if via_inverse:
                # Edge was found through inverse lookup: the stored edge is
                # subj_idx -> obj_idx, but the query direction is reversed.
                # So subj_idx corresponds to the query's object node and
                # obj_idx corresponds to the query's subject node.
                discovered_subjects.add(graph.get_node_id(obj_idx))
                discovered_objects.add(graph.get_node_id(subj_idx))
            else:
                discovered_subjects.add(graph.get_node_id(subj_idx))
                discovered_objects.add(graph.get_node_id(obj_idx))

        # Update node IDs in subgraph
        if len(discovered_subjects) > 0:
            subqgraph["nodes"][next_edge["subject"]]["ids"] = list(discovered_subjects)
        if len(discovered_objects) > 0:
            subqgraph["nodes"][next_edge["object"]]["ids"] = list(discovered_objects)

        # Remove processed edge
        subqgraph["edges"].pop(next_edge_id)

        # Remove orphaned nodes
        remove_orphaned(subqgraph)

        if verbose:
            print(f"  Remaining edges: {len(subqgraph['edges'])}")

    # Reconstruct complete paths from edge results
    if verbose:
        print(f"\nReconstructing complete paths...")

    path_data = reconstruct_paths(
        graph, query_graph, edge_results, original_edges, verbose,
        edge_inverse_preds=edge_inverse_preds
    )

    num_paths = len(path_data) if path_data is not None else 0

    if verbose:
        print(f"Found {num_paths:,} complete paths")

    t_post_start = time.perf_counter()

    response = {
        "message": {
            "query_graph": original_query_graph,
            "knowledge_graph": {
                "nodes": {},
                "edges": {},
            },
            "results": [],
            "auxiliary_graphs": {},
        }
    }

    if num_paths == 0:
        t_built = time.perf_counter()
        if verbose:
            print(f"  Post-processing total: {t_built - t_post_start:.2f}s")
    else:
        _build_response(
            graph, response, path_data, query_graph, num_paths,
            t_post_start, gc_monitor, verbose,
        )

    # GC summary is printed after GC is re-enabled in the caller's finally block.
    # The monitor is still accumulating events until stop() is called there.
    gc_summary = gc_monitor.summary()
    if verbose and gc_summary and gc_summary["total_time"] > 0.1:
        print(f"  [GC Summary] {gc_summary['total_collections']} collections, "
              f"{gc_summary['total_time']:.2f}s total, "
              f"{gc_summary['total_collected']} objects collected")

    return response


def _build_response(graph, response, path_data, query_graph, num_paths,
                    t_post_start, gc_monitor, verbose):
    """Build the TRAPI response from path data."""
    # Extract arrays and metadata from PathArrays for efficient access
    pa_nodes = path_data.paths_nodes
    pa_preds = path_data.paths_preds
    pa_via_inv = path_data.paths_via_inverse
    pa_fwd_eidx = path_data.paths_fwd_edge_idx
    node_cache = path_data.node_cache
    idx_to_predicate = path_data.idx_to_predicate
    qnode_to_col = path_data.qnode_to_col
    qedge_to_col = path_data.qedge_to_col
    col_to_qnode = path_data.col_to_qnode
    col_to_qedge = path_data.col_to_qedge
    pa_num_node_cols = path_data.num_node_cols
    pa_num_edges = path_data.num_edges
    lightweight = path_data.lightweight

    # Pre-compute subclass metadata for result building
    superclass_qnodes = {
        qnode_id for qnode_id, qnode in query_graph["nodes"].items()
        if qnode.get("_superclass")
    }
    subclass_qedges = {
        qedge_id for qedge_id, qedge in query_graph["edges"].items()
        if qedge.get("_subclass")
    }
    qnode_to_superclass = {}
    qedge_attached_subclass = defaultdict(list)
    for qedge_id, qedge in query_graph["edges"].items():
        if qedge.get("_subclass"):
            child_qnode = qedge["subject"]
            parent_qnode = qedge["object"]
            qnode_to_superclass[child_qnode] = parent_qnode
    for qedge_id, qedge in query_graph["edges"].items():
        if qedge.get("_subclass"):
            continue
        subj = qedge["subject"]
        obj = qedge["object"]
        if subj in qnode_to_superclass:
            qedge_attached_subclass[qedge_id].append(
                ("subject", f"{subj}_subclass_edge", qnode_to_superclass[subj])
            )
        if obj in qnode_to_superclass:
            qedge_attached_subclass[qedge_id].append(
                ("object", f"{obj}_subclass_edge", qnode_to_superclass[obj])
            )

    # Group paths by unique node binding combinations using numpy arrays.
    # Stores path *indices* (ints) instead of enriched dicts (~15 GB savings
    # for 5M paths).
    node_binding_groups = defaultdict(list)

    for path_idx in range(num_paths):
        key_pairs = []
        for col in range(pa_num_node_cols):
            qnode_id = col_to_qnode[col]
            if qnode_id in superclass_qnodes:
                continue
            node_idx = int(pa_nodes[path_idx, col])
            bound_id = node_cache[node_idx]["id"]
            if qnode_id in qnode_to_superclass:
                sc_qnode = qnode_to_superclass[qnode_id]
                if sc_qnode in qnode_to_col:
                    sc_col = qnode_to_col[sc_qnode]
                    sc_node_idx = int(pa_nodes[path_idx, sc_col])
                    sc_id = node_cache[sc_node_idx]["id"]
                    if sc_id != bound_id:
                        bound_id = sc_id
            key_pairs.append((qnode_id, bound_id))
        node_key = tuple(sorted(key_pairs))
        node_binding_groups[node_key].append(path_idx)

    t_grouped = time.perf_counter()
    if verbose:
        print(f"  Grouped into {len(node_binding_groups):,} unique node paths ({t_grouped - t_post_start:.2f}s)")

    # GC is already disabled for the entire query (see top of lookup()).
    # Build results -- one per unique node binding combination.
    # Edge dicts are created only for unique edges (not per-path).
    for node_key, path_indices in node_binding_groups.items():
        first_idx = path_indices[0]

        result = {
            "node_bindings": {},
            "analyses": [{
                "resource_id": "infores:gandalf",
                "edge_bindings": {},
            }],
        }

        # Add node bindings -- skip superclass nodes, substitute IDs
        for col in range(pa_num_node_cols):
            qnode_id = col_to_qnode[col]
            if qnode_id in superclass_qnodes:
                continue

            node_idx = int(pa_nodes[first_idx, col])
            node = node_cache[node_idx]
            response["message"]["knowledge_graph"]["nodes"][node["id"]] = node

            bound_id = node["id"]
            if qnode_id in qnode_to_superclass:
                sc_qnode = qnode_to_superclass[qnode_id]
                if sc_qnode in qnode_to_col:
                    sc_col = qnode_to_col[sc_qnode]
                    sc_node_idx = int(pa_nodes[first_idx, sc_col])
                    sc_node = node_cache[sc_node_idx]
                    if sc_node["id"] != node["id"]:
                        bound_id = sc_node["id"]
                        response["message"]["knowledge_graph"]["nodes"][sc_node["id"]] = sc_node

            result["node_bindings"][qnode_id] = [
                {"id": bound_id, "attributes": []},
            ]

        # Aggregate edge bindings from all paths in group.
        # Edge dicts are created only for unique (subj, pred, obj,
        # qualifiers, sources) combinations -- not for every path.
        edge_bindings_by_qedge = defaultdict(list)
        edge_seen_keys = defaultdict(set)

        for path_idx in path_indices:
            for col in range(pa_num_edges):
                qedge_id = col_to_qedge[col]
                pred_idx = int(pa_preds[path_idx, col])
                predicate = idx_to_predicate[pred_idx]
                is_inverse = bool(pa_via_inv[path_idx, col])
                fwd_eidx = int(pa_fwd_eidx[path_idx, col])

                # Determine actual (stored) edge direction
                edge_def = query_graph["edges"][qedge_id]
                subj_col_e = qnode_to_col[edge_def["subject"]]
                obj_col_e = qnode_to_col[edge_def["object"]]
                query_subj_idx = int(pa_nodes[path_idx, subj_col_e])
                query_obj_idx = int(pa_nodes[path_idx, obj_col_e])

                if is_inverse:
                    actual_subj_idx, actual_obj_idx = query_obj_idx, query_subj_idx
                else:
                    actual_subj_idx, actual_obj_idx = query_subj_idx, query_obj_idx

                subj_id = node_cache[actual_subj_idx]["id"]
                obj_id = node_cache[actual_obj_idx]["id"]

                # Compute dedup key
                if lightweight or fwd_eidx < 0:
                    edge_key = (subj_id, predicate, obj_id, (), ())
                else:
                    quals = graph.edge_properties.get_qualifiers(fwd_eidx)
                    quals_key = tuple(
                        sorted(
                            (q.get("qualifier_type_id", ""), q.get("qualifier_value", ""))
                            for q in (quals or [])
                        )
                    )
                    sources = graph.edge_properties.get_sources(fwd_eidx)
                    sources_key = tuple(
                        sorted(
                            (s.get("resource_id", ""), s.get("resource_role", ""))
                            for s in (sources or [])
                        )
                    )
                    edge_key = (subj_id, predicate, obj_id, quals_key, sources_key)

                if edge_key not in edge_seen_keys[qedge_id]:
                    edge_seen_keys[qedge_id].add(edge_key)
                    # Build edge dict only for unique edges
                    if lightweight:
                        edge_props = {
                            "predicate": predicate,
                            "subject": subj_id,
                            "object": obj_id,
                        }
                    else:
                        if fwd_eidx < 0:
                            edge_props = {}
                        else:
                            edge_props = graph.get_edge_properties_by_index(fwd_eidx).copy()
                        edge_props["predicate"] = predicate
                        edge_props["subject"] = subj_id
                        edge_props["object"] = obj_id

                    if fwd_eidx >= 0:
                        orig_id = graph.get_edge_id(fwd_eidx)
                        if orig_id is not None:
                            edge_props["_edge_id"] = orig_id

                    edge_bindings_by_qedge[qedge_id].append(edge_props)

        # Add edges to knowledge graph and result bindings
        for edge_id, edges in edge_bindings_by_qedge.items():
            if edge_id in subclass_qedges:
                continue

            result["analyses"][0]["edge_bindings"][edge_id] = []

            for edge in edges:
                edge_kg_id = edge.pop("_edge_id", None) or str(uuid.uuid4())[:8]
                response["message"]["knowledge_graph"]["edges"][edge_kg_id] = edge

                attached = qedge_attached_subclass.get(edge_id, [])
                if attached:
                    subclass_edge_kg_ids = []
                    superclass_node_overrides = {}

                    for (which_end, sc_edge_id, sc_qnode_id) in attached:
                        sc_edges = edge_bindings_by_qedge.get(sc_edge_id, [])
                        for sc_edge in sc_edges:
                            if sc_edge["subject"] == sc_edge["object"]:
                                continue
                            sc_kg_id = sc_edge.get("_edge_id") or str(uuid.uuid4())[:8]
                            response["message"]["knowledge_graph"]["edges"][sc_kg_id] = sc_edge
                            subclass_edge_kg_ids.append(sc_kg_id)

                        # Get superclass node ID for endpoint override
                        if sc_qnode_id in qnode_to_col:
                            sc_col = qnode_to_col[sc_qnode_id]
                            sc_node_idx = int(pa_nodes[first_idx, sc_col])
                            superclass_node_overrides[which_end] = node_cache[sc_node_idx]["id"]

                    if subclass_edge_kg_ids:
                        composite_edge_ids = [edge_kg_id] + subclass_edge_kg_ids
                        composite_edge_id = "_".join(composite_edge_ids)
                        aux_graph_id = f"aux_{composite_edge_id}"

                        if aux_graph_id not in response["message"]["auxiliary_graphs"]:
                            response["message"]["auxiliary_graphs"][aux_graph_id] = {
                                "edges": composite_edge_ids,
                                "attributes": [],
                            }

                        if composite_edge_id not in response["message"]["knowledge_graph"]["edges"]:
                            inferred_edge = {
                                "subject": superclass_node_overrides.get("subject", edge["subject"]),
                                "predicate": edge["predicate"],
                                "object": superclass_node_overrides.get("object", edge["object"]),
                                "attributes": [
                                    {
                                        "attribute_type_id": "biolink:knowledge_level",
                                        "value": "logical_entailment",
                                    },
                                    {
                                        "attribute_type_id": "biolink:agent_type",
                                        "value": "automated_agent",
                                    },
                                    {
                                        "attribute_type_id": "biolink:support_graphs",
                                        "value": [aux_graph_id],
                                    },
                                ],
                                "sources": [
                                    {
                                        "resource_id": "infores:gandalf",
                                        "resource_role": "primary_knowledge_source",
                                    }
                                ],
                            }
                            response["message"]["knowledge_graph"]["edges"][composite_edge_id] = inferred_edge

                        result["analyses"][0]["edge_bindings"][edge_id].append(
                            {"id": composite_edge_id, "attributes": []}
                        )
                    else:
                        result["analyses"][0]["edge_bindings"][edge_id].append(
                            {"id": edge_kg_id, "attributes": []}
                        )
                else:
                    result["analyses"][0]["edge_bindings"][edge_id].append(
                        {"id": edge_kg_id, "attributes": []}
                    )

        response["message"]["results"].append(result)

    # Free path arrays now that results are built
    del path_data

    # Strip internal _edge_id markers from KG edges so they don't leak
    # into the TRAPI response.
    for edge in response["message"]["knowledge_graph"]["edges"].values():
        edge.pop("_edge_id", None)

    t_built = time.perf_counter()
    if verbose:
        print(f"  Built {len(response['message']['results']):,} results ({t_built - t_grouped:.2f}s)")
        print(f"  Post-processing total: {t_built - t_post_start:.2f}s")


def _rewrite_for_subclass(query_graph, subclass_depth=1):
    """Rewrite query graph to add subclass expansion for pinned nodes.

    For each pinned node (one with ``ids``), creates a synthetic superclass
    node that holds the original IDs and a variable-depth ``subclass_of``
    edge connecting the original node to the superclass.  This allows the
    search to match both the exact node and any of its subclass descendants.

    Mirrors the reasoner-transpiler ``match_query`` rewriting logic.

    Args:
        query_graph: The query graph dict (mutated in-place).
        subclass_depth: Maximum number of ``subclass_of`` hops to traverse.
    """
    nodes = query_graph["nodes"]
    edges = query_graph["edges"]

    # Nodes already involved in explicit subclass_of / superclass_of edges
    # should not be rewritten (user specified them intentionally).
    excluded: set[str] = set()
    for edge in edges.values():
        preds = edge.get("predicates", [])
        if "biolink:subclass_of" in preds or "biolink:superclass_of" in preds:
            excluded.add(edge["subject"])
            excluded.add(edge["object"])

    pinned_qnodes = [
        qnode_id
        for qnode_id, qnode in list(nodes.items())
        if qnode.get("ids") and qnode_id not in excluded
    ]

    for qnode_id in pinned_qnodes:
        original = nodes[qnode_id]

        superclass_id = f"{qnode_id}_superclass"
        nodes[superclass_id] = {
            "ids": original.pop("ids"),
            "_superclass": True,
        }
        # Move categories to superclass node if present
        if "categories" in original:
            nodes[superclass_id]["categories"] = original.pop("categories")

        subclass_edge_id = f"{qnode_id}_subclass_edge"
        edges[subclass_edge_id] = {
            "subject": qnode_id,
            "object": superclass_id,
            "predicates": ["biolink:subclass_of"],
            "_subclass": True,
            "_subclass_depth": subclass_depth,
        }
