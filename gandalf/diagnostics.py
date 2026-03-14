"""General Diagnostic tools for different path types."""
import logging
from collections import Counter, defaultdict

import numpy as np

from gandalf.graph import CSRGraph

logger = logging.getLogger(__name__)


def diagnose_path_explosion(graph: CSRGraph, start_id, end_id):
    """
    Diagnose why so many paths exist between two nodes.
    """
    start_idx = graph.get_node_idx(start_id)
    end_idx = graph.get_node_idx(end_id)

    logger.info("=== PATH EXPLOSION DIAGNOSIS ===")
    logger.info("Start: %s", start_id)
    logger.info("End: %s", end_id)
    logger.info("")

    # 1. Check degrees
    start_deg = graph.degree(start_idx)
    end_deg = graph.degree(end_idx)

    logger.info("1. DEGREE ANALYSIS")
    logger.info("   Start node degree: %s", f"{start_deg:,}")
    logger.info("   End node degree: %s", f"{end_deg:,}")
    logger.info(
        "   Naive estimate (deg_start × avg_deg × deg_end): ~%s paths",
        f"{start_deg * 20 * end_deg:,}",
    )
    logger.info("")

    # 2. Analyze 1-hop neighborhoods
    start_neighbors = graph.neighbors(start_idx)
    end_neighbors = graph.neighbors(end_idx)

    logger.info("2. ONE-HOP NEIGHBORHOODS")
    logger.info("   Start has %s direct neighbors", f"{len(start_neighbors):,}")
    logger.info("   End has %s direct neighbors", f"{len(end_neighbors):,}")

    # Check for direct connection (2-hop would be even more explosive)
    if end_idx in start_neighbors:
        logger.info("   ⚠️  WARNING: Direct edge exists between start and end!")
    logger.info("")

    # 3. Analyze 2-hop reachability
    logger.info("3. TWO-HOP ANALYSIS FROM START")
    two_hop_nodes = set()
    hop1_to_hop2_count = defaultdict(int)

    for n1_idx in start_neighbors[: min(100, len(start_neighbors))]:  # Sample first 100
        n1_neighbors = graph.neighbors(n1_idx)
        hop1_to_hop2_count[n1_idx] = len(n1_neighbors)
        for n2_idx in n1_neighbors:
            two_hop_nodes.add(n2_idx)

    avg_2hop_fanout = (
        np.mean(list(hop1_to_hop2_count.values())) if hop1_to_hop2_count else 0
    )
    logger.info("   Nodes reachable in 2 hops (sampled): %s", f"{len(two_hop_nodes):,}")
    logger.info("   Average fanout at hop 2: %s", f"{avg_2hop_fanout:.1f}")
    logger.info(
        "   Estimated full 2-hop reachable: ~%s",
        f"{len(start_neighbors) * avg_2hop_fanout:,.0f}",
    )
    logger.info("")

    # 4. Check overlap between 2-hop from start and 1-hop from end
    logger.info("4. MIDDLE NODE OVERLAP")
    overlap_count = sum(1 for node in two_hop_nodes if node in end_neighbors)
    overlap_pct = (overlap_count / len(two_hop_nodes) * 100) if two_hop_nodes else 0
    logger.info("   Nodes that connect start (2-hop) to end (1-hop): %s", f"{overlap_count:,}")
    logger.info("   Overlap percentage: %s%%", f"{overlap_pct:.1f}")
    logger.info("")

    # 5. Path multiplicity analysis
    logger.info("5. PATH MULTIPLICITY")
    logger.info("   Analyzing how many ways to reach middle nodes...")

    # Count how many 1-hop nodes lead to each 2-hop node
    two_hop_incoming = defaultdict(int)
    for n1_idx in start_neighbors[: min(1000, len(start_neighbors))]:
        for n2_idx in graph.neighbors(n1_idx):
            two_hop_incoming[n2_idx] += 1

    multiplicities = []
    if two_hop_incoming:
        multiplicities = list(two_hop_incoming.values())
        logger.info("   Average ways to reach a 2-hop node: %s", f"{np.mean(multiplicities):.1f}")
        logger.info("   Max ways to reach a 2-hop node: %s", np.max(multiplicities))
        logger.info("   Median: %s", f"{np.median(multiplicities):.1f}")

        # Show distribution
        mult_dist = Counter(multiplicities)
        logger.info("   Distribution of multiplicities:")
        for mult in sorted(mult_dist.keys())[:10]:
            logger.info("      %s paths to node: %s nodes", mult, f"{mult_dist[mult]:,}")
    logger.info("")

    # 6. Compute actual path count with formula
    logger.info("6. ACTUAL PATH COUNT CALCULATION")
    actual_paths = compute_path_count_fast(graph, start_idx, end_idx)
    logger.info("   Actual 3-hop paths: %s", f"{actual_paths:,}")
    logger.info("")

    # 7. Find heaviest contributors
    logger.info("7. HEAVIEST MIDDLE NODES (Top 10)")
    middle_node_contributions = defaultdict(int)

    # For each potential middle node (2-hop from start, 1-hop from end)
    end_neighbors_set = set(end_neighbors)

    for n1_idx in start_neighbors[: min(1000, len(start_neighbors))]:
        for n2_idx in graph.neighbors(n1_idx):
            if n2_idx in end_neighbors_set:
                middle_node_contributions[n2_idx] += 1

    top_contributors = sorted(
        middle_node_contributions.items(), key=lambda x: x[1], reverse=True
    )[:10]

    for rank, (node_idx, count) in enumerate(top_contributors, 1):
        node_id = graph.get_node_id(node_idx)
        node_deg = graph.degree(node_idx)
        logger.info("   %s. %s", rank, node_id)
        logger.info("      Contributes %s paths (degree: %s)", f"{count:,}", f"{node_deg:,}")
    logger.info("")

    # 8. Recommendations
    logger.info("8. RECOMMENDATIONS")
    if start_deg > 1000 or end_deg > 1000:
        logger.info("   ⚠️  High-degree nodes detected!")
        logger.info("      Consider filtering edges by predicate type before querying")

    if avg_2hop_fanout > 50:
        logger.info("   ⚠️  High fanout at hop 2!")
        logger.info("      Many hub nodes in the path - consider constraining middle nodes")

    if overlap_pct > 50:
        logger.info("   ℹ️  High overlap suggests these nodes are well-connected")
        logger.info("      Paths may be redundant - consider deduplication by middle nodes")

    return {
        "start_degree": start_deg,
        "end_degree": end_deg,
        "two_hop_reachable": len(two_hop_nodes),
        "middle_node_overlap": overlap_count,
        "avg_multiplicity": np.mean(multiplicities) if two_hop_incoming else 0,
        "total_paths": actual_paths,
        "top_contributors": top_contributors,
    }


def compute_path_count_fast(graph: CSRGraph, start_idx, end_idx):
    """
    Compute exact 3-hop path count without enumerating them all.
    Much faster than generating all paths.
    """
    start_neighbors = graph.neighbors(start_idx)
    end_neighbors_set = set(graph.neighbors(end_idx))

    path_count = 0
    for n1_idx in start_neighbors:
        if n1_idx == end_idx:
            continue
        for n2_idx in graph.neighbors(n1_idx):
            if n2_idx == start_idx:
                continue
            if n2_idx in end_neighbors_set:
                path_count += 1

    return path_count


def analyze_node_types(graph: CSRGraph, start_id, end_id, max_sample=1000):
    """
    Analyze what types of nodes appear in the paths.
    Helps understand if certain node types are causing explosion.
    """
    start_idx = graph.get_node_idx(start_id)
    end_idx = graph.get_node_idx(end_id)

    logger.info("=== NODE TYPE ANALYSIS ===")

    # Sample some paths
    start_neighbors = graph.neighbors(start_idx)
    end_neighbors_set = set(graph.neighbors(end_idx))

    hop1_categories = []
    hop2_categories = []

    sampled = 0
    for n1_idx in start_neighbors:
        if sampled >= max_sample:
            break
        n1_cat = graph.get_node_property(n1_idx, "category", "")

        for n2_idx in graph.neighbors(n1_idx):
            if sampled >= max_sample:
                break
            if n2_idx == start_idx or n2_idx not in end_neighbors_set:
                continue

            n2_cat = graph.get_node_property(n2_idx, "category", "")

            hop1_categories.append(n1_cat)
            hop2_categories.append(n2_cat)
            sampled += 1

    logger.info("Sampled %s paths", f"{sampled:,}")
    logger.info("")

    logger.info("Top categories at HOP 1 (from start):")
    for cat, count in Counter(hop1_categories).most_common(10):
        logger.info("  %s: %s", cat, f"{count:,}")
    logger.info("")

    logger.info("Top categories at HOP 2 (to end):")
    for cat, count in Counter(hop2_categories).most_common(10):
        logger.info("  %s: %s", cat, f"{count:,}")
    logger.info("")


def analyze_predicates(graph: CSRGraph, start_id, end_id, max_sample=1000):
    """
    Analyze what predicates appear in the paths.
    Helps identify if certain relationship types dominate.
    """
    start_idx = graph.get_node_idx(start_id)
    end_idx = graph.get_node_idx(end_id)

    logger.info("=== PREDICATE ANALYSIS ===")

    start_neighbors = graph.neighbors(start_idx)
    end_neighbors_set = set(graph.neighbors(end_idx))

    edge1_predicates = []
    edge2_predicates = []
    edge3_predicates = []

    sampled = 0
    for n1_idx in start_neighbors:
        if sampled >= max_sample:
            break

        pred1 = graph.get_edge_property(start_idx, n1_idx, "predicate")

        for n2_idx in graph.neighbors(n1_idx):
            if sampled >= max_sample:
                break
            if n2_idx == start_idx or n2_idx not in end_neighbors_set:
                continue

            pred2 = graph.get_edge_property(n1_idx, n2_idx, "predicate")
            pred3 = graph.get_edge_property(n2_idx, end_idx, "predicate")

            edge1_predicates.append(pred1)
            edge2_predicates.append(pred2)
            edge3_predicates.append(pred3)
            sampled += 1

    logger.info("Sampled %s paths", f"{sampled:,}")
    logger.info("")

    logger.info("Top predicates at EDGE 1 (start → hop1):")
    for pred, count in Counter(edge1_predicates).most_common(10):
        logger.info("  %s: %s", pred, f"{count:,}")
    logger.info("")

    logger.info("Top predicates at EDGE 2 (hop1 → hop2):")
    for pred, count in Counter(edge2_predicates).most_common(10):
        logger.info("  %s: %s", pred, f"{count:,}")
    logger.info("")

    logger.info("Top predicates at EDGE 3 (hop2 → end):")
    for pred, count in Counter(edge3_predicates).most_common(10):
        logger.info("  %s: %s", pred, f"{count:,}")
    logger.info("")

    # Predicate combinations
    logger.info("Top predicate PATTERNS (edge1 → edge2 → edge3):")
    patterns = [
        f"{e1} → {e2} → {e3}"
        for e1, e2, e3 in zip(edge1_predicates, edge2_predicates, edge3_predicates)
    ]
    for pattern, count in Counter(patterns).most_common(10):
        logger.info("  %s: %s", pattern, f"{count:,}")
