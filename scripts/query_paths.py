#!/usr/bin/env python3
"""
CLI tool to query paths in knowledge graphs.

Example:
    gandalf-query --graph graph_mmap/ --start "CHEBI:45783" --end "MONDO:0004979"
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from gandalf import (
    CSRGraph,
    find_3hop_paths_filtered,
    find_3hop_paths_with_properties,
)
from gandalf.logging_config import configure_logging

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Query 3-hop paths in a knowledge graph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic query
  gandalf-query --graph graph_mmap/ --start "CHEBI:45783" --end "MONDO:0004979"

  # With property enrichment
  gandalf-query --graph graph_mmap/ --start "CHEBI:45783" --end "MONDO:0004979" \\
      --with-properties --limit 100

  # Filter by predicates
  gandalf-query --graph graph_mmap/ --start "CHEBI:45783" --end "MONDO:0004979" \\
      --allowed-predicates biolink:treats biolink:affects

  # Exclude predicates
  gandalf-query --graph graph_mmap/ --start "CHEBI:45783" --end "MONDO:0004979" \\
      --excluded-predicates biolink:subclass_of

  # Save results
  gandalf-query --graph graph_mmap/ --start "CHEBI:45783" --end "MONDO:0004979" \\
      --output results.json
        """,
    )

    parser.add_argument(
        "--graph", "-g", required=True, type=Path, help="Path to graph directory"
    )

    parser.add_argument("--start", "-s", required=True, help="Start node ID")

    parser.add_argument("--end", "-e", required=True, help="End node ID")

    parser.add_argument(
        "--output", "-o", type=Path, help="Output JSON file for results"
    )

    parser.add_argument(
        "--with-properties",
        action="store_true",
        help="Include node and edge properties in results",
    )

    parser.add_argument("--limit", type=int, help="Limit number of paths to return")

    parser.add_argument(
        "--allowed-predicates",
        nargs="+",
        help="Only include edges with these predicates",
    )

    parser.add_argument(
        "--excluded-predicates", nargs="+", help="Exclude edges with these predicates"
    )

    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress progress output"
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable debug logging"
    )

    args = parser.parse_args()

    if args.verbose:
        configure_logging(logging.DEBUG)
    elif args.quiet:
        configure_logging(logging.WARNING)
    else:
        configure_logging(logging.INFO)

    # Validate graph file
    if not args.graph.exists():
        logger.error("Graph file not found: %s", args.graph)
        sys.exit(1)

    # Load graph
    logger.info("Loading graph from %s", args.graph)

    try:
        graph = CSRGraph.load_mmap(str(args.graph))
    except Exception as e:
        logger.error("Error loading graph: %s", e)
        sys.exit(1)

    # Query paths
    logger.info("Finding paths: %s -> %s", args.start, args.end)

    start_time = time.time()

    try:
        # Check if filtering is needed
        if args.allowed_predicates or args.excluded_predicates:
            logger.info("Applying predicate filters...")

            paths = find_3hop_paths_filtered(
                graph,
                args.start,
                args.end,
                allowed_predicates=set(args.allowed_predicates)
                if args.allowed_predicates
                else None,
                excluded_predicates=set(args.excluded_predicates)
                if args.excluded_predicates
                else None,
            )
        else:
            # Standard search
            if args.with_properties:
                paths = find_3hop_paths_with_properties(
                    graph,
                    args.start,
                    args.end,
                    max_paths=args.limit,
                )
            else:
                # Get just IDs
                start_idx = graph.get_node_idx(args.start)
                end_idx = graph.get_node_idx(args.end)

                if start_idx is None or end_idx is None:
                    logger.error("Start or end node not found in graph")
                    sys.exit(1)

                from gandalf.search import _find_3hop_paths_directed_idx

                paths_idx = _find_3hop_paths_directed_idx(graph, start_idx, end_idx)

                # Convert to IDs
                paths = [[graph.get_node_id(idx) for idx in path] for path in paths_idx]

                if args.limit:
                    paths = paths[: args.limit]

        elapsed = time.time() - start_time

        # Report results
        logger.info("Found %s paths in %.1f seconds", f"{len(paths):,}", elapsed)

        # Output results
        if args.output:
            output_data = {
                "query": {
                    "start": args.start,
                    "end": args.end,
                    "filters": {
                        "allowed_predicates": args.allowed_predicates,
                        "excluded_predicates": args.excluded_predicates,
                    },
                },
                "num_paths": len(paths),
                "elapsed_seconds": elapsed,
                "paths": paths,
            }

            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(output_data, f, indent=2)

            logger.info("Results saved to %s", args.output)
        else:
            # Print to stdout
            if args.with_properties:
                print(json.dumps(paths, indent=2))
            else:
                # Simple format for ID-only paths
                for i, path in enumerate(paths[:20], 1):  # Show first 20
                    print(f"{i}. {' -> '.join([path['n0']['id'], path['n1']['id'], path['n2']['id'], path['n3']['id']])}")

                if len(paths) > 20:
                    print(f"... and {len(paths) - 20:,} more paths")

    except Exception as e:
        logger.error("Error querying paths: %s", e)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
