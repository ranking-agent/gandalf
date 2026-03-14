#!/usr/bin/env python3
"""
CLI tool to diagnose path explosion in knowledge graphs.

Example:
    kg-diagnose --graph graph_mmap/ --start "CHEBI:45783" --end "MONDO:0004979"
"""

import argparse
import sys
from pathlib import Path

from gandalf import (
    CSRGraph,
    analyze_node_types,
    analyze_predicates,
    diagnose_path_explosion,
)


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose path explosion in knowledge graphs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full diagnosis
  gandalf-diagnose --graph graph_mmap/ --start "CHEBI:45783" --end "MONDO:0004979"

  # Just path count analysis
  gandalf-diagnose --graph graph_mmap/ --start "CHEBI:45783" --end "MONDO:0004979" \\
      --skip-node-types --skip-predicates

  # Focus on predicates
  gandalf-diagnose --graph graph_mmap/ --start "CHEBI:45783" --end "MONDO:0004979" \\
      --skip-node-types --sample 10000
        """,
    )

    parser.add_argument(
        "--graph", "-g", required=True, type=Path, help="Path to graph directory"
    )

    parser.add_argument("--start", "-s", required=True, help="Start node ID")

    parser.add_argument("--end", "-e", required=True, help="End node ID")

    parser.add_argument(
        "--sample",
        type=int,
        default=1000,
        help="Number of paths to sample for analysis (default: 1000)",
    )

    parser.add_argument(
        "--skip-node-types", action="store_true", help="Skip node type analysis"
    )

    parser.add_argument(
        "--skip-predicates", action="store_true", help="Skip predicate analysis"
    )

    args = parser.parse_args()

    # Validate graph file
    if not args.graph.exists():
        print(f"Error: Graph file not found: {args.graph}", file=sys.stderr)
        sys.exit(1)

    # Load graph
    print(f"Loading graph from {args.graph}\n")

    try:
        graph = CSRGraph.load_mmap(str(args.graph))
    except Exception as e:
        print(f"Error loading graph: {e}", file=sys.stderr)
        sys.exit(1)

    # Run diagnostics
    try:
        # Main diagnosis
        results = diagnose_path_explosion(graph, args.start, args.end)

        # Node type analysis
        if not args.skip_node_types:
            print()
            analyze_node_types(graph, args.start, args.end, max_sample=args.sample)

        # Predicate analysis
        if not args.skip_predicates:
            print()
            analyze_predicates(graph, args.start, args.end, max_sample=args.sample)

        # Summary recommendations
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)

        total_paths = results["total_paths"]

        if total_paths > 1_000_000:
            print("⚠️  SEVERE path explosion detected!")
            print(
                f"   {total_paths:,} paths is likely too many to process efficiently."
            )
            print()
            print("Recommended actions:")
            print("  1. Exclude ontology predicates (biolink:subclass_of)")
            print("  2. Filter by allowed predicates only")
            print("  3. Constrain intermediate node types")
            print()
            print("Example:")
            print("  gandalf-build --edges edges.jsonl --output graph_filtered/ \\")
            print("      --exclude-predicates biolink:subclass_of")

        elif total_paths > 10_000:
            print("⚠️  High path count detected")
            print(f"   {total_paths:,} paths may be manageable with filtering.")
            print()
            print("Consider:")
            print("  1. Filter by predicate types")
            print("  2. Apply business logic filters early")
            print("  3. Sample and rank before enrichment")

        else:
            print(f"✓ Reasonable path count: {total_paths:,}")
            print("  This should be fast to process with property enrichment.")

        print()

    except Exception as e:
        print(f"Error during diagnosis: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
