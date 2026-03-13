#!/usr/bin/env python3
"""
CLI tool to build knowledge graphs from JSONL files.

Example:
    kg-build --edges data/edges.jsonl --output data/graph.pkl
"""

import argparse
import sys
from pathlib import Path

from gandalf import build_graph_from_jsonl


def main():
    parser = argparse.ArgumentParser(
        description="Build a knowledge graph from JSONL files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  kg-build --edges edges.jsonl --nodes nodes.jsonl --output graph.pkl
        """,
    )

    parser.add_argument(
        "--edges", required=True, type=Path, help="Path to edges JSONL file"
    )

    parser.add_argument(
        "--nodes", required=True, type=Path, help="Path to nodes JSONL file"
    )

    parser.add_argument(
        "--output", "-o", required=True, type=Path, help="Output path for pickled graph"
    )

    args = parser.parse_args()

    # Validate input files
    if not args.edges.exists():
        print(f"Error: Edge file not found: {args.edges}", file=sys.stderr)
        sys.exit(1)

    if not args.nodes.exists():
        print(f"Error: Node file not found: {args.nodes}", file=sys.stderr)
        sys.exit(1)

    # Create output directory if needed
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Build graph
    print(f"Building graph from {args.edges}")
    print(f"Loading nodes from {args.nodes}")

    try:
        graph = build_graph_from_jsonl(
            edge_jsonl_path=str(args.edges),
            node_jsonl_path=str(args.nodes),
        )

        # Save graph
        print(f"\nSaving graph to {args.output}")
        graph.save_mmap(str(args.output))

        print("\n✓ Graph built successfully!")
        # print(f"  Nodes: {graph.num_nodes:,}")
        # print(f"  Edges: {len(graph.edge_dst):,}")

    except Exception as e:
        print(f"Error building graph: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
