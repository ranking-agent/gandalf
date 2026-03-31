#!/usr/bin/env python3
"""
CLI tool to build knowledge graphs from JSONL files.

Example:
    kg-build --edges data/edges.jsonl --output data/graph_mmap/
"""

import argparse
import logging
import sys
from pathlib import Path

from gandalf import build_graph_from_jsonl
from gandalf.logging_config import configure_logging

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Build a knowledge graph from JSONL files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  kg-build --edges edges.jsonl --nodes nodes.jsonl --output graph_mmap/
        """,
    )

    parser.add_argument(
        "--edges", required=True, type=Path, help="Path to edges JSONL file"
    )

    parser.add_argument(
        "--nodes", required=True, type=Path, help="Path to nodes JSONL file"
    )

    parser.add_argument(
        "--output", "-o", required=True, type=Path, help="Output directory for graph"
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable debug logging"
    )

    args = parser.parse_args()

    configure_logging(logging.DEBUG if args.verbose else logging.INFO)

    # Validate input files
    if not args.edges.exists():
        logger.error("Edge file not found: %s", args.edges)
        sys.exit(1)

    if not args.nodes.exists():
        logger.error("Node file not found: %s", args.nodes)
        sys.exit(1)

    # Create output directory if needed
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Build graph
    logger.info("Building graph from %s", args.edges)
    logger.info("Loading nodes from %s", args.nodes)

    try:
        graph = build_graph_from_jsonl(
            edge_jsonl_path=str(args.edges),
            node_jsonl_path=str(args.nodes),
        )

        # Save graph
        logger.info("Saving graph to %s", args.output)
        graph.save_mmap(str(args.output))

        logger.info("Graph built successfully!")

    except Exception as e:
        logger.error("Error building graph: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
