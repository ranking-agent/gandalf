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
from gandalf.node_annotations import DEFAULT_ANNOTATION_BATCH_SIZE

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Build a knowledge graph from JSONL files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  kg-build --edges edges.jsonl --nodes nodes.jsonl --output graph_mmap/

  # Also fetch node annotations from the Translator Annotator service
  kg-build --edges edges.jsonl --nodes nodes.jsonl --output graph_mmap/ --annotate
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
        "--annotate",
        action="store_true",
        help=(
            "Annotate nodes via the Translator Annotator (biothings_annotator) "
            "and store the results as 'biothings_annotations' node attributes. "
            "Requires network access and `pip install -r requirements-annotate.txt`"
        ),
    )

    parser.add_argument(
        "--annotate-batch-size",
        type=int,
        default=DEFAULT_ANNOTATION_BATCH_SIZE,
        help="CURIEs per Annotator request (default: %(default)s)",
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
    if args.annotate:
        logger.info("Node annotation enabled (biothings_annotator)")

    try:
        graph = build_graph_from_jsonl(
            edge_jsonl_path=str(args.edges),
            node_jsonl_path=str(args.nodes),
            annotate_nodes=args.annotate,
            annotation_batch_size=args.annotate_batch_size,
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
