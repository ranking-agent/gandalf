#!/usr/bin/env python3
"""
CLI tool to build a knowledge graph from already-normalized documents in MongoDB.

The MongoDB collections must contain node and edge documents already in
gandalf's normalized form (an upstream pipeline performs normalization before
loading them into MongoDB). Requires the ``mongo`` extra (``pip install
gandalf[mongo]``).

Example:
    gandalf-build-mongo \
        --mongo-uri mongodb://localhost:27017 \
        --db kg \
        --nodes-collection nodes \
        --edges-collection edges \
        --output data/graph_mmap/
"""

import argparse
import logging
import sys
from pathlib import Path

from gandalf import build_graph_from_mongo
from gandalf.logging_config import configure_logging

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Build a knowledge graph from normalized MongoDB documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  gandalf-build-mongo --mongo-uri mongodb://localhost:27017 \\
      --db kg --nodes-collection nodes --edges-collection edges \\
      --output graph_mmap/
        """,
    )

    parser.add_argument("--mongo-uri", required=True, help="MongoDB connection URI")

    parser.add_argument("--db", required=True, help="MongoDB database name")

    parser.add_argument(
        "--nodes-collection",
        required=True,
        help="Collection of normalized node documents",
    )

    parser.add_argument(
        "--edges-collection",
        required=True,
        help="Collection of normalized edge documents",
    )

    parser.add_argument(
        "--output", "-o", required=True, type=Path, help="Output directory for graph"
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable debug logging"
    )

    args = parser.parse_args()

    configure_logging(logging.DEBUG if args.verbose else logging.INFO)

    # Create output directory if needed
    args.output.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Building graph from MongoDB %s (db=%s)", args.mongo_uri, args.db)
    logger.info(
        "Collections: nodes=%s, edges=%s",
        args.nodes_collection,
        args.edges_collection,
    )

    try:
        graph = build_graph_from_mongo(
            mongo_uri=args.mongo_uri,
            db=args.db,
            nodes_collection=args.nodes_collection,
            edges_collection=args.edges_collection,
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
