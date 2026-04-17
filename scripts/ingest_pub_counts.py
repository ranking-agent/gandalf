#!/usr/bin/env python3
"""CLI: ingest per-node and/or per-edge publication counts into a graph directory.

The inputs are JSONL files produced outside Gandalf.  See
``gandalf.metadata.pub_counts`` for the full contract; summary:

  Per-node:  {"node_id": "<id from nodes.jsonl>", "count": <uint32>}
  Per-edge:  {"edge_id": "<id from edges.jsonl>", "count": <uint32>}

Any unknown ID, duplicate ID, or uncovered node/edge raises — failures are
loud by design while the pipeline is being stood up.

Example:

  gandalf-ingest-pub-counts \\
      --graph-dir data/graph_mmap/ \\
      --node-counts counts/nodes.jsonl \\
      --edge-counts counts/edges.jsonl
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from gandalf.logging_config import configure_logging
from gandalf.metadata.pub_counts import (
    PubCountIngestError,
    ingest_edge_pub_counts,
    ingest_node_pub_counts,
)

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest pre-computed publication counts into a Gandalf graph directory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--graph-dir",
        required=True,
        type=Path,
        help="Path to the built graph (directory written by gandalf-build).",
    )
    parser.add_argument(
        "--node-counts",
        type=Path,
        default=None,
        help="Optional JSONL file with per-node counts.",
    )
    parser.add_argument(
        "--edge-counts",
        type=Path,
        default=None,
        help="Optional JSONL file with per-edge counts.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable debug logging."
    )

    args = parser.parse_args()
    configure_logging(logging.DEBUG if args.verbose else logging.INFO)

    if args.node_counts is None and args.edge_counts is None:
        parser.error("At least one of --node-counts or --edge-counts must be provided.")

    if not args.graph_dir.exists():
        logger.error("Graph directory not found: %s", args.graph_dir)
        sys.exit(1)

    try:
        if args.node_counts is not None:
            if not args.node_counts.exists():
                logger.error("Node counts file not found: %s", args.node_counts)
                sys.exit(1)
            ingest_node_pub_counts(args.graph_dir, args.node_counts)

        if args.edge_counts is not None:
            if not args.edge_counts.exists():
                logger.error("Edge counts file not found: %s", args.edge_counts)
                sys.exit(1)
            ingest_edge_pub_counts(args.graph_dir, args.edge_counts)
    except PubCountIngestError as e:
        logger.error("Pub count ingest failed: %s", e)
        sys.exit(2)


if __name__ == "__main__":
    main()
