#!/usr/bin/env python3
"""CLI: derive per-node publication counts from a PublicationsIndex and ingest them.

Reads a ``PublicationsIndex`` LMDB (built by
``gandalf-build-publications-index``) plus the graph's ``nodes.jsonl``,
unions each node's PMIDs across its ``equivalent_identifiers``, and
writes ``node_pub_counts.npy`` + manifest entry alongside the graph.

Example:

  gandalf-build-pub-counts \\
      --graph-dir data/graph_mmap/ \\
      --index data/publications_index.lmdb \\
      --nodes data/nodes.jsonl
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from gandalf.logging_config import configure_logging
from gandalf.metadata.pub_counts import PubCountIngestError
from gandalf.metadata.publications.derive import derive_and_ingest_node_pub_counts

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Derive per-node publication counts from a PublicationsIndex "
            "and write them into a Gandalf graph."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--graph-dir", required=True, type=Path, help="Path to the built graph directory."
    )
    parser.add_argument(
        "--index",
        required=True,
        type=Path,
        help="Path to the PublicationsIndex LMDB directory.",
    )
    parser.add_argument(
        "--nodes",
        required=True,
        type=Path,
        help=(
            "Path to the same nodes.jsonl used when building the graph — "
            "needed for equivalent_identifiers."
        ),
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable debug logging."
    )

    args = parser.parse_args()
    configure_logging(logging.DEBUG if args.verbose else logging.INFO)

    for label, p in (("graph-dir", args.graph_dir), ("index", args.index), ("nodes", args.nodes)):
        if not p.exists():
            logger.error("--%s path not found: %s", label, p)
            sys.exit(1)

    try:
        derive_and_ingest_node_pub_counts(
            graph_dir=args.graph_dir,
            index_path=args.index,
            nodes_jsonl=args.nodes,
        )
    except PubCountIngestError as e:
        logger.error("Pub count derivation failed: %s", e)
        sys.exit(2)


if __name__ == "__main__":
    main()
