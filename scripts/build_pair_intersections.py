#!/usr/bin/env python3
"""CLI: derive pair-intersection counts from a PublicationsIndex and ingest them.

For every pair of graph nodes (a, b) that share at least one publication,
writes the shared-publication count into a sparse CSR store alongside
the graph.  Pair scoping and sharing semantics:

  * Pairs are unordered — we only store (a, b) with a < b.
  * Counts union across ``equivalent_identifiers`` — a PMID that mentions
    CHEBI:6801 and DRUGBANK:DB00331 counts once toward pairs involving
    the Metformin node, not twice.
  * Only pairs with count > 0 are stored.

Example:

  gandalf-build-pair-intersections \\
      --graph-dir data/graph_mmap/ \\
      --index data/publications_index.lmdb \\
      --nodes data/nodes.jsonl

Memory note: the default in-memory accumulator suits small-to-medium
graphs.  At full PubTator scale (hundreds of millions of non-zero pairs)
you will want a SQLite-backed or external-sort accumulator — the derive
layer accepts any ``PairCountAccumulator`` subclass.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from gandalf.logging_config import configure_logging
from gandalf.metadata.publications.derive import derive_and_ingest_pair_intersections

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Derive pairwise publication intersections from a "
            "PublicationsIndex and write them to a Gandalf graph."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--graph-dir",
        required=True,
        type=Path,
        help="Path to the built graph directory.",
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

    for label, p in (
        ("graph-dir", args.graph_dir),
        ("index", args.index),
        ("nodes", args.nodes),
    ):
        if not p.exists():
            logger.error("--%s path not found: %s", label, p)
            sys.exit(1)

    try:
        derive_and_ingest_pair_intersections(
            graph_dir=args.graph_dir,
            index_path=args.index,
            nodes_jsonl=args.nodes,
        )
    except Exception as e:
        logger.error("Pair intersection derivation failed: %s", e)
        sys.exit(2)


if __name__ == "__main__":
    main()
