#!/usr/bin/env python3
"""CLI: build a PublicationsIndex LMDB from a PubTator3 bulk TSV.

The index is the source-agnostic store used by downstream derivations
(node publication counts, future pair intersections).  PubTator is the
first adapter; adding another source is a new parser module under
``gandalf/metadata/publications/`` — this script only sees the tuple
stream, so it doesn't need to change.

Example:

  gandalf-build-publications-index \\
      --pubtator data/bioconcepts2pubtator3central.gz \\
      --nodes data/nodes.jsonl \\
      --output data/publications_index.lmdb

``--nodes`` is optional; when supplied, the index is filtered to CURIEs
that appear as a graph node's id or an equivalent_identifier.  Without
it the index stores every PubTator CURIE (larger, but reusable across
graphs).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from gandalf.logging_config import configure_logging
from gandalf.metadata.publications.derive import collect_tracked_curies
from gandalf.metadata.publications.index import PublicationsIndex
from gandalf.metadata.publications.pubtator import iter_pubtator_annotations

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a PublicationsIndex LMDB from PubTator3 bulk data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pubtator",
        required=True,
        type=Path,
        help="Path to a PubTator annotations TSV (.gz supported).",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        type=Path,
        help="Directory where the LMDB environment will be written.",
    )
    parser.add_argument(
        "--nodes",
        type=Path,
        default=None,
        help=(
            "Optional nodes.jsonl used to filter the index to graph-relevant "
            "CURIEs (union of id + equivalent_identifiers)."
        ),
    )
    parser.add_argument(
        "--entity-types",
        type=str,
        default=None,
        help=(
            "Optional comma-separated list of PubTator entity types to keep "
            "(e.g. 'Gene,Chemical,Disease').  Case-insensitive."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace any existing index at the output path.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable debug logging."
    )

    args = parser.parse_args()
    configure_logging(logging.DEBUG if args.verbose else logging.INFO)

    if not args.pubtator.exists():
        logger.error("PubTator file not found: %s", args.pubtator)
        sys.exit(1)

    tracked = None
    if args.nodes is not None:
        if not args.nodes.exists():
            logger.error("Nodes file not found: %s", args.nodes)
            sys.exit(1)
        logger.info("Collecting tracked CURIEs from %s...", args.nodes)
        tracked = collect_tracked_curies(args.nodes)
        logger.info("  tracked %s unique CURIEs", f"{len(tracked):,}")

    entity_types = None
    if args.entity_types:
        entity_types = {t.strip() for t in args.entity_types.split(",") if t.strip()}

    PublicationsIndex.build(
        args.output,
        iter_pubtator_annotations(
            args.pubtator, tracked_curies=tracked, entity_types=entity_types
        ),
        tracked_curies=None,  # filtering already applied by the parser
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
