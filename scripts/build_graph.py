#!/usr/bin/env python3
"""
CLI tool to build knowledge graphs from JSONL files.

Example:
    kg-build --edges data/edges.jsonl --output data/graph_mmap/
"""

import argparse
import logging
import sys
from contextlib import nullcontext
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gandalf import build_graph_from_jsonl
from gandalf.backends.qlever import build_qlever_backend_from_jsonl
from gandalf.kgx_input import resolved_kgx_input
from gandalf.logging_config import configure_logging

logger = logging.getLogger(__name__)


def _build_csr_backend(edges: Path, nodes: Path, output_dir: Path) -> None:
    graph = build_graph_from_jsonl(
        edge_jsonl_path=str(edges),
        node_jsonl_path=str(nodes),
    )
    try:
        logger.info("Saving graph to %s", output_dir)
        graph.save_mmap(str(output_dir))
    finally:
        graph.close()


def _build_both_backends(
    *,
    edges: Path,
    nodes: Path,
    output_root: Path,
    dataset_name: str | None,
    run_qlever_index: bool,
    qlever_binary: str,
    stxxl_memory: str,
) -> None:
    csr_output_dir = output_root / "csr"
    qlever_output_dir = output_root / "qlever"
    qlever_dataset_name = dataset_name or output_root.name or "gandalf"

    _build_csr_backend(edges, nodes, csr_output_dir)

    logger.info(
        "Building QLever artifacts in %s using shared CSR artifacts from %s",
        qlever_output_dir,
        csr_output_dir,
    )
    build_qlever_backend_from_jsonl(
        edge_jsonl_path=edges,
        node_jsonl_path=nodes,
        output_dir=qlever_output_dir,
        dataset_name=qlever_dataset_name,
        run_index=run_qlever_index,
        qlever_binary=qlever_binary,
        stxxl_memory=stxxl_memory,
        shared_artifact_dir=csr_output_dir,
    )


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
        "--input",
        type=Path,
        default=None,
        help="Path to a KGX directory or a KGX `.tar.zst` archive",
    )

    parser.add_argument(
        "--edges", type=Path, default=None, help="Path to edges JSONL file"
    )

    parser.add_argument(
        "--nodes", type=Path, default=None, help="Path to nodes JSONL file"
    )

    parser.add_argument(
        "--output", "-o", required=True, type=Path, help="Output directory for graph"
    )

    parser.add_argument(
        "--backend",
        choices=("csr", "qlever", "both"),
        default="csr",
        help="Execution backend artifact set to build. Default: csr",
    )

    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help=(
            "Dataset name used for QLever artifact names. "
            "Default: output directory name"
        ),
    )

    parser.add_argument(
        "--skip-qlever-index",
        action="store_true",
        help="In qlever/both mode, export RDF and metadata but skip `qlever index`.",
    )

    parser.add_argument(
        "--qlever-binary",
        type=str,
        default="qlever",
        help="QLever executable to use in qlever mode. Default: qlever",
    )

    parser.add_argument(
        "--qlever-stxxl-memory",
        type=str,
        default="32G",
        help="Value for `qlever index --stxxl-memory` in qlever mode. Default: 32G",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable debug logging"
    )

    args = parser.parse_args()

    configure_logging(logging.DEBUG if args.verbose else logging.INFO)

    if args.input is not None and (args.edges is not None or args.nodes is not None):
        parser.error("Use `--input` or `--edges`/`--nodes`, not both")

    if args.input is None and (args.edges is None or args.nodes is None):
        parser.error("Provide either `--input` or both `--edges` and `--nodes`")

    if args.input is None:
        if not args.edges.exists():
            logger.error("Edge file not found: %s", args.edges)
            sys.exit(1)

        if not args.nodes.exists():
            logger.error("Node file not found: %s", args.nodes)
            sys.exit(1)

        input_context = nullcontext((args.edges, args.nodes, None))
    else:
        input_context = resolved_kgx_input(args.input)

    # Create output directory if needed
    args.output.parent.mkdir(parents=True, exist_ok=True)

    try:
        with input_context as (edges_path, nodes_path, inferred_dataset_name):
            dataset_name = args.dataset_name or inferred_dataset_name

            # Build graph
            logger.info("Building %s backend from %s", args.backend, edges_path)
            logger.info("Loading nodes from %s", nodes_path)

            if args.backend == "csr":
                _build_csr_backend(edges_path, nodes_path, args.output)
            elif args.backend == "qlever":
                build_qlever_backend_from_jsonl(
                    edge_jsonl_path=edges_path,
                    node_jsonl_path=nodes_path,
                    output_dir=args.output,
                    dataset_name=dataset_name,
                    run_index=not args.skip_qlever_index,
                    qlever_binary=args.qlever_binary,
                    stxxl_memory=args.qlever_stxxl_memory,
                )
            else:
                _build_both_backends(
                    edges=edges_path,
                    nodes=nodes_path,
                    output_root=args.output,
                    dataset_name=dataset_name,
                    run_qlever_index=not args.skip_qlever_index,
                    qlever_binary=args.qlever_binary,
                    stxxl_memory=args.qlever_stxxl_memory,
                )

            logger.info("Artifacts built successfully!")

    except Exception as e:
        logger.error("Error building graph: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
