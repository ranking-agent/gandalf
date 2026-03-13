"""CLI utility to convert graph formats."""

import argparse
import sys
from pathlib import Path

from gandalf.graph import CSRGraph


def convert_pickle_to_mmap(pickle_path: str, output_dir: str) -> None:
    """
    Convert a pickle-format graph to memory-mapped format.

    Args:
        pickle_path: Path to existing .pkl graph file
        output_dir: Directory to save mmap-format graph
    """
    pickle_path = Path(pickle_path)
    output_dir = Path(output_dir)

    if not pickle_path.exists():
        print(f"Error: Pickle file not found: {pickle_path}")
        sys.exit(1)

    print(f"Loading pickle graph from {pickle_path}...")
    graph = CSRGraph.load(pickle_path)

    print(f"\nConverting to mmap format...")
    graph.save_mmap(output_dir)

    print(f"\nConversion complete!")
    print(f"  Source: {pickle_path}")
    print(f"  Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Gandalf graph between formats"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Convert pickle to mmap
    pickle_to_mmap = subparsers.add_parser(
        "pickle-to-mmap",
        help="Convert pickle graph to memory-mapped format"
    )
    pickle_to_mmap.add_argument(
        "pickle_path",
        help="Path to source pickle file (.pkl)"
    )
    pickle_to_mmap.add_argument(
        "output_dir",
        help="Directory for mmap output files"
    )

    args = parser.parse_args()

    if args.command == "pickle-to-mmap":
        convert_pickle_to_mmap(args.pickle_path, args.output_dir)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
