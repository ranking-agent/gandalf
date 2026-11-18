"""Each node's categories, as an index into the distinct category lists.

Nodes share a few hundred distinct category lists, so one small integer per
node says which list it has.  That array is memory-mapped (``NODE_SETS_FILE``,
shared by every worker through the page cache, like the CSR arrays) and the
lists themselves ride in ``metadata.pkl``.

``gandalf-build`` writes it.  A graph built before it existed can be given
one without a rebuild::

    python -m gandalf.node_categories /data/graph_mmap

A traversal uses it to test a whole step's candidate nodes against a QNode's
categories with one array lookup.  Without it, each candidate node's record
is read from the node store and unpacked -- name, attributes and all -- just
to look at its categories.
"""

import argparse
import pickle
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

from gandalf.node_store import NodeStore

#: The per-node array, in a graph directory.
NODE_SETS_FILE = "node_category_sets.npy"

#: The distinct category lists, in ``metadata.pkl``.
METADATA_KEY = "node_category_sets"


class NodeCategoryIndex:
    """Which of the distinct category lists each node has.

    Attributes:
        set_of_node: For each node index, an index into ``category_sets``.
        category_sets: The distinct category lists, as tuples.

    Examples:
        >>> index = NodeCategoryIndex.build(
        ...     4,
        ...     {
        ...         0: {"categories": ["biolink:Gene", "biolink:NamedThing"]},
        ...         1: {"categories": ["biolink:Disease"]},
        ...         2: {"categories": ["biolink:Gene", "biolink:NamedThing"]},
        ...     },
        ... )
        >>> index.category_sets
        [('biolink:Gene', 'biolink:NamedThing'), ('biolink:Disease',), ()]
        >>> index.any_of(np.array([0, 1, 2, 3, 1]), ["biolink:Gene"]).tolist()
        [True, False, True, False, False]
        >>> index.any_of(np.array([1, 3]), ["biolink:Disease", "biolink:Gene"]).tolist()
        [True, False]
    """

    __slots__ = ("set_of_node", "category_sets")

    def __init__(self, set_of_node: np.ndarray, category_sets: Sequence[Sequence[str]]):
        self.set_of_node = set_of_node
        self.category_sets = [tuple(categories) for categories in category_sets]

    @classmethod
    def build(
        cls, num_nodes: int, node_properties: Mapping[int, dict]
    ) -> "NodeCategoryIndex":
        """Index the ``categories`` of every node's stored properties.

        Args:
            num_nodes: How many nodes the graph has.
            node_properties: ``node_idx -> properties``, as written to the
                node store.  A node without properties has no categories.
        """
        return cls.from_items(num_nodes, node_properties.items())

    @classmethod
    def from_items(
        cls, num_nodes: int, items: Iterable[tuple[int, dict]]
    ) -> "NodeCategoryIndex":
        """:meth:`build` from ``(node_idx, properties)`` pairs, read one at a
        time, so a node store can be indexed without holding its records."""
        interned: dict[tuple, int] = {}
        set_of_node = np.full(num_nodes, -1, dtype=np.int32)
        for node_idx, properties in items:
            categories = tuple(properties.get("categories", ()))
            set_of_node[node_idx] = interned.setdefault(categories, len(interned))
        absent = set_of_node < 0
        if absent.any():
            set_of_node[absent] = interned.setdefault((), len(interned))
        # Number the lists by the first node that has each, so the index does
        # not depend on the order the nodes were read in.
        codes, first_node = np.unique(set_of_node, return_index=True)
        new_code = np.empty(len(interned), dtype=np.int32)
        new_code[codes[np.argsort(first_node)]] = np.arange(len(codes))
        category_sets = sorted(
            interned, key=lambda categories: new_code[interned[categories]]
        )
        # The narrowest type that holds every index: the array is one entry
        # per node.
        dtype = np.uint16 if len(interned) <= np.iinfo(np.uint16).max else np.int32
        return cls(new_code[set_of_node].astype(dtype), category_sets)

    def any_of(self, node_idxes: np.ndarray, categories: Iterable[str]) -> np.ndarray:
        """Whether each node has at least one of *categories*.

        Returns:
            A boolean array parallel to *node_idxes*.
        """
        wanted = set(categories)
        passes = np.fromiter(
            (
                not wanted.isdisjoint(category_set)
                for category_set in self.category_sets
            ),
            dtype=np.bool_,
            count=len(self.category_sets),
        )
        mask: np.ndarray = passes[self.set_of_node[node_idxes]]
        return mask

    def save(self, directory: Path) -> None:
        """Write the per-node array into a graph directory."""
        np.save(directory / NODE_SETS_FILE, self.set_of_node)

    def metadata(self) -> dict:
        """What ``metadata.pkl`` must carry alongside the saved array."""
        return {METADATA_KEY: self.category_sets}


def add_to_graph(directory: Path) -> NodeCategoryIndex:
    """Give an already-built graph its category index, without a rebuild.

    Reads every node's categories from the graph's node store and writes
    ``NODE_SETS_FILE`` and the ``metadata.pkl`` entry, as ``save_mmap`` does
    for a new graph.  Running it again rewrites both.  Processes serving the
    graph pick the index up when they next load it.

    Args:
        directory: A graph directory written by ``gandalf-build``.

    Returns:
        The index written.
    """
    directory = Path(directory)
    with open(directory / "metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    with NodeStore(directory / "node_store.lmdb", readonly=True) as store:
        index = NodeCategoryIndex.from_items(
            metadata["num_nodes"], store.iter_properties()
        )
    index.save(directory)
    metadata.update(index.metadata())
    with open(directory / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
    return index


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command line for :func:`add_to_graph`."""
    parser = argparse.ArgumentParser(
        description="Add the node category index to an already-built graph."
    )
    parser.add_argument("graph", type=Path, help="graph directory (gandalf-build)")
    args = parser.parse_args(argv)
    index = add_to_graph(args.graph)
    print(
        f"Wrote {args.graph / NODE_SETS_FILE}: {len(index.set_of_node):,} nodes, "
        f"{len(index.category_sets):,} distinct category lists, "
        f"{index.set_of_node.nbytes / 2**20:.1f} MB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
