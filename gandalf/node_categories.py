"""Each node's categories, as an index into the distinct category lists.

Nodes share a few hundred distinct category lists, so one small integer per
node says which list it has.  That array is memory-mapped (``NODE_SETS_FILE``,
shared by every worker through the page cache, like the CSR arrays) and the
lists themselves ride in ``metadata.pkl``.

A traversal uses it to test a whole step's candidate nodes against a QNode's
categories with one array lookup.  Without it, each candidate node's record
is read from the node store and unpacked -- name, attributes and all -- just
to look at its categories.
"""

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

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
        interned: dict[tuple, int] = {}
        set_of_node = np.zeros(num_nodes, dtype=np.int32)
        for node_idx in range(num_nodes):
            categories = tuple(node_properties.get(node_idx, {}).get("categories", ()))
            set_of_node[node_idx] = interned.setdefault(categories, len(interned))
        # The narrowest type that holds every index: the array is one entry
        # per node.
        dtype = np.uint16 if len(interned) <= np.iinfo(np.uint16).max else np.int32
        return cls(set_of_node.astype(dtype), list(interned))

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
