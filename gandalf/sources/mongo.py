"""MongoDB graph source (wrapper 2).

Reads node and edge documents that an upstream pipeline has *already* normalized
into gandalf's internal form (TRAPI-style ``sources``/``qualifiers``/
``attributes`` lists, node ``categories``). This source performs NO
transformation: it only drops the Mongo-internal ``_id`` and validates each
record against the normalized contract before yielding it.

Because nothing here triggers ``gandalf.normalize``, a pure-Mongo build never
loads BMT or fetches the Biolink schema. The source therefore trusts that the
upstream pipeline used the correct Biolink version for qualifier classification.

``pymongo`` is imported lazily inside ``__init__`` so the rest of gandalf (and
the jsonl path) never require it. Install it with the ``mongo`` extra.
"""

import logging
from typing import Iterator, Tuple

from gandalf.sources.base import (
    GraphSource,
    validate_normalized_edge,
    validate_normalized_node,
)

logger = logging.getLogger(__name__)

# Edge cursors must use a stable sort so Pass 1 (triples) and Pass 2 (full edges)
# visit edges in the same order. Sorting by the immutable ``_id`` guarantees this.
_EDGE_SORT_KEY = "_id"


class MongoSource(GraphSource):
    """A :class:`GraphSource` backed by a MongoDB database of normalized docs."""

    def __init__(
        self, *, uri: str, db: str, nodes_collection: str, edges_collection: str
    ):
        import pymongo  # local import: optional dependency (`pip install gandalf[mongo]`)

        self._client = pymongo.MongoClient(uri)
        self._db = self._client[db]
        self._nodes = self._db[nodes_collection]
        self._edges = self._db[edges_collection]

    @classmethod
    def from_collections(cls, nodes_collection, edges_collection) -> "MongoSource":
        """Build a source from pre-constructed collections (no client/pymongo).

        Intended for testing with fakes/mongomock; the ``__init__`` path is used
        in production to construct the real client.
        """
        self = cls.__new__(cls)
        self._client = None
        self._db = None
        self._nodes = nodes_collection
        self._edges = edges_collection
        return self

    def iter_edge_triples(self) -> Iterator[Tuple[str, str, str]]:
        cursor = self._edges.find(
            {}, {"subject": 1, "object": 1, "predicate": 1, "_id": 1}
        ).sort(_EDGE_SORT_KEY, 1)
        for doc in cursor:
            yield doc["subject"], doc["object"], doc["predicate"]

    def iter_edges(self) -> Iterator[dict]:
        cursor = self._edges.find({}).sort(_EDGE_SORT_KEY, 1)
        for doc in cursor:
            doc.pop("_id", None)
            validate_normalized_edge(doc)
            yield doc

    def iter_nodes(self) -> Iterator[dict]:
        for doc in self._nodes.find({}):
            doc.pop("_id", None)
            validate_normalized_node(doc)
            yield doc

    def close(self) -> None:
        """Release the underlying MongoDB client connection, if any."""
        if self._client is not None:
            self._client.close()
