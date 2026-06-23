"""KGX jsonl graph source (wrapper 1).

Reads KGX-formatted ``nodes.jsonl`` / ``edges.jsonl`` directly and applies
gandalf's normalization (``gandalf.normalize``) to produce normalized,
validated records. This reproduces the loader's historical behavior.
"""

import logging
from typing import Iterator, Optional, Tuple

import orjson

from gandalf.normalize import normalize_edge, normalize_node
from gandalf.sources.base import (
    GraphSource,
    validate_normalized_edge,
    validate_normalized_node,
)

logger = logging.getLogger(__name__)


class KGXJsonlSource(GraphSource):
    """A :class:`GraphSource` backed by KGX jsonl files.

    ``node_jsonl_path`` is optional; when absent, ``iter_nodes()`` yields nothing
    (the graph is built from edge endpoints only).
    """

    def __init__(self, edge_jsonl_path, node_jsonl_path: Optional[str] = None):
        self.edge_path = str(edge_jsonl_path)
        self.node_path = str(node_jsonl_path) if node_jsonl_path else None

    def iter_edge_triples(self) -> Iterator[Tuple[str, str, str]]:
        with open(self.edge_path, "r", encoding="utf-8") as f:
            for line in f:
                data = orjson.loads(line)
                yield data["subject"], data["object"], data["predicate"]

    def iter_edges(self) -> Iterator[dict]:
        with open(self.edge_path, "r", encoding="utf-8") as f:
            for line in f:
                edge = normalize_edge(orjson.loads(line))
                validate_normalized_edge(edge)
                yield edge

    def iter_nodes(self) -> Iterator[dict]:
        if not self.node_path:
            return
        with open(self.node_path, "r", encoding="utf-8") as f:
            for line in f:
                node = normalize_node(orjson.loads(line))
                # Mirror the historical loader guard: skip records without an id.
                if not node.get("id"):
                    continue
                validate_normalized_node(node)
                yield node
