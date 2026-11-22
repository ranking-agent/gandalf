"""min_information_content filter plugin.

Filters out nodes whose `information_content` attribute is below a threshold.
Active when `filter_config["min_information_content"]` is a number.

Besides the per-node closure, the filter has a ``mask`` method that judges a
whole traversal step's nodes at once: from the graph's information content
column (``gandalf.attribute_columns``) where it has one, else with one
batched node-store read.
"""

import numpy as np

from gandalf.search.node_filters import register_node_filter

INFORMATION_CONTENT = "biolink:information_content"


def _information_content(attributes):
    """The value of the first information_content attribute, or None."""
    for attr in attributes:
        if attr.get("attribute_type_id") == INFORMATION_CONTENT:
            return attr.get("value")
    return None


def _get_information_content(graph, node_idx):
    return _information_content(graph.get_node_property(node_idx, "attributes", []))


class _MinInformationContent:
    """Passes nodes whose information content is at least *threshold*."""

    __slots__ = ("threshold",)

    def __init__(self, threshold):
        self.threshold = threshold

    def _passes(self, ic) -> bool:
        return ic is not None and ic >= self.threshold

    def __call__(self, graph, node_idx) -> bool:
        return self._passes(_get_information_content(graph, node_idx))

    def mask(self, graph, node_idxes: np.ndarray) -> np.ndarray:
        """:meth:`__call__` for many nodes: from the graph's information
        content column where it has one, else read in one batch."""
        columns = getattr(graph, "attribute_columns", None)
        column = (
            columns.node_numeric.get(INFORMATION_CONTENT)
            if columns is not None
            else None
        )
        if column is not None:
            passes: np.ndarray
            passes, fallback = column.at_least(node_idxes, self.threshold)
            for i in np.flatnonzero(fallback).tolist():
                passes[i] = self(graph, int(node_idxes[i]))
            return passes
        nodes = node_idxes.tolist()
        props = graph.get_all_node_properties_batch(nodes)
        return np.fromiter(
            (
                self._passes(
                    _information_content(props.get(node, {}).get("attributes", []))
                )
                for node in nodes
            ),
            dtype=np.bool_,
            count=len(nodes),
        )


def _factory(cfg):
    threshold = cfg.get("min_information_content")
    if threshold is None:
        return None
    return _MinInformationContent(threshold)


register_node_filter("min_information_content", _factory)
