"""min_information_content filter plugin.

Filters out nodes whose `information_content` attribute is below a threshold.
Active when `filter_config["min_information_content"]` is a number.
"""

from gandalf.search.node_filters import register_node_filter


def _get_information_content(graph, node_idx):
    attrs = graph.get_node_property(node_idx, "attributes", [])
    for attr in attrs:
        if attr.get("original_attribute_name") == "information_content":
            return attr.get("value")
    return None


def _factory(cfg):
    threshold = cfg.get("min_information_content")
    if threshold is None:
        return None

    def _filter(graph, node_idx):
        ic = _get_information_content(graph, node_idx)
        return ic is not None and ic >= threshold

    return _filter


register_node_filter("min_information_content", _factory)
