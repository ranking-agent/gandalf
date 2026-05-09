"""max_node_degree filter plugin.

Filters out nodes whose total degree (in + out) exceeds a threshold.
Active when `filter_config["max_node_degree"]` is an int.
"""

from gandalf.search.node_filters import register_node_filter


def _node_total_degree(graph, node_idx):
    out_deg = int(graph.fwd_offsets[node_idx + 1] - graph.fwd_offsets[node_idx])
    in_deg = int(graph.rev_offsets[node_idx + 1] - graph.rev_offsets[node_idx])
    return out_deg + in_deg


def _factory(cfg):
    threshold = cfg.get("max_node_degree")
    if threshold is None:
        return None

    def _filter(graph, node_idx):
        return _node_total_degree(graph, node_idx) <= threshold

    return _filter


register_node_filter("max_node_degree", _factory)
