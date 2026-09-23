""""""

import math
from collections import defaultdict

N = 1_000_000  # total number of nodes
R = 25  # number of edges per node


def get_next_qedge(qgraph, bound_counts=None):
    """Get next qedge to solve.

    Args:
        qgraph: The (remaining) query graph.
        bound_counts: Optional ``{qnode_id: n}`` giving the number of graph
            nodes already bound to a qnode, for callers that track bindings
            outside the qgraph.  Overrides the count derived from that
            qnode's ``ids``.
    """
    num_ids = get_num_ids(qgraph)
    if bound_counts:
        num_ids.update(
            (qnode_id, n) for qnode_id, n in bound_counts.items() if qnode_id in num_ids
        )
    adjacency_mat = get_adjacency_matrix(qgraph)
    pinnednesses = {
        qnode_id: -compute_log_expected_n(adjacency_mat, num_ids, qnode_id)
        for qnode_id in qgraph["nodes"]
    }
    efforts = {
        qedge_id: math.log(num_ids[qedge["subject"]])
        + math.log(num_ids[qedge["object"]])
        for qedge_id, qedge in qgraph["edges"].items()
    }
    edge_priorities = {
        qedge_id: pinnednesses[qedge["subject"]]
        + pinnednesses[qedge["object"]]
        - efforts[qedge_id]
        for qedge_id, qedge in qgraph["edges"].items()
    }
    qedge_id = max(edge_priorities, key=edge_priorities.get)
    return qedge_id, qgraph["edges"][qedge_id]


def get_pinnedness(qgraph, qnode_id):
    """Get pinnedness of each node."""
    adjacency_mat = get_adjacency_matrix(qgraph)
    num_ids = get_num_ids(qgraph)
    return -compute_log_expected_n(
        adjacency_mat,
        num_ids,
        qnode_id,
    )


def compute_log_expected_n(adjacency_mat, num_ids, qnode_id, last=None, level=0):
    """Compute the log of the expected number of unique knodes bound to the specified qnode in the final results."""
    log_expected_n = math.log(num_ids[qnode_id])
    if level < 10:
        for neighbor, num_edges in adjacency_mat[qnode_id].items():
            if neighbor == last:
                continue
            # ignore contributions >0 - edges should only _further_ constrain n
            log_expected_n += num_edges * min(
                max(
                    compute_log_expected_n(
                        adjacency_mat,
                        num_ids,
                        neighbor,
                        qnode_id,
                        level + 1,
                    ),
                    0,
                )
                + math.log(R / N),
                0,
            )
    return log_expected_n


def get_adjacency_matrix(qgraph):
    """Get adjacency matrix."""
    A = defaultdict(lambda: defaultdict(int))
    for qedge in qgraph["edges"].values():
        A[qedge["subject"]][qedge["object"]] += 1
        A[qedge["object"]][qedge["subject"]] += 1
    return A


def get_num_ids(qgraph):
    """Get the number of ids bound to each node (``N`` for an unpinned node).

    Reads the counts without copying the id lists, which after a traversal
    step can hold every node the step discovered.

    >>> get_num_ids({"nodes": {
    ...     "a": {"ids": ["X:1", "X:2"]},
    ...     "b": {},
    ...     "c": {"set_interpretation": "MANY", "member_ids": ["X:3"]},
    ... }})
    {'a': 2, 'b': 1000000, 'c': 1}
    """
    return {qnode_id: _num_ids(qnode) for qnode_id, qnode in qgraph["nodes"].items()}


def _num_ids(qnode):
    """Count the ids bound to one qnode, treating a multi-curie qnode by its members."""
    if (
        qnode.get("set_interpretation") == "MANY"
        and len(qnode.get("member_ids") or []) > 0
    ):
        return len(qnode["member_ids"])
    if qnode.get("ids") is not None:
        return len(qnode["ids"])
    return N


def connected_edges(qgraph, node_id):
    """Find edges connected to node."""
    outgoing = []
    incoming = []
    for edge_id, edge in qgraph["edges"].items():
        if node_id == edge["subject"]:
            outgoing.append(edge_id)
        if node_id == edge["object"]:
            incoming.append(edge_id)
    return outgoing, incoming


def remove_orphaned(qgraph):
    """Remove nodes with degree 0."""
    qgraph["nodes"] = {
        node_id: node
        for node_id, node in qgraph["nodes"].items()
        if any(connected_edges(qgraph, node_id))
    }
