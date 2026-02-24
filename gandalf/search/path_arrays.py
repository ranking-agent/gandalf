"""Compact path representation using numpy arrays."""


class PathArrays:
    """Compact path representation using numpy arrays instead of Python dicts.

    Keeps all path data in four dense numpy arrays (~50 bytes/path) rather
    than enriched Python dicts (~3 KB/path).  For 5M paths this reduces
    memory from ~15 GB of dicts to ~250 MB of arrays.
    """

    __slots__ = (
        'paths_nodes', 'paths_preds', 'paths_via_inverse',
        'paths_fwd_edge_idx', 'node_cache', 'idx_to_predicate',
        'qnode_to_col', 'qedge_to_col', 'col_to_qnode', 'col_to_qedge',
        'num_node_cols', 'num_edges', 'lightweight',
    )

    def __init__(self, *, paths_nodes, paths_preds, paths_via_inverse,
                 paths_fwd_edge_idx, node_cache, idx_to_predicate,
                 qnode_to_col, qedge_to_col, col_to_qnode, col_to_qedge,
                 num_node_cols, num_edges, lightweight):
        self.paths_nodes = paths_nodes
        self.paths_preds = paths_preds
        self.paths_via_inverse = paths_via_inverse
        self.paths_fwd_edge_idx = paths_fwd_edge_idx
        self.node_cache = node_cache
        self.idx_to_predicate = idx_to_predicate
        self.qnode_to_col = qnode_to_col
        self.qedge_to_col = qedge_to_col
        self.col_to_qnode = col_to_qnode
        self.col_to_qedge = col_to_qedge
        self.num_node_cols = num_node_cols
        self.num_edges = num_edges
        self.lightweight = lightweight

    def __len__(self):
        return len(self.paths_nodes)
