"""Node-filter registry for path traversal.

Filters are plain closures `(graph, node_idx) -> bool` that return True iff a
node passes. They are built once per query from a config dict and applied per
candidate node during traversal.

Registration lives in `gandalf.plugins`. This module owns the type aliases,
the registry list, and the build/apply helpers used by the query engine.
"""

from typing import Any, Callable, Optional

NodeFilter = Callable[[Any, int], bool]
NodeFilterFactory = Callable[[dict], Optional[NodeFilter]]


_REGISTRY: list[tuple[str, NodeFilterFactory]] = []


def register_node_filter(name: str, factory: NodeFilterFactory) -> None:
    """Register a NodeFilter factory under a name.

    A factory takes the per-query config dict and returns either a closure
    (the filter) or None when the filter is inactive for this query (its
    config key is missing or set to None).
    """
    _REGISTRY.append((name, factory))


def registered_filter_names() -> list[str]:
    """Names of all registered filter factories, in registration order."""
    return [name for name, _ in _REGISTRY]


def build_node_filters(cfg: dict) -> list[NodeFilter]:
    """Build the list of active NodeFilters for a single query.

    Walks the registry in registration order and asks each factory for a
    filter. Inactive factories return None and are skipped.
    """
    filters: list[NodeFilter] = []
    for _, factory in _REGISTRY:
        f = factory(cfg)
        if f is not None:
            filters.append(f)
    return filters


def apply_node_filters(filters: list[NodeFilter], graph, node_idx: int) -> bool:
    """True iff the node passes every filter. Empty list is vacuously True."""
    for f in filters:
        if not f(graph, node_idx):
            return False
    return True
