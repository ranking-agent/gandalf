"""Node-enricher registry.

Enrichers run once after the graph is loaded (either from JSONL or from
mmap). Each writes a single key into `graph.traversal_metadata`, which is
**separate from the TRAPI attribute list** and is never read during
response building. Enrichers must not mutate `graph.node_properties`,
the LMDB stores, or any other graph field.

Registration lives in plugin modules under `gandalf.plugins`. This module
owns the registry list and the runner.
"""

from typing import Callable

NodeEnricher = Callable[[object], None]  # (CSRGraph) -> None


_ENRICHERS: list[tuple[str, NodeEnricher]] = []


def register_node_enricher(name: str, fn: NodeEnricher) -> None:
    """Register an enricher under a name. Called by plugin modules at import time."""
    _ENRICHERS.append((name, fn))


def registered_enricher_names() -> list[str]:
    """Names of all registered enrichers, in registration order."""
    return [name for name, _ in _ENRICHERS]


def run_enrichers(graph) -> None:
    """Run every registered enricher against the graph, in registration order."""
    for _, fn in _ENRICHERS:
        fn(graph)
