"""Node-enricher registry.

Enrichers run once after the graph is loaded (either from JSONL or from
mmap) and write a single key into `graph.traversal_metadata`, which is
**separate from the TRAPI attribute list** and is never read during
response building. Enrichers must not mutate `graph.node_properties`,
the LMDB stores, or any other graph field.

**Convention:** an enricher's registered name MUST equal the
`graph.traversal_metadata` key it owns. ``run_enrichers`` uses this to
skip enrichers whose data has already been loaded from disk (or
otherwise populated), so persisted metadata is not recomputed.

Registration lives in plugin modules under `gandalf.plugins`. This module
owns the registry list and the runner.
"""

from typing import Callable

NodeEnricher = Callable[[object], None]  # (CSRGraph) -> None


_ENRICHERS: list[tuple[str, NodeEnricher]] = []


def register_node_enricher(name: str, fn: NodeEnricher) -> None:
    """Register an enricher under a name. Called by plugin modules at import time.

    The ``name`` must match the ``graph.traversal_metadata`` key the enricher
    writes; ``run_enrichers`` skips enrichers whose name is already present
    in the metadata dict (e.g. loaded from disk).
    """
    _ENRICHERS.append((name, fn))


def registered_enricher_names() -> list[str]:
    """Names of all registered enrichers, in registration order."""
    return [name for name, _ in _ENRICHERS]


def run_enrichers(graph) -> None:
    """Run every registered enricher whose key is missing from traversal_metadata.

    Enrichers whose registered name is already a key in
    ``graph.traversal_metadata`` are skipped; their data is assumed to have
    been loaded from disk. Run order is registration order.
    """
    for name, fn in _ENRICHERS:
        if name in graph.traversal_metadata:
            continue
        fn(graph)
