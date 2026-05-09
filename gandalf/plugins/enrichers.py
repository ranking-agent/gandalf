"""Node-enricher registry.

Enrichers run after the graph is loaded (either from JSONL or from mmap)
and write into ``graph.traversal_metadata`` — an LMDB-backed
``TraversalMetadataStore`` whose contents are **separate from the TRAPI
attribute list** and are never read during response building. Enrichers
must not mutate ``graph.node_properties``, the other LMDB stores, or
any other graph field.

**Namespace convention:** an enricher's registered name MUST equal the
``traversal_metadata`` namespace it owns. ``run_enrichers`` uses this to
skip enrichers whose data is already in the store (e.g. loaded from
disk), so persisted metadata is not recomputed.

Registration lives in plugin modules under ``gandalf.plugins``. This
module owns the registry list and the runner.
"""

import logging
from typing import Callable

logger = logging.getLogger(__name__)

NodeEnricher = Callable[[object], None]  # (CSRGraph) -> None


_ENRICHERS: list[tuple[str, NodeEnricher]] = []


def register_node_enricher(name: str, fn: NodeEnricher) -> None:
    """Register an enricher under a name. Called by plugin modules at import time.

    The ``name`` must match the ``traversal_metadata`` namespace the
    enricher writes (i.e. the first argument to ``store.put(...)``);
    ``run_enrichers`` skips enrichers whose namespace already has data.
    """
    _ENRICHERS.append((name, fn))


def registered_enricher_names() -> list[str]:
    """Names of all registered enrichers, in registration order."""
    return [name for name, _ in _ENRICHERS]


def run_enrichers(graph) -> None:
    """Populate any missing namespaces in ``graph.traversal_metadata``.

    For each registered enricher, in registration order:
      * If the store already has data under the enricher's namespace, skip
        it (this is the common load-time path: data was persisted).
      * Else if the store is readonly (loaded from disk and the namespace
        is missing), log a warning and skip — the user should rebuild the
        graph to include this plugin's data.
      * Otherwise run the enricher; it will write into the store.
    """
    store = graph.traversal_metadata
    for name, fn in _ENRICHERS:
        if name in store:
            continue
        if getattr(store, "is_readonly", False):
            logger.warning(
                "Plugin %r registered but missing from the saved traversal "
                "metadata store. Rebuild the graph to include it.",
                name,
            )
            continue
        fn(graph)
