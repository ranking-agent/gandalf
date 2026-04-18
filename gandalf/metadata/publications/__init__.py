"""Source-agnostic publication indexing.

A ``PublicationsIndex`` maps each CURIE string to the set of PMIDs that
mention it (and the inverse map, PMID to CURIEs).  Any data source that
can yield ``(pmid, curie)`` tuples can populate the index — currently only
PubTator3 is wired up, but new sources plug in by adding a parser module
alongside ``pubtator.py`` that yields the same tuple stream.

Downstream derivations (node publication counts, pair intersections) read
from the index and don't care which source produced it.
"""

from gandalf.metadata.publications.derive import (
    collect_tracked_curies,
    derive_and_ingest_node_pub_counts,
    iter_node_equivalents,
    iter_node_pub_counts,
)
from gandalf.metadata.publications.index import PublicationsIndex
from gandalf.metadata.publications.pubtator import iter_pubtator_annotations

__all__ = [
    "PublicationsIndex",
    "collect_tracked_curies",
    "derive_and_ingest_node_pub_counts",
    "iter_node_equivalents",
    "iter_node_pub_counts",
    "iter_pubtator_annotations",
]
