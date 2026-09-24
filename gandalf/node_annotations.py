"""Build-time node annotation via the Translator ``biothings_annotator`` package.

The Translator Annotator service (https://github.com/biothings/biothings_annotator)
resolves a CURIE such as ``NCBIGene:1017`` into a rich annotation object
(gene summaries, chemical structures, disease metadata, ...).  Gandalf calls it
once, at graph *build* time, and stores whatever comes back on the node so that
queries never pay for a network round trip.

Annotations are attached to a node's ``attributes`` list as a single TRAPI
Attribute whose ``attribute_type_id`` is ``biothings_annotations`` -- the same
shape ``Annotator.annotate_trapi`` produces, so downstream consumers see the
identical structure whether the annotation was added at build time by Gandalf or
at query time by the Annotator service itself.

``biothings_annotator`` is an optional dependency (it is not published to PyPI);
it is imported lazily so that only builds that ask for annotation need it::

    pip install -r requirements-annotate.txt

Usage from the loader::

    build_graph_from_jsonl(edges, nodes, annotate_nodes=True)

or from the CLI::

    gandalf-build --edges edges.jsonl --nodes nodes.jsonl -o graph/ --annotate
"""

from __future__ import annotations

import asyncio
import logging
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Tuple,
)

from gandalf.biolink import NAMED_THING

logger = logging.getLogger(__name__)

# TRAPI attribute_type_id used by the Annotator service for its payload.
BIOTHINGS_ANNOTATIONS_ATTRIBUTE_TYPE_ID = "biothings_annotations"

# Number of CURIEs sent to the Annotator in a single call.
DEFAULT_ANNOTATION_BATCH_SIZE = 1000

_MISSING_PACKAGE_MESSAGE = (
    "Node annotation requires the 'biothings_annotator' package, which is not "
    "installed.  It is not on PyPI; install it with "
    "`pip install -r requirements-annotate.txt`."
)


def _import_annotator() -> Tuple[Any, Mapping[str, dict]]:
    """Import ``biothings_annotator`` lazily.

    Returns:
        A ``(Annotator, BIOLINK_PREFIX_to_BioThings)`` tuple: the annotator
        class and the package's CURIE-prefix -> BioThings-source mapping.

    Raises:
        ImportError: If the optional package is not installed, with an
            actionable install hint.
    """
    try:
        from biothings_annotator import Annotator, BIOLINK_PREFIX_to_BioThings
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(_MISSING_PACKAGE_MESSAGE) from exc
    return Annotator, BIOLINK_PREFIX_to_BioThings


def annotatable_curies(node_ids: Iterable[str]) -> List[str]:
    """Filter *node_ids* down to CURIEs the Annotator can actually look up.

    The Annotator only knows a fixed set of CURIE prefixes (NCBIGene, CHEBI,
    MONDO, HP, ...); everything else is a guaranteed miss.  Filtering up front
    keeps a whole-graph annotation run from spending its time on prefixes the
    service will never resolve.  The prefix set comes from the package itself,
    so it stays correct as the Annotator adds sources.

    Args:
        node_ids: Node ID strings, e.g. the keys of ``node_id_to_idx``.

    Returns:
        The subset of *node_ids* whose prefix the Annotator supports, in input
        order and de-duplicated.
    """
    _, prefix_map = _import_annotator()
    seen = set()
    curies = []
    for node_id in node_ids:
        prefix = node_id.split(":", 1)[0]
        if prefix in prefix_map and node_id not in seen:
            seen.add(node_id)
            curies.append(node_id)
    return curies


def clean_annotation(result: Any) -> Optional[Any]:
    """Reduce one Annotator result to the annotation worth storing.

    The Annotator returns ``{}`` for a node it has nothing for, and hit objects
    carrying ``notfound`` / ``skipped`` markers for CURIEs whose backend was
    unavailable.  None of those are annotations, so they are dropped rather than
    written onto the graph as empty attributes.

    Args:
        result: A single value from ``Annotator.annotate_curie_list``: an
            annotation dict, a list of hit dicts, or an empty/placeholder value.

    Returns:
        The annotation to store, or ``None`` when there is nothing available.

    Examples:
        >>> clean_annotation({"symbol": "TP53"})
        {'symbol': 'TP53'}
        >>> clean_annotation({}) is None
        True
        >>> clean_annotation([{"query": "FOO:1", "notfound": True}]) is None
        True
        >>> clean_annotation([{"symbol": "TP53"}, {"notfound": True}])
        [{'symbol': 'TP53'}]
    """
    if isinstance(result, dict):
        if not result or result.get("notfound"):
            return None
        return result
    if isinstance(result, list):
        hits = [
            hit
            for hit in result
            if isinstance(hit, dict) and hit and not hit.get("notfound")
        ]
        return hits or None
    return None


def attach_annotations(
    node_properties: MutableMapping[int, dict],
    node_id_to_idx: Mapping[str, int],
    annotations: Mapping[str, Any],
) -> int:
    """Write *annotations* onto *node_properties* as TRAPI attributes.

    Each annotated node gets exactly one ``biothings_annotations`` attribute:
    any attribute already carrying that ``attribute_type_id`` is replaced, so
    annotating twice is idempotent.  Nodes present in the graph but absent from
    the node file get a properties entry created for them, in the same shape
    the loader builds: no ``name`` key at all (TRAPI 2.0 admits no nulls, so an
    unknown name must stay absent) and ``categories`` defaulted to the Biolink
    root class (it is required with a ``minItems`` of 1).

    Args:
        node_properties: Loader property map, ``node_idx -> {name, categories,
            attributes}``.  Mutated in place.
        node_id_to_idx: The graph's node ID -> index vocabulary.
        annotations: ``curie -> annotation`` as produced by
            :func:`clean_annotation`.

    Returns:
        The number of nodes that received an annotation.

    Examples:
        >>> props = {0: {"name": "TP53", "categories": ["biolink:Gene"],
        ...               "attributes": []}}
        >>> attach_annotations(props, {"NCBIGene:7157": 0},
        ...                    {"NCBIGene:7157": {"symbol": "TP53"}})
        1
        >>> props[0]["attributes"]
        [{'attribute_type_id': 'biothings_annotations', 'value': {'symbol': 'TP53'}}]
    """
    annotated = 0
    for curie, annotation in annotations.items():
        node_idx = node_id_to_idx.get(curie)
        if node_idx is None:
            continue
        properties = node_properties.setdefault(
            node_idx, {"categories": [NAMED_THING], "attributes": []}
        )
        attributes = properties.setdefault("attributes", [])
        attributes[:] = [
            attribute
            for attribute in attributes
            if attribute.get("attribute_type_id")
            != BIOTHINGS_ANNOTATIONS_ATTRIBUTE_TYPE_ID
        ]
        attributes.append(
            {
                "attribute_type_id": BIOTHINGS_ANNOTATIONS_ATTRIBUTE_TYPE_ID,
                "value": annotation,
            }
        )
        annotated += 1
    return annotated


async def _annotate_batches(
    curies: List[str], batch_size: int, on_batch: Callable[[Dict[str, Any]], None]
) -> None:
    """Annotate *curies* batch by batch, handing each batch to *on_batch*.

    One Annotator instance serves every batch so its backend-discovery cache is
    reused.  Batches are handed off as they arrive rather than accumulated, so
    a whole-graph run never holds more than one batch of annotations at a time.
    """
    annotator_cls, _ = _import_annotator()
    annotator = annotator_cls()

    for start in range(0, len(curies), batch_size):
        batch = curies[start : start + batch_size]
        results = await annotator.annotate_curie_list(batch)
        annotations = {}
        for curie, result in results.items():
            annotation = clean_annotation(result)
            if annotation is not None:
                annotations[curie] = annotation
        on_batch(annotations)
        logger.debug(
            "  annotated %s/%s CURIEs...",
            f"{min(start + batch_size, len(curies)):,}",
            f"{len(curies):,}",
        )


def _run_annotation(
    node_ids: Iterable[str],
    batch_size: int,
    on_batch: Callable[[Dict[str, Any]], None],
) -> int:
    """Filter *node_ids*, annotate them, and stream each batch to *on_batch*.

    Returns:
        The number of annotatable CURIEs that were queried (not the number
        that came back with an annotation).
    """
    curies = annotatable_curies(node_ids)
    if not curies:
        logger.info("No annotatable CURIE prefixes found; skipping annotation")
        return 0

    logger.info("Annotating %s nodes via biothings_annotator...", f"{len(curies):,}")
    asyncio.run(_annotate_batches(curies, batch_size, on_batch))
    return len(curies)


def fetch_annotations(
    node_ids: Iterable[str], batch_size: int = DEFAULT_ANNOTATION_BATCH_SIZE
) -> Dict[str, Any]:
    """Fetch annotations for every annotatable CURIE in *node_ids*.

    Args:
        node_ids: Node ID strings to annotate.  Unsupported prefixes are
            dropped by :func:`annotatable_curies` before any request is made.
        batch_size: CURIEs per Annotator call.

    Returns:
        ``curie -> annotation`` for the nodes that had one.  CURIEs the
        Annotator has nothing for are absent, not present-and-empty.
    """
    annotations: Dict[str, Any] = {}
    queried = _run_annotation(node_ids, batch_size, annotations.update)
    logger.info(
        "  Retrieved annotations for %s/%s nodes",
        f"{len(annotations):,}",
        f"{queried:,}",
    )
    return annotations


def annotate_node_properties(
    node_properties: MutableMapping[int, dict],
    node_id_to_idx: Mapping[str, int],
    batch_size: int = DEFAULT_ANNOTATION_BATCH_SIZE,
) -> int:
    """Annotate every node in the graph vocabulary and store the results.

    This is the entry point the loader uses: it fetches annotations for the
    graph's node IDs and attaches them to *node_properties* in place.

    Args:
        node_properties: Loader property map, ``node_idx -> {name, categories,
            attributes}``.  Mutated in place.
        node_id_to_idx: The graph's node ID -> index vocabulary.
        batch_size: CURIEs per Annotator call.

    Returns:
        The number of nodes that received an annotation.

    Note:
        Annotations are attached one batch at a time, so a whole-graph run adds
        only a batch of annotations to peak memory on top of the properties
        themselves.
    """
    annotated = 0

    def attach_batch(annotations: Dict[str, Any]) -> None:
        nonlocal annotated
        annotated += attach_annotations(node_properties, node_id_to_idx, annotations)

    queried = _run_annotation(node_id_to_idx.keys(), batch_size, attach_batch)
    logger.info(
        "  Retrieved annotations for %s/%s nodes", f"{annotated:,}", f"{queried:,}"
    )
    return annotated
