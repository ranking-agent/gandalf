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

The Annotator fans each call out to the BioThings APIs (mygene.info,
mydisease.info, ...), any of which can answer 500 for a while, or for one
particular CURIE.  A whole-graph run must not die of either, so each batch
is retried with backoff (:func:`annotate_with_retry`), a batch that keeps
failing is split in half until the CURIEs the service rejects are isolated
and left unannotated (:func:`annotate_resilient`), and a service that fails
everything, even one CURIE at a time, aborts the build with
:class:`AnnotatorUnavailable` rather than quietly producing a graph with no
annotations.  An :class:`AnnotationCache` (``--annotation-cache``) records
every answer as it arrives, so a run that is aborted, or a later rebuild,
fetches only what it does not have yet.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Tuple,
    Union,
)

import orjson

from gandalf.biolink import NAMED_THING

logger = logging.getLogger(__name__)

# TRAPI attribute_type_id used by the Annotator service for its payload.
BIOTHINGS_ANNOTATIONS_ATTRIBUTE_TYPE_ID = "biothings_annotations"

# Number of CURIEs sent to the Annotator in a single call.
DEFAULT_ANNOTATION_BATCH_SIZE = 1000

# How many times one batch is tried before it is split, and the first wait
# between tries (doubling each time, up to _MAX_BACKOFF_SECONDS).
DEFAULT_ANNOTATION_ATTEMPTS = 5
DEFAULT_ANNOTATION_BACKOFF_SECONDS = 2.0
_MAX_BACKOFF_SECONDS = 60.0

#: One Annotator call: a batch of CURIEs to ``curie -> raw result``.
Annotate = Callable[[List[str]], Awaitable[Dict[str, Any]]]
#: ``asyncio.sleep`` or a stand-in, so the retry policy is testable in no time.
Sleep = Callable[[float], Awaitable[None]]


class AnnotatorUnavailable(RuntimeError):
    """The Annotator fails every CURIE, even one at a time: it is down.

    Raised instead of finishing the build with the remaining nodes
    unannotated, which nothing downstream would notice.
    """


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


async def annotate_with_retry(
    annotate: Annotate,
    batch: List[str],
    *,
    attempts: int = DEFAULT_ANNOTATION_ATTEMPTS,
    backoff_seconds: float = DEFAULT_ANNOTATION_BACKOFF_SECONDS,
    sleep: Sleep = asyncio.sleep,
) -> Dict[str, Any]:
    """Call *annotate* on *batch* up to *attempts* times, backing off between.

    A BioThings backend answering 500 for a minute is the common case this
    absorbs: the waits double from *backoff_seconds* (2, 4, 8, 16 s ...),
    capped at a minute.  The last failure is raised as it came.
    """
    for attempt in range(1, attempts + 1):
        try:
            return await annotate(batch)
        except Exception as exc:
            if attempt == attempts:
                raise
            delay = min(backoff_seconds * 2 ** (attempt - 1), _MAX_BACKOFF_SECONDS)
            logger.warning(
                "Annotator call for %s CURIEs failed (%s: %s); retry %d/%d in %.0fs",
                f"{len(batch):,}",
                type(exc).__name__,
                str(exc)[:200],
                attempt + 1,
                attempts,
                delay,
            )
            await sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover


async def annotate_resilient(
    annotate: Annotate,
    batch: List[str],
    *,
    attempts: int = DEFAULT_ANNOTATION_ATTEMPTS,
    backoff_seconds: float = DEFAULT_ANNOTATION_BACKOFF_SECONDS,
    failed: List[str],
    sleep: Sleep = asyncio.sleep,
) -> Dict[str, Any]:
    """Annotate *batch*, retrying, then splitting it to isolate what the service rejects.

    A batch that still fails after its retries is split in half and each
    half tried again (with fewer attempts, since the retries just showed the
    failure persists), down to single CURIEs.  A CURIE that fails alone is
    appended to *failed* and left unannotated: the service cannot handle
    that one, and a graph without its annotation beats no graph.  When both
    halves of a batch fail outright the service is failing everything, and
    :class:`AnnotatorUnavailable` is raised.

    Returns:
        ``curie -> raw result`` for the CURIEs that were annotated.
    """
    try:
        return await annotate_with_retry(
            annotate,
            batch,
            attempts=attempts,
            backoff_seconds=backoff_seconds,
            sleep=sleep,
        )
    except Exception as exc:
        if len(batch) == 1:
            logger.warning(
                "Annotator cannot annotate %s (%s: %s); leaving it unannotated",
                batch[0],
                type(exc).__name__,
                str(exc)[:200],
            )
            failed.append(batch[0])
            return {}
        logger.warning(
            "Annotator failed a batch of %s after %d attempts; splitting it to "
            "find the CURIEs it rejects",
            f"{len(batch):,}",
            attempts,
        )
        mid = len(batch) // 2
        results: Dict[str, Any] = {}
        halves_lost = 0
        for half in (batch[:mid], batch[mid:]):
            failed_before = len(failed)
            results.update(
                await annotate_resilient(
                    annotate,
                    half,
                    attempts=min(attempts, 2),
                    backoff_seconds=backoff_seconds,
                    failed=failed,
                    sleep=sleep,
                )
            )
            if len(failed) - failed_before == len(half):
                halves_lost += 1
        if halves_lost == 2:
            raise AnnotatorUnavailable(
                f"The Annotator failed every CURIE of a batch of {len(batch):,}, "
                "even one at a time, after retries: the service is unavailable. "
                "Re-run the build when it is back; with --annotation-cache the "
                "CURIEs already annotated are not fetched again."
            ) from exc
        return results


class AnnotationCache:
    """Every Annotator answer so far, as JSON lines, so nothing is fetched twice.

    Each line is ``{"curie": ..., "annotation": ...}`` with ``null`` for a
    CURIE the service had nothing for (so it is not asked again either).
    CURIEs the service *failed* on are not recorded: they are retried next
    run.  Opening the cache reads what is there; :meth:`record` appends.

    Examples:
        >>> import tempfile, os
        >>> path = os.path.join(tempfile.mkdtemp(), "annotations.jsonl")
        >>> cache = AnnotationCache(path)
        >>> cache.record({"NCBIGene:1": {"symbol": "A1BG"}, "MONDO:1": None})
        >>> sorted(AnnotationCache(path).known)
        ['MONDO:1', 'NCBIGene:1']
        >>> AnnotationCache(path).annotations()
        {'NCBIGene:1': {'symbol': 'A1BG'}}
    """

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self.known: Dict[str, Any] = {}
        if self.path.exists():
            with open(self.path, "rb") as f:
                for line in f:
                    if line.strip():
                        entry = orjson.loads(line)
                        self.known[entry["curie"]] = entry.get("annotation")
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def annotations(self) -> Dict[str, Any]:
        """The cached CURIEs that have an annotation."""
        return {c: a for c, a in self.known.items() if a is not None}

    def record(self, answers: Mapping[str, Any]) -> None:
        """Append *answers* (``curie -> annotation or None``) and remember them."""
        with open(self.path, "ab") as f:
            for curie, annotation in answers.items():
                f.write(orjson.dumps({"curie": curie, "annotation": annotation}))
                f.write(b"\n")
        self.known.update(answers)


async def _annotate_batches(
    curies: List[str],
    batch_size: int,
    on_batch: Callable[[Dict[str, Any]], None],
    *,
    annotate: Optional[Annotate] = None,
    attempts: int = DEFAULT_ANNOTATION_ATTEMPTS,
    backoff_seconds: float = DEFAULT_ANNOTATION_BACKOFF_SECONDS,
    cache: Optional[AnnotationCache] = None,
    sleep: Sleep = asyncio.sleep,
) -> List[str]:
    """Annotate *curies* batch by batch, handing each batch to *on_batch*.

    One Annotator instance serves every batch so its backend-discovery cache is
    reused.  Batches are handed off as they arrive rather than accumulated, so
    a whole-graph run never holds more than one batch of annotations at a time.

    Args:
        annotate: The call to make per batch; the real Annotator's
            ``annotate_curie_list`` when None.
        cache: Answers already in hand are handed to *on_batch* first and
            not fetched; every new answer is recorded as it arrives.

    Returns:
        The CURIEs the service could not annotate after retries.
    """
    if annotate is None:
        annotator_cls, _ = _import_annotator()
        annotator = annotator_cls()
        annotate = annotator.annotate_curie_list

    if cache is not None:
        cached = cache.annotations()
        if cached:
            on_batch({c: a for c, a in cached.items() if c in set(curies)})
        pending = [c for c in curies if c not in cache.known]
        logger.info(
            "  %s CURIEs already in %s; fetching %s",
            f"{len(curies) - len(pending):,}",
            cache.path,
            f"{len(pending):,}",
        )
        curies = pending

    failed: List[str] = []
    for start in range(0, len(curies), batch_size):
        batch = curies[start : start + batch_size]
        results = await annotate_resilient(
            annotate,
            batch,
            attempts=attempts,
            backoff_seconds=backoff_seconds,
            failed=failed,
            sleep=sleep,
        )
        answers = {curie: clean_annotation(result) for curie, result in results.items()}
        if cache is not None:
            cache.record(answers)
        on_batch({c: a for c, a in answers.items() if a is not None})
        logger.debug(
            "  annotated %s/%s CURIEs...",
            f"{min(start + batch_size, len(curies)):,}",
            f"{len(curies):,}",
        )
    if failed:
        logger.warning(
            "  %s CURIEs could not be annotated and were left without "
            "annotations (first few: %s)",
            f"{len(failed):,}",
            ", ".join(failed[:5]),
        )
    return failed


def _run_annotation(
    node_ids: Iterable[str],
    batch_size: int,
    on_batch: Callable[[Dict[str, Any]], None],
    *,
    attempts: int = DEFAULT_ANNOTATION_ATTEMPTS,
    cache_path: Optional[Union[str, Path]] = None,
    annotate: Optional[Annotate] = None,
) -> int:
    """Filter *node_ids*, annotate them, and stream each batch to *on_batch*.

    Returns:
        The number of annotatable CURIEs that were queried (not the number
        that came back with an annotation).
    """
    curies = annotatable_curies(node_ids) if annotate is None else sorted(set(node_ids))
    if not curies:
        logger.info("No annotatable CURIE prefixes found; skipping annotation")
        return 0

    logger.info("Annotating %s nodes via biothings_annotator...", f"{len(curies):,}")
    cache = AnnotationCache(cache_path) if cache_path else None
    asyncio.run(
        _annotate_batches(
            curies,
            batch_size,
            on_batch,
            annotate=annotate,
            attempts=attempts,
            cache=cache,
        )
    )
    return len(curies)


def fetch_annotations(
    node_ids: Iterable[str],
    batch_size: int = DEFAULT_ANNOTATION_BATCH_SIZE,
    *,
    attempts: int = DEFAULT_ANNOTATION_ATTEMPTS,
    cache_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Fetch annotations for every annotatable CURIE in *node_ids*.

    Args:
        node_ids: Node ID strings to annotate.  Unsupported prefixes are
            dropped by :func:`annotatable_curies` before any request is made.
        batch_size: CURIEs per Annotator call.
        attempts: Tries per batch before it is split (see
            :func:`annotate_resilient`).
        cache_path: An :class:`AnnotationCache` file to read and extend.

    Returns:
        ``curie -> annotation`` for the nodes that had one.  CURIEs the
        Annotator has nothing for are absent, not present-and-empty.
    """
    annotations: Dict[str, Any] = {}
    queried = _run_annotation(
        node_ids,
        batch_size,
        annotations.update,
        attempts=attempts,
        cache_path=cache_path,
    )
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
    *,
    attempts: int = DEFAULT_ANNOTATION_ATTEMPTS,
    cache_path: Optional[Union[str, Path]] = None,
) -> int:
    """Annotate every node in the graph vocabulary and store the results.

    This is the entry point the loader uses: it fetches annotations for the
    graph's node IDs and attaches them to *node_properties* in place.

    Args:
        node_properties: Loader property map, ``node_idx -> {name, categories,
            attributes}``.  Mutated in place.
        node_id_to_idx: The graph's node ID -> index vocabulary.
        batch_size: CURIEs per Annotator call.
        attempts: Tries per batch before it is split (see
            :func:`annotate_resilient`).
        cache_path: An :class:`AnnotationCache` file to read and extend, so
            a rebuild (or a run the service interrupted) fetches only what
            it does not have.

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

    queried = _run_annotation(
        node_id_to_idx.keys(),
        batch_size,
        attach_batch,
        attempts=attempts,
        cache_path=cache_path,
    )
    logger.info(
        "  Retrieved annotations for %s/%s nodes", f"{annotated:,}", f"{queried:,}"
    )
    return annotated
