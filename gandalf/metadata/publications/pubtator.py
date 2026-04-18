"""PubTator3 bulk annotation file parser.

Adapter that turns a PubTator TSV (``bioconcepts2pubtatorcentral`` /
``bioconcepts2pubtator3central`` family) into a stream of ``(pmid, curie)``
tuples consumable by ``PublicationsIndex.build``.

Accepts both the annotations-only TSV and the mixed "BioC-text" form
where title / abstract lines are interleaved with annotation lines.
Annotation lines look like:

    <pmid>\t<start>\t<end>\t<mention>\t<type>\t<concept_id[;concept_id...]>

Concept IDs for some types arrive bare (e.g. NCBI Gene IDs and NCBI
Taxonomy species IDs), so we apply per-type prefix normalization to
produce canonical CURIEs (``NCBIGene:5468``, ``NCBITaxon:9606``, etc.).
Values that already carry a prefix pass through untouched.

Other sources can plug in by writing a sibling module that yields the
same ``(pmid, curie)`` tuple stream — the index and downstream derivations
don't care where the data came from.
"""

from __future__ import annotations

import gzip
import logging
import re
from pathlib import Path
from typing import Iterator, Optional, Set, TextIO, Tuple

logger = logging.getLogger(__name__)


# Map PubTator entity types to the CURIE prefix that should be applied
# when the concept_id arrives without one.  Anything not listed leaves
# the concept_id unprefixed (we still emit it, but downstream matching
# will require the source document to carry a CURIE already).
_DEFAULT_PREFIX_BY_TYPE = {
    "gene": "NCBIGene",
    "species": "NCBITaxon",
    "cellline": "CVCL",
    "chemical": "MESH",
    "disease": "MESH",
    "snp": "RS",
    "mutation": "RS",
    "proteinmutation": "RS",
    "dnamutation": "RS",
}


_NUMERIC_ID = re.compile(r"^\d+$")


def _normalize_concept_id(concept_id: str, entity_type: str) -> Optional[str]:
    """Return a canonical CURIE for a raw PubTator concept_id, or None to skip."""
    concept_id = concept_id.strip()
    if not concept_id or concept_id == "-":
        return None
    # Already a CURIE (has a namespace prefix).
    if ":" in concept_id:
        return concept_id
    prefix = _DEFAULT_PREFIX_BY_TYPE.get(entity_type.strip().lower())
    if prefix is None:
        return None
    if not _NUMERIC_ID.match(concept_id) and prefix in ("NCBIGene", "NCBITaxon"):
        # Non-numeric bare value for a type we expected to be numeric —
        # skip rather than emit a garbage CURIE.
        return None
    return f"{prefix}:{concept_id}"


def _open_text(path: Path) -> TextIO:
    """Open a text file, transparently handling .gz compression."""
    if str(path).endswith(".gz"):
        return gzip.open(path, mode="rt", encoding="utf-8", errors="replace")
    return open(path, mode="r", encoding="utf-8", errors="replace")


def iter_pubtator_annotations(
    path: Path,
    tracked_curies: Optional[Set[str]] = None,
    entity_types: Optional[Set[str]] = None,
) -> Iterator[Tuple[int, str]]:
    """Stream ``(pmid, curie)`` tuples from a PubTator TSV file.

    Args:
        path: Path to a PubTator annotations file.  ``.gz`` is handled.
        tracked_curies: If supplied, only tuples whose CURIE is in this
            set are yielded.  The filter is applied after normalization.
        entity_types: If supplied, only annotations whose entity type
            (case-insensitive) is in this set are emitted.

    Yields:
        ``(pmid, curie)`` — CURIEs are canonicalized and may be yielded
        multiple times for the same PMID when the source lists multiple
        concept IDs on one line.
    """
    path = Path(path)
    allowed_types = (
        {t.lower() for t in entity_types} if entity_types is not None else None
    )

    skipped_unparseable = 0
    with _open_text(path) as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\r\n")
            if not line:
                continue
            # Skip BioC title/abstract text lines like "12345|t|...".
            if "|" in line and line.split("\t", 1)[0].find("|") != -1:
                head = line.split("|", 2)
                if len(head) >= 2 and head[1] in ("t", "a"):
                    continue

            fields = line.split("\t")
            # Canonical annotation row has 6 fields:
            #   pmid, start, end, mention, type, concept_id(s)
            # Some variants only carry 5 (no mention) or 7 (trailing resource).
            if len(fields) < 5:
                skipped_unparseable += 1
                continue
            pmid_str = fields[0].strip()
            if not pmid_str.isdigit():
                skipped_unparseable += 1
                continue
            pmid = int(pmid_str)
            entity_type = fields[4] if len(fields) >= 6 else fields[3]
            concept_field = fields[5] if len(fields) >= 6 else fields[4]
            if allowed_types is not None and entity_type.strip().lower() not in allowed_types:
                continue

            for raw_id in _split_concept_ids(concept_field):
                curie = _normalize_concept_id(raw_id, entity_type)
                if curie is None:
                    continue
                if tracked_curies is not None and curie not in tracked_curies:
                    continue
                yield pmid, curie

    if skipped_unparseable:
        logger.warning(
            "PubTator parser: skipped %s unparseable line(s) in %s",
            f"{skipped_unparseable:,}",
            path,
        )


def _split_concept_ids(field: str) -> Iterator[str]:
    """PubTator often concatenates IDs with ``;`` or ``,``."""
    for part in re.split(r"[;,|]", field):
        part = part.strip()
        if part:
            yield part
