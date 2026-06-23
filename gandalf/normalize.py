"""Normalization / standardization of raw KGX records into gandalf's internal form.

This module owns everything that restructures a raw KGX node/edge dict into the
normalized form the loader consumes: TRAPI-style ``sources``/``qualifiers``/
``attributes`` lists, ``biolink:`` prefixing, the Biolink-Model-derived qualifier
field set, and the ``category`` -> ``categories`` rename for nodes.

The KGX jsonl source (``gandalf.sources.kgx_jsonl``) calls ``normalize_edge`` /
``normalize_node`` to produce records in the contract documented in
``gandalf.sources.base``. The MongoDB source does NOT use this module: its
documents are already stored in normalized form by an upstream pipeline.
"""

import logging
from typing import Optional

import orjson
from bmt.toolkit import Toolkit

from gandalf.biolink import make_toolkit
from gandalf.config import settings

logger = logging.getLogger(__name__)


# Fields that are structural (not stored as properties)
_CORE_FIELDS = {
    "id",
    "category",
    "subject",
    "object",
    "predicate",
    "sources",
    "primary_knowledge_source",
    "aggregator_knowledge_source",
}

# Node fields that become top-level TRAPI Node properties (not attributes).
# These are *raw* KGX field names ("category" is singular here); the normalized
# node form uses "categories" (plural).
_CORE_NODE_FIELDS = {"id", "name", "category"}

# Fallback qualifier fields, used only if the Biolink Model cannot be loaded via
# BMT at runtime. The authoritative set is derived from the model (see
# ``_get_qualifier_fields``); this static list is just a safety net. It mirrors
# the descendants of the abstract ``qualifier`` slot in the pinned Biolink
# version (settings.biolink_version, currently 4.3.2).
_FALLBACK_QUALIFIER_FIELDS = {
    "anatomical_context_qualifier",
    "aspect_qualifier",
    "causal_mechanism_qualifier",
    "context_qualifier",
    "derivative_qualifier",
    "direction_qualifier",
    "disease_context_qualifier",
    "form_or_variant_qualifier",
    "frequency_qualifier",
    "object_aspect_qualifier",
    "object_context_qualifier",
    "object_derivative_qualifier",
    "object_direction_qualifier",
    "object_form_or_variant_qualifier",
    "object_part_qualifier",
    "object_specialization_qualifier",
    "onset_qualifier",
    "part_qualifier",
    "population_context_qualifier",
    "qualified_predicate",
    "response_context_qualifier",
    "response_target_context_qualifier",
    "severity_qualifier",
    "sex_qualifier",
    "specialization_qualifier",
    "species_context_qualifier",
    "stage_qualifier",
    "statement_qualifier",
    "subject_aspect_qualifier",
    "subject_context_qualifier",
    "subject_derivative_qualifier",
    "subject_direction_qualifier",
    "subject_form_or_variant_qualifier",
    "subject_part_qualifier",
    "subject_specialization_qualifier",
    "temporal_context_qualifier",
    "temporal_interval_qualifier",
}

# Module-level BMT instance and cached qualifier-field set (lazily initialized).
_bmt: Optional[Toolkit] = None
_qualifier_fields: Optional[set] = None


def _get_bmt() -> Toolkit:
    """Get or create the module-level BMT instance (pinned biolink version)."""
    global _bmt
    if _bmt is None:
        _bmt = make_toolkit()
    return _bmt


def _get_qualifier_fields() -> set:
    """Return the set of top-level edge field names that are Biolink qualifiers.

    Derived from the Biolink Model: the snake_case names of every descendant of
    the abstract ``qualifier`` slot (e.g. ``object_aspect_qualifier``,
    ``frequency_qualifier``, ``qualified_predicate``). This keeps the loader in
    sync with the model instead of relying on a hardcoded list. The result is
    cached after the first build. If BMT or the model is unavailable, fall back
    to ``_FALLBACK_QUALIFIER_FIELDS`` so loading never fails.
    """
    global _qualifier_fields
    if _qualifier_fields is None:
        try:
            descendants = _get_bmt().get_descendants("qualifier", formatted=True)
            # ``formatted=True`` yields e.g. "biolink:object_aspect_qualifier";
            # strip the prefix to match the snake_case top-level JSONL keys.
            fields = {d.split(":", 1)[-1] for d in descendants}
            fields.discard("qualifier")  # abstract root, never a real edge field
            _qualifier_fields = fields or set(_FALLBACK_QUALIFIER_FIELDS)
        except Exception:
            logger.warning(
                "Could not load Biolink qualifier slots from BMT; "
                "using fallback qualifier set",
                exc_info=True,
            )
            _qualifier_fields = set(_FALLBACK_QUALIFIER_FIELDS)
    return _qualifier_fields


def _extract_sources(data):
    """Extract normalized source list from edge data.

    Ensures every source has an ``upstream_resource_ids`` list (defaults to
    ``[]``) and prepends an ``infores:gandalf`` aggregator_knowledge_source
    whose upstream points to the top of the existing source chain (i.e. the
    source(s) not referenced in any other source's upstream_resource_ids).

    This function is pure: it does not mutate ``data``.
    """
    raw = data.get("sources", [])

    if len(raw) == 0:
        # this is most likely from an automat kgx
        sources = [
            {
                "resource_id": data["primary_knowledge_source"],
                "resource_role": "primary_knowledge_source",
                "upstream_resource_ids": [],
            }
        ]

        if (
            "aggregator_knowledge_source" in data
            and len(data["aggregator_knowledge_source"]) > 0
        ):
            # Reverse a copy so we never mutate the caller's dict.
            aggregators = list(reversed(data["aggregator_knowledge_source"]))
            previous_source = data["primary_knowledge_source"]
            for aggregator_source in aggregators:
                sources.append(
                    {
                        "resource_id": aggregator_source,
                        "resource_role": "aggregator_knowledge_source",
                        "upstream_resource_ids": [previous_source],
                    }
                )
                previous_source = aggregator_source
    else:
        # new translatorkg format

        # Normalize: guarantee upstream_resource_ids on every source
        sources = [
            {
                "resource_id": s["resource_id"],
                "resource_role": s["resource_role"],
                "upstream_resource_ids": s.get("upstream_resource_ids", []),
            }
            for s in raw
        ]

    # Find the top of the source chain: sources whose resource_id is NOT
    # referenced in any other source's upstream_resource_ids.  These are the
    # "leaf" providers that no one else aggregates from yet.
    all_upstream = {uid for s in sources for uid in s["upstream_resource_ids"]}
    top_ids = [
        s["resource_id"] for s in sources if s["resource_id"] not in all_upstream
    ]

    # Prepend gandalf as aggregator_knowledge_source
    gandalf_source = {
        "resource_id": settings.infores,
        "resource_role": "aggregator_knowledge_source",
        "upstream_resource_ids": top_ids,
    }

    return [gandalf_source] + sources


def _ensure_biolink_prefix(value):
    """Ensure a CURIE-like value carries the ``biolink:`` prefix.

    Mirrors retriever's ``biolink.ensure_prefix`` (tier 1): strip any existing
    prefix, then prepend ``biolink:``. Used for ``qualified_predicate`` values,
    which are Biolink predicate CURIEs.
    """
    local = value.split(":", 1)[1] if ":" in value else value
    return f"biolink:{local}"


def _extract_qualifiers(data):
    """Extract qualifiers.

    Format: top-level fields (object_aspect_qualifier, etc.). The set of
    qualifier field names is derived from the Biolink Model via
    ``_get_qualifier_fields``.

    A TRAPI ``qualifier_value`` must be a scalar string. This matches the tier 1
    (BioPack/retriever Elasticsearch) driver so the same edge produces identical
    qualifiers across tiers: a non-string value is coerced to a JSON string via
    ``orjson`` (yielding exactly one qualifier entry per type, not split), and
    ``qualified_predicate`` values are normalized to carry the ``biolink:``
    prefix.
    """
    qualifier_fields = _get_qualifier_fields()
    qualifiers = []
    for field in qualifier_fields:
        if field in data:
            value = data[field]
            if not isinstance(value, str):
                value = orjson.dumps(value).decode()
            if field == "qualified_predicate":
                value = _ensure_biolink_prefix(value)
            qualifiers.append(
                {
                    "qualifier_type_id": f"biolink:{field}",
                    "qualifier_value": value,
                }
            )

    return qualifiers


def _extract_attributes(data):
    """Extract attributes (everything not in core/qualifier/source fields).

    Publications are included as a TRAPI Attribute with
    ``attribute_type_id`` of ``biolink:publications``.
    """
    qualifier_fields = _get_qualifier_fields()
    attributes = []
    for field, value in data.items():
        if field in _CORE_FIELDS or field in qualifier_fields or field == "qualifiers":
            continue
        attributes.append(
            {
                "attribute_type_id": f"biolink:{field}",
                "value": value,
                "original_attribute_name": field,
            }
        )
    return attributes


def _extract_node_attributes(node_data):
    """Extract node attributes as TRAPI Attribute objects.

    Any field not in ``_CORE_NODE_FIELDS`` (id, name, category) is converted
    to a TRAPI-compliant Attribute dict with ``attribute_type_id``, ``value``,
    and ``original_attribute_name``.
    """
    attributes = []
    for field, value in node_data.items():
        if field in _CORE_NODE_FIELDS:
            continue
        attributes.append(
            {
                "attribute_type_id": "biolink:Attribute",
                "value": value,
                "original_attribute_name": field,
            }
        )
    return attributes


def normalize_edge(raw: dict) -> dict:
    """Restructure a raw KGX edge dict into gandalf's normalized edge form.

    Produces the contract documented in ``gandalf.sources.base`` (subject,
    object, predicate, id, plus the normalized sources/qualifiers/attributes
    lists). Pure: does not mutate ``raw``.
    """
    return {
        "subject": raw["subject"],
        "object": raw["object"],
        "predicate": raw["predicate"],
        "id": raw.get("id"),
        "sources": _extract_sources(raw),
        "qualifiers": _extract_qualifiers(raw),
        "attributes": _extract_attributes(raw),
    }


def normalize_node(raw: dict) -> dict:
    """Restructure a raw KGX node dict into gandalf's normalized node form.

    Renames raw ``category`` (singular) to ``categories`` (plural) and converts
    non-core fields to TRAPI Attribute dicts. Pure: does not mutate ``raw``.
    """
    return {
        "id": raw.get("id"),
        "name": raw.get("name"),
        "categories": raw.get("category", []),
        "attributes": _extract_node_attributes(raw),
    }
