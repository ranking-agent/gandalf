"""Literature co-occurrence response annotator.

Reference annotator plugin that demonstrates the response-annotator
extension point. Active when a request includes
``annotator_config["literature_cooccurrence"]``.

For each request, the plugin:

1. Collects every node CURIE in
   ``response["message"]["knowledge_graph"]["nodes"]``.
2. Posts the CURIE list to the configured co-occurrence service in a
   single batched call.
3. Receives per-node total counts and per-pair counts.
4. Appends a ``biolink:occurrences_in_literature`` attribute to each
   node's ``attributes`` list.
5. For each node pair whose count exceeds ``min_cooccurrence``, inserts
   a new edge into ``knowledge_graph.edges`` with a namespaced ID
   (``"litcoocc:<hex>"``). These are KG-only "bonus" edges; no result
   bindings are added.

Per-request config shape::

    annotator_config = {
        "literature_cooccurrence": {
            "service_url": "https://...",     # optional override
            "min_cooccurrence": 50,           # default 50
            "include_pair_edges": True,       # default True
            "timeout_s": 5.0,                 # default 5.0
            "infores_id": "infores:...",      # default infores:literature_cooccurrence
        }
    }

Failures (timeout, 5xx, malformed response) are logged at WARNING level
and the response is returned unmodified. The framework's outer
``try``/``except`` is a backstop, not a primary error path.
"""

from __future__ import annotations

import logging
import uuid

import httpx

from gandalf.config import settings
from gandalf.search.response_annotators import register_response_annotator

logger = logging.getLogger(__name__)


_PLUGIN_KEY = "literature_cooccurrence"
_DEFAULT_MIN_COOCCURRENCE = 50
_DEFAULT_TIMEOUT_S = 5.0
_DEFAULT_INFORES = "infores:literature_cooccurrence"
_PREDICATE = "biolink:occurs_together_in_literature_with"
_NODE_ATTRIBUTE_TYPE = "biolink:occurrences_in_literature"


# Per-process cache: (service_url, frozenset(curies)) -> (node_counts, pair_counts).
# Reuses results across requests that share node sets in this worker.
_CACHE: dict[tuple[str, frozenset], tuple[dict, dict]] = {}


def _factory(cfg):
    settings_for_request = cfg.get(_PLUGIN_KEY)
    if settings_for_request is None:
        return None

    if not isinstance(settings_for_request, dict):
        raise ValueError(
            f"annotator_config[{_PLUGIN_KEY!r}] must be a dict, got "
            f"{type(settings_for_request).__name__}"
        )

    service_url = (
        settings_for_request.get("service_url")
        or settings.cooccurrence_service_url
    )
    min_cooccurrence = int(
        settings_for_request.get("min_cooccurrence", _DEFAULT_MIN_COOCCURRENCE)
    )
    include_pair_edges = bool(
        settings_for_request.get("include_pair_edges", True)
    )
    timeout_s = float(settings_for_request.get("timeout_s", _DEFAULT_TIMEOUT_S))
    infores_id = settings_for_request.get("infores_id") or _DEFAULT_INFORES

    if not service_url:
        logger.warning(
            "literature_cooccurrence annotator skipped: no service_url configured"
        )
        return None

    def _annotator(response: dict, graph) -> None:
        kg = response.get("message", {}).get("knowledge_graph", {})
        nodes = kg.get("nodes")
        if not nodes:
            return
        edges = kg.setdefault("edges", {})

        curies = sorted(nodes.keys())
        try:
            node_counts, pair_counts = _fetch_cooccurrence(
                service_url, curies, timeout_s
            )
        except Exception as exc:
            logger.warning(
                "literature_cooccurrence service call failed: %s", exc
            )
            return

        _attach_node_counts(nodes, node_counts, infores_id)

        if include_pair_edges:
            _insert_pair_edges(
                edges, pair_counts, min_cooccurrence, infores_id
            )

    return _annotator


def _fetch_cooccurrence(
    service_url: str, curies: list[str], timeout_s: float
) -> tuple[dict, dict]:
    """Fetch per-node and per-pair counts. Returns (node_counts, pair_counts).

    The service is expected to accept ``{"curies": [...]}`` and return
    ``{"node_counts": {curie: int}, "pair_counts": {"a\\tb": int}}``.
    Cached per-process by ``(service_url, frozenset(curies))``.
    """
    key = (service_url, frozenset(curies))
    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    with httpx.Client(timeout=timeout_s) as client:
        resp = client.post(service_url, json={"curies": curies})
        resp.raise_for_status()
        payload = resp.json()

    node_counts = payload.get("node_counts") or {}
    pair_counts = payload.get("pair_counts") or {}
    if not isinstance(node_counts, dict) or not isinstance(pair_counts, dict):
        raise ValueError(
            "literature_cooccurrence response must contain dict node_counts / pair_counts"
        )

    _CACHE[key] = (node_counts, pair_counts)
    return node_counts, pair_counts


def _attach_node_counts(
    nodes: dict, node_counts: dict, infores_id: str
) -> None:
    for curie, count in node_counts.items():
        node = nodes.get(curie)
        if node is None:
            continue
        attrs = node.setdefault("attributes", [])
        attrs.append(
            {
                "attribute_type_id": _NODE_ATTRIBUTE_TYPE,
                "value": count,
                "value_type_id": "linkml:Integer",
                "attribute_source": infores_id,
            }
        )


def _insert_pair_edges(
    edges: dict,
    pair_counts: dict,
    min_cooccurrence: int,
    infores_id: str,
) -> None:
    for pair_key, count in pair_counts.items():
        if count < min_cooccurrence:
            continue
        if not isinstance(pair_key, str) or "\t" not in pair_key:
            continue
        subject, obj = pair_key.split("\t", 1)
        edge_id = "litcoocc:" + uuid.uuid4().hex[:12]
        edges[edge_id] = {
            "subject": subject,
            "object": obj,
            "predicate": _PREDICATE,
            "sources": [
                {
                    "resource_id": infores_id,
                    "resource_role": "primary_knowledge_source",
                }
            ],
            "attributes": [
                {
                    "attribute_type_id": _NODE_ATTRIBUTE_TYPE,
                    "value": count,
                    "value_type_id": "linkml:Integer",
                    "attribute_source": infores_id,
                }
            ],
        }


register_response_annotator(_PLUGIN_KEY, _factory)
