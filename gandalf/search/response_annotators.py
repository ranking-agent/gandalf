"""Response-annotator registry for post-lookup response mutation.

Annotators run after ``lookup()`` returns a TRAPI response. Each annotator is
a closure ``(response, graph) -> None`` that mutates the response in place —
typically to attach extra node attributes, add bonus edges to the knowledge
graph, or otherwise enrich the response with data from external sources.

Registration lives in plugin modules under ``gandalf.plugins``. This module
owns the type aliases, the registry list, and the build helper used by
``gandalf.annotate.annotate_response``.

Annotators are opt-in per request: each plugin's factory pulls its own key
out of the per-request ``annotator_config`` dict and returns ``None`` when
the key is missing — exactly the same pattern as ``NodeFilter`` factories.
"""

from typing import Any, Callable, Optional

ResponseAnnotator = Callable[[dict, Any], None]
ResponseAnnotatorFactory = Callable[[dict], Optional[ResponseAnnotator]]


_REGISTRY: list[tuple[str, ResponseAnnotatorFactory]] = []


def register_response_annotator(
    name: str, factory: ResponseAnnotatorFactory
) -> None:
    """Register a ResponseAnnotator factory under a name.

    A factory takes the per-request ``annotator_config`` dict and returns
    either a closure (the annotator) or None when the annotator is inactive
    for this request (its config key is missing).
    """
    _REGISTRY.append((name, factory))


def registered_annotator_names() -> list[str]:
    """Names of all registered annotator factories, in registration order."""
    return [name for name, _ in _REGISTRY]


def build_response_annotators(
    cfg: dict,
) -> list[tuple[str, ResponseAnnotator]]:
    """Build the list of active (name, ResponseAnnotator) pairs for one request.

    Walks the registry in registration order and asks each factory for an
    annotator. Inactive factories return ``None`` and are skipped.
    """
    active: list[tuple[str, ResponseAnnotator]] = []
    for name, factory in _REGISTRY:
        ann = factory(cfg)
        if ann is not None:
            active.append((name, ann))
    return active
