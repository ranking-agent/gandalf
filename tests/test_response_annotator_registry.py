"""Unit tests for the ResponseAnnotator registry.

Mirrors ``test_node_filter_registry.py``: exercises
``register_response_annotator`` / ``build_response_annotators`` /
``registered_annotator_names`` in isolation, without depending on a
concrete graph or the built-in plugins. Each test stashes and restores
the global registry to keep tests independent.
"""

import pytest

from gandalf.search import response_annotators as ra


@pytest.fixture
def isolated_registry():
    saved = list(ra._REGISTRY)
    ra._REGISTRY.clear()
    try:
        yield
    finally:
        ra._REGISTRY.clear()
        ra._REGISTRY.extend(saved)


def test_build_response_annotators_empty_when_no_factories(isolated_registry):
    assert ra.build_response_annotators({}) == []
    assert ra.build_response_annotators({"anything": {}}) == []


def test_build_response_annotators_skips_inactive_factories(isolated_registry):
    def _factory(cfg):
        return None

    ra.register_response_annotator("never", _factory)
    assert ra.build_response_annotators({"never": {"x": 1}}) == []


def test_build_response_annotators_returns_active_annotator(isolated_registry):
    def _factory(cfg):
        if cfg.get("active") is None:
            return None

        def _ann(response, graph):
            response["touched"] = True

        return _ann

    ra.register_response_annotator("active", _factory)
    pairs = ra.build_response_annotators({"active": {}})
    assert len(pairs) == 1
    name, ann = pairs[0]
    assert name == "active"
    response: dict = {}
    ann(response, None)
    assert response["touched"] is True


def test_build_response_annotators_preserves_registration_order(isolated_registry):
    def _make(name):
        def _factory(cfg):
            def _ann(response, graph):
                response.setdefault("order", []).append(name)

            return _ann

        return _factory

    ra.register_response_annotator("a", _make("a"))
    ra.register_response_annotator("b", _make("b"))
    ra.register_response_annotator("c", _make("c"))

    pairs = ra.build_response_annotators({"a": {}, "b": {}, "c": {}})
    response: dict = {}
    for _, ann in pairs:
        ann(response, None)
    assert response["order"] == ["a", "b", "c"]


def test_registered_annotator_names_returns_registration_order(isolated_registry):
    ra.register_response_annotator("first", lambda cfg: None)
    ra.register_response_annotator("second", lambda cfg: None)
    assert ra.registered_annotator_names() == ["first", "second"]


def test_builtin_literature_cooccurrence_registers_on_import():
    """Importing gandalf.plugins must register the reference annotator."""
    import gandalf.plugins  # noqa: F401

    assert "literature_cooccurrence" in ra.registered_annotator_names()
