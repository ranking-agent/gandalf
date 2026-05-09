"""Unit tests for the NodeFilter registry.

Exercises ``register_node_filter`` / ``build_node_filters`` / ``apply_node_filters``
in isolation, without depending on a concrete graph or the built-in plugins.
Each test stashes and restores the global registry to keep tests independent.
"""

import pytest

from gandalf.search import node_filters as nf


@pytest.fixture
def isolated_registry():
    saved = list(nf._REGISTRY)
    nf._REGISTRY.clear()
    try:
        yield
    finally:
        nf._REGISTRY.clear()
        nf._REGISTRY.extend(saved)


def test_build_node_filters_empty_when_no_factories_registered(isolated_registry):
    assert nf.build_node_filters({}) == []
    assert nf.build_node_filters({"anything": 1}) == []


def test_build_node_filters_skips_inactive_factories(isolated_registry):
    def _factory(cfg):
        return None  # always inactive

    nf.register_node_filter("never", _factory)
    assert nf.build_node_filters({"never": 5}) == []


def test_build_node_filters_returns_active_filter(isolated_registry):
    def _factory(cfg):
        threshold = cfg.get("min_x")
        if threshold is None:
            return None
        return lambda graph, node_idx: node_idx >= threshold

    nf.register_node_filter("min_x", _factory)
    filters = nf.build_node_filters({"min_x": 3})
    assert len(filters) == 1
    assert filters[0](None, 5) is True
    assert filters[0](None, 1) is False


def test_build_node_filters_preserves_registration_order(isolated_registry):
    calls = []

    def _make(name):
        def _factory(cfg):
            def _filter(graph, node_idx):
                calls.append(name)
                return True
            return _filter
        return _factory

    nf.register_node_filter("a", _make("a"))
    nf.register_node_filter("b", _make("b"))
    nf.register_node_filter("c", _make("c"))

    filters = nf.build_node_filters({"a": 1, "b": 1, "c": 1})
    assert nf.apply_node_filters(filters, None, 0) is True
    assert calls == ["a", "b", "c"]


def test_apply_node_filters_short_circuits(isolated_registry):
    counter = {"calls": 0}

    def _spy(graph, node_idx):
        counter["calls"] += 1
        return True

    def _reject(graph, node_idx):
        return False

    # _reject runs first, so _spy must never be called
    assert nf.apply_node_filters([_reject, _spy], None, 0) is False
    assert counter["calls"] == 0


def test_apply_node_filters_empty_list_is_vacuously_true(isolated_registry):
    assert nf.apply_node_filters([], None, 0) is True


def test_registered_filter_names_returns_registration_order(isolated_registry):
    nf.register_node_filter("first", lambda cfg: None)
    nf.register_node_filter("second", lambda cfg: None)
    assert nf.registered_filter_names() == ["first", "second"]
