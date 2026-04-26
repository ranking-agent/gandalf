"""Tests for the in-repo plugin system: enricher registry, built-in plugins,
and the LMDB-backed ``TraversalMetadataStore``.

Confirms that:
- ``register_node_enricher`` adds to the registry and ``run_enrichers``
  invokes each in registration order, skipping namespaces that already
  hold data (the load-from-disk path).
- Enrichers writing to the store do not leak into the TRAPI attribute path
  (``graph.get_node_property(..., "attributes")``).
- The package import populates both the filter and enricher registries
  with the expected built-in names.
- The store round-trips through ``save_to`` / ``open_readonly``.
"""

from types import SimpleNamespace

import pytest

from gandalf.plugins import enrichers as enr
from gandalf.plugins.traversal_metadata_store import TraversalMetadataStore
from gandalf.search import node_filters as nf


@pytest.fixture
def isolated_enricher_registry():
    saved = list(enr._ENRICHERS)
    enr._ENRICHERS.clear()
    try:
        yield
    finally:
        enr._ENRICHERS.clear()
        enr._ENRICHERS.extend(saved)


@pytest.fixture
def store(tmp_path):
    """A writable TraversalMetadataStore anchored at a tmp_path subdirectory."""
    s = TraversalMetadataStore.open_writable(tmp_path / "store.lmdb")
    try:
        yield s
    finally:
        s.close()


def test_register_node_enricher_adds_in_order(isolated_enricher_registry):
    enr.register_node_enricher("a", lambda g: None)
    enr.register_node_enricher("b", lambda g: None)
    assert enr.registered_enricher_names() == ["a", "b"]


def test_run_enrichers_invokes_in_registration_order(
    isolated_enricher_registry, store
):
    calls = []
    enr.register_node_enricher("first", lambda g: calls.append("first"))
    enr.register_node_enricher("second", lambda g: calls.append("second"))

    fake_graph = SimpleNamespace(traversal_metadata=store)
    enr.run_enrichers(fake_graph)
    assert calls == ["first", "second"]


def test_run_enrichers_skips_namespaces_already_in_store(
    isolated_enricher_registry, store
):
    """Already-populated namespaces (e.g. loaded from disk) must not be recomputed."""
    calls = []

    def _record(name):
        def _enrich(g):
            calls.append(name)
            g.traversal_metadata.put(name, 0, "computed")
        return _enrich

    enr.register_node_enricher("alpha", _record("alpha"))
    enr.register_node_enricher("beta", _record("beta"))

    # Simulate "alpha" already loaded from disk.
    store.put("alpha", 0, "from_disk")

    fake_graph = SimpleNamespace(traversal_metadata=store)
    enr.run_enrichers(fake_graph)

    assert calls == ["beta"]
    assert store.get("alpha", 0) == "from_disk"
    assert store.get("beta", 0) == "computed"


def test_run_enrichers_warns_on_readonly_with_missing_namespace(
    isolated_enricher_registry, tmp_path, caplog
):
    """A readonly store + missing namespace logs a warning and does not crash."""
    # Build & save a store containing only "alpha".
    src = TraversalMetadataStore.open_writable(tmp_path / "src.lmdb")
    src.put("alpha", 0, "v")
    src.close()

    ro = TraversalMetadataStore.open_readonly(tmp_path / "src.lmdb")
    enr.register_node_enricher(
        "beta", lambda g: pytest.fail("should not run on readonly store")
    )

    fake_graph = SimpleNamespace(traversal_metadata=ro)
    with caplog.at_level("WARNING"):
        enr.run_enrichers(fake_graph)

    assert any("beta" in msg for msg in caplog.messages)
    ro.close()


def test_enricher_writes_to_store_without_touching_attributes(
    isolated_enricher_registry, store
):
    """Enrichers must write only to traversal_metadata.

    A regression in this test means a plugin's data could leak into the TRAPI
    response path.
    """

    class FakeGraph:
        def __init__(self, store):
            self.traversal_metadata = store
            # Stand-in for node_properties; mimics the read path used by
            # response enrichment.
            self._attrs_by_node = {0: [{"original_attribute_name": "name", "value": "x"}]}

        def get_node_property(self, node_idx, key, default=None):
            if key == "attributes":
                return list(self._attrs_by_node.get(node_idx, []))
            return default

    def _enrich(graph):
        graph.traversal_metadata.put("my_metric", 0, 42)

    enr.register_node_enricher("my_metric", _enrich)

    g = FakeGraph(store)
    enr.run_enrichers(g)

    assert store.get("my_metric", 0) == 42
    # The TRAPI attribute list for node 0 is unchanged.
    assert g.get_node_property(0, "attributes") == [
        {"original_attribute_name": "name", "value": "x"}
    ]


def test_builtin_plugins_register_filter_names_on_import():
    """Importing gandalf.plugins must register the two built-in filters."""
    import gandalf.plugins  # noqa: F401  ensure plugins package is loaded

    names = nf.registered_filter_names()
    assert "max_node_degree" in names
    assert "min_information_content" in names


# ---------------------------------------------------------------------------
# TraversalMetadataStore round-trip and shape
# ---------------------------------------------------------------------------


def test_store_put_get_round_trip(store):
    store.put("ns", 0, "hello")
    store.put("ns", 1, [1, 2, 3])
    store.put("ns", None, {"shape": [10, 8]})  # plugin-global

    assert store.get("ns", 0) == "hello"
    assert store.get("ns", 1) == [1, 2, 3]
    assert store.get("ns", None) == {"shape": [10, 8]}
    assert store.get("ns", 999, default="missing") == "missing"


def test_store_namespace_membership(store):
    assert "ns" not in store
    store.put("ns", 0, 1)
    assert "ns" in store
    assert "other" not in store


def test_store_iter_namespace(store):
    store.put("ns", 0, "a")
    store.put("ns", 5, "b")
    store.put("ns", None, "global")
    store.put("other_ns", 0, "shouldnt-show")

    pairs = sorted(store.iter_namespace("ns"), key=lambda kv: (kv[0] is not None, kv[0]))
    assert pairs == [(None, "global"), (0, "a"), (5, "b")]


def test_store_save_to_and_open_readonly(tmp_path):
    """A writable store, saved into a graph directory, reopens as readonly."""
    src = TraversalMetadataStore.open_writable(tmp_path / "src.lmdb")
    src.put("plugin_a", 0, [1, 2, 3])
    src.put("plugin_a", 1, [4, 5, 6])
    src.put("plugin_b", None, "hello")

    graph_dir = tmp_path / "graph"
    graph_dir.mkdir()
    src.save_to(graph_dir)
    src.close()

    assert (graph_dir / "traversal_metadata.lmdb").exists()

    ro = TraversalMetadataStore.open_readonly(graph_dir / "traversal_metadata.lmdb")
    try:
        assert "plugin_a" in ro
        assert "plugin_b" in ro
        assert "plugin_c" not in ro
        assert ro.get("plugin_a", 0) == [1, 2, 3]
        assert ro.get("plugin_a", 1) == [4, 5, 6]
        assert ro.get("plugin_b", None) == "hello"
        assert ro.is_readonly is True
    finally:
        ro.close()


def test_store_readonly_rejects_writes(tmp_path):
    src = TraversalMetadataStore.open_writable(tmp_path / "src.lmdb")
    src.put("ns", 0, "v")
    src.close()

    ro = TraversalMetadataStore.open_readonly(tmp_path / "src.lmdb")
    try:
        with pytest.raises(RuntimeError):
            ro.put("ns", 1, "v2")
    finally:
        ro.close()


def test_store_unopened_acts_empty():
    """A lazy writable store reports empty until something is written."""
    s = TraversalMetadataStore.open_writable()  # no path -> lazy temp dir
    try:
        assert "anything" not in s
        assert s.get("anything", 0, default=None) is None
        assert s.namespaces() == set()
    finally:
        s.close()


def test_store_put_many_bulk_writes(store):
    n = store.put_many("bulk", ((i, i * i) for i in range(1000)))
    assert n == 1000
    assert store.get("bulk", 0) == 0
    assert store.get("bulk", 999) == 999 * 999
    assert "bulk" in store
