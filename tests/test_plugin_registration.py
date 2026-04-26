"""Tests for the in-repo plugin system: enricher registry and built-in plugins.

Confirms that:
- ``register_node_enricher`` adds to the registry and ``run_enrichers`` invokes
  each in registration order.
- Enrichers writing to ``graph.traversal_metadata`` do not leak into the
  TRAPI-attribute path (``graph.get_node_property(..., "attributes")``).
- The package import populates both the filter and enricher registries with
  the expected built-in names.
"""

from types import SimpleNamespace

import pytest

from gandalf.plugins import enrichers as enr
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


def test_register_node_enricher_adds_in_order(isolated_enricher_registry):
    enr.register_node_enricher("a", lambda g: None)
    enr.register_node_enricher("b", lambda g: None)
    assert enr.registered_enricher_names() == ["a", "b"]


def test_run_enrichers_invokes_in_registration_order(isolated_enricher_registry):
    calls = []
    enr.register_node_enricher("first", lambda g: calls.append("first"))
    enr.register_node_enricher("second", lambda g: calls.append("second"))

    fake_graph = SimpleNamespace(traversal_metadata={})
    enr.run_enrichers(fake_graph)
    assert calls == ["first", "second"]


def test_enricher_writes_to_traversal_metadata_without_touching_attributes(
    isolated_enricher_registry,
):
    """Enrichers must write only to graph.traversal_metadata.

    A regression in this test means a plugin's data could leak into the TRAPI
    response path.
    """
    captured_attrs = []

    class FakeGraph:
        def __init__(self):
            self.traversal_metadata: dict = {}
            # Stand-in for node_properties; mimics the read path used by
            # response enrichment.
            self._attrs_by_node = {0: [{"original_attribute_name": "name", "value": "x"}]}

        def get_node_property(self, node_idx, key, default=None):
            captured_attrs.append((node_idx, key))
            if key == "attributes":
                return list(self._attrs_by_node.get(node_idx, []))
            return default

    def _enrich(graph):
        graph.traversal_metadata["my_metric"] = {0: 42}

    enr.register_node_enricher("my_metric", _enrich)

    g = FakeGraph()
    enr.run_enrichers(g)

    assert g.traversal_metadata == {"my_metric": {0: 42}}
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
