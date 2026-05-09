"""Tests for the literature_cooccurrence reference annotator.

Mocks ``_fetch_cooccurrence`` so tests don't hit the network. Validates:

* Per-node attributes are attached.
* Pair edges with a count above the threshold are inserted.
* Pair edges below the threshold are skipped.
* ``include_pair_edges=False`` keeps node attribution but suppresses edges.
* Factory returns None when neither config nor settings supplies a URL.
"""

import pytest

from gandalf.plugins import literature_cooccurrence_annotator as plugin


@pytest.fixture(autouse=True)
def _clear_cache():
    plugin._CACHE.clear()
    yield
    plugin._CACHE.clear()


@pytest.fixture
def fake_fetch(monkeypatch):
    """Replace _fetch_cooccurrence with a callable returning fixed payloads."""
    calls: list[tuple] = []

    def _install(node_counts: dict, pair_counts: dict):
        def _fake(service_url, curies, timeout_s):
            calls.append((service_url, tuple(curies), timeout_s))
            return node_counts, pair_counts

        monkeypatch.setattr(plugin, "_fetch_cooccurrence", _fake)
        return calls

    return _install


def _response_with_nodes(*curies: str) -> dict:
    return {
        "message": {
            "knowledge_graph": {
                "nodes": {c: {"name": c} for c in curies},
                "edges": {},
            },
            "results": [],
            "auxiliary_graphs": {},
        },
        "logs": [],
    }


def test_factory_returns_none_when_no_service_url(monkeypatch):
    monkeypatch.setattr(plugin.settings, "cooccurrence_service_url", "")
    ann = plugin._factory({"literature_cooccurrence": {}})
    assert ann is None


def test_factory_returns_none_when_key_missing():
    ann = plugin._factory({})
    assert ann is None


def test_attaches_node_counts(fake_fetch):
    fake_fetch(
        node_counts={"CURIE:A": 100, "CURIE:B": 200},
        pair_counts={},
    )
    ann = plugin._factory(
        {"literature_cooccurrence": {"service_url": "http://stub"}}
    )
    response = _response_with_nodes("CURIE:A", "CURIE:B")
    ann(response, None)

    nodes = response["message"]["knowledge_graph"]["nodes"]
    a_attrs = nodes["CURIE:A"]["attributes"]
    b_attrs = nodes["CURIE:B"]["attributes"]
    assert a_attrs[0]["attribute_type_id"] == "biolink:occurrences_in_literature"
    assert a_attrs[0]["value"] == 100
    assert b_attrs[0]["value"] == 200


def test_inserts_pair_edges_above_threshold(fake_fetch):
    fake_fetch(
        node_counts={"CURIE:A": 1, "CURIE:B": 1, "CURIE:C": 1},
        pair_counts={
            "CURIE:A\tCURIE:B": 100,  # above default 50
            "CURIE:A\tCURIE:C": 10,  # below default 50
        },
    )
    ann = plugin._factory(
        {"literature_cooccurrence": {"service_url": "http://stub"}}
    )
    response = _response_with_nodes("CURIE:A", "CURIE:B", "CURIE:C")
    ann(response, None)

    edges = response["message"]["knowledge_graph"]["edges"]
    pair_edges = [e for e in edges.values() if e["predicate"] == plugin._PREDICATE]
    assert len(pair_edges) == 1
    assert pair_edges[0]["subject"] == "CURIE:A"
    assert pair_edges[0]["object"] == "CURIE:B"
    # Edge IDs are namespaced.
    assert all(eid.startswith("litcoocc:") for eid in edges)


def test_min_cooccurrence_setting_overrides_default(fake_fetch):
    fake_fetch(
        node_counts={"CURIE:A": 1, "CURIE:B": 1},
        pair_counts={"CURIE:A\tCURIE:B": 5},
    )
    ann = plugin._factory(
        {
            "literature_cooccurrence": {
                "service_url": "http://stub",
                "min_cooccurrence": 5,
            }
        }
    )
    response = _response_with_nodes("CURIE:A", "CURIE:B")
    ann(response, None)

    edges = response["message"]["knowledge_graph"]["edges"]
    assert len(edges) == 1


def test_include_pair_edges_false_skips_edges(fake_fetch):
    fake_fetch(
        node_counts={"CURIE:A": 1, "CURIE:B": 1},
        pair_counts={"CURIE:A\tCURIE:B": 1000},
    )
    ann = plugin._factory(
        {
            "literature_cooccurrence": {
                "service_url": "http://stub",
                "include_pair_edges": False,
            }
        }
    )
    response = _response_with_nodes("CURIE:A", "CURIE:B")
    ann(response, None)

    edges = response["message"]["knowledge_graph"]["edges"]
    assert edges == {}
    # Node counts still attached.
    assert response["message"]["knowledge_graph"]["nodes"]["CURIE:A"]["attributes"]


def test_service_failure_is_swallowed(monkeypatch):
    def _boom(service_url, curies, timeout_s):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(plugin, "_fetch_cooccurrence", _boom)
    ann = plugin._factory(
        {"literature_cooccurrence": {"service_url": "http://stub"}}
    )
    response = _response_with_nodes("CURIE:A")
    # Plugin must not raise; response must remain unmodified.
    ann(response, None)
    nodes = response["message"]["knowledge_graph"]["nodes"]
    assert "attributes" not in nodes["CURIE:A"]


def test_non_dict_settings_raises(monkeypatch):
    monkeypatch.setattr(plugin.settings, "cooccurrence_service_url", "http://stub")
    with pytest.raises(ValueError):
        plugin._factory({"literature_cooccurrence": "not a dict"})
