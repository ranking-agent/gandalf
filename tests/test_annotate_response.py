"""Tests for the annotate_response runner.

Confirms that ``annotate_response``:

* No-ops when no annotators are registered or activated.
* Calls active annotators with ``(response, graph)`` and lets them
  mutate node attributes / KG edges.
* Emits a TRAPI debug log entry per successful annotator.
* Catches per-annotator exceptions, logs an error entry, and continues
  running subsequent annotators.
"""

import pytest

from gandalf import annotate_response
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


def _empty_response() -> dict:
    return {
        "message": {
            "knowledge_graph": {"nodes": {}, "edges": {}},
            "results": [],
            "auxiliary_graphs": {},
        },
        "logs": [],
    }


def test_annotate_response_with_empty_config_is_noop(isolated_registry):
    response = _empty_response()
    out = annotate_response(response, None, {})
    assert out is response
    assert response["logs"] == []


def test_annotate_response_no_active_annotators_is_noop(isolated_registry):
    ra.register_response_annotator("nope", lambda cfg: None)
    response = _empty_response()
    annotate_response(response, None, {"nope": {}})
    assert response["logs"] == []


def test_annotate_response_runs_noop_annotator_and_logs_debug(isolated_registry):
    def _factory(cfg):
        if cfg.get("noop") is None:
            return None
        return lambda response, graph: None

    ra.register_response_annotator("noop", _factory)
    response = _empty_response()
    annotate_response(response, None, {"noop": {}})

    assert len(response["logs"]) == 1
    entry = response["logs"][0]
    assert entry["level"] == "DEBUG"
    assert "noop" in entry["message"]


def test_annotator_can_mutate_node_attributes(isolated_registry):
    def _factory(cfg):
        if cfg.get("attach") is None:
            return None

        def _ann(response, graph):
            nodes = response["message"]["knowledge_graph"]["nodes"]
            for node in nodes.values():
                node.setdefault("attributes", []).append(
                    {"attribute_type_id": "biolink:foo", "value": 1}
                )

        return _ann

    ra.register_response_annotator("attach", _factory)
    response = _empty_response()
    response["message"]["knowledge_graph"]["nodes"] = {
        "CURIE:1": {"name": "node 1"},
        "CURIE:2": {"name": "node 2", "attributes": []},
    }
    annotate_response(response, None, {"attach": {}})

    nodes = response["message"]["knowledge_graph"]["nodes"]
    assert nodes["CURIE:1"]["attributes"][0]["attribute_type_id"] == "biolink:foo"
    assert nodes["CURIE:2"]["attributes"][0]["value"] == 1


def test_annotator_can_add_edges_without_touching_results(isolated_registry):
    def _factory(cfg):
        if cfg.get("add_edge") is None:
            return None

        def _ann(response, graph):
            edges = response["message"]["knowledge_graph"]["edges"]
            edges["bonus:1"] = {
                "subject": "A",
                "object": "B",
                "predicate": "biolink:related_to",
            }

        return _ann

    ra.register_response_annotator("add_edge", _factory)
    response = _empty_response()
    response["message"]["results"] = [{"node_bindings": {}, "analyses": []}]
    annotate_response(response, None, {"add_edge": {}})

    edges = response["message"]["knowledge_graph"]["edges"]
    assert "bonus:1" in edges
    # Results are untouched.
    assert response["message"]["results"] == [{"node_bindings": {}, "analyses": []}]


def test_failing_annotator_is_isolated_and_subsequent_annotators_run(
    isolated_registry,
):
    def _factory_boom(cfg):
        if cfg.get("boom") is None:
            return None

        def _ann(response, graph):
            raise RuntimeError("kaboom")

        return _ann

    def _factory_after(cfg):
        if cfg.get("after") is None:
            return None

        def _ann(response, graph):
            response["after_ran"] = True

        return _ann

    ra.register_response_annotator("boom", _factory_boom)
    ra.register_response_annotator("after", _factory_after)

    response = _empty_response()
    annotate_response(response, None, {"boom": {}, "after": {}})

    # The successor ran.
    assert response["after_ran"] is True
    # Two log entries: one ERROR for boom, one DEBUG for after.
    levels = [e["level"] for e in response["logs"]]
    assert "ERROR" in levels
    assert "DEBUG" in levels
    err_entry = next(e for e in response["logs"] if e["level"] == "ERROR")
    assert "boom" in err_entry["message"]
    assert err_entry["code"] == "AnnotatorError"
