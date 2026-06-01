"""Tests for query-graph None-field normalization.

Regression coverage for the default (non-validating) request path: a client
that sends an optional field explicitly as ``null`` (e.g. ``"ids": null`` on an
unpinned node) used to leave the key present with a ``None`` value, which broke
the ``node.get("ids", [])`` idiom downstream (``len(None)`` -> 500). See
``gandalf.request_validation.normalize_query_graph``.
"""

from gandalf.request_validation import normalize_query_graph


class TestNormalizeQueryGraph:
    """Tests for normalize_query_graph."""

    def test_none_node_ids_dropped(self):
        # Pinned -> unpinned: the unpinned object node arrives with ids=None.
        qg = {
            "nodes": {
                "n0": {"ids": ["MONDO:0004975"], "categories": None},
                "n1": {"ids": None, "categories": ["biolink:Gene"]},
            },
            "edges": {
                "e0": {
                    "subject": "n0",
                    "object": "n1",
                    "predicates": ["biolink:affects"],
                    "qualifier_constraints": None,
                }
            },
        }
        normalize_query_graph(qg)

        # None-valued optional fields are removed entirely, not set to [].
        assert "ids" not in qg["nodes"]["n1"]
        assert "categories" not in qg["nodes"]["n0"]
        assert "qualifier_constraints" not in qg["edges"]["e0"]

        # Present, non-None values are untouched.
        assert qg["nodes"]["n0"]["ids"] == ["MONDO:0004975"]
        assert qg["nodes"]["n1"]["categories"] == ["biolink:Gene"]
        assert qg["edges"]["e0"]["predicates"] == ["biolink:affects"]

    def test_absent_fields_unaffected(self):
        qg = {
            "nodes": {"n0": {"categories": ["biolink:Disease"]}},
            "edges": {"e0": {"subject": "n0", "object": "n0"}},
        }
        normalize_query_graph(qg)
        assert "ids" not in qg["nodes"]["n0"]
        assert qg["nodes"]["n0"]["categories"] == ["biolink:Disease"]

    def test_empty_list_preserved(self):
        # An explicit empty list is a valid, distinct value -- keep it.
        qg = {"nodes": {"n0": {"ids": []}}, "edges": {}}
        normalize_query_graph(qg)
        assert qg["nodes"]["n0"]["ids"] == []

    def test_missing_containers_no_error(self):
        normalize_query_graph({})  # should not raise
        normalize_query_graph({"nodes": None, "edges": None})  # should not raise
