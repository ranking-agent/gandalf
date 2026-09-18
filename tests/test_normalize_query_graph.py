"""Tests for query-graph null / empty-container normalization.

Regression coverage for the default (non-validating) request path: a client
that sends an optional field explicitly as ``null`` (e.g. ``"ids": null`` on an
unpinned node) used to leave the key present with a ``None`` value, which broke
the ``node.get("ids", [])`` idiom downstream (``len(None)`` -> 500). See
``gandalf.request_validation.normalize_query_graph``.

The same pass also drops empty containers for the properties TRAPI 2.0 gives a
``minItems`` / ``minProperties`` of 1, because the query graph is echoed back
in the response and would otherwise be invalid there.
"""

import pytest

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

    @pytest.mark.parametrize("prop", ["ids", "categories", "member_ids", "constraints"])
    def test_empty_qnode_list_dropped(self, prop):
        """TRAPI 2.0 gives each of these a minItems of 1.

        The query graph is echoed back in the response, so an empty list on
        the way in would be an invalid message.query_graph on the way out.
        """
        qg = {"nodes": {"n0": {prop: []}}, "edges": {}}
        normalize_query_graph(qg)
        assert prop not in qg["nodes"]["n0"]

    def test_empty_qedge_predicates_dropped(self):
        qg = {
            "nodes": {"n0": {}, "n1": {}},
            "edges": {"e0": {"subject": "n0", "object": "n1", "predicates": []}},
        }
        normalize_query_graph(qg)
        assert "predicates" not in qg["edges"]["e0"]

    def test_empty_qedge_constraints_dropped(self):
        """QEdgeConstraints has a minProperties of 1, its lists a minItems of 1."""
        qg = {
            "nodes": {"n0": {}, "n1": {}},
            "edges": {
                "e0": {
                    "subject": "n0",
                    "object": "n1",
                    "constraints": {"qualifiers": [], "attributes": []},
                }
            },
        }
        normalize_query_graph(qg)
        assert "constraints" not in qg["edges"]["e0"]

    def test_empty_qpath_constraints_dropped(self):
        qg = {
            "nodes": {"n0": {}, "n1": {}},
            "paths": {
                "p0": {
                    "subject": "n0",
                    "object": "n1",
                    "predicates": [],
                    "constraints": [],
                }
            },
        }
        normalize_query_graph(qg)
        assert "predicates" not in qg["paths"]["p0"]
        assert "constraints" not in qg["paths"]["p0"]

    def test_other_empty_values_preserved(self):
        """Only the properties 2.0 forbids empty are dropped."""
        qg = {"nodes": {"n0": {"is_set": False, "set_interpretation": ""}}, "edges": {}}
        normalize_query_graph(qg)
        assert qg["nodes"]["n0"] == {"is_set": False, "set_interpretation": ""}

    def test_missing_containers_no_error(self):
        normalize_query_graph({})  # should not raise
        normalize_query_graph({"nodes": None, "edges": None})  # should not raise
