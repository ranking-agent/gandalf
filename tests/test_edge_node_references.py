"""Tests for query-graph edge/node reference validation."""

import pytest
from fastapi import HTTPException

from gandalf.request_validation import validate_edge_node_references


class TestEdgeNodeReferenceValidation:
    """Tests for validate_edge_node_references."""

    def test_valid_references_pass(self):
        qg = {
            "nodes": {"n0": {"ids": ["A"]}, "n1": {"categories": ["biolink:Gene"]}},
            "edges": {"e0": {"subject": "n0", "object": "n1"}},
        }
        validate_edge_node_references(qg)  # should not raise

    def test_no_edges_passes(self):
        qg = {"nodes": {"n0": {"ids": ["A"]}}}
        validate_edge_node_references(qg)  # should not raise

    def test_missing_subject_rejected(self):
        qg = {
            "nodes": {"n1": {"categories": ["biolink:Gene"]}},
            "edges": {"e0": {"subject": "n0", "object": "n1"}},
        }
        with pytest.raises(HTTPException) as exc_info:
            validate_edge_node_references(qg)
        assert exc_info.value.status_code == 400
        assert "subject" in str(exc_info.value.detail)
        assert "n0" in str(exc_info.value.detail)

    def test_missing_object_rejected(self):
        qg = {
            "nodes": {"n0": {"ids": ["A"]}},
            "edges": {"e0": {"subject": "n0", "object": "n1"}},
        }
        with pytest.raises(HTTPException) as exc_info:
            validate_edge_node_references(qg)
        assert exc_info.value.status_code == 400
        assert "object" in str(exc_info.value.detail)
        assert "n1" in str(exc_info.value.detail)

    def test_none_reference_rejected(self):
        qg = {
            "nodes": {"n0": {"ids": ["A"]}},
            "edges": {"e0": {"subject": "n0"}},
        }
        with pytest.raises(HTTPException) as exc_info:
            validate_edge_node_references(qg)
        assert exc_info.value.status_code == 400
