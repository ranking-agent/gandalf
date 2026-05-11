"""Tests for the gandalf request profiler."""

import json

import pytest
from fastapi.testclient import TestClient

from gandalf.profiler import (
    NullProfiler,
    Profiler,
    current_profiler,
    set_profiler,
)
from gandalf.search import lookup
from tests.search_fixtures import graph  # noqa: F401


_ONE_HOP_BOTH_PINNED = {
    "message": {
        "query_graph": {
            "nodes": {
                "n0": {"ids": ["CHEBI:6801"]},
                "n1": {"ids": ["MONDO:0005148"]},
            },
            "edges": {
                "e0": {
                    "subject": "n0",
                    "object": "n1",
                    "predicates": ["biolink:treats"],
                },
            },
        },
    },
}


_ONE_HOP_PINNED_TO_GENE = {
    "message": {
        "query_graph": {
            "nodes": {
                "n0": {"ids": ["CHEBI:6801"]},
                "n1": {"categories": ["biolink:Gene"]},
            },
            "edges": {
                "e0": {
                    "subject": "n0",
                    "object": "n1",
                    "predicates": ["biolink:affects"],
                },
            },
        },
    },
}


def _profile_summary(response):
    """Return the parsed ProfileSummary tree from a response, or None."""
    summaries = [
        e for e in response.get("logs", []) if e.get("code") == "ProfileSummary"
    ]
    if not summaries:
        return None
    return json.loads(summaries[0]["message"])


class TestNullProfiler:
    """A NullProfiler is the default and is a true no-op."""

    def test_default_current_profiler_is_null(self):
        assert isinstance(current_profiler(), NullProfiler)
        assert current_profiler().enabled is False

    def test_null_methods_do_nothing(self):
        prof = NullProfiler()
        with prof.stage("a"):
            prof.event("ev", x=1)
            prof.add_metric("k", 1)
            prof.incr("c")
            with prof.lmdb_call("get", 5):
                pass
        assert prof.to_log_entries() == []
        assert prof.to_dict() == {}


class TestProfilerCore:
    """Direct tests of the Profiler class (no graph needed)."""

    def test_stage_records_duration_and_nesting(self):
        prof = Profiler(root_name="root")
        with prof.stage("outer"):
            with prof.stage("inner"):
                prof.add_metric("hits", 3)
        tree = prof.to_dict()
        assert tree["name"] == "root"
        assert len(tree["children"]) == 1
        outer = tree["children"][0]
        assert outer["name"] == "outer"
        assert outer["duration_ms"] >= 0
        assert len(outer["children"]) == 1
        inner = outer["children"][0]
        assert inner["name"] == "inner"
        assert inner["metrics"]["hits"] == 3
        # Nested stage cannot exceed its parent
        assert inner["duration_ms"] <= outer["duration_ms"] + 1.0

    def test_lmdb_call_aggregates_into_root(self):
        prof = Profiler()
        with prof.stage("step"):
            with prof.lmdb_call("get_batch", 12):
                pass
            with prof.lmdb_call("get_batch", 8):
                pass
            with prof.lmdb_call("get", 1):
                pass
        tree = prof.to_dict()
        lmdb = tree["lmdb"]
        assert lmdb["calls"] == 3
        assert lmdb["total_keys"] == 21
        assert sorted(lmdb["batch_sizes"]) == [1, 8, 12]
        assert lmdb["by_kind"]["get_batch"]["calls"] == 2
        assert lmdb["by_kind"]["get"]["calls"] == 1

    def test_to_log_entries_contains_summary(self):
        prof = Profiler()
        with prof.stage("alpha", n=3):
            with prof.stage("beta"):
                pass
        entries = prof.to_log_entries()
        codes = [e["code"] for e in entries]
        assert "ProfileSummary" in codes
        assert codes.count("ProfileSummary") == 1
        # Each non-root stage gets a ProfileStage entry
        assert codes.count("ProfileStage") == 2
        summary = next(e for e in entries if e["code"] == "ProfileSummary")
        tree = json.loads(summary["message"])
        assert tree["name"] == "lookup"  # default root_name
        assert tree["children"][0]["name"] == "alpha"
        assert tree["children"][0]["fields"]["n"] == 3

    def test_set_profiler_contextvar_scoping(self):
        outer = current_profiler()
        prof = Profiler()
        with set_profiler(prof):
            assert current_profiler() is prof
        assert current_profiler() is outer


class TestProfilerInLookup:
    """End-to-end tests through ``lookup()``."""

    def test_no_profile_emits_no_profile_entries(self, graph, bmt):
        response = lookup(graph, _ONE_HOP_BOTH_PINNED, bmt=bmt)
        codes = {e.get("code") for e in response.get("logs", [])}
        assert "ProfileStage" not in codes
        assert "ProfileSummary" not in codes

    def test_profile_kwarg_emits_summary_with_expected_stages(self, graph, bmt):
        response = lookup(
            graph, _ONE_HOP_BOTH_PINNED, bmt=bmt, profile=True
        )
        tree = _profile_summary(response)
        assert tree is not None, "Expected a ProfileSummary log entry"
        assert tree["name"] == "lookup"
        child_names = {c["name"] for c in tree["children"]}
        assert "bmt_init" in child_names
        assert "qedge" in child_names
        assert "reconstruct" in child_names
        # build_response only runs when there are paths
        assert "build_response" in child_names

    def test_profile_response_shape_unchanged(self, graph, bmt):
        baseline = lookup(graph, _ONE_HOP_PINNED_TO_GENE, bmt=bmt)
        profiled = lookup(
            graph, _ONE_HOP_PINNED_TO_GENE, bmt=bmt, profile=True
        )
        # Same number of results, same node bindings — only logs differ.
        assert len(profiled["message"]["results"]) == len(
            baseline["message"]["results"]
        )
        baseline_ids = sorted(
            r["node_bindings"]["n1"][0]["id"] for r in baseline["message"]["results"]
        )
        profiled_ids = sorted(
            r["node_bindings"]["n1"][0]["id"] for r in profiled["message"]["results"]
        )
        assert baseline_ids == profiled_ids

    def test_qedge_contains_query_subtree(self, graph, bmt):
        # Disable subclass rewriting so the only qedge is the real one and
        # we always see a query_{forward,backward,both_pinned} child.
        response = lookup(
            graph,
            _ONE_HOP_PINNED_TO_GENE,
            bmt=bmt,
            subclass=False,
            profile=True,
        )
        tree = _profile_summary(response)
        qedge_nodes = [c for c in tree["children"] if c["name"] == "qedge"]
        assert qedge_nodes
        # Find any qedge whose subtree contains a query_* stage.
        all_sub_names = set()
        query_stages = []
        for qe in qedge_nodes:
            for sub in qe["children"]:
                all_sub_names.add(sub["name"])
                if sub["name"].startswith("query_"):
                    query_stages.append(sub)
                    # query_* duration should be <= parent qedge duration
                    assert sub["duration_ms"] <= qe["duration_ms"] + 1.0
        assert all_sub_names & {"query_forward", "query_backward", "query_both_pinned"}
        # Diagnostic fields/metrics so callers can see why a stage was slow.
        for qs in query_stages:
            fields = qs.get("fields", {})
            assert "n_predicates" in fields
            assert "check_inverse" in fields
            metrics = qs.get("metrics", {})
            # _record_traversal_metrics always writes these two.
            assert "total_neighbors" in metrics
            assert "slow_nodes" in metrics

    def test_lmdb_metrics_present_when_profile_on(self, graph, bmt):
        response = lookup(graph, _ONE_HOP_BOTH_PINNED, bmt=bmt, profile=True)
        tree = _profile_summary(response)
        lmdb = tree.get("lmdb")
        assert lmdb is not None
        assert lmdb["calls"] >= 0  # may be zero on a fixture without LMDB
        assert "batch_sizes" in lmdb
        assert "by_kind" in lmdb


class TestProfilerOverHTTP:
    """Smoke-test the FastAPI ``/query`` endpoint with profile=true."""

    @pytest.fixture
    def client(self, graph, bmt, monkeypatch):
        """Patch the server module's GRAPH/BMT and return a TestClient."""
        # Skip the module-level graph preload + OpenTelemetry init so we
        # don't need a real graph or the otel instrumentation packages.
        monkeypatch.setenv("GANDALF_SKIP_PRELOAD", "true")
        monkeypatch.setenv("GANDALF_OTEL_ENABLED", "false")
        try:
            from gandalf import server as gandalf_server
        except ModuleNotFoundError as exc:
            pytest.skip(f"server import failed: {exc}")

        monkeypatch.setattr(gandalf_server, "GRAPH", graph)
        monkeypatch.setattr(gandalf_server, "BMT", bmt)
        return TestClient(gandalf_server.APP)

    def test_query_param_profile_true(self, client):
        resp = client.post("/query?profile=true", json=_ONE_HOP_BOTH_PINNED)
        assert resp.status_code == 200
        body = resp.json()
        codes = {e.get("code") for e in body.get("logs", [])}
        assert "ProfileSummary" in codes

    def test_body_field_profile_true(self, client):
        body_in = dict(_ONE_HOP_BOTH_PINNED)
        body_in["profile"] = True
        resp = client.post("/query", json=body_in)
        assert resp.status_code == 200
        body = resp.json()
        codes = {e.get("code") for e in body.get("logs", [])}
        assert "ProfileSummary" in codes

    def test_no_profile_by_default(self, client):
        resp = client.post("/query", json=_ONE_HOP_BOTH_PINNED)
        assert resp.status_code == 200
        body = resp.json()
        codes = {e.get("code") for e in body.get("logs", [])}
        assert "ProfileSummary" not in codes
