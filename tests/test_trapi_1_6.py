"""Tests for the TRAPI 1.6 protocol surface (the TRAPI 1.6 branch).

This file is the 2.0 line's ``tests/test_trapi_2_0.py``, renamed so changes
made there still merge into it.  It covers what differs between the versions
rather than what the search engine does: the Response envelope, the binding
and Edge shapes 1.6 serves, the 1.6 QEdge constraint lists, the ``parameters``
object (gandalf's own under 1.6, including the ``timeout`` budget and its
409), and the rejection of the 2.0 request fields 1.6 does not have.

Internally a response is still built in the 2.0 shape and converted on the
way out (``gandalf.trapi_1_6``), and ``lookup()`` still reads the internal
(2.0) ``constraints`` object, so the 2.0 line's allow/deny constraint tests
are kept below: they exercise code both lines share.  The null and
empty-container rules further down are the 2.0 line's too; 1.6 is laxer on
both, so holding to them keeps 1.6 responses valid as well.
"""

import logging

import pytest
from fastapi.testclient import TestClient

from gandalf.search import lookup, reconstruct
from gandalf.search.edge_constraints import (
    AllowDeny,
    ConstraintError,
    EdgeConstraints,
    SourcesConstraint,
)
from gandalf.trapi import (
    SCHEMA_VERSION,
    Deadline,
    QueryTimeout,
    TimeoutNotSatisfiable,
    resolve_timeout,
    response_results,
)
from tests.search_fixtures import graph  # noqa: F401
from tests.trapi_1_6_helpers import agent_type, knowledge_level

METFORMIN = "CHEBI:6801"


def affects_query(constraints=None, qedge_fields=None, **top_level) -> dict:
    """A ``CHEBI:6801 --affects--> Gene`` query with optional QEdge constraints.

    *constraints* is the internal (2.0) ``constraints`` object; *qedge_fields*
    holds 1.6 properties such as ``qualifier_constraints``.
    """
    qedge = {
        "subject": "n0",
        "object": "n1",
        "predicates": ["biolink:affects"],
        **(qedge_fields or {}),
    }
    if constraints is not None:
        qedge["constraints"] = constraints
    return {
        "message": {
            "query_graph": {
                "nodes": {
                    "n0": {"ids": [METFORMIN]},
                    "n1": {"categories": ["biolink:Gene"]},
                },
                "edges": {"e0": qedge},
            },
        },
        **top_level,
    }


def qualifier_set(**qualifiers) -> dict:
    """A 1.6 QualifierConstraint from ``type=value`` pairs (types sans prefix)."""
    return {
        "qualifier_set": [
            {"qualifier_type_id": f"biolink:{type_id}", "qualifier_value": value}
            for type_id, value in qualifiers.items()
        ]
    }


def bound_genes(response: dict) -> set:
    """The n1 CURIEs bound across a response's results."""
    return {
        node_id
        for result in response["message"]["results"]
        for node_id in [b["id"] for b in result["node_bindings"]["n1"]]
    }


@pytest.fixture
def server(graph, bmt, monkeypatch):  # noqa: F811
    """The server module patched onto the fixture graph, plus a TestClient."""
    monkeypatch.setenv("GANDALF_SKIP_PRELOAD", "true")
    monkeypatch.setenv("GANDALF_OTEL_ENABLED", "false")
    from gandalf import server as gandalf_server

    monkeypatch.setattr(gandalf_server, "GRAPH", graph)
    monkeypatch.setattr(gandalf_server, "BMT", bmt)
    return gandalf_server, TestClient(gandalf_server.APP)


# ---------------------------------------------------------------------------
# Response envelope
# ---------------------------------------------------------------------------


class TestResponseEnvelope:
    """TRAPI 1.6 Response metadata."""

    def test_response_reports_versions(self, graph, bmt):  # noqa: F811
        response = lookup(graph, affects_query(), bmt=bmt)
        assert response["schema_version"] == SCHEMA_VERSION
        assert response["status"] == "Success"
        assert response["biolink_version"]

    def test_schema_version_is_1_6(self, graph, bmt):  # noqa: F811
        assert lookup(graph, affects_query(), bmt=bmt)["schema_version"] == "1.6.0"

    def test_response_does_not_repeat_parameters(self, graph, bmt):  # noqa: F811
        """Repeating ``parameters`` is a 2.0 rule; 1.6 has no such object."""
        parameters = {"subclass": False, "bypass_cache": True}
        response = lookup(
            graph, affects_query(parameters=parameters), bmt=bmt, subclass=False
        )
        assert "parameters" not in response

    def test_response_omits_parameters_when_not_given(self, graph, bmt):  # noqa: F811
        assert "parameters" not in lookup(graph, affects_query(), bmt=bmt)

    def test_data_release_versions_from_graph_metadata(
        self, graph, bmt, monkeypatch  # noqa: F811
    ):
        from gandalf import trapi

        monkeypatch.setattr(
            graph, "graph_metadata", {"version": "2026_06_21"}, raising=False
        )
        response = lookup(graph, affects_query(), bmt=bmt)
        assert response["data_release_versions"] == {
            trapi.settings.infores: "2026_06_21"
        }

    def test_data_release_versions_omitted_without_metadata_version(
        self, graph, bmt, monkeypatch  # noqa: F811
    ):
        monkeypatch.setattr(graph, "graph_metadata", None, raising=False)
        assert "data_release_versions" not in lookup(graph, affects_query(), bmt=bmt)
        monkeypatch.setattr(graph, "graph_metadata", {"name": "kg"})
        assert "data_release_versions" not in lookup(graph, affects_query(), bmt=bmt)


class TestBindingShape:
    """1.6 binds a list of ``{"id", "attributes"}`` objects per QNode / QEdge."""

    @pytest.mark.parametrize("results_as_json", [False, True])
    def test_node_binding_is_a_list_of_objects(
        self, graph, bmt, results_as_json  # noqa: F811
    ):
        response = lookup(
            graph, affects_query(), bmt=bmt, results_as_json=results_as_json
        )
        result = response_results(response)[0]
        assert result["node_bindings"]["n0"] == [{"id": METFORMIN, "attributes": []}]

    @pytest.mark.parametrize("results_as_json", [False, True])
    def test_edge_binding_is_a_list_of_objects(
        self, graph, bmt, results_as_json  # noqa: F811
    ):
        response = lookup(
            graph, affects_query(), bmt=bmt, results_as_json=results_as_json
        )
        result = response_results(response)[0]
        binding = result["analyses"][0]["edge_bindings"]["e0"]
        assert binding
        for bound in binding:
            assert set(bound) == {"id", "attributes"}
            assert bound["id"] in response["message"]["knowledge_graph"]["edges"]

    def test_every_result_carries_analyses(self, graph, bmt):  # noqa: F811
        """Result.analyses is required in 1.6, even when it is empty."""
        from gandalf.trapi_1_6 import to_trapi_1_6

        response = to_trapi_1_6(
            {"message": {"results": [{"node_bindings": {"n0": {"ids": ["A"]}}}]}}
        )
        assert response["message"]["results"][0]["analyses"] == []

    def test_auxiliary_graphs_carry_attributes(self, graph, bmt):  # noqa: F811
        """AuxiliaryGraph.attributes is required in 1.6."""
        response = lookup(
            graph,
            {
                "message": {
                    "query_graph": {
                        "nodes": {
                            "n0": {"ids": [METFORMIN]},
                            "n1": {"ids": ["MONDO:0005015"]},
                        },
                        "edges": {
                            "e0": {
                                "subject": "n0",
                                "object": "n1",
                                "predicates": ["biolink:treats"],
                            }
                        },
                    }
                }
            },
            bmt=bmt,
        )
        aux_graphs = response["message"]["auxiliary_graphs"]
        assert aux_graphs
        for aux_graph in aux_graphs.values():
            assert aux_graph["attributes"] == []


class TestEdgeProperties:
    """1.6 carries knowledge_level and agent_type as Edge attributes."""

    @pytest.mark.parametrize("dehydrated", [False, True])
    @pytest.mark.parametrize("attributes_as_json", [False, True])
    def test_every_edge_carries_knowledge_level_and_agent_type(
        self, graph, bmt, dehydrated, attributes_as_json  # noqa: F811
    ):
        response = lookup(
            graph,
            affects_query(),
            bmt=bmt,
            dehydrated=dehydrated,
            attributes_as_json=attributes_as_json,
        )
        edges = response["message"]["knowledge_graph"]["edges"]
        assert edges
        for edge in edges.values():
            assert knowledge_level(edge) == "knowledge_assertion"
            assert agent_type(edge) == "manual_agent"

    def test_they_are_not_also_edge_properties(self, graph, bmt):  # noqa: F811
        """Each lives in one place: the attribute list, once."""
        response = lookup(graph, affects_query(), bmt=bmt)
        for edge in response["message"]["knowledge_graph"]["edges"].values():
            assert "knowledge_level" not in edge
            assert "agent_type" not in edge
            type_ids = [a["attribute_type_id"] for a in edge["attributes"]]
            assert type_ids.count("biolink:knowledge_level") == 1
            assert type_ids.count("biolink:agent_type") == 1

    def test_stored_json_attributes_serialize_as_decoded(
        self, graph, bmt
    ):  # noqa: F811
        """Moving them into attributes still held as the stored JSON gives
        the same bytes as moving them into the decoded list."""
        import orjson

        from gandalf.trapi import to_fragments

        decoded = lookup(graph, affects_query(), bmt=bmt)
        stored = lookup(graph, affects_query(), bmt=bmt, attributes_as_json=True)
        to_fragments(stored)
        assert orjson.dumps(stored["message"]["knowledge_graph"]) == orjson.dumps(
            decoded["message"]["knowledge_graph"]
        )


class TestLibraryHelpersRead16Bindings:
    """``gandalf.validation`` reads results after they are converted."""

    def test_validate_trapi_response(self, graph, bmt):  # noqa: F811
        from gandalf.validation import validate_trapi_response

        response = lookup(graph, affects_query(), bmt=bmt)
        outcome = validate_trapi_response(graph, response)
        assert outcome.total_paths == len(response["message"]["results"])
        assert outcome.valid_paths == outcome.total_paths

    def test_compare_trapi_messages(self, graph, bmt):  # noqa: F811
        from gandalf.validation import compare_trapi_messages

        everything = lookup(graph, affects_query(), bmt=bmt)["message"]
        nothing = {**everything, "results": []}
        missing = compare_trapi_messages(everything, nothing)
        assert len(missing) == len(everything["results"])
        assert all("?" not in path for path in missing)
        assert compare_trapi_messages(everything, everything) == []


class TestRehydratingA16DehydratedResponse:
    """A dehydrated 1.6 edge's only attributes are knowledge_level and
    agent_type, which must not read as "attributes already present"."""

    def test_stored_attributes_are_filled_in(self, graph, bmt):  # noqa: F811
        from gandalf.enrichment import enrich_knowledge_graph

        full = lookup(graph, affects_query(), bmt=bmt)
        response = lookup(graph, affects_query(), bmt=bmt, dehydrated=True)
        enrich_knowledge_graph(response, graph)
        edges = response["message"]["knowledge_graph"]["edges"]
        assert edges
        for edge_id, edge in edges.items():
            assert (
                edge["attributes"]
                == full["message"]["knowledge_graph"]["edges"][edge_id]["attributes"]
            ), edge_id
            assert edge["sources"]


class TestTRAPI16QEdgeConstraints:
    """1.6 ``qualifier_constraints`` / ``attribute_constraints`` filter as the
    internal ``constraints`` object does."""

    def test_qualifier_constraints_filter(self, graph, bmt):  # noqa: F811
        trapi_1_6 = lookup(
            graph,
            affects_query(
                qedge_fields={
                    "qualifier_constraints": [
                        qualifier_set(
                            object_aspect_qualifier="activity",
                            object_direction_qualifier="increased",
                        )
                    ]
                }
            ),
            bmt=bmt,
        )
        internal = lookup(
            graph,
            affects_query(
                {
                    "qualifiers": [
                        {
                            "biolink:object_aspect_qualifier": "activity",
                            "biolink:object_direction_qualifier": "increased",
                        }
                    ]
                }
            ),
            bmt=bmt,
        )
        assert bound_genes(trapi_1_6) == bound_genes(internal)
        assert bound_genes(trapi_1_6) < bound_genes(
            lookup(graph, affects_query(), bmt=bmt)
        )

    def test_qualifier_constraints_are_ored(self, graph, bmt):  # noqa: F811
        one = lookup(
            graph,
            affects_query(
                qedge_fields={
                    "qualifier_constraints": [
                        qualifier_set(object_direction_qualifier="increased")
                    ]
                }
            ),
            bmt=bmt,
        )
        both = lookup(
            graph,
            affects_query(
                qedge_fields={
                    "qualifier_constraints": [
                        qualifier_set(object_direction_qualifier="increased"),
                        qualifier_set(object_direction_qualifier="decreased"),
                    ]
                }
            ),
            bmt=bmt,
        )
        assert bound_genes(one) < bound_genes(both)

    def test_attribute_constraints_filter(self, graph, bmt):  # noqa: F811
        constraint = {
            "id": "biolink:publications",
            "name": "publications",
            "operator": "==",
            "value": "PMID:00000000",
        }
        response = lookup(
            graph,
            affects_query(qedge_fields={"attribute_constraints": [constraint]}),
            bmt=bmt,
        )
        assert response["message"]["results"] == []

    def test_query_graph_is_echoed_as_sent(self, graph, bmt):  # noqa: F811
        constraints = [qualifier_set(object_aspect_qualifier="activity")]
        response = lookup(
            graph,
            affects_query(qedge_fields={"qualifier_constraints": constraints}),
            bmt=bmt,
        )
        qedge = response["message"]["query_graph"]["edges"]["e0"]
        assert qedge["qualifier_constraints"] == constraints
        assert "constraints" not in qedge

    @pytest.mark.parametrize(
        "qualifier_constraints",
        [
            "activity",
            [{"biolink:object_aspect_qualifier": "activity"}],
            [{"qualifier_set": [{"qualifier_type_id": "biolink:x"}]}],
            [
                {
                    "qualifier_set": [
                        {"qualifier_type_id": "biolink:x", "qualifier_value": "a"},
                        {"qualifier_type_id": "biolink:x", "qualifier_value": "b"},
                    ]
                }
            ],
        ],
        ids=["not-a-list", "2.0-mapping", "no-value", "type-twice"],
    )
    def test_malformed_qualifier_constraints_rejected(self, qualifier_constraints):
        with pytest.raises(ConstraintError):
            EdgeConstraints.parse({"qualifier_constraints": qualifier_constraints})

    def test_both_spellings_at_once_rejected(self):
        with pytest.raises(ConstraintError, match="not both"):
            EdgeConstraints.parse(
                {
                    "qualifier_constraints": [
                        qualifier_set(object_aspect_qualifier="a")
                    ],
                    "constraints": {"qualifiers": [{"biolink:x": "b"}]},
                }
            )


# ---------------------------------------------------------------------------
# QEdge constraints
# ---------------------------------------------------------------------------


class TestAllowDenyParsing:
    @pytest.mark.parametrize(
        "raw",
        [
            "not-an-object",
            {"values": ["a"]},
            {"behavior": "MAYBE", "values": ["a"]},
            {"behavior": "ALLOW"},
            {"behavior": "ALLOW", "values": []},
        ],
    )
    def test_malformed_allow_deny_rejected(self, raw):
        with pytest.raises(ConstraintError):
            AllowDeny.parse(raw, "knowledge_level")

    def test_unknown_constraint_key_rejected(self):
        """A constraint this server does not implement is an error, not a no-op."""
        with pytest.raises(ConstraintError, match="colour"):
            EdgeConstraints.parse({"constraints": {"colour": "blue"}})

    def test_no_constraints_is_falsy(self):
        assert not EdgeConstraints.parse({"subject": "n0", "object": "n1"})

    def test_sources_primary_only_must_be_boolean(self):
        with pytest.raises(ConstraintError, match="primary_only"):
            SourcesConstraint.parse(
                {"behavior": "ALLOW", "values": ["infores:ctd"], "primary_only": "yes"}
            )


class TestKnowledgeLevelAndAgentTypeConstraints:
    """The fixture's affects edges are all knowledge_assertion / manual_agent.

    Internal (2.0) constraints only: a 1.6 client cannot send these.
    """

    @pytest.mark.parametrize(
        "constraints, expect_results",
        [
            (
                {"knowledge_level": {"behavior": "ALLOW", "values": ["prediction"]}},
                False,
            ),
            (
                {
                    "knowledge_level": {
                        "behavior": "ALLOW",
                        "values": ["knowledge_assertion"],
                    }
                },
                True,
            ),
            (
                {
                    "knowledge_level": {
                        "behavior": "DENY",
                        "values": ["knowledge_assertion"],
                    }
                },
                False,
            ),
            ({"agent_type": {"behavior": "ALLOW", "values": ["manual_agent"]}}, True),
            (
                {"agent_type": {"behavior": "DENY", "values": ["text_mining_agent"]}},
                True,
            ),
            ({"agent_type": {"behavior": "DENY", "values": ["manual_agent"]}}, False),
        ],
    )
    def test_filtering(self, graph, bmt, constraints, expect_results):  # noqa: F811
        response = lookup(graph, affects_query(constraints), bmt=bmt)
        assert bool(response["message"]["results"]) is expect_results

    def test_both_constraints_are_anded(self, graph, bmt):  # noqa: F811
        """Every constraint given must hold, so one failing rules the edge out."""
        response = lookup(
            graph,
            affects_query(
                {
                    "knowledge_level": {
                        "behavior": "ALLOW",
                        "values": ["knowledge_assertion"],
                    },
                    "agent_type": {"behavior": "ALLOW", "values": ["automated_agent"]},
                }
            ),
            bmt=bmt,
        )
        assert response["message"]["results"] == []


class TestSourcesConstraint:
    """The ``CHEBI:6801 --affects--> Gene`` fixture edges name two sources.

    Internal (2.0) constraints only: a 1.6 client cannot send these.

    PPARG, INSR, GCK and TNF come from infores:ctd; INSR has a second edge from
    infores:hetio.  Gandalf adds itself as an aggregator on every edge.
    """

    CTD_GENES = {
        "NCBIGene:5468",
        "NCBIGene:3643",
        "NCBIGene:2645",
        "NCBIGene:7124",
    }
    HETIO_GENES = {"NCBIGene:3643"}

    def test_allow_keeps_only_the_named_source(self, graph, bmt):  # noqa: F811
        response = lookup(
            graph,
            affects_query(
                {"sources": {"behavior": "ALLOW", "values": ["infores:hetio"]}}
            ),
            bmt=bmt,
        )
        assert bound_genes(response) == self.HETIO_GENES

    def test_allow_rejects_unlisted_source(self, graph, bmt):  # noqa: F811
        response = lookup(
            graph,
            affects_query(
                {"sources": {"behavior": "ALLOW", "values": ["infores:nowhere"]}}
            ),
            bmt=bmt,
        )
        assert response["message"]["results"] == []

    def test_deny_excludes_the_named_source(self, graph, bmt):  # noqa: F811
        """Denying ctd leaves only the hetio edge."""
        response = lookup(
            graph,
            affects_query({"sources": {"behavior": "DENY", "values": ["infores:ctd"]}}),
            bmt=bmt,
        )
        assert bound_genes(response) == self.HETIO_GENES

    def test_allow_several_sources_is_a_union(self, graph, bmt):  # noqa: F811
        response = lookup(
            graph,
            affects_query(
                {
                    "sources": {
                        "behavior": "ALLOW",
                        "values": ["infores:ctd", "infores:hetio"],
                    }
                }
            ),
            bmt=bmt,
        )
        assert bound_genes(response) == self.CTD_GENES | self.HETIO_GENES

    def test_primary_only_ignores_aggregators(self, graph, bmt):  # noqa: F811
        """Gandalf adds itself as an aggregator on every edge it serves.

        Denying that aggregator would empty the response, but with
        ``primary_only`` the constraint looks at the primary source alone and
        the edges survive.
        """
        from gandalf.config import settings

        denied = {
            "behavior": "DENY",
            "values": [settings.infores],
            "primary_only": True,
        }
        response = lookup(graph, affects_query({"sources": denied}), bmt=bmt)
        assert bound_genes(response)

        denied_everywhere = {"behavior": "DENY", "values": [settings.infores]}
        response = lookup(graph, affects_query({"sources": denied_everywhere}), bmt=bmt)
        assert response["message"]["results"] == []


# ---------------------------------------------------------------------------
# Timeouts
# ---------------------------------------------------------------------------


class TestResolveTimeout:
    @pytest.mark.parametrize("requested", [-1, -0.5, 0])
    def test_non_positive_disables_timeout(self, requested):
        assert resolve_timeout({"timeout": requested}) is None

    def test_positive_is_the_budget(self):
        assert resolve_timeout({"timeout": 12.5}) == 12.5

    def test_absent_uses_server_default(self, monkeypatch):
        from gandalf import trapi

        monkeypatch.setattr(trapi.settings, "query_timeout", 30.0)
        assert resolve_timeout(None) == 30.0
        assert resolve_timeout({}) == 30.0

    @pytest.mark.parametrize("requested", [0.001, "soon", True])
    def test_unsatisfiable_or_malformed_rejected(self, requested):
        with pytest.raises(TimeoutNotSatisfiable):
            resolve_timeout({"timeout": requested})


class TestDeadline:
    def test_no_budget_never_expires(self):
        deadline = Deadline(None)
        assert not deadline
        assert deadline.expired is False
        deadline.check("anything")

    def test_spent_budget_raises(self):
        deadline = Deadline(-1.0)
        with pytest.raises(QueryTimeout, match="reconstruction"):
            deadline.check("reconstruction")

    def test_lookup_returns_timeout_status(self, graph, bmt):  # noqa: F811
        """An overrun is a TRAPI outcome, not an exception out of lookup()."""
        response = lookup(
            graph, affects_query(), bmt=bmt, deadline=Deadline(-1.0), log_level="DEBUG"
        )
        assert response["status"] == "Timeout"
        assert response["message"]["results"] == []
        assert response["message"]["query_graph"]["nodes"]["n0"]["ids"] == [METFORMIN]
        assert any("timeout" in entry["message"] for entry in response["logs"])


class TestPathLimit:
    """Over ``max_path_limit`` the query is refused, never truncated."""

    def _paths(self, graph, bmt):  # noqa: F811
        return len(lookup(graph, affects_query(), bmt=bmt)["message"]["results"])

    def test_over_the_limit_is_refused_with_no_results(
        self, graph, bmt, monkeypatch  # noqa: F811
    ):
        assert self._paths(graph, bmt) > 1
        monkeypatch.setattr(reconstruct, "MAX_PATH_LIMIT", 1)
        response = lookup(graph, affects_query(), bmt=bmt)
        assert response["status"] == "QueryTooLarge"
        assert response["message"]["results"] == []
        assert response["message"]["query_graph"]["nodes"]["n0"]["ids"] == [METFORMIN]
        assert "limit of 1" in response["description"]
        assert any("too large" in entry["message"] for entry in response["logs"])

    def test_under_the_limit_is_answered_in_full(
        self, graph, bmt, monkeypatch  # noqa: F811
    ):
        # The limit is on intermediate paths, which bound the results.
        full = lookup(graph, affects_query(), bmt=bmt)["message"]["results"]
        monkeypatch.setattr(reconstruct, "MAX_PATH_LIMIT", 10_000)
        response = lookup(graph, affects_query(), bmt=bmt)
        assert response.get("status", "Success") == "Success"
        assert len(response["message"]["results"]) == len(full)

    def test_endpoint_answers_200_with_the_refusal(
        self, server, monkeypatch  # noqa: F811
    ):
        _, client = server
        monkeypatch.setattr(reconstruct, "MAX_PATH_LIMIT", 1)
        resp = client.post("/query", json=affects_query())
        assert resp.status_code == 200, resp.text
        assert resp.json()["status"] == "QueryTooLarge"
        assert resp.json()["message"]["results"] == []


# ---------------------------------------------------------------------------
# Endpoint behaviour
# ---------------------------------------------------------------------------


class TestEndpoints:
    def test_unsatisfiable_timeout_is_409(self, server):
        _, client = server
        resp = client.post("/query", json=affects_query(parameters={"timeout": 0.001}))
        assert resp.status_code == 409
        assert "timeout" in resp.json()["detail"]

    def test_generous_timeout_is_accepted(self, server):
        _, client = server
        resp = client.post("/query", json=affects_query(parameters={"timeout": 600}))
        assert resp.status_code == 200, resp.text
        assert "parameters" not in resp.json()

    def test_qualifier_constraints_served(self, server):
        _, client = server
        body = affects_query(
            qedge_fields={
                "qualifier_constraints": [
                    qualifier_set(object_aspect_qualifier="activity")
                ]
            }
        )
        resp = client.post("/query", json=body)
        assert resp.status_code == 200, resp.text
        results = resp.json()["message"]["results"]
        assert results
        assert results[0]["node_bindings"]["n0"] == [
            {"id": METFORMIN, "attributes": []}
        ]

    def test_malformed_constraint_is_400(self, server):
        _, client = server
        resp = client.post(
            "/query",
            json=affects_query(
                qedge_fields={"qualifier_constraints": [{"qualifier_set": "x"}]}
            ),
        )
        assert resp.status_code == 400
        assert "qualifier_set" in resp.json()["detail"]

    def test_2_0_edge_constraints_are_rejected(self, server):
        """Silently ignoring a filter the client asked for is worse than a 400."""
        _, client = server
        body = affects_query(
            {"qualifiers": [{"biolink:object_aspect_qualifier": "activity"}]}
        )
        resp = client.post("/query", json=body)
        assert resp.status_code == 400
        detail = resp.json()["detail"]
        assert "TRAPI 1.6" in detail
        assert "qualifier_constraints" in detail

    def test_2_0_pathfinder_constraint_is_rejected(self, server):
        _, client = server
        body = affects_query()
        body["message"]["query_graph"]["paths"] = {
            "p0": {
                "subject": "n0",
                "object": "n1",
                "constraints": [{"required_intermediate_categories": ["biolink:Gene"]}],
            }
        }
        resp = client.post("/query", json=body)
        assert resp.status_code == 400
        assert "intermediate_categories" in resp.json()["detail"]

    def test_log_level_read_from_the_top_level(self, server):
        """TRAPI 1.6 puts log_level on the Query itself."""
        _, client = server
        resp = client.post("/query", json=affects_query(log_level="DEBUG"))
        assert resp.status_code == 200, resp.text
        assert len(resp.json()["logs"]) > 0

    def test_log_level_read_from_parameters(self, server):
        """Still honoured where the 2.0 line reads it."""
        _, client = server
        resp = client.post(
            "/query", json=affects_query(parameters={"log_level": "DEBUG"})
        )
        assert resp.status_code == 200, resp.text
        assert len(resp.json()["logs"]) > 0

    def test_openapi_declares_trapi_1_6(self, server):
        gandalf_server, _ = server
        info = gandalf_server.APP.openapi()["info"]
        assert info["x-trapi"]["version"] == "1.6.0"


class TestStrictValidationMode:
    """``validate_responses`` routes requests and responses through Pydantic.

    This path must not turn unset optional properties into nulls, and must
    hand ``log_level`` on as the plain string that ``logging.setLevel`` and
    the JSON encoder both require.
    """

    @pytest.fixture
    def strict_server(self, graph, bmt, monkeypatch):  # noqa: F811
        """A client with inbound request validation switched on.

        Only the request half: the route's ``response_model`` is bound at
        import time from the same flag, so the outbound half is covered by the
        model-level assertions below.
        """
        monkeypatch.setenv("GANDALF_SKIP_PRELOAD", "true")
        monkeypatch.setenv("GANDALF_OTEL_ENABLED", "false")
        from gandalf import server as gandalf_server

        monkeypatch.setattr(gandalf_server, "GRAPH", graph)
        monkeypatch.setattr(gandalf_server, "BMT", bmt)
        monkeypatch.setattr(gandalf_server, "_validate", True)
        return TestClient(gandalf_server.APP)

    def test_validated_request_round_trips(self, strict_server):
        resp = strict_server.post(
            "/query",
            json=affects_query(log_level="ERROR", parameters={"timeout": 60}),
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert "parameters" not in body
        assert body["message"]["results"]

    def test_validated_log_level_is_a_plain_string(self):
        """``logging.setLevel`` rejects a str subclass, enum members included."""
        import logging

        from gandalf.models import TRAPIQuery

        validated = TRAPIQuery.model_validate(
            affects_query(log_level="ERROR")
        ).model_dump(exclude_none=True)
        log_level = validated["log_level"]
        assert type(log_level) is str
        logging.Logger("test").setLevel(log_level)

    def test_unset_optional_properties_are_absent_not_null(self):
        from gandalf.models import TRAPIResponse

        dumped = TRAPIResponse(
            message={}, status="Success", schema_version="1.6.0"
        ).model_dump(exclude_none=True)
        assert dumped == {
            "message": {},
            "status": "Success",
            "schema_version": "1.6.0",
        }


# ---------------------------------------------------------------------------
# Nulls and empty containers
# ---------------------------------------------------------------------------


@pytest.fixture
def sparse_graph(tmp_path):
    """A graph whose A:1 node has neither a name nor a category."""
    from gandalf.loader import build_graph_from_jsonl

    nodes = tmp_path / "nodes.jsonl"
    edges = tmp_path / "edges.jsonl"
    nodes.write_text(
        '{"id": "A:1"}\n' '{"id": "B:1", "name": "Bee", "category": ["biolink:Gene"]}\n'
    )
    edges.write_text(
        '{"id": "e1", "subject": "A:1", "object": "B:1", '
        '"predicate": "biolink:affects", '
        '"primary_knowledge_source": "infores:test"}\n'
    )
    return build_graph_from_jsonl(str(edges), str(nodes))


def sparse_query() -> dict:
    return {
        "message": {
            "query_graph": {
                "nodes": {"n0": {"ids": ["A:1"]}, "n1": {}},
                "edges": {"e0": {"subject": "n0", "object": "n1"}},
            }
        }
    }


def _null_paths(value, path=""):
    """Yield the path of every null found anywhere inside *value*."""
    if value is None:
        yield path or "<root>"
    elif isinstance(value, dict):
        for key, sub in value.items():
            yield from _null_paths(sub, f"{path}/{key}")
    elif isinstance(value, list):
        for i, sub in enumerate(value):
            yield from _null_paths(sub, f"{path}/{i}")


class TestNoNullsSerialized:
    """TRAPI 2.0 is OpenAPI 3.1 and dropped every ``nullable``.

    A typed property therefore does not admit null: an absent value has to be
    an absent property.
    """

    @pytest.mark.parametrize("dehydrated", [False, True])
    def test_nameless_node_carries_no_name_property(
        self, sparse_graph, bmt, dehydrated
    ):
        response = lookup(sparse_graph, sparse_query(), bmt=bmt, dehydrated=dehydrated)
        node = response["message"]["knowledge_graph"]["nodes"]["A:1"]
        assert "name" not in node
        # ... while a node that has one still reports it
        assert response["message"]["knowledge_graph"]["nodes"]["B:1"]["name"] == "Bee"

    def test_log_entry_outside_the_trapi_enum_carries_no_level(
        self, graph, bmt  # noqa: F811
    ):
        """LogLevel has no CRITICAL member, so such a record reports no level."""
        from gandalf.logging_config import TRAPILogCollector

        collector = TRAPILogCollector()
        logger = logging.getLogger("gandalf.test.levels")
        logger.addHandler(collector)
        logger.setLevel(logging.DEBUG)
        try:
            logger.critical("a critical message")
            logger.warning("a warning message")
        finally:
            logger.removeHandler(collector)

        entries = collector.get_logs()
        assert "level" not in entries[0]
        assert entries[1]["level"] == "WARNING"

    @pytest.mark.parametrize("dehydrated", [False, True])
    def test_whole_response_serializes_no_nulls(
        self, graph, bmt, dehydrated  # noqa: F811
    ):
        """Nothing anywhere in a response may be null.

        The one place 2.0 would permit it is ``Attribute.value``, which
        declares no ``type`` -- no fixture attribute has a null value, so a
        blanket assertion is the right guard here.
        """
        response = lookup(
            graph, affects_query(), bmt=bmt, dehydrated=dehydrated, log_level="DEBUG"
        )
        assert list(_null_paths(response)) == []

    def test_meta_attribute_omits_what_it_does_not_know(self):
        """MetaAttribute.attribute_source / constraint_name are typed strings.

        The meta-KG scan finds a source for some attributes and not others,
        and gandalf never computes a constraint_name, so neither may be
        serialized as null.
        """
        from gandalf.graph import _meta_attributes

        known, unknown = _meta_attributes(
            {
                ("biolink:p_value", "infores:ctd", "p-value"),
                ("biolink:score", None, "score"),
            }
        )
        assert known["attribute_source"] == "infores:ctd"
        assert "attribute_source" not in unknown
        assert "constraint_name" not in known
        assert "constraint_name" not in unknown

    def test_meta_knowledge_graph_serializes_no_nulls(self, graph, bmt):  # noqa: F811
        graph.build_metadata()
        nulls = list(_null_paths(graph.meta_kg))
        assert nulls == []


class TestEmptyContainersOnlyWhereAllowed:
    """2.0 forbids an empty value on some properties and requires it on others."""

    def test_unqualified_edge_carries_no_qualifiers_property(
        self, graph, bmt  # noqa: F811
    ):
        """Edge.qualifiers has a minItems of 1."""
        treats = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": [METFORMIN]},
                        "n1": {"ids": ["MONDO:0005148"]},
                    },
                    "edges": {
                        "e0": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:treats"],
                        }
                    },
                }
            }
        }
        edges = lookup(graph, treats, bmt=bmt)["message"]["knowledge_graph"]["edges"]
        assert edges
        assert all("qualifiers" not in edge for edge in edges.values())

    def test_qualified_edge_still_reports_its_qualifiers(
        self, graph, bmt
    ):  # noqa: F811
        edges = lookup(graph, affects_query(), bmt=bmt)["message"]["knowledge_graph"][
            "edges"
        ]
        assert any(edge.get("qualifiers") for edge in edges.values())

    def test_primary_source_carries_no_upstream_resource_ids(
        self, graph, bmt  # noqa: F811
    ):
        """RetrievalSource.upstream_resource_ids has a minItems of 1."""
        edges = lookup(graph, affects_query(), bmt=bmt)["message"]["knowledge_graph"][
            "edges"
        ]
        assert edges
        for edge in edges.values():
            for source in edge["sources"]:
                if source["resource_role"] == "primary_knowledge_source":
                    assert "upstream_resource_ids" not in source
                else:
                    assert source["upstream_resource_ids"]

    def test_no_auxiliary_graphs_property_when_nothing_was_inferred(
        self, graph, bmt  # noqa: F811
    ):
        """Message.auxiliary_graphs has a minProperties of 1."""
        assert (
            "auxiliary_graphs" not in lookup(graph, affects_query(), bmt=bmt)["message"]
        )

    def test_auxiliary_graphs_present_when_subclass_inference_fires(
        self, graph, bmt  # noqa: F811
    ):
        response = lookup(
            graph,
            {
                "message": {
                    "query_graph": {
                        "nodes": {
                            "n0": {"ids": [METFORMIN]},
                            "n1": {"ids": ["MONDO:0005015"]},
                        },
                        "edges": {
                            "e0": {
                                "subject": "n0",
                                "object": "n1",
                                "predicates": ["biolink:treats"],
                            }
                        },
                    }
                }
            },
            bmt=bmt,
            subclass=True,
            subclass_depth=1,
        )
        assert response["message"]["auxiliary_graphs"]

    def test_nameless_node_still_reports_a_category(self, sparse_graph, bmt):
        """Node.categories is required with a minItems of 1, so it cannot be [].

        Omitting it is not an option either, so a record with no category is
        served as the Biolink root class.
        """
        response = lookup(sparse_graph, sparse_query(), bmt=bmt)
        node = response["message"]["knowledge_graph"]["nodes"]["A:1"]
        assert node["categories"] == ["biolink:NamedThing"]

    def test_empty_results_stay_an_empty_list(self, graph, bmt):  # noqa: F811
        """2.0 *requires* the empty form here, so this must not be pruned.

        "If Results are expected ... and no Results are available, this
        property SHOULD be an array with 0 Results in it."
        """
        response = lookup(
            graph,
            {
                "message": {
                    "query_graph": {
                        "nodes": {
                            "n0": {"ids": ["NOSUCH:0000"]},
                            "n1": {"categories": ["biolink:Gene"]},
                        },
                        "edges": {"e0": {"subject": "n0", "object": "n1"}},
                    }
                }
            },
            bmt=bmt,
        )
        assert response["message"]["results"] == []
        # KnowledgeGraph.nodes is required and has no minimum, so it stays too.
        assert response["message"]["knowledge_graph"] == {"nodes": {}, "edges": {}}

    def test_edge_attributes_may_be_empty(self, graph, bmt):  # noqa: F811
        """2.0 sets no minimum on Edge.attributes, so [] is a valid value."""
        from gandalf.trapi import prune_edge

        assert prune_edge({"attributes": []}) == {"attributes": []}


class TestRehydrateKeepsClientValuesValid:
    """The rehydrate path serves a knowledge graph the client handed back.

    A property that arrives present-but-null, or with an empty value 2.0
    forbids, must not pass through: it has to read as absent so the stored
    value fills it.
    """

    def _rehydrate(self, graph, nodes, edges):  # noqa: F811
        from gandalf.enrichment import enrich_knowledge_graph

        message = {"message": {"knowledge_graph": {"nodes": nodes, "edges": edges}}}
        enrich_knowledge_graph(message, graph)
        return message["message"]["knowledge_graph"]

    def test_client_null_name_is_replaced_by_the_stored_one(
        self, graph, bmt
    ):  # noqa: F811
        kg = self._rehydrate(graph, {METFORMIN: {"name": None}}, {})
        assert kg["nodes"][METFORMIN]["name"] == "Metformin"

    def test_client_empty_categories_replaced_by_the_stored_ones(
        self, graph, bmt  # noqa: F811
    ):
        kg = self._rehydrate(graph, {METFORMIN: {"categories": []}}, {})
        assert kg["nodes"][METFORMIN]["categories"]
        assert "biolink:Drug" in kg["nodes"][METFORMIN]["categories"]

    def test_unknown_node_still_served_validly(self, graph, bmt):  # noqa: F811
        """A node the graph does not know has nothing to fill from."""
        kg = self._rehydrate(
            graph, {"SYNTHETIC:1": {"name": None, "categories": []}}, {}
        )
        node = kg["nodes"]["SYNTHETIC:1"]
        assert "name" not in node
        assert node["categories"] == ["biolink:NamedThing"]

    def test_client_null_edge_properties_are_filled_from_the_graph(
        self, graph, bmt  # noqa: F811
    ):
        kg = self._rehydrate(
            graph,
            {},
            {
                "x": {
                    "subject": METFORMIN,
                    "object": "MONDO:0005148",
                    "predicate": "biolink:treats",
                    "attributes": None,
                    "qualifiers": [],
                }
            },
        )
        edge = kg["edges"]["x"]
        assert edge["attributes"] is not None
        assert "qualifiers" not in edge
        assert edge["sources"]

    def test_no_nulls_survive_rehydration(self, graph, bmt):  # noqa: F811
        kg = self._rehydrate(
            graph,
            {
                METFORMIN: {"name": None, "categories": []},
                "SYNTHETIC:1": {"name": None},
            },
            {
                "x": {
                    "subject": METFORMIN,
                    "object": "MONDO:0005148",
                    "predicate": "biolink:treats",
                    "attributes": None,
                }
            },
        )
        assert list(_null_paths(kg)) == []


class TestNonExecutableQueryGraphRejected:
    """A query graph with nothing to execute is a 400, not a 500 or an echo.

    QueryGraph.edges has a minProperties of 1, so an empty map could not be
    echoed back validly anyway, and a missing one used to reach the planner
    and raise KeyError behind an opaque 500.
    """

    @pytest.mark.parametrize(
        "query_graph",
        [
            {"nodes": {"n0": {"ids": [METFORMIN]}}, "edges": {}},
            {"nodes": {"n0": {"ids": [METFORMIN]}}},
            {
                "nodes": {"n0": {}, "n1": {}},
                "paths": {"p0": {"subject": "n0", "object": "n1"}},
            },
        ],
        ids=["empty-edges", "no-edges", "paths-only"],
    )
    def test_rejected_with_400(self, server, query_graph):
        _, client = server
        resp = client.post("/query", json={"message": {"query_graph": query_graph}})
        assert resp.status_code == 400
        assert "edges" in resp.json()["detail"]
