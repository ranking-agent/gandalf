"""Tests for the TRAPI 2.0 protocol surface.

Covers what 2.0 changed rather than what the search engine does: the Response
envelope, the ``parameters`` object (including the ``timeout`` budget and its
409), the new QEdge allow/deny constraints, and the rejection of the 1.x
request fields 2.0 renamed.
"""

import logging

import pytest
from fastapi.testclient import TestClient

from gandalf.search import lookup
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
)
from tests.search_fixtures import graph  # noqa: F401

METFORMIN = "CHEBI:6801"


def affects_query(constraints=None, **top_level) -> dict:
    """A ``CHEBI:6801 --affects--> Gene`` query with optional QEdge constraints."""
    qedge = {
        "subject": "n0",
        "object": "n1",
        "predicates": ["biolink:affects"],
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


def bound_genes(response: dict) -> set:
    """The n1 CURIEs bound across a response's results."""
    return {
        node_id
        for result in response["message"]["results"]
        for node_id in result["node_bindings"]["n1"]["ids"]
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
    """TRAPI 2.0 Response metadata."""

    def test_response_reports_versions(self, graph, bmt):  # noqa: F811
        response = lookup(graph, affects_query(), bmt=bmt)
        assert response["schema_version"] == SCHEMA_VERSION
        assert response["status"] == "Success"
        assert response["biolink_version"]

    def test_response_repeats_parameters(self, graph, bmt):  # noqa: F811
        """TRAPI 2.0: the server MUST repeat the parameters it is given."""
        parameters = {"subclass": False, "bypass_cache": True}
        response = lookup(
            graph, affects_query(parameters=parameters), bmt=bmt, subclass=False
        )
        assert response["parameters"] == parameters

    def test_response_omits_parameters_when_not_given(self, graph, bmt):  # noqa: F811
        assert "parameters" not in lookup(graph, affects_query(), bmt=bmt)

    def test_data_release_versions_reported_when_configured(
        self, graph, bmt, monkeypatch  # noqa: F811
    ):
        from gandalf import trapi

        monkeypatch.setattr(
            trapi.settings, "data_release_versions", '{"translator_kg": "2026_06_21"}'
        )
        response = lookup(graph, affects_query(), bmt=bmt)
        assert response["data_release_versions"] == {"translator_kg": "2026_06_21"}

    def test_data_release_versions_omitted_when_unconfigured(
        self, graph, bmt  # noqa: F811
    ):
        assert "data_release_versions" not in lookup(graph, affects_query(), bmt=bmt)


class TestBindingShape:
    """2.0 folds each binding's ids into one object per QNode / QEdge."""

    def test_node_binding_is_an_object_with_ids(self, graph, bmt):  # noqa: F811
        result = lookup(graph, affects_query(), bmt=bmt)["message"]["results"][0]
        assert result["node_bindings"]["n0"] == {"ids": [METFORMIN]}

    def test_edge_binding_is_an_object_with_ids(self, graph, bmt):  # noqa: F811
        result = lookup(graph, affects_query(), bmt=bmt)["message"]["results"][0]
        binding = result["analyses"][0]["edge_bindings"]["e0"]
        assert set(binding) == {"ids"}
        assert len(binding["ids"]) >= 1


class TestEdgeProperties:
    """2.0 promotes knowledge_level and agent_type to required Edge properties."""

    @pytest.mark.parametrize("dehydrated", [False, True])
    def test_every_edge_carries_knowledge_level_and_agent_type(
        self, graph, bmt, dehydrated  # noqa: F811
    ):
        response = lookup(graph, affects_query(), bmt=bmt, dehydrated=dehydrated)
        edges = response["message"]["knowledge_graph"]["edges"]
        assert edges
        for edge in edges.values():
            assert edge["knowledge_level"] == "knowledge_assertion"
            assert edge["agent_type"] == "manual_agent"

    def test_knowledge_level_is_not_also_an_attribute(self, graph, bmt):  # noqa: F811
        """They live in one place now, not duplicated into the attribute list."""
        response = lookup(graph, affects_query(), bmt=bmt)
        for edge in response["message"]["knowledge_graph"]["edges"].values():
            type_ids = {a["attribute_type_id"] for a in edge.get("attributes", [])}
            assert "biolink:knowledge_level" not in type_ids
            assert "biolink:agent_type" not in type_ids


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
    """The fixture's affects edges are all knowledge_assertion / manual_agent."""

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
        assert resp.json()["parameters"] == {"timeout": 600}

    def test_malformed_constraint_is_400(self, server):
        _, client = server
        resp = client.post(
            "/query",
            json=affects_query({"knowledge_level": {"behavior": "SOMETIMES"}}),
        )
        assert resp.status_code == 400
        assert "behavior" in resp.json()["detail"]

    @pytest.mark.parametrize(
        "retired, replacement",
        [
            ("qualifier_constraints", "constraints.qualifiers"),
            ("attribute_constraints", "constraints.attributes"),
        ],
    )
    def test_retired_1x_edge_fields_are_rejected(self, server, retired, replacement):
        """Silently ignoring a filter the client asked for is worse than a 400."""
        _, client = server
        body = affects_query()
        body["message"]["query_graph"]["edges"]["e0"][retired] = []
        resp = client.post("/query", json=body)
        assert resp.status_code == 400
        assert replacement in resp.json()["detail"]

    def test_retired_pathfinder_constraint_is_rejected(self, server):
        _, client = server
        body = affects_query()
        body["message"]["query_graph"]["paths"] = {
            "p0": {
                "subject": "n0",
                "object": "n1",
                "constraints": [{"intermediate_categories": ["biolink:Gene"]}],
            }
        }
        resp = client.post("/query", json=body)
        assert resp.status_code == 400
        assert "required_intermediate_categories" in resp.json()["detail"]

    def test_log_level_read_from_parameters(self, server):
        _, client = server
        resp = client.post(
            "/query", json=affects_query(parameters={"log_level": "DEBUG"})
        )
        assert resp.status_code == 200, resp.text
        assert len(resp.json()["logs"]) > 0

    def test_openapi_declares_trapi_2_0(self, server):
        gandalf_server, _ = server
        info = gandalf_server.APP.openapi()["info"]
        assert info["x-trapi"]["version"] == "2.0.0"


class TestStrictValidationMode:
    """``validate_responses`` routes requests and responses through Pydantic.

    2.0 dropped ``nullable`` everywhere, so this path must not turn unset
    optional properties into nulls, and must hand ``log_level`` on as the plain
    string that ``logging.setLevel`` and the JSON encoder both require.
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
            json=affects_query(parameters={"log_level": "ERROR", "timeout": 60}),
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["parameters"]["log_level"] == "ERROR"
        assert body["message"]["results"]

    def test_validated_log_level_is_a_plain_string(self):
        """``logging.setLevel`` rejects a str subclass, enum members included."""
        import logging

        from gandalf.models import TRAPIQuery

        validated = TRAPIQuery.model_validate(
            affects_query(parameters={"log_level": "ERROR"})
        ).model_dump(exclude_none=True)
        log_level = validated["parameters"]["log_level"]
        assert type(log_level) is str
        logging.Logger("test").setLevel(log_level)

    def test_unset_optional_properties_are_absent_not_null(self):
        from gandalf.models import TRAPIResponse

        dumped = TRAPIResponse(
            message={}, status="Success", schema_version="2.0.0"
        ).model_dump(exclude_none=True)
        assert dumped == {
            "message": {},
            "status": "Success",
            "schema_version": "2.0.0",
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
