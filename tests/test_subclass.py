"""Tests for subclass reasoning feature."""

import os

import pytest

from tests.search_fixtures import graph  # noqa: F401

from gandalf.loader import build_graph_from_jsonl
from gandalf.search import lookup

_FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


@pytest.fixture
def multi_child_graph():
    """Graph with one superclass that has two children sharing a phenotype.

    - MONDO:0011122 (Umbrella Disease) is the superclass D
    - MONDO:0044444 (Target Disease) is the shared object nA
    - MONDO:0022222 (Child A) --subclass_of--> D, --has_phenotype--> nA
    - MONDO:0033333 (Child B) --subclass_of--> D, --has_phenotype--> nA
    - CHEBI:9999 --treats--> nA

    There is deliberately no direct ``D --has_phenotype--> nA`` edge, so
    subclass inference (not direct-edge priority) drives the result.
    """
    return build_graph_from_jsonl(
        os.path.join(_FIXTURES_DIR, "subclass_multi_edges.jsonl"),
        os.path.join(_FIXTURES_DIR, "subclass_multi_nodes.jsonl"),
    )


def _inferred_edges(response):
    """Return KG edges carrying a biolink:support_graphs attribute (inferred)."""
    return {
        eid: edge
        for eid, edge in response["message"]["knowledge_graph"]["edges"].items()
        if any(
            a.get("attribute_type_id") == "biolink:support_graphs"
            for a in edge.get("attributes", [])
        )
    }


class TestSubclassHandling:
    """Tests for subclass reasoning feature.

    The test fixtures contain:
    - MONDO:0005148 (Type 2 Diabetes) --subclass_of--> MONDO:0005015 (Diabetes Mellitus)
    - MONDO:0005015 (Diabetes Mellitus) --subclass_of--> MONDO:0004995 (Cardiovascular Disease)
    - CHEBI:6801 (Metformin) --treats--> MONDO:0005148 (Type 2 Diabetes)
    - CHEBI:6801 (Metformin) --treats--> MONDO:0005015 (Diabetes Mellitus)
    """

    def test_subclass_off(self, graph, bmt):
        """Without subclass=True, querying for Diabetes Mellitus only returns exact matches."""
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["CHEBI:6801"]},
                        "n1": {"ids": ["MONDO:0005015"]},  # Diabetes Mellitus
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

        response = lookup(graph, query, bmt=bmt, subclass=False)
        results = response["message"]["results"]

        # Only exact match: Metformin treats Diabetes Mellitus
        assert len(results) == 1
        assert results[0]["node_bindings"]["n1"][0]["id"] == "MONDO:0005015"

    def test_subclass_depth_one_expands_to_children(self, graph, bmt):
        """With subclass=True, querying for Diabetes Mellitus also finds Type 2 Diabetes results.

        Diabetes Mellitus (MONDO:0005015) has child Type 2 Diabetes (MONDO:0005148).
        Metformin treats both, so we should see results for both.
        """
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["CHEBI:6801"]},
                        "n1": {"ids": ["MONDO:0005015"]},  # Diabetes Mellitus (parent)
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

        response = lookup(graph, query, bmt=bmt, subclass=True, subclass_depth=1)
        results = response["message"]["results"]

        # Should find results for both Diabetes Mellitus (direct) and Type 2 Diabetes (subclass)
        assert len(results) == 1

        # Node bindings should reference the originally queried ID (superclass)
        # for results that came via subclass expansion
        bound_ids = {r["node_bindings"]["n1"][0]["id"] for r in results}
        assert "MONDO:0005015" in bound_ids

    def test_subclass_depth_zero_is_identity(self, graph, bmt):
        """With subclass_depth=0, only the exact node matches (equivalent to no subclass)."""
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["CHEBI:6801"]},
                        "n1": {"ids": ["MONDO:0005015"]},
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

        response_no_subclass = lookup(graph, query, bmt=bmt, subclass=False)
        response_depth_zero = lookup(
            graph, query, bmt=bmt, subclass=True, subclass_depth=0
        )

        results_no = response_no_subclass["message"]["results"]
        results_zero = response_depth_zero["message"]["results"]

        # Both should return 1 result: exact match only
        assert len(results_no) == len(results_zero) == 1

    def test_subclass_skips_explicit_hierarchy_edges(self, graph, bmt):
        """Nodes already in explicit subclass_of edges are not rewritten."""
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["MONDO:0005148"]},  # Type 2 Diabetes
                        "n1": {"ids": ["MONDO:0005015"]},  # Diabetes Mellitus
                    },
                    "edges": {
                        "e0": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:subclass_of"],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, bmt=bmt, subclass=True, subclass_depth=1)
        results = response["message"]["results"]

        # Should find the explicit edge without creating synthetic superclass nodes
        assert len(results) == 1
        # Node bindings should use the exact queried IDs (no rewriting happened)
        assert results[0]["node_bindings"]["n0"][0]["id"] == "MONDO:0005148"
        assert results[0]["node_bindings"]["n1"][0]["id"] == "MONDO:0005015"

    def test_subclass_auxiliary_graphs_present(self, graph, bmt):
        """Results from subclass expansion should include auxiliary_graphs.

        Regression test for BioPack-team/retriever#192: a Tier 0 (Gandalf)
        query for a drug treating a parent disease must surface the subclass
        support graph for the child-disease evidence, exactly as Tier 1 does.
        Metformin treats Type 2 Diabetes (a child of Diabetes Mellitus), so the
        response must contain at least one auxiliary graph even though a direct
        Metformin --treats--> Diabetes Mellitus edge also exists.
        """
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["CHEBI:6801"]},
                        "n1": {"ids": ["MONDO:0005015"]},  # Diabetes Mellitus (parent)
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

        response = lookup(graph, query, bmt=bmt, subclass=True, subclass_depth=1)
        aux_graphs = response["message"]["auxiliary_graphs"]

        # auxiliary_graphs should exist and be populated from the subclass path.
        assert isinstance(aux_graphs, dict)
        assert len(aux_graphs) > 0, (
            "Expected subclass support graphs to be present for the child "
            "disease evidence (regression for retriever#192)"
        )

        # Each auxiliary graph holds at least the real edge + the subclass edge.
        for ag_id, ag in aux_graphs.items():
            assert "edges" in ag
            assert len(ag["edges"]) >= 2

    def test_subclass_inferred_edges_have_logical_entailment(self, graph, bmt):
        """Inferred composite edges should have knowledge_level=logical_entailment."""
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["CHEBI:6801"]},
                        "n1": {"ids": ["MONDO:0005015"]},
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

        response = lookup(graph, query, bmt=bmt, subclass=True, subclass_depth=1)
        kg_edges = response["message"]["knowledge_graph"]["edges"]

        # Find inferred edges (those with support_graphs attribute)
        inferred_edges = [
            e
            for e in kg_edges.values()
            if any(
                attr.get("attribute_type_id") == "biolink:support_graphs"
                for attr in e.get("attributes", [])
            )
        ]

        # There should be at least one inferred edge (from subclass expansion),
        # coexisting with the direct Metformin --treats--> Diabetes Mellitus edge.
        assert len(inferred_edges) > 0, (
            "Expected at least one subclass-inferred edge (regression for "
            "retriever#192)"
        )
        for edge in inferred_edges:
            attr_map = {a["attribute_type_id"]: a["value"] for a in edge["attributes"]}
            assert attr_map["biolink:knowledge_level"] == "logical_entailment"
            assert attr_map["biolink:agent_type"] == "automated_agent"

    def test_subclass_node_binding_uses_superclass_id(self, graph, bmt):
        """When a result comes via subclass, node binding should reference the queried (superclass) ID."""
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["CHEBI:6801"]},
                        "n1": {"ids": ["MONDO:0005015"]},  # Diabetes Mellitus (parent)
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

        response = lookup(graph, query, bmt=bmt, subclass=True, subclass_depth=1)
        results = response["message"]["results"]

        # All results should have n0 bound to Metformin
        for result in results:
            assert result["node_bindings"]["n0"][0]["id"] == "CHEBI:6801"

        # n1 bindings: direct match uses MONDO:0005015, subclass match also uses MONDO:0005015
        # (the superclass ID, since that's what was queried)
        n1_ids = {r["node_bindings"]["n1"][0]["id"] for r in results}
        assert "MONDO:0005015" in n1_ids

    def test_subclass_superclass_nodes_hidden_from_bindings(self, graph, bmt):
        """Synthetic superclass nodes should not appear in result node_bindings."""
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["CHEBI:6801"]},
                        "n1": {"ids": ["MONDO:0005015"]},
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

        response = lookup(graph, query, bmt=bmt, subclass=True, subclass_depth=1)
        results = response["message"]["results"]

        for result in results:
            # Only original query node IDs should be in bindings
            assert set(result["node_bindings"].keys()) == {"n0", "n1"}
            # No "_superclass" keys
            for key in result["node_bindings"]:
                assert "_superclass" not in key

    def test_subclass_subclass_edges_hidden_from_bindings(self, graph, bmt):
        """Synthetic subclass edges should not appear in result edge_bindings."""
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["CHEBI:6801"]},
                        "n1": {"ids": ["MONDO:0005015"]},
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

        response = lookup(graph, query, bmt=bmt, subclass=True, subclass_depth=1)
        results = response["message"]["results"]

        for result in results:
            edge_binding_keys = set(result["analyses"][0]["edge_bindings"].keys())
            # Only original query edge IDs should be in bindings
            assert "e0" in edge_binding_keys
            # No "_subclass_edge" keys
            for key in edge_binding_keys:
                assert "_subclass" not in key

    def test_subclass_two_hop_with_expansion(self, graph, bmt):
        """Two-hop query with subclass expansion on one end.

        Query: Metformin --treats--> ? --has_phenotype--> Hypoglycemia
        With subclass on, the disease node should expand to include subclasses.
        Type 2 Diabetes has phenotype Hypoglycemia, and Type 2 Diabetes is
        a subclass of Diabetes Mellitus.
        """
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["CHEBI:6801"]},
                        "n1": {"categories": ["biolink:Disease"]},
                        "n2": {"ids": ["HP:0001943"]},  # Hypoglycemia
                    },
                    "edges": {
                        "e0": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:treats"],
                        },
                        "e1": {
                            "subject": "n1",
                            "object": "n2",
                            "predicates": ["biolink:has_phenotype"],
                        },
                    },
                },
            },
        }

        # Without subclass: n1 must have treats edges AND has_phenotype edges
        response_no = lookup(graph, query, bmt=bmt, subclass=False)
        results_no = response_no["message"]["results"]

        # With subclass: same query but subclass expansion might find more paths
        response_yes = lookup(graph, query, bmt=bmt, subclass=True, subclass_depth=1)
        results_yes = response_yes["message"]["results"]

        # Both should find results through Type 2 Diabetes
        # (Metformin treats T2D, T2D has_phenotype Hypoglycemia)
        assert len(results_no) >= 1
        assert len(results_yes) >= 1

    def test_subclass_response_has_auxiliary_graphs_key(self, graph, bmt):
        """Even without subclass, response should have auxiliary_graphs key."""
        query = {
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

        response = lookup(graph, query, bmt=bmt)
        assert "auxiliary_graphs" in response["message"]
        assert isinstance(response["message"]["auxiliary_graphs"], dict)


class TestSubclassMultipleChildrenDistinctDerivations:
    """Regression tests for issue #39.

    When a pinned superclass has two children that each support the same
    inferred edge (same subject/predicate/object), each child forms a distinct
    derivation.  Hydration must keep the two inferred edges distinct, give each
    its own support graph containing exactly that child's underlying edges, and
    must not reference the same KG edge twice in one QEdge binding or drop a
    derivation.  The bug attached every sibling's ``subclass_of`` edge to every
    base edge, collapsing the derivations in the hydrated (``dehydrated:false``)
    response.
    """

    C1 = "MONDO:0022222"
    C2 = "MONDO:0033333"
    D = "MONDO:0011122"
    NA = "MONDO:0044444"

    def _assert_two_distinct_derivations(self, response, base_qedge_id):
        kg_edges = response["message"]["knowledge_graph"]["edges"]
        aux_graphs = response["message"]["auxiliary_graphs"]

        # Exactly one result: both children collapse into one node-binding group
        # under the superclass D.
        results = response["message"]["results"]
        assert len(results) == 1

        # Two distinct inferred composite edges (one per child derivation).
        inferred = _inferred_edges(response)
        assert (
            len(inferred) == 2
        ), f"expected 2 inferred edges, got {len(inferred)}: {list(inferred)}"

        # Two auxiliary graphs, each holding exactly its own [phenotype, subclass]
        # pair -- no cross-contamination with the sibling's subclass_of edge.
        aux_ids_from_inferred = set()
        child_subjects = set()
        for edge in inferred.values():
            sg = next(
                a["value"]
                for a in edge["attributes"]
                if a["attribute_type_id"] == "biolink:support_graphs"
            )
            assert len(sg) == 1
            aux_ids_from_inferred.add(sg[0])

        assert len(aux_ids_from_inferred) == 2, "inferred edges share a support graph"
        assert len(aux_graphs) == 2

        for aux_id in aux_ids_from_inferred:
            member_ids = aux_graphs[aux_id]["edges"]
            assert len(member_ids) == 2, (
                f"aux graph {aux_id} should contain exactly the phenotype edge "
                f"and its subclass edge, got {member_ids}"
            )
            members = [kg_edges[mid] for mid in member_ids]
            preds = {m["predicate"] for m in members}
            assert preds == {"biolink:has_phenotype", "biolink:subclass_of"}

            pheno = next(
                m for m in members if m["predicate"] == "biolink:has_phenotype"
            )
            sub = next(m for m in members if m["predicate"] == "biolink:subclass_of")
            # Both underlying edges must belong to the SAME child.
            assert pheno["subject"] == sub["subject"], (
                f"support graph {aux_id} mixes children: "
                f"{pheno['subject']} vs {sub['subject']}"
            )
            assert sub["object"] == self.D
            child_subjects.add(pheno["subject"])

        # Both children are represented -- neither derivation dropped.
        assert child_subjects == {self.C1, self.C2}

        # The QEdge binding references two distinct inferred edges, no duplicates.
        bindings = results[0]["analyses"][0]["edge_bindings"][base_qedge_id]
        ids = [b["id"] for b in bindings]
        assert len(ids) == 2
        assert len(set(ids)) == 2, f"duplicate edge binding ids: {ids}"
        assert set(ids) == set(inferred)

    @pytest.mark.parametrize("dehydrated", [False, True])
    def test_one_hop_two_children_stay_distinct(
        self, multi_child_graph, bmt, dehydrated
    ):
        """Single-edge query: D --has_phenotype--> nA with subclass expansion."""
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": [self.D]},
                        "n1": {"ids": [self.NA]},
                    },
                    "edges": {
                        "e0": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:has_phenotype"],
                        },
                    },
                },
            },
        }

        response = lookup(
            multi_child_graph,
            query,
            bmt=bmt,
            subclass=True,
            subclass_depth=1,
            dehydrated=dehydrated,
        )
        self._assert_two_distinct_derivations(response, "e0")

    @pytest.mark.parametrize("dehydrated", [False, True])
    def test_two_hop_two_children_stay_distinct(
        self, multi_child_graph, bmt, dehydrated
    ):
        """Two-hop query mirroring issue #39: SN --treats--> nA, D --has_phenotype--> nA.

        The subclass-expanded ``has_phenotype`` edge is the second hop; both
        children must still yield distinct, correctly-scoped inferred edges in
        both dehydrated modes.
        """
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "SN": {"ids": ["CHEBI:9999"]},
                        "nA": {"ids": [self.NA]},
                        "ON": {"ids": [self.D]},
                    },
                    "edges": {
                        "eA": {
                            "subject": "SN",
                            "object": "nA",
                            "predicates": ["biolink:treats"],
                        },
                        "eB": {
                            "subject": "ON",
                            "object": "nA",
                            "predicates": ["biolink:has_phenotype"],
                        },
                    },
                },
            },
        }

        response = lookup(
            multi_child_graph,
            query,
            bmt=bmt,
            subclass=True,
            subclass_depth=1,
            dehydrated=dehydrated,
        )
        self._assert_two_distinct_derivations(response, "eB")
