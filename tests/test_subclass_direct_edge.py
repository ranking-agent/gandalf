"""Tests: a direct edge and subclass-inferred edges coexist in the result.

When subclass expansion is enabled and a direct edge already connects the
originally-queried (superclass) nodes, the direct edge is bound as-is AND any
additional knowledge reachable only through child (subclass) nodes is surfaced
as inferred composite edges backed by subclass support graphs.

This mirrors the Tier 1 reference behavior (retriever's ``solve_subclass_edges``):
a direct assertion never suppresses the subclass-derived evidence.  Regression
test for BioPack-team/retriever#192, where Tier 0 (Gandalf) returned no
auxiliary graphs while Tier 1 correctly returned subclass support graphs for
the same query.

Test graph relationships used:
- CHEBI:6801 (Metformin) --treats--> MONDO:0005148 (Type 2 Diabetes)
- CHEBI:6801 (Metformin) --treats--> MONDO:0005015 (Diabetes Mellitus)
- MONDO:0005148 --subclass_of--> MONDO:0005015
- MONDO:0005015 --subclass_of--> MONDO:0004995
- MONDO:0005148 --has_phenotype--> HP:0001943 (Hypoglycemia)
"""

from tests.search_fixtures import graph  # noqa: F401

from gandalf.config import settings
from gandalf.search import lookup

TREATS_QUERY = {
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


def _get_inferred_edges(response):
    """Return KG edges that have a biolink:support_graphs attribute (inferred)."""
    return {
        eid: edge
        for eid, edge in response["message"]["knowledge_graph"]["edges"].items()
        if any(
            a.get("attribute_type_id") == "biolink:support_graphs"
            for a in edge.get("attributes", [])
        )
    }


def _has_support_graph(edge):
    return any(
        a.get("attribute_type_id") == "biolink:support_graphs"
        for a in edge.get("attributes", [])
    )


class TestDirectEdgeCoexistsWithSubclass:
    """When a direct edge exists between the queried nodes, it is emitted
    alongside the subclass-expanded (inferred) edges rather than suppressing
    them."""

    def test_direct_edge_bound_directly(self, graph, bmt):
        """The direct treats edge is present in the KG and bound in the result
        as a plain (non-inferred) edge."""
        response = lookup(graph, TREATS_QUERY, bmt=bmt, subclass=True, subclass_depth=1)

        kg_edges = response["message"]["knowledge_graph"]["edges"]
        results = response["message"]["results"]
        assert len(results) == 1

        bound_ids = {
            eb["id"]
            for result in results
            for eb in result["analyses"][0]["edge_bindings"]["e0"]
        }

        direct_bound = [
            eid
            for eid in bound_ids
            if eid in kg_edges
            and kg_edges[eid]["subject"] == "CHEBI:6801"
            and kg_edges[eid]["object"] == "MONDO:0005015"
            and kg_edges[eid]["predicate"] == "biolink:treats"
            and not _has_support_graph(kg_edges[eid])
        ]
        assert len(direct_bound) >= 1, (
            "Expected the direct CHEBI:6801 --treats--> MONDO:0005015 edge to be "
            "bound directly (without a support graph)"
        )

    def test_subclass_of_edges_present_when_direct_exists(self, graph, bmt):
        """subclass_of edges appear in the KG as part of the support graphs,
        even though a direct edge also connects the queried nodes."""
        response = lookup(graph, TREATS_QUERY, bmt=bmt, subclass=True, subclass_depth=1)

        kg_edges = response["message"]["knowledge_graph"]["edges"]
        subclass_edges = [
            e for e in kg_edges.values() if e["predicate"] == "biolink:subclass_of"
        ]
        assert len(subclass_edges) > 0, (
            "Expected subclass_of edges in KG so the subclass-derived evidence "
            "is not lost when a direct edge exists"
        )

    def test_child_node_present_when_direct_exists(self, graph, bmt):
        """The child node providing the subclass path appears in the KG."""
        response = lookup(graph, TREATS_QUERY, bmt=bmt, subclass=True, subclass_depth=1)

        kg_nodes = set(response["message"]["knowledge_graph"]["nodes"].keys())
        assert "MONDO:0005148" in kg_nodes, (
            "Child node MONDO:0005148 should appear in KG as part of the "
            "subclass support graph"
        )

    def test_inferred_edges_present_when_direct_exists(self, graph, bmt):
        """Composite inferred edges are created for the child paths even though
        a direct edge exists."""
        response = lookup(graph, TREATS_QUERY, bmt=bmt, subclass=True, subclass_depth=1)

        inferred = _get_inferred_edges(response)
        assert len(inferred) > 0, (
            "Expected inferred composite edges from subclass expansion to "
            "coexist with the direct edge"
        )
        # Every inferred edge connects the originally-queried (superclass) nodes.
        for eid, edge in inferred.items():
            assert edge["subject"] == "CHEBI:6801", (
                f"Inferred edge {eid} subject should be CHEBI:6801, got "
                f"{edge['subject']}"
            )
            assert edge["object"] == "MONDO:0005015", (
                f"Inferred edge {eid} object should be MONDO:0005015, got "
                f"{edge['object']}"
            )

    def test_auxiliary_graphs_present_when_direct_exists(self, graph, bmt):
        """auxiliary_graphs are populated with subclass support graphs."""
        response = lookup(graph, TREATS_QUERY, bmt=bmt, subclass=True, subclass_depth=1)

        aux_graphs = response["message"]["auxiliary_graphs"]
        assert len(aux_graphs) > 0, (
            "Expected non-empty auxiliary_graphs when a subclass path exists, "
            "regardless of a direct edge also being present"
        )
        # Each support graph carries at least the real edge and the subclass edge.
        for ag_id, ag in aux_graphs.items():
            assert len(ag["edges"]) >= 2, (
                f"Support graph {ag_id} should contain at least the real edge "
                f"and a subclass_of edge"
            )

    def test_inferred_edge_binding_pairs_correct_subclass_edge(self, graph, bmt):
        """Each inferred composite's support graph pairs the child edge with the
        subclass_of edge of *that same* child (no cross-child mispairing)."""
        response = lookup(graph, TREATS_QUERY, bmt=bmt, subclass=True, subclass_depth=1)

        kg_edges = response["message"]["knowledge_graph"]["edges"]
        aux_graphs = response["message"]["auxiliary_graphs"]

        for ag_id, ag in aux_graphs.items():
            edge_ids = ag["edges"]
            subclass_children = {
                kg_edges[eid]["subject"]
                for eid in edge_ids
                if kg_edges[eid]["predicate"] == "biolink:subclass_of"
            }
            # The non-subclass (knowledge) edges in the same support graph must
            # be incident to the very child asserted by the subclass_of edge.
            for eid in edge_ids:
                edge = kg_edges[eid]
                if edge["predicate"] == "biolink:subclass_of":
                    continue
                incident = {edge["subject"], edge["object"]}
                assert incident & subclass_children, (
                    f"Support graph {ag_id}: knowledge edge {eid} "
                    f"({edge['subject']} -> {edge['object']}) is not incident to "
                    f"its subclass child(ren) {subclass_children}"
                )


class TestSubclassCompositeWhenNoDirectEdge:
    """When NO direct edge exists between the queried nodes, subclass-expanded
    edges produce composite inferred edges as before."""

    def test_composite_created_when_only_subclass_path_exists(self, graph, bmt):
        """Composite inferred edge should be created when no direct edge exists.

        Query: HP:0001943 --related_to--> MONDO:0005015 with subclass=True.
        No direct edge between HP:0001943 and MONDO:0005015 exists.
        Subclass path: MONDO:0005148 --has_phenotype--> HP:0001943 (inverse)
                     + MONDO:0005148 --subclass_of--> MONDO:0005015
        A composite inferred edge should be created.
        """
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["HP:0001943"]},  # Hypoglycemia
                        "n1": {"ids": ["MONDO:0005015"]},  # Diabetes Mellitus
                    },
                    "edges": {
                        "e0": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:related_to"],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, bmt=bmt, subclass=True, subclass_depth=1)

        results = response["message"]["results"]
        assert len(results) >= 1

        inferred = _get_inferred_edges(response)
        assert len(inferred) > 0, (
            "Expected inferred composite edges when no direct edge exists "
            "and the path goes through subclass expansion"
        )

        # The inferred edge should connect the queried nodes
        for eid, edge in inferred.items():
            endpoints = {edge["subject"], edge["object"]}
            assert "HP:0001943" in endpoints, (
                f"Inferred edge {eid} missing HP:0001943: "
                f"{edge['subject']} -> {edge['object']}"
            )
            assert "MONDO:0005015" in endpoints, (
                f"Inferred edge {eid} missing MONDO:0005015: "
                f"{edge['subject']} -> {edge['object']}"
            )

    def test_inferred_edge_primary_source_is_obie(self, graph, bmt):
        """Subclass-inferred composite edges must credit infores:obie.

        The logical entailment comes from ontology-based subclass inference,
        not from Gandalf's own graph, so the primary_knowledge_source is
        infores:obie and Gandalf appears only as the aggregator that returned
        it (upstream of the obie primary).
        """
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["HP:0001943"]},
                        "n1": {"ids": ["MONDO:0005015"]},
                    },
                    "edges": {
                        "e0": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:related_to"],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, bmt=bmt, subclass=True, subclass_depth=1)

        inferred = _get_inferred_edges(response)
        assert len(inferred) > 0, "Expected at least one inferred composite edge"

        for eid, edge in inferred.items():
            sources = edge["sources"]

            primary = [
                s for s in sources if s["resource_role"] == "primary_knowledge_source"
            ]
            assert len(primary) == 1, (
                f"Inferred edge {eid} must have exactly one "
                f"primary_knowledge_source, found {len(primary)}"
            )
            assert primary[0]["resource_id"] == settings.subclass_inference_infores, (
                f"Inferred edge {eid} primary source should be "
                f"{settings.subclass_inference_infores}, got "
                f"{primary[0]['resource_id']}"
            )

            # Gandalf itself must remain in the provenance as an aggregator
            # sitting on top of the obie primary source.
            aggregator = next(
                (s for s in sources if s["resource_id"] == settings.infores), None
            )
            assert aggregator is not None, (
                f"Inferred edge {eid} should record {settings.infores} as an "
                "aggregator_knowledge_source"
            )
            assert aggregator["resource_role"] == "aggregator_knowledge_source"
            assert (
                settings.subclass_inference_infores
                in aggregator["upstream_resource_ids"]
            )

    def test_subclass_of_edge_present_when_no_direct(self, graph, bmt):
        """subclass_of edge should appear in KG when no direct edge exists.

        The subclass_of edge is part of the support graph for the composite.
        """
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["HP:0001943"]},
                        "n1": {"ids": ["MONDO:0005015"]},
                    },
                    "edges": {
                        "e0": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:related_to"],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, bmt=bmt, subclass=True, subclass_depth=1)

        kg_edges = response["message"]["knowledge_graph"]["edges"]
        subclass_edges = [
            e for e in kg_edges.values() if e["predicate"] == "biolink:subclass_of"
        ]
        assert len(subclass_edges) > 0, (
            "Expected subclass_of edges in KG when no direct edge exists "
            "and the result goes through subclass expansion"
        )

    def test_child_node_present_when_no_direct(self, graph, bmt):
        """Child node should appear in KG when no direct edge exists.

        MONDO:0005148 (Type 2 Diabetes) is the child that provides the path.
        """
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["HP:0001943"]},
                        "n1": {"ids": ["MONDO:0005015"]},
                    },
                    "edges": {
                        "e0": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:related_to"],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, bmt=bmt, subclass=True, subclass_depth=1)

        kg_nodes = set(response["message"]["knowledge_graph"]["nodes"].keys())
        assert "MONDO:0005148" in kg_nodes, (
            "Child node MONDO:0005148 should appear in KG when no direct edge "
            "exists and the result goes through subclass expansion"
        )


class TestDirectEdgeCoexistsWithSubclassInverse:
    """Same coexistence guarantee for queries whose edge is found via inverse."""

    def test_inverse_direct_and_subclass_coexist(self, graph, bmt):
        """Inverse edge: direct edge and subclass inference coexist.

        Query: MONDO:0005015 --related_to--> CHEBI:6801 (disease -> chemical).
        Stored edge: CHEBI:6801 --treats--> MONDO:0005015 (found via inverse).
        A direct match exists, and subclass-expanded paths through MONDO:0005148
        add inferred edges; both are surfaced.
        """
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["MONDO:0005015"]},
                        "n1": {"ids": ["CHEBI:6801"]},
                    },
                    "edges": {
                        "e0": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:related_to"],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, bmt=bmt, subclass=True, subclass_depth=1)

        results = response["message"]["results"]
        assert len(results) >= 1

        # subclass_of edges should be present (support graph evidence)
        kg_edges = response["message"]["knowledge_graph"]["edges"]
        subclass_edges = [
            e for e in kg_edges.values() if e["predicate"] == "biolink:subclass_of"
        ]
        assert len(subclass_edges) > 0, (
            "Expected subclass_of edges with inverse lookup when a subclass "
            "path exists alongside the direct edge"
        )

        # Inferred composite edges connect the queried nodes.
        inferred = _get_inferred_edges(response)
        assert len(inferred) > 0, (
            "Expected inferred edges with inverse lookup when a subclass path "
            "exists alongside the direct edge"
        )
        for eid, edge in inferred.items():
            endpoints = {edge["subject"], edge["object"]}
            assert endpoints == {"MONDO:0005015", "CHEBI:6801"}, (
                f"Inferred edge {eid} should connect the queried nodes, got "
                f"{edge['subject']} -> {edge['object']}"
            )

        # Child node present in KG.
        kg_nodes = set(response["message"]["knowledge_graph"]["nodes"].keys())
        assert "MONDO:0005148" in kg_nodes
