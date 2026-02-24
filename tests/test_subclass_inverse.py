"""Tests for composite edge direction with inverse edges and subclass expansion.

Regression tests for a bug where inferred/composite edges had wrong endpoints
when the underlying edge was found via inverse lookup and the query had
subclass expansion.  The root cause was that superclass_node_overrides used
query-direction labels ("subject"/"object") but edge dicts stored endpoints
in the *stored* edge direction, which is swapped for inverse matches.

Test graph relationships used:
- CHEBI:6801 (Metformin) --treats--> MONDO:0005148 (Type 2 Diabetes)
- CHEBI:6801 (Metformin) --treats--> MONDO:0005015 (Diabetes Mellitus)
- MONDO:0005148 --has_phenotype--> HP:0001943 (Hypoglycemia)
- MONDO:0005148 --subclass_of--> MONDO:0005015
- MONDO:0005015 --subclass_of--> MONDO:0004995
"""

from tests.search_fixtures import graph  # noqa: F401

from gandalf.search import lookup


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


def _assert_all_results_connected(response):
    """Assert every result's edge bindings reference KG edges whose endpoints
    appear in the result's node bindings."""
    kg_edges = response["message"]["knowledge_graph"]["edges"]
    for ri, result in enumerate(response["message"]["results"]):
        bound_node_ids = set()
        for bindings in result["node_bindings"].values():
            for b in bindings:
                bound_node_ids.add(b["id"])

        for analysis in result.get("analyses", []):
            for qeid, ebindings in analysis.get("edge_bindings", {}).items():
                for eb in ebindings:
                    eid = eb["id"]
                    kg_edge = kg_edges.get(eid)
                    assert kg_edge is not None, (
                        f"result[{ri}] references missing KG edge {eid}"
                    )
                    subj = kg_edge["subject"]
                    obj = kg_edge["object"]
                    assert subj in bound_node_ids, (
                        f"result[{ri}] edge {eid}: subject {subj} not in "
                        f"bound nodes {bound_node_ids}"
                    )
                    assert obj in bound_node_ids, (
                        f"result[{ri}] edge {eid}: object {obj} not in "
                        f"bound nodes {bound_node_ids}"
                    )


class TestSubclassInverseEdgeDirection:
    """Tests that composite edges have correct endpoints when the underlying
    edge is found via inverse lookup combined with subclass expansion."""

    def test_inverse_edge_with_subclass_on_subject(self, graph, bmt):
        """Inverse edge + subclass on subject should produce correct composite.

        Query: n0 (disease, pinned MONDO:0005015) -> n1 (chemical, pinned CHEBI:6801)
        predicate: related_to (wildcard, matches all predicates in both directions)

        Stored edge: CHEBI:6801 --treats--> MONDO:0005148 (chemical -> disease)
        Query direction: disease -> chemical (opposite of stored)
        This edge is found via inverse lookup.

        With subclass on n0, MONDO:0005015 expands to include MONDO:0005148.
        The composite/inferred edge should connect:
          subject = MONDO:0005015 (superclass, queried ID)
          object  = CHEBI:6801 (chemical)

        Before the fix, the inverse caused subject/object to be swapped in the
        override logic, producing a disease-to-disease composite edge.
        """
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["MONDO:0005015"]},   # Diabetes Mellitus
                        "n1": {"ids": ["CHEBI:6801"]},       # Metformin
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

        response = lookup(graph, query, bmt=bmt, verbose=False,
                          subclass=True, subclass_depth=1)
        results = response["message"]["results"]
        assert len(results) >= 1

        inferred = _get_inferred_edges(response)
        # There should be at least one inferred edge from subclass expansion
        assert len(inferred) > 0, "Expected inferred composite edges from subclass expansion"

        for eid, edge in inferred.items():
            # The inferred edge should connect the superclass disease to the
            # chemical, NOT disease-to-disease.
            endpoints = {edge["subject"], edge["object"]}
            assert "CHEBI:6801" in endpoints, (
                f"Inferred edge {eid} missing CHEBI:6801: "
                f"{edge['subject']} -> {edge['object']}"
            )
            assert "MONDO:0005015" in endpoints, (
                f"Inferred edge {eid} missing MONDO:0005015: "
                f"{edge['subject']} -> {edge['object']}"
            )

        _assert_all_results_connected(response)

    def test_inverse_edge_with_subclass_on_object(self, graph, bmt):
        """Inverse edge + subclass on object should produce correct composite.

        Query: n0 (phenotype, pinned HP:0001943) -> n1 (disease, pinned MONDO:0005015)
        predicate: related_to

        Stored edge: MONDO:0005148 --has_phenotype--> HP:0001943 (disease -> phenotype)
        Query direction: phenotype -> disease (opposite of stored)
        This edge is found via inverse lookup.

        With subclass on n1, MONDO:0005015 expands to include MONDO:0005148.
        The composite/inferred edge should connect:
          subject = HP:0001943 (phenotype)
          object  = MONDO:0005015 (superclass disease)

        Before the fix, the override replaced the wrong endpoint, producing
        a disease-to-disease composite edge.
        """
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["HP:0001943"]},       # Hypoglycemia
                        "n1": {"ids": ["MONDO:0005015"]},    # Diabetes Mellitus
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

        response = lookup(graph, query, bmt=bmt, verbose=False,
                          subclass=True, subclass_depth=1)
        results = response["message"]["results"]
        assert len(results) >= 1

        inferred = _get_inferred_edges(response)
        assert len(inferred) > 0, "Expected inferred composite edges from subclass expansion"

        for eid, edge in inferred.items():
            # Inferred edge should connect phenotype to superclass disease,
            # NOT disease-to-disease.
            endpoints = {edge["subject"], edge["object"]}
            assert "HP:0001943" in endpoints, (
                f"Inferred edge {eid} missing HP:0001943: "
                f"{edge['subject']} -> {edge['object']}"
            )
            assert "MONDO:0005015" in endpoints, (
                f"Inferred edge {eid} missing MONDO:0005015: "
                f"{edge['subject']} -> {edge['object']}"
            )

        _assert_all_results_connected(response)

    def test_forward_edge_with_subclass_still_works(self, graph, bmt):
        """Forward edge + subclass should still produce correct composite (control test).

        Query: n0 (chemical, pinned CHEBI:6801) -> n1 (disease, pinned MONDO:0005015)
        predicate: treats

        Stored edge: CHEBI:6801 --treats--> MONDO:0005148 (same direction as query)
        Forward match — no inverse involved.

        The composite edge should connect:
          subject = CHEBI:6801 (chemical)
          object  = MONDO:0005015 (superclass disease)
        """
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

        response = lookup(graph, query, bmt=bmt, verbose=False,
                          subclass=True, subclass_depth=1)
        results = response["message"]["results"]
        assert len(results) >= 1

        inferred = _get_inferred_edges(response)

        for eid, edge in inferred.items():
            # Inferred edge: chemical -> superclass disease
            endpoints = {edge["subject"], edge["object"]}
            assert "CHEBI:6801" in endpoints, (
                f"Inferred edge {eid} missing CHEBI:6801: "
                f"{edge['subject']} -> {edge['object']}"
            )
            assert "MONDO:0005015" in endpoints, (
                f"Inferred edge {eid} missing MONDO:0005015: "
                f"{edge['subject']} -> {edge['object']}"
            )

        _assert_all_results_connected(response)

    def test_no_internal_markers_in_response(self, graph, bmt):
        """Internal _query_subject/_query_object markers must not leak into response."""
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

        response = lookup(graph, query, bmt=bmt, verbose=False,
                          subclass=True, subclass_depth=1)
        for eid, edge in response["message"]["knowledge_graph"]["edges"].items():
            assert "_query_subject" not in edge, (
                f"KG edge {eid} leaks _query_subject"
            )
            assert "_query_object" not in edge, (
                f"KG edge {eid} leaks _query_object"
            )
            assert "_edge_id" not in edge, (
                f"KG edge {eid} leaks _edge_id"
            )

    def test_two_hop_inverse_with_subclass(self, graph, bmt):
        """Two-hop query where one hop is inverse + subclass expanded.

        Query: n0 (chemical, pinned CHEBI:6801) -> n1 (disease) -> n2 (phenotype, pinned HP:0001943)
        e0: treats (forward match)
        e1: has_phenotype (forward match)
        subclass on n2 shouldn't apply (phenotype isn't pinned with subclass)

        Flip e1 to test inverse:
        n0 (chemical) -> n1 (disease, subclass MONDO:0005015) <- n2 (phenotype)
        Not directly testable because the query graph direction must be specified.

        Instead: n0 (CHEBI:6801) -> n1 (gene) -> n2 (disease, MONDO:0005015)
        e0: affects (forward), e1: related_to (finds gene_associated_with_condition forward)

        Validate overall connectivity for the two-hop case.
        """
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["CHEBI:6801"]},
                        "n1": {"categories": ["biolink:Gene"]},
                        "n2": {"ids": ["MONDO:0005015"]},
                    },
                    "edges": {
                        "e0": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:affects"],
                        },
                        "e1": {
                            "subject": "n1",
                            "object": "n2",
                            "predicates": ["biolink:related_to"],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, bmt=bmt, verbose=False,
                          subclass=True, subclass_depth=1)
        results = response["message"]["results"]
        assert len(results) >= 1

        # All results must have connected edges
        _assert_all_results_connected(response)

        # Node bindings should use queried IDs
        for result in results:
            assert result["node_bindings"]["n0"][0]["id"] == "CHEBI:6801"
            assert result["node_bindings"]["n2"][0]["id"] == "MONDO:0005015"

    def test_all_kg_edge_endpoints_in_kg_nodes(self, graph, bmt):
        """Every KG edge's subject and object must exist in KG nodes."""
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

        response = lookup(graph, query, bmt=bmt, verbose=False,
                          subclass=True, subclass_depth=1)
        kg_nodes = set(response["message"]["knowledge_graph"]["nodes"].keys())
        kg_edges = response["message"]["knowledge_graph"]["edges"]

        for eid, edge in kg_edges.items():
            assert edge["subject"] in kg_nodes, (
                f"KG edge {eid}: subject {edge['subject']} not in KG nodes"
            )
            assert edge["object"] in kg_nodes, (
                f"KG edge {eid}: object {edge['object']} not in KG nodes"
            )
