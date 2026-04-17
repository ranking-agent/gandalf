"""Tests for node-level set_interpretation behavior."""

import pytest

from tests.search_fixtures import graph  # noqa: F401

from gandalf.request_validation import validate_set_interpretation
from gandalf.search import lookup

# ---------------------------------------------------------------------------
# Server-level validation
# ---------------------------------------------------------------------------


class TestSetInterpretationValidation:
    """Tests for validate_set_interpretation."""

    def test_batch_accepted(self):
        qg = {"nodes": {"n0": {"ids": ["A"], "set_interpretation": "BATCH"}}}
        validate_set_interpretation(qg)  # should not raise

    def test_all_accepted(self):
        qg = {"nodes": {"n0": {"ids": ["A", "B"], "set_interpretation": "ALL"}}}
        validate_set_interpretation(qg)  # should not raise

    def test_collate_accepted(self):
        qg = {
            "nodes": {
                "n0": {
                    "categories": ["biolink:Gene"],
                    "set_interpretation": "COLLATE",
                }
            }
        }
        validate_set_interpretation(qg)  # should not raise

    def test_many_rejected(self):
        from fastapi import HTTPException

        qg = {"nodes": {"n0": {"set_interpretation": "MANY"}}}
        with pytest.raises(HTTPException) as exc_info:
            validate_set_interpretation(qg)
        assert exc_info.value.status_code == 422
        assert "MANY" in str(exc_info.value.detail)

    def test_all_without_ids_rejected(self):
        from fastapi import HTTPException

        qg = {
            "nodes": {
                "n0": {
                    "categories": ["biolink:Gene"],
                    "set_interpretation": "ALL",
                }
            }
        }
        with pytest.raises(HTTPException) as exc_info:
            validate_set_interpretation(qg)
        assert exc_info.value.status_code == 422

    def test_collate_with_ids_rejected(self):
        from fastapi import HTTPException

        qg = {"nodes": {"n0": {"ids": ["A"], "set_interpretation": "COLLATE"}}}
        with pytest.raises(HTTPException) as exc_info:
            validate_set_interpretation(qg)
        assert exc_info.value.status_code == 422

    def test_no_interpretation_passes(self):
        qg = {"nodes": {"n0": {"ids": ["A"]}, "n1": {"categories": ["biolink:Gene"]}}}
        validate_set_interpretation(qg)  # should not raise


# ---------------------------------------------------------------------------
# BATCH mode (default behavior, regression)
# ---------------------------------------------------------------------------


class TestBatchMode:
    """Verify BATCH mode produces one binding per node per result."""

    def test_batch_default_one_hop(self, graph, bmt):
        """Default (absent set_interpretation) = BATCH: one result per gene."""
        query = {
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
                        }
                    },
                }
            }
        }
        response = lookup(graph, query, bmt=bmt, subclass=False)
        results = response["message"]["results"]

        # Metformin affects 4 genes: PPARG, INSR, GCK, TNF
        assert len(results) == 4
        for r in results:
            assert len(r["node_bindings"]["n1"]) == 1

    def test_batch_explicit_same_as_default(self, graph, bmt):
        """Explicit BATCH produces same results as default."""
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["CHEBI:6801"]},
                        "n1": {
                            "categories": ["biolink:Gene"],
                            "set_interpretation": "BATCH",
                        },
                    },
                    "edges": {
                        "e0": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:affects"],
                        }
                    },
                }
            }
        }
        response = lookup(graph, query, bmt=bmt, subclass=False)
        results = response["message"]["results"]
        assert len(results) == 4
        for r in results:
            assert len(r["node_bindings"]["n1"]) == 1


# ---------------------------------------------------------------------------
# ALL mode
# ---------------------------------------------------------------------------


class TestAllMode:
    """Tests for set_interpretation ALL."""

    def test_all_mode_both_ids_found(self, graph, bmt):
        """ALL with two gene IDs that both have 'affects' edges from Metformin.

        Should produce 1 result where n1 has 2 bindings.
        """
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["CHEBI:6801"]},
                        "n1": {
                            "ids": ["NCBIGene:5468", "NCBIGene:3643"],
                            "set_interpretation": "ALL",
                        },
                    },
                    "edges": {
                        "e0": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:affects"],
                        }
                    },
                }
            }
        }
        response = lookup(graph, query, bmt=bmt, subclass=False)
        results = response["message"]["results"]

        assert len(results) == 1
        n1_bindings = results[0]["node_bindings"]["n1"]
        assert len(n1_bindings) == 2
        bound_ids = {b["id"] for b in n1_bindings}
        assert bound_ids == {"NCBIGene:5468", "NCBIGene:3643"}

    def test_all_mode_missing_id_returns_no_results(self, graph, bmt):
        """ALL with one valid and one nonexistent ID -> 0 results."""
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["CHEBI:6801"]},
                        "n1": {
                            "ids": ["NCBIGene:5468", "NONEXISTENT:0000"],
                            "set_interpretation": "ALL",
                        },
                    },
                    "edges": {
                        "e0": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:affects"],
                        }
                    },
                }
            }
        }
        response = lookup(graph, query, bmt=bmt, subclass=False)
        results = response["message"]["results"]
        assert len(results) == 0

    def test_all_mode_single_id_same_as_batch(self, graph, bmt):
        """ALL with a single ID behaves like BATCH (1 result, 1 binding)."""
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["CHEBI:6801"]},
                        "n1": {
                            "ids": ["NCBIGene:5468"],
                            "set_interpretation": "ALL",
                        },
                    },
                    "edges": {
                        "e0": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:affects"],
                        }
                    },
                }
            }
        }
        response = lookup(graph, query, bmt=bmt, subclass=False)
        results = response["message"]["results"]
        assert len(results) == 1
        assert len(results[0]["node_bindings"]["n1"]) == 1

    def test_all_mode_kg_contains_all_ids(self, graph, bmt):
        """ALL mode should include all bound IDs in the knowledge graph."""
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["CHEBI:6801"]},
                        "n1": {
                            "ids": ["NCBIGene:5468", "NCBIGene:3643"],
                            "set_interpretation": "ALL",
                        },
                    },
                    "edges": {
                        "e0": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:affects"],
                        }
                    },
                }
            }
        }
        response = lookup(graph, query, bmt=bmt, subclass=False)
        kg_nodes = response["message"]["knowledge_graph"]["nodes"]
        assert "NCBIGene:5468" in kg_nodes
        assert "NCBIGene:3643" in kg_nodes

    def test_all_mode_two_hop(self, graph, bmt):
        """ALL on intermediate node in two-hop query.

        n0=Metformin -affects-> n1(ALL: PPARG,INSR) -gene_assoc-> n2=T2D
        Both PPARG and INSR are associated with T2D, so 1 result expected.
        """
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["CHEBI:6801"]},
                        "n1": {
                            "ids": ["NCBIGene:5468", "NCBIGene:3643"],
                            "set_interpretation": "ALL",
                        },
                        "n2": {"ids": ["MONDO:0005148"]},
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
                            "predicates": ["biolink:gene_associated_with_condition"],
                        },
                    },
                }
            }
        }
        response = lookup(graph, query, bmt=bmt, subclass=False)
        results = response["message"]["results"]

        assert len(results) == 1
        n1_bindings = results[0]["node_bindings"]["n1"]
        bound_ids = {b["id"] for b in n1_bindings}
        assert bound_ids == {"NCBIGene:5468", "NCBIGene:3643"}


# ---------------------------------------------------------------------------
# COLLATE mode
# ---------------------------------------------------------------------------


class TestCollateMode:
    """Tests for set_interpretation COLLATE."""

    def test_collate_merges_results(self, graph, bmt):
        """COLLATE on unpinned gene node merges all genes into one result.

        Without COLLATE: 4 results (one per gene).
        With COLLATE: 1 result with all 4 genes in n1 bindings.
        """
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["CHEBI:6801"]},
                        "n1": {
                            "categories": ["biolink:Gene"],
                            "set_interpretation": "COLLATE",
                        },
                    },
                    "edges": {
                        "e0": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:affects"],
                        }
                    },
                }
            }
        }
        response = lookup(graph, query, bmt=bmt, subclass=False)
        results = response["message"]["results"]

        assert len(results) == 1
        n1_bindings = results[0]["node_bindings"]["n1"]
        bound_ids = {b["id"] for b in n1_bindings}
        assert bound_ids == {
            "NCBIGene:5468",
            "NCBIGene:3643",
            "NCBIGene:2645",
            "NCBIGene:7124",
        }

    def test_collate_kg_contains_all_nodes(self, graph, bmt):
        """COLLATE result should include all collated nodes in the KG."""
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["CHEBI:6801"]},
                        "n1": {
                            "categories": ["biolink:Gene"],
                            "set_interpretation": "COLLATE",
                        },
                    },
                    "edges": {
                        "e0": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:affects"],
                        }
                    },
                }
            }
        }
        response = lookup(graph, query, bmt=bmt, subclass=False)
        kg_nodes = response["message"]["knowledge_graph"]["nodes"]
        for gene_id in [
            "NCBIGene:5468",
            "NCBIGene:3643",
            "NCBIGene:2645",
            "NCBIGene:7124",
        ]:
            assert gene_id in kg_nodes

    def test_collate_two_hop_groups_by_batch_nodes(self, graph, bmt):
        """COLLATE on middle node groups by the pinned endpoints.

        n0=Metformin -affects-> n1(COLLATE Gene) -gene_assoc-> n2=T2D
        Genes connected both ways: PPARG, INSR, GCK
        Should produce 1 result with 3 genes in n1.
        """
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["CHEBI:6801"]},
                        "n1": {
                            "categories": ["biolink:Gene"],
                            "set_interpretation": "COLLATE",
                        },
                        "n2": {"ids": ["MONDO:0005148"]},
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
                            "predicates": ["biolink:gene_associated_with_condition"],
                        },
                    },
                }
            }
        }
        response = lookup(graph, query, bmt=bmt, subclass=False)
        results = response["message"]["results"]

        assert len(results) == 1
        n1_bindings = results[0]["node_bindings"]["n1"]
        bound_ids = {b["id"] for b in n1_bindings}
        # PPARG, INSR, GCK all connect Metformin to T2D
        assert bound_ids == {"NCBIGene:5468", "NCBIGene:3643", "NCBIGene:2645"}
