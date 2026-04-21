"""Tests for QLever runtime response assembly."""

import re
from pathlib import Path

import pytest

from gandalf.backends.qlever.edge_lookup import EdgeIdLookup
from gandalf.backends.qlever.runtime import (
    answer_trapi_request,
    build_trapi_query,
    curie_to_iri,
    normalize_trapi_request,
    qedge_binding_var,
    qedge_orientation_var,
    qedge_predicate_var,
    qnode_binding_var,
    qnode_constant_id,
)
from gandalf.graph import CSRGraph
from gandalf.loader import build_graph_from_jsonl
from gandalf.search.expanders import PredicateExpander, QualifierExpander
from gandalf.search.lookup import lookup
from tests.search_fixtures import graph  # noqa: F401
from tests.search_fixtures import EDGES_FILE, NODES_FILE


@pytest.fixture
def qlever_runtime_artifacts(tmp_path) -> tuple[CSRGraph, Path]:
    graph = build_graph_from_jsonl(EDGES_FILE, NODES_FILE)
    output_dir = tmp_path / "csr"
    edge_lookup_path = tmp_path / "edge_id_to_idx.lmdb"
    try:
        graph.save_mmap(output_dir)
        EdgeIdLookup.build_from_graph(graph, edge_lookup_path).close()
    finally:
        graph.close()

    runtime_graph = CSRGraph.load_mmap(output_dir)
    try:
        yield runtime_graph, edge_lookup_path
    finally:
        runtime_graph.close()


def _columns_for(normalized):
    columns = []
    for qnode in normalized["qnodes"].values():
        if qnode_constant_id(qnode) is None:
            columns.append(qnode_binding_var(qnode))
    for qedge in normalized["qedges"].values():
        columns.append(qedge_binding_var(qedge))
    for qedge in normalized["qedges"].values():
        if not qedge.get("_subclass", False):
            columns.append(qedge_predicate_var(qedge))
    for qedge in normalized["qedges"].values():
        if not qedge.get("_subclass", False):
            columns.append(qedge_orientation_var(qedge))
    return {column: index for index, column in enumerate(columns)}


def _encode_subclass_binding(value):
    if not value:
        return '""'
    if isinstance(value, list):
        return '"' + "||".join(curie_to_iri(item) for item in value) + '"'
    return '"' + curie_to_iri(value) + '"'


def _build_row(
    normalized,
    node_bindings,
    edge_bindings,
    predicates,
    orientations,
):
    columns = _columns_for(normalized)
    row = [""] * len(columns)

    for qnode_id, node_id in node_bindings.items():
        qnode = normalized["qnodes"][qnode_id]
        if qnode_constant_id(qnode) is None:
            row[columns[qnode_binding_var(qnode)]] = curie_to_iri(node_id)

    for qedge_id, qedge in normalized["qedges"].items():
        edge_value = edge_bindings.get(
            qedge_id,
            "" if qedge.get("_subclass", False) else None,
        )
        if qedge.get("_subclass", False):
            row[columns[qedge_binding_var(qedge)]] = _encode_subclass_binding(edge_value)
            continue
        if edge_value is None:
            raise KeyError(f"Missing edge binding for {qedge_id}")
        row[columns[qedge_binding_var(qedge)]] = curie_to_iri(edge_value)
        row[columns[qedge_predicate_var(qedge)]] = curie_to_iri(predicates[qedge_id])
        row[columns[qedge_orientation_var(qedge)]] = (
            f'"{orientations.get(qedge_id, "forward")}"'
        )

    return columns, row


def _run_runtime(
    query,
    row_specs,
    qlever_runtime_artifacts,
    bmt,
    *,
    subclass=True,
    subclass_depth=1,
    dehydrated=None,
    max_node_degree=None,
    min_information_content=None,
):
    normalized = normalize_trapi_request(
        query,
        subclass=subclass,
        subclass_depth=subclass_depth,
    )
    columns = _columns_for(normalized)
    rows = []
    for spec in row_specs:
        _, row = _build_row(
            normalized,
            spec["node_bindings"],
            spec["edge_bindings"],
            spec.get("predicates", {}),
            spec.get("orientations", {}),
        )
        rows.append(row)

    runtime_graph, edge_lookup_path = qlever_runtime_artifacts
    with EdgeIdLookup(edge_lookup_path, readonly=True) as edge_lookup:
        return answer_trapi_request(
            query,
            graph=runtime_graph,
            edge_lookup=edge_lookup,
            predicate_expander=PredicateExpander(bmt),
            qualifier_expander=QualifierExpander(bmt),
            run_qlever_query=lambda _query: (columns, rows),
            resource_id="infores:gandalf",
            subclass=subclass,
            subclass_depth=subclass_depth,
            max_node_degree=max_node_degree,
            min_information_content=min_information_content,
            dehydrated=dehydrated,
            bmt=bmt,
        )


def test_qlever_runtime_matches_csr_for_dehydrated_duplicate_edge_collapse(
    graph,
    bmt,
    qlever_runtime_artifacts,
):
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
                    }
                },
            }
        }
    }
    row_specs = [
        {
            "node_bindings": {"n0": "CHEBI:6801", "n1": "MONDO:0005148"},
            "edge_bindings": {
                "e0": "CHEBI:6801-biolink:treats-MONDO:0005148-drugcentral",
            },
            "predicates": {"e0": "biolink:treats"},
        },
        {
            "node_bindings": {"n0": "CHEBI:6801", "n1": "MONDO:0005148"},
            "edge_bindings": {
                "e0": "CHEBI:6801-biolink:treats-MONDO:0005148-chembl",
            },
            "predicates": {"e0": "biolink:treats"},
        },
        {
            "node_bindings": {"n0": "CHEBI:6801", "n1": "MONDO:0005148"},
            "edge_bindings": {
                "e0": "CHEBI:6801-biolink:ameliorates_condition-MONDO:0005148-clinicaltrials",
            },
            "predicates": {"e0": "biolink:ameliorates_condition"},
        },
        {
            "node_bindings": {"n0": "CHEBI:6801", "n1": "MONDO:0005148"},
            "edge_bindings": {
                "e0": "CHEBI:6801-biolink:preventative_for_condition-MONDO:0005148-pubmed",
            },
            "predicates": {"e0": "biolink:preventative_for_condition"},
        },
    ]

    actual = _run_runtime(
        query,
        row_specs,
        qlever_runtime_artifacts,
        bmt,
        subclass=True,
        subclass_depth=1,
        dehydrated=True,
    )
    expected = lookup(
        graph,
        query,
        bmt=bmt,
        subclass=True,
        subclass_depth=1,
        dehydrated=True,
    )

    assert actual["message"]["knowledge_graph"] == expected["message"]["knowledge_graph"]
    assert actual["message"]["results"][0]["node_bindings"] == expected["message"]["results"][0]["node_bindings"]
    assert {
        binding["id"]
        for binding in actual["message"]["results"][0]["analyses"][0]["edge_bindings"]["e0"]
    } == {
        binding["id"]
        for binding in expected["message"]["results"][0]["analyses"][0]["edge_bindings"]["e0"]
    }
    assert actual["message"]["auxiliary_graphs"] == expected["message"]["auxiliary_graphs"]


def test_qlever_runtime_matches_csr_for_direct_edge_precedence(
    graph,
    bmt,
    qlever_runtime_artifacts,
):
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
                    }
                },
            }
        }
    }
    row_specs = [
        {
            "node_bindings": {"n0": "CHEBI:6801", "n1": "MONDO:0005148"},
            "edge_bindings": {
                "e0": "CHEBI:6801-biolink:treats-MONDO:0005148-drugcentral",
                "n1_subclass_edge": [
                    "MONDO:0005148-biolink:subclass_of-MONDO:0005015-mondo"
                ],
            },
            "predicates": {"e0": "biolink:treats"},
        },
        {
            "node_bindings": {"n0": "CHEBI:6801", "n1": "MONDO:0005015"},
            "edge_bindings": {
                "e0": "CHEBI:6801-biolink:treats-MONDO:0005015-drugcentral",
            },
            "predicates": {"e0": "biolink:treats"},
        },
    ]

    actual = _run_runtime(
        query,
        row_specs,
        qlever_runtime_artifacts,
        bmt,
        subclass=True,
        subclass_depth=1,
    )
    expected = lookup(
        graph,
        query,
        bmt=bmt,
        subclass=True,
        subclass_depth=1,
    )

    assert actual["message"] == expected["message"]


def test_qlever_runtime_matches_csr_for_inverse_subclass_inference(
    graph,
    bmt,
    qlever_runtime_artifacts,
):
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
                    }
                },
            }
        }
    }
    row_specs = [
        {
            "node_bindings": {"n0": "HP:0001943", "n1": "MONDO:0005148"},
            "edge_bindings": {
                "e0": "MONDO:0005148-biolink:has_phenotype-HP:0001943-hpo",
                "n1_subclass_edge": [
                    "MONDO:0005148-biolink:subclass_of-MONDO:0005015-mondo"
                ],
            },
            "predicates": {"e0": "biolink:has_phenotype"},
            "orientations": {"e0": "reverse"},
        }
    ]

    actual = _run_runtime(
        query,
        row_specs,
        qlever_runtime_artifacts,
        bmt,
        subclass=True,
        subclass_depth=1,
    )
    expected = lookup(
        graph,
        query,
        bmt=bmt,
        subclass=True,
        subclass_depth=1,
    )

    assert actual["message"] == expected["message"]


def test_qlever_runtime_applies_qualifier_filters_from_csr_properties(
    graph,
    bmt,
    qlever_runtime_artifacts,
):
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
                        "qualifier_constraints": [
                            {
                                "qualifier_set": [
                                    {
                                        "qualifier_type_id": "biolink:object_aspect_qualifier",
                                        "qualifier_value": "activity",
                                    }
                                ]
                            }
                        ],
                    }
                },
            }
        }
    }
    row_specs = [
        {
            "node_bindings": {"n0": "CHEBI:6801", "n1": "NCBIGene:3643"},
            "edge_bindings": {
                "e0": "CHEBI:6801-biolink:affects-NCBIGene:3643-ctd-activity-increased",
            },
            "predicates": {"e0": "biolink:affects"},
        },
        {
            "node_bindings": {"n0": "CHEBI:6801", "n1": "NCBIGene:2645"},
            "edge_bindings": {
                "e0": "CHEBI:6801-biolink:affects-NCBIGene:2645-ctd-activity-decreased",
            },
            "predicates": {"e0": "biolink:affects"},
        },
        {
            "node_bindings": {"n0": "CHEBI:6801", "n1": "NCBIGene:7124"},
            "edge_bindings": {
                "e0": "CHEBI:6801-biolink:affects-NCBIGene:7124-ctd-abundance-increased",
            },
            "predicates": {"e0": "biolink:affects"},
        },
    ]

    actual = _run_runtime(
        query,
        row_specs,
        qlever_runtime_artifacts,
        bmt,
        subclass=False,
    )
    expected = lookup(
        graph,
        query,
        bmt=bmt,
        subclass=False,
    )

    assert actual["message"]["knowledge_graph"] == expected["message"]["knowledge_graph"]
    assert {
        result["node_bindings"]["n1"][0]["id"]
        for result in actual["message"]["results"]
    } == {
        result["node_bindings"]["n1"][0]["id"]
        for result in expected["message"]["results"]
    }
    assert (
        "CHEBI:6801-biolink:affects-NCBIGene:7124-ctd-abundance-increased"
        not in actual["message"]["knowledge_graph"]["edges"]
    )


def test_build_trapi_query_pushes_expanded_qualifier_constraints_into_sparql(bmt):
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
                        "qualifier_constraints": [
                            {
                                "qualifier_set": [
                                    {
                                        "qualifier_type_id": "biolink:object_aspect_qualifier",
                                        "qualifier_value": "activity_or_abundance",
                                    },
                                    {
                                        "qualifier_type_id": "biolink:object_direction_qualifier",
                                        "qualifier_value": "increased",
                                    },
                                ]
                            }
                        ],
                    }
                },
            }
        }
    }

    normalized = normalize_trapi_request(
        query,
        subclass=False,
        subclass_depth=0,
    )
    qualifier_expander = QualifierExpander(bmt)
    expanded_qualifier_constraints = {
        qedge_id: qualifier_expander.expand_qualifier_constraints(
            qedge.get("qualifier_constraints", [])
        )
        for qedge_id, qedge in normalized["original_qedges"].items()
    }

    sparql = build_trapi_query(
        normalized,
        PredicateExpander(bmt),
        expanded_qualifier_constraints,
    )

    assert "<https://w3id.org/biolink/vocab/object_aspect_qualifier>" in sparql
    assert "<https://w3id.org/biolink/vocab/object_direction_qualifier>" in sparql
    assert '"activity"' in sparql
    assert '"abundance"' in sparql
    assert '"increased"' in sparql
    assert "VALUES ?qualifier_value_" in sparql


def test_build_trapi_query_inlines_subclass_constraints_into_incident_qedge(bmt):
    query = {
        "message": {
            "query_graph": {
                "nodes": {
                    "n0": {"categories": ["biolink:ChemicalEntity"]},
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
    }

    normalized = normalize_trapi_request(
        query,
        subclass=True,
        subclass_depth=1,
    )
    sparql = build_trapi_query(
        normalized,
        PredicateExpander(bmt),
        {
            qedge_id: []
            for qedge_id in normalized["original_qedges"]
        },
    )

    assert sparql.count('BIND("" AS ?edge_1_n1_subclass_edge)') == 2
    assert sparql.count("?edge_1_n1_subclass_edge_hop_0 a rdf:Statement ;") == 2
    assert re.search(
        re.escape("?edge_0_e0 a rdf:Statement ;")
        + r"[\s\S]*?"
        + re.escape('BIND("" AS ?edge_1_n1_subclass_edge)'),
        sparql,
    )


def test_build_trapi_query_inlines_shared_subclass_constraint_once(bmt):
    query = {
        "message": {
            "query_graph": {
                "nodes": {
                    "n0": {"categories": ["biolink:ChemicalEntity"]},
                    "n1": {"ids": ["MONDO:0005015"]},
                    "n2": {"categories": ["biolink:Gene"]},
                },
                "edges": {
                    "e0": {
                        "subject": "n0",
                        "object": "n1",
                        "predicates": ["biolink:treats"],
                    },
                    "e1": {
                        "subject": "n2",
                        "object": "n1",
                        "predicates": ["biolink:genetic_association"],
                    },
                },
            }
        }
    }

    normalized = normalize_trapi_request(
        query,
        subclass=True,
        subclass_depth=1,
    )
    sparql = build_trapi_query(
        normalized,
        PredicateExpander(bmt),
        {
            qedge_id: []
            for qedge_id in normalized["original_qedges"]
        },
    )

    assert sparql.count('BIND("" AS ?edge_2_n1_subclass_edge)') == 2
    assert sparql.count("?edge_2_n1_subclass_edge_hop_0 a rdf:Statement ;") == 2


def test_build_trapi_query_omits_duplicate_global_node_filters(bmt):
    query = {
        "message": {
            "query_graph": {
                "nodes": {
                    "n0": {
                        "ids": ["NCBIGene:1017"],
                        "categories": ["biolink:Gene"],
                    },
                    "n1": {"categories": ["biolink:DiseaseOrPhenotypicFeature"]},
                },
                "edges": {
                    "e0": {
                        "subject": "n0",
                        "object": "n1",
                        "predicates": ["biolink:genetic_association"],
                    }
                },
            }
        }
    }

    normalized = normalize_trapi_request(
        query,
        subclass=False,
        subclass_depth=0,
    )
    sparql = build_trapi_query(
        normalized,
        PredicateExpander(bmt),
        {
            qedge_id: []
            for qedge_id in normalized["original_qedges"]
        },
    )

    assert (
        sparql.count(
            "?node_0_n0 <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> ?node_category_0_n0 ."
        )
        == 1
    )
    assert (
        sparql.count(
            "?node_1_n1 <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> ?node_category_1_n1 ."
        )
        == 1
    )
