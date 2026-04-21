"""QLever query transpilation and Gandalf-compatible runtime integration."""

import json
import logging
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote

from bmt.toolkit import Toolkit

from gandalf.backends.qlever.edge_lookup import EdgeIdLookup
from gandalf.graph import CSRGraph
from gandalf.search.attribute_constraints import matches_attribute_constraints
from gandalf.search.expanders import PredicateExpander, QualifierExpander
from gandalf.search.gc_utils import GCMonitor
from gandalf.search.lookup import build_response_from_edge_results, prepare_query_graph
from gandalf.search.qualifiers import edge_matches_qualifier_constraints
from gandalf.search.query_edge import _passes_node_filters

logger = logging.getLogger(__name__)

RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
RDFS_NS = "http://www.w3.org/2000/01/rdf-schema#"
IDENTIFIERS_ORG = "https://identifiers.org/"
BIOLINK_VOCAB = "https://w3id.org/biolink/vocab/"
BIOLINK_SUBCLASS_OF = BIOLINK_VOCAB + "subclass_of"
RDF_TYPE = RDF_NS + "type"
RDF_STATEMENT = RDF_NS + "Statement"
RDF_PROPERTY = RDF_NS + "Property"
RDF_SUBJECT = RDF_NS + "subject"
RDF_PREDICATE = RDF_NS + "predicate"
RDF_OBJECT = RDF_NS + "object"
RDFS_CLASS = RDFS_NS + "Class"
XSD_NS = "http://www.w3.org/2001/XMLSchema#"
NON_ALPHANUMERIC_RE = re.compile(r"[^A-Za-z0-9_]+")
BIOLINK_NAMED_THING = "biolink:NamedThing"


def curie_to_iri(value: str) -> str:
    if value.startswith(("http://", "https://", "urn:")):
        return value
    if value.startswith("biolink:"):
        return BIOLINK_VOCAB + value.split(":", 1)[1]
    return IDENTIFIERS_ORG + quote(value, safe=":/._-")


def iri_to_curie(value: str) -> str:
    if value.startswith(IDENTIFIERS_ORG):
        return unquote(value[len(IDENTIFIERS_ORG) :])
    if value.startswith(BIOLINK_VOCAB):
        return "biolink:" + value[len(BIOLINK_VOCAB) :]
    return value


def iri_term(iri: str) -> str:
    return f"<{iri}>"


def values_clause(variable: str, iris: list[str]) -> str:
    return f"VALUES {variable} {{ {' '.join(iri_term(iri) for iri in iris)} }}"


def values_term_clause(variable: str, terms: list[str]) -> str:
    return f"VALUES {variable} {{ {' '.join(terms)} }}"


def strip_typed_literal(value: str | None) -> str:
    if value is None:
        return ""
    if value.startswith('"') and '"^^<' in value:
        return value.split('"^^<', 1)[0][1:]
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    return value


def safe_var_suffix(value: str) -> str:
    suffix = NON_ALPHANUMERIC_RE.sub("_", value)
    if not suffix:
        return "x"
    if suffix[0].isdigit():
        return "x_" + suffix
    return suffix


def qnode_binding_var(qnode: dict[str, Any]) -> str:
    return f"?node_{qnode['index']}_{safe_var_suffix(qnode['qnode_id'])}"


def qnode_category_var(qnode: dict[str, Any]) -> str:
    return f"?node_category_{qnode['index']}_{safe_var_suffix(qnode['qnode_id'])}"


def qnode_constant_id(qnode: dict[str, Any]) -> str | None:
    constant_id = qnode.get("_constant_id")
    return constant_id if isinstance(constant_id, str) else None


def qnode_constant_term(qnode: dict[str, Any]) -> str:
    constant_id = qnode_constant_id(qnode)
    if constant_id is None:
        raise ValueError(f"QNode {qnode['qnode_id']} does not have a constant binding")
    return iri_term(curie_to_iri(constant_id))


def qedge_binding_var(qedge: dict[str, Any]) -> str:
    return f"?edge_{qedge['index']}_{safe_var_suffix(qedge['qedge_id'])}"


def qedge_predicate_var(qedge: dict[str, Any]) -> str:
    return f"?predicate_{qedge['index']}_{safe_var_suffix(qedge['qedge_id'])}"


def qedge_qualifier_value_var(
    qedge: dict[str, Any],
    constraint_index: int,
    filter_index: int,
) -> str:
    return f"?qualifier_value_{qedge['index']}_{constraint_index}_{filter_index}"


def qedge_orientation_var(qedge: dict[str, Any]) -> str:
    return f"?orientation_{qedge['index']}_{safe_var_suffix(qedge['qedge_id'])}"


def _index_records(
    records: dict[str, dict[str, Any]],
    *,
    id_field: str,
) -> dict[str, dict[str, Any]]:
    indexed_records: dict[str, dict[str, Any]] = {}
    for index, (record_id, record) in enumerate(records.items()):
        indexed_record = dict(record)
        indexed_record[id_field] = record_id
        indexed_record["index"] = index
        indexed_records[record_id] = indexed_record
    return indexed_records


def normalize_trapi_request(
    request: dict[str, Any],
    *,
    subclass: bool,
    subclass_depth: int,
) -> dict[str, Any]:
    original_query_graph, query_graph = prepare_query_graph(
        request,
        subclass=subclass,
        subclass_depth=subclass_depth,
    )
    original_qnodes = {
        qnode_id: dict(qnode)
        for qnode_id, qnode in original_query_graph["nodes"].items()
    }
    original_qedges = {
        qedge_id: dict(qedge)
        for qedge_id, qedge in original_query_graph["edges"].items()
    }
    qnodes = _index_records(query_graph["nodes"], id_field="qnode_id")
    for qnode in qnodes.values():
        ids = qnode.get("ids") or []
        if qnode.get("_superclass") and len(ids) == 1 and isinstance(ids[0], str):
            qnode["_constant_id"] = ids[0]
    qedges = _index_records(query_graph["edges"], id_field="qedge_id")

    referenced_qnodes = {
        qnode_id
        for qedge in qedges.values()
        for qnode_id in (qedge["subject"], qedge["object"])
    }
    original_referenced_qnodes = {
        qnode_id
        for qedge in original_qedges.values()
        for qnode_id in (qedge["subject"], qedge["object"])
    }

    return {
        "original_query_graph": original_query_graph,
        "query_graph": query_graph,
        "original_qnodes": original_qnodes,
        "original_qedges": original_qedges,
        "qnodes": qnodes,
        "qedges": qedges,
        "referenced_qnodes": referenced_qnodes,
        "original_referenced_qnodes": original_referenced_qnodes,
    }


def effective_qnode_categories(categories: list[str]) -> list[str]:
    filtered = [category for category in categories if category != BIOLINK_NAMED_THING]
    if filtered:
        return filtered
    if categories == [BIOLINK_NAMED_THING]:
        return []
    return categories


def append_node_filters(
    lines: list[str],
    qnode: dict[str, Any],
    *,
    indent: str = "  ",
) -> None:
    constant_term = (
        qnode_constant_term(qnode) if qnode_constant_id(qnode) is not None else None
    )
    variable = constant_term or qnode_binding_var(qnode)

    if constant_term is None:
        ids = [curie_to_iri(value) for value in qnode.get("ids", [])]
        if ids:
            lines.append(f"{indent}{values_clause(variable, ids)}")

    categories = [
        curie_to_iri(category)
        for category in effective_qnode_categories(qnode.get("categories", []))
    ]
    if categories:
        category_var = qnode_category_var(qnode)
        lines.append(f"{indent}{values_clause(category_var, categories)}")
        lines.append(f"{indent}{variable} <{RDF_TYPE}> {category_var} .")


def subclass_attachment_metadata(
    normalized_request: dict[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], set[str]]:
    qnode_to_subclass_qedge: dict[str, dict[str, Any]] = {}
    for qedge in normalized_request["qedges"].values():
        if qedge.get("_subclass", False):
            qnode_to_subclass_qedge[qedge["subject"]] = qedge

    qedge_attached_subclass_qedges: dict[str, list[dict[str, Any]]] = {}
    inlined_subclass_qedge_ids: set[str] = set()
    for qedge in normalized_request["qedges"].values():
        if qedge.get("_subclass", False):
            continue
        attached: list[dict[str, Any]] = []
        seen_qnodes: set[str] = set()
        for qnode_id in (qedge["subject"], qedge["object"]):
            if qnode_id in seen_qnodes:
                continue
            subclass_qedge = qnode_to_subclass_qedge.get(qnode_id)
            if (
                subclass_qedge is None
                or subclass_qedge["qedge_id"] in inlined_subclass_qedge_ids
            ):
                continue
            attached.append(subclass_qedge)
            inlined_subclass_qedge_ids.add(subclass_qedge["qedge_id"])
            seen_qnodes.add(qnode_id)
        if attached:
            qedge_attached_subclass_qedges[qedge["qedge_id"]] = attached

    return qedge_attached_subclass_qedges, inlined_subclass_qedge_ids


def inline_filtered_qnode_ids(
    normalized_request: dict[str, Any],
    qedge_attached_subclass_qedges: dict[str, list[dict[str, Any]]],
) -> set[str]:
    filtered_qnode_ids: set[str] = set()
    for qedge in normalized_request["qedges"].values():
        if qedge.get("_subclass", False):
            continue
        filtered_qnode_ids.add(qedge["subject"])
        filtered_qnode_ids.add(qedge["object"])

    for attached_subclass_qedges in qedge_attached_subclass_qedges.values():
        for subclass_qedge in attached_subclass_qedges:
            filtered_qnode_ids.add(subclass_qedge["object"])

    return filtered_qnode_ids


def indented_lines(lines: list[str], indent: str) -> list[str]:
    return [f"{indent}{line}" if line else line for line in lines]


def qedge_subquery_vars(
    normalized_request: dict[str, Any],
    qedge: dict[str, Any],
    attached_subclass_qedges: list[dict[str, Any]],
) -> list[str]:
    vars: list[str] = []
    seen: set[str] = set()

    def add(variable: str | None) -> None:
        if variable is None or variable in seen:
            return
        seen.add(variable)
        vars.append(variable)

    for qnode_id in (qedge["subject"], qedge["object"]):
        qnode = normalized_request["qnodes"][qnode_id]
        if qnode_constant_id(qnode) is None:
            add(qnode_binding_var(qnode))

    add(qedge_binding_var(qedge))
    if not qedge.get("_subclass", False):
        add(qedge_predicate_var(qedge))
        add(qedge_orientation_var(qedge))

    for subclass_qedge in attached_subclass_qedges:
        child_qnode = normalized_request["qnodes"][subclass_qedge["subject"]]
        superclass_qnode = normalized_request["qnodes"][subclass_qedge["object"]]
        if qnode_constant_id(child_qnode) is None:
            add(qnode_binding_var(child_qnode))
        if qnode_constant_id(superclass_qnode) is None:
            add(qnode_binding_var(superclass_qnode))
        add(qedge_binding_var(subclass_qedge))

    return vars


def build_qedge_subquery_lines(
    normalized_request: dict[str, Any],
    qedge: dict[str, Any],
    block_lines: list[str],
    attached_subclass_qedges: list[dict[str, Any]],
) -> list[str]:
    select_vars = qedge_subquery_vars(
        normalized_request,
        qedge,
        attached_subclass_qedges,
    )
    lines = [
        "  {",
        "    SELECT DISTINCT " + " ".join(select_vars),
        "    WHERE {",
    ]
    lines.extend(indented_lines(block_lines, "    "))
    lines.extend(
        [
            "    }",
            "  }",
        ]
    )
    return lines


def sparql_string_literal(value: str) -> str:
    escaped = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    return f'"{escaped}"'


def sparql_literal(value: Any) -> str:
    if isinstance(value, bool):
        literal = "true" if value else "false"
        return f'"{literal}"^^<{XSD_NS}boolean>'
    if isinstance(value, int):
        return f'"{value}"^^<{XSD_NS}integer>'
    if isinstance(value, float):
        return f'"{value}"^^<{XSD_NS}double>'
    return sparql_string_literal(str(value))


def qualifier_predicate_iri(qualifier_type_id: str) -> str:
    return curie_to_iri(qualifier_type_id)


def qualifier_value_terms(qualifier_filter: dict[str, Any]) -> list[str]:
    values = qualifier_filter.get("qualifier_values")
    if values:
        return [sparql_literal(value) for value in values]
    return [sparql_literal(qualifier_filter["qualifier_value"])]


def build_qualifier_set_lines(
    qedge: dict[str, Any],
    qualifier_set: list[dict[str, Any]],
    constraint_index: int,
    indent: str = "    ",
) -> list[str]:
    lines: list[str] = []
    for filter_index, qualifier_filter in enumerate(qualifier_set):
        value_var = qedge_qualifier_value_var(qedge, constraint_index, filter_index)
        lines.append(
            f"{indent}{values_term_clause(value_var, qualifier_value_terms(qualifier_filter))}"
        )
        lines.append(
            f"{indent}{qedge_binding_var(qedge)} <{qualifier_predicate_iri(qualifier_filter['qualifier_type_id'])}> {value_var} ."
        )
    return lines


def build_qualifier_constraint_union_lines(
    qedge: dict[str, Any],
    qualifier_constraints: list[dict[str, Any]],
    indent: str = "    ",
) -> list[str]:
    branches: list[list[str]] = []
    for constraint_index, constraint in enumerate(qualifier_constraints):
        qualifier_set = constraint.get("qualifier_set", [])
        if not qualifier_set:
            continue
        branch_lines = [f"{indent}{{"]
        branch_lines.extend(
            build_qualifier_set_lines(
                qedge,
                qualifier_set,
                constraint_index,
                indent=indent + "  ",
            )
        )
        branch_lines.append(f"{indent}}}")
        branches.append(branch_lines)

    if not branches:
        return []

    lines: list[str] = []
    for index, branch_lines in enumerate(branches):
        if index:
            lines.append(f"{indent}UNION")
        lines.extend(branch_lines)
    return lines


def build_qedge_mode_lines(
    normalized_request: dict[str, Any],
    qedge: dict[str, Any],
    mode: dict[str, Any],
    qualifier_constraints: list[dict[str, Any]],
    attached_subclass_qedges: list[dict[str, Any]],
) -> list[str]:
    subject_var = qnode_binding_var(normalized_request["qnodes"][qedge["subject"]])
    object_var = qnode_binding_var(normalized_request["qnodes"][qedge["object"]])
    predicate_var = qedge_predicate_var(qedge)
    orientation = "reverse" if mode["reverse"] else "forward"
    statement_subject_var = object_var if mode["reverse"] else subject_var
    statement_object_var = subject_var if mode["reverse"] else object_var

    lines = [
        "  {",
        f"    {qedge_binding_var(qedge)} a rdf:Statement ;",
        f"      rdf:subject {statement_subject_var} ;",
        f"      rdf:predicate {predicate_var} ;",
        f"      rdf:object {statement_object_var} .",
    ]
    predicate_iris = [curie_to_iri(value) for value in mode["predicates"]]
    if predicate_iris:
        lines.append(f"    {values_clause(predicate_var, predicate_iris)}")
    seen_qnodes: set[str] = set()
    for qnode_id in (qedge["subject"], qedge["object"]):
        if qnode_id in seen_qnodes:
            continue
        append_node_filters(
            lines,
            normalized_request["qnodes"][qnode_id],
            indent="    ",
        )
        seen_qnodes.add(qnode_id)
    seen_superclass_qnodes: set[str] = set()
    for subclass_qedge in attached_subclass_qedges:
        superclass_qnode = normalized_request["qnodes"][subclass_qedge["object"]]
        if superclass_qnode["qnode_id"] not in seen_superclass_qnodes:
            append_node_filters(
                lines,
                superclass_qnode,
                indent="    ",
            )
            seen_superclass_qnodes.add(superclass_qnode["qnode_id"])
        lines.extend(
            build_subclass_union_lines(
                normalized_request,
                subclass_qedge,
                indent="    ",
            )
        )
    lines.extend(
        build_qualifier_constraint_union_lines(
            qedge,
            qualifier_constraints,
            indent="    ",
        )
    )
    lines.append(f'    BIND("{orientation}" AS {qedge_orientation_var(qedge)})')
    lines.append("  }")
    return lines


def build_subclass_union_lines(
    normalized_request: dict[str, Any],
    qedge: dict[str, Any],
    *,
    indent: str = "  ",
) -> list[str]:
    actual_var = qnode_binding_var(normalized_request["qnodes"][qedge["subject"]])
    superclass_qnode = normalized_request["qnodes"][qedge["object"]]
    superclass_constant = (
        qnode_constant_term(superclass_qnode)
        if qnode_constant_id(superclass_qnode) is not None
        else None
    )
    superclass_var = superclass_constant or qnode_binding_var(superclass_qnode)
    path_var = qedge_binding_var(qedge)
    max_path_length = qedge["_subclass_depth"]

    branches: list[str] = []
    for depth in range(max_path_length + 1):
        branch_lines = [f"{indent}{{"]
        if depth == 0:
            if superclass_constant is not None:
                branch_lines.append(
                    f"{indent}  VALUES {actual_var} {{ {superclass_constant} }}"
                )
            else:
                branch_lines.append(
                    f"{indent}  FILTER({actual_var} = {superclass_var})"
                )
            branch_lines.append(f'{indent}  BIND("" AS {path_var})')
        else:
            edge_vars: list[str] = []
            previous_node = actual_var
            for hop_index in range(depth):
                edge_var = f"{path_var}_hop_{hop_index}"
                edge_vars.append(edge_var)
                next_node = (
                    superclass_var
                    if hop_index == depth - 1
                    else f"{path_var}_node_{hop_index}"
                )
                branch_lines.extend(
                    [
                        f"{indent}  {edge_var} a rdf:Statement ;",
                        f"{indent}    rdf:subject {previous_node} ;",
                        f"{indent}    rdf:predicate <{BIOLINK_SUBCLASS_OF}> ;",
                        f"{indent}    rdf:object {next_node} .",
                    ]
                )
                previous_node = next_node
            if len(edge_vars) > 1:
                for left_index in range(len(edge_vars)):
                    for right_index in range(left_index + 1, len(edge_vars)):
                        branch_lines.append(
                            f"{indent}  FILTER({edge_vars[left_index]} != {edge_vars[right_index]})"
                        )
            branch_lines.append(f"{indent}  BIND(STR({edge_vars[0]}) AS {path_var})")
        branch_lines.append(f"{indent}}}")
        branches.append("\n".join(branch_lines))

    union_lines: list[str] = []
    for index, branch in enumerate(branches):
        if index:
            union_lines.append(f"{indent}UNION")
        union_lines.extend(branch.splitlines())
    return union_lines


def predicate_match_modes(
    predicate_expander: PredicateExpander,
    predicates: list[str],
) -> list[dict[str, Any]]:
    forward_predicates, inverse_predicates = predicate_expander.expand_predicates(
        predicates
    )
    if not predicates or "biolink:related_to" in predicates:
        return [
            {"reverse": False, "predicates": []},
            {"reverse": True, "predicates": []},
        ]
    if inverse_predicates is None:
        return [{"reverse": False, "predicates": forward_predicates}]
    return [
        {"reverse": False, "predicates": forward_predicates},
        {"reverse": True, "predicates": inverse_predicates},
    ]


def build_qedge_union_lines(
    normalized_request: dict[str, Any],
    qedge: dict[str, Any],
    predicate_expander: PredicateExpander,
    expanded_qualifier_constraints: dict[str, list[dict[str, Any]]],
    qedge_attached_subclass_qedges: dict[str, list[dict[str, Any]]],
) -> list[str]:
    modes = predicate_match_modes(predicate_expander, qedge.get("predicates", []))
    qualifier_constraints = expanded_qualifier_constraints.get(qedge["qedge_id"], [])
    attached_subclass_qedges = qedge_attached_subclass_qedges.get(
        qedge["qedge_id"], []
    )
    lines: list[str] = []
    for mode_index, mode in enumerate(modes):
        if mode_index:
            lines.append("  UNION")
        lines.extend(
            build_qedge_mode_lines(
                normalized_request,
                qedge,
                mode,
                qualifier_constraints,
                attached_subclass_qedges,
            )
        )
    return lines


def build_trapi_query(
    normalized_request: dict[str, Any],
    predicate_expander: PredicateExpander,
    expanded_qualifier_constraints: dict[str, list[dict[str, Any]]],
) -> str:
    qedge_attached_subclass_qedges, inlined_subclass_qedge_ids = (
        subclass_attachment_metadata(normalized_request)
    )
    globally_filtered_qnode_ids = inline_filtered_qnode_ids(
        normalized_request,
        qedge_attached_subclass_qedges,
    )
    referenced_qnodes = normalized_request["referenced_qnodes"]
    qnode_vars = [
        qnode_binding_var(qnode)
        for qnode_id, qnode in normalized_request["qnodes"].items()
        if qnode_id in referenced_qnodes and qnode_constant_id(qnode) is None
    ]
    qedge_vars = [
        qedge_binding_var(qedge) for qedge in normalized_request["qedges"].values()
    ]
    predicate_vars = [
        qedge_predicate_var(qedge)
        for qedge in normalized_request["qedges"].values()
        if not qedge.get("_subclass", False)
    ]
    orientation_vars = [
        qedge_orientation_var(qedge)
        for qedge in normalized_request["qedges"].values()
        if not qedge.get("_subclass", False)
    ]

    lines = [
        f"PREFIX rdf: <{RDF_NS}>",
        "",
        (
            "SELECT DISTINCT "
            + " ".join(qnode_vars + qedge_vars + predicate_vars + orientation_vars)
        ),
        "WHERE {",
    ]

    for qedge in normalized_request["qedges"].values():
        if qedge.get("_subclass", False):
            if qedge["qedge_id"] in inlined_subclass_qedge_ids:
                continue
            lines.extend(build_subclass_union_lines(normalized_request, qedge))
            continue
        qedge_lines = build_qedge_union_lines(
            normalized_request,
            qedge,
            predicate_expander,
            expanded_qualifier_constraints,
            qedge_attached_subclass_qedges,
        )
        attached_subclass_qedges = qedge_attached_subclass_qedges.get(
            qedge["qedge_id"], []
        )
        if attached_subclass_qedges:
            lines.extend(
                build_qedge_subquery_lines(
                    normalized_request,
                    qedge,
                    qedge_lines,
                    attached_subclass_qedges,
                )
            )
            continue
        lines.extend(qedge_lines)

    for qnode_id in referenced_qnodes:
        if qnode_id in globally_filtered_qnode_ids:
            continue
        append_node_filters(lines, normalized_request["qnodes"][qnode_id])

    lines.append("}")
    return "\n".join(lines) + "\n"


def _row_node_id(
    row: list[str],
    qnode: dict[str, Any],
    qnode_binding_indexes: dict[str, int | None],
) -> str | None:
    constant_id = qnode_constant_id(qnode)
    if constant_id is not None:
        return constant_id

    binding_index = qnode_binding_indexes.get(qnode["qnode_id"])
    if binding_index is None:
        return None
    return iri_to_curie(row[binding_index])


def _row_node_idx(
    row: list[str],
    qnode: dict[str, Any],
    qnode_binding_indexes: dict[str, int | None],
    graph: CSRGraph,
    node_idx_cache: dict[str, int | None],
) -> int | None:
    node_id = _row_node_id(row, qnode, qnode_binding_indexes)
    if node_id is None:
        return None
    if node_id not in node_idx_cache:
        node_idx_cache[node_id] = graph.get_node_idx(node_id)
    return node_idx_cache[node_id]


def _node_attributes(
    graph: CSRGraph,
    node_idx: int,
    node_attributes_cache: dict[int, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    if node_idx not in node_attributes_cache:
        node_attributes_cache[node_idx] = list(
            graph.get_node_property(node_idx, "attributes", [])
        )
    return node_attributes_cache[node_idx]


def _edge_index(
    edge_id: str,
    edge_lookup: EdgeIdLookup,
    edge_idx_cache: dict[str, int | None],
) -> int | None:
    if edge_id not in edge_idx_cache:
        edge_idx_cache[edge_id] = edge_lookup.get(edge_id)
    return edge_idx_cache[edge_id]


def _edge_properties(
    graph: CSRGraph,
    fwd_edge_idx: int,
    edge_properties_cache: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    if fwd_edge_idx not in edge_properties_cache:
        edge_properties_cache[fwd_edge_idx] = graph.get_edge_properties_by_index(
            fwd_edge_idx
        )
    return edge_properties_cache[fwd_edge_idx]


def _subclass_edge_index(
    row: list[str],
    qedge: dict[str, Any],
    qedge_binding_indexes: dict[str, int],
    edge_lookup: EdgeIdLookup,
    edge_idx_cache: dict[str, int | None],
) -> int | None:
    raw_value = strip_typed_literal(row[qedge_binding_indexes[qedge["qedge_id"]]])
    if not raw_value:
        return -1
    first_edge_iri = raw_value.split("||", 1)[0]
    return _edge_index(iri_to_curie(first_edge_iri), edge_lookup, edge_idx_cache)


def build_edge_results(
    rows: list[list[str]],
    row_column_indexes: dict[str, int],
    normalized_request: dict[str, Any],
    graph: CSRGraph,
    edge_lookup: EdgeIdLookup,
    expanded_qualifier_constraints: dict[str, list[dict[str, Any]]],
    *,
    max_node_degree=None,
    min_information_content=None,
) -> dict[str, list[tuple[int, str, int, bool, int]]]:
    qnode_binding_indexes = {
        qnode_id: row_column_indexes.get(qnode_binding_var(qnode))
        for qnode_id, qnode in normalized_request["qnodes"].items()
    }
    qedge_binding_indexes = {
        qedge_id: row_column_indexes[qedge_binding_var(qedge)]
        for qedge_id, qedge in normalized_request["qedges"].items()
    }
    qedge_orientation_indexes = {
        qedge_id: row_column_indexes[qedge_orientation_var(qedge)]
        for qedge_id, qedge in normalized_request["qedges"].items()
        if not qedge.get("_subclass", False)
    }
    edge_results = {
        qedge_id: [] for qedge_id in normalized_request["qedges"]
    }
    edge_seen = {qedge_id: set() for qedge_id in normalized_request["qedges"]}

    node_idx_cache: dict[str, int | None] = {}
    node_attributes_cache: dict[int, list[dict[str, Any]]] = {}
    edge_idx_cache: dict[str, int | None] = {}
    edge_properties_cache: dict[int, dict[str, Any]] = {}

    for row in rows:
        resolved_node_indices: dict[str, int] = {}
        row_valid = True

        for qnode_id in normalized_request["referenced_qnodes"]:
            qnode = normalized_request["qnodes"][qnode_id]
            node_idx = _row_node_idx(
                row,
                qnode,
                qnode_binding_indexes,
                graph,
                node_idx_cache,
            )
            if node_idx is None:
                row_valid = False
                break
            resolved_node_indices[qnode_id] = node_idx
        if not row_valid:
            continue

        for qnode_id in normalized_request["original_referenced_qnodes"]:
            qnode = normalized_request["original_qnodes"][qnode_id]
            node_idx = resolved_node_indices[qnode_id]
            if not _passes_node_filters(
                graph,
                node_idx,
                max_node_degree,
                min_information_content,
            ):
                row_valid = False
                break
            if qnode.get("constraints") and not matches_attribute_constraints(
                _node_attributes(graph, node_idx, node_attributes_cache),
                qnode.get("constraints", []),
            ):
                row_valid = False
                break
        if not row_valid:
            continue

        row_matches: dict[str, tuple[int, str, int, bool, int]] = {}
        for qedge_id, qedge in normalized_request["qedges"].items():
            if qedge.get("_subclass", False):
                fwd_edge_idx = _subclass_edge_index(
                    row,
                    qedge,
                    qedge_binding_indexes,
                    edge_lookup,
                    edge_idx_cache,
                )
                if fwd_edge_idx is None:
                    row_valid = False
                    break
                row_matches[qedge_id] = (
                    resolved_node_indices[qedge["subject"]],
                    "biolink:subclass_of",
                    resolved_node_indices[qedge["object"]],
                    False,
                    fwd_edge_idx,
                )
                continue

            edge_id = iri_to_curie(row[qedge_binding_indexes[qedge_id]])
            fwd_edge_idx = _edge_index(edge_id, edge_lookup, edge_idx_cache)
            if fwd_edge_idx is None:
                row_valid = False
                break

            edge_properties = _edge_properties(
                graph,
                fwd_edge_idx,
                edge_properties_cache,
            )
            if not matches_attribute_constraints(
                edge_properties.get("attributes", []),
                normalized_request["original_qedges"][qedge_id].get(
                    "attribute_constraints", []
                ),
            ):
                row_valid = False
                break
            if not edge_matches_qualifier_constraints(
                edge_properties.get("qualifiers", []),
                expanded_qualifier_constraints[qedge_id],
            ):
                row_valid = False
                break

            orientation = strip_typed_literal(row[qedge_orientation_indexes[qedge_id]])
            via_inverse = orientation == "reverse"
            query_subj_idx = resolved_node_indices[qedge["subject"]]
            query_obj_idx = resolved_node_indices[qedge["object"]]
            subj_idx, obj_idx = (
                (query_obj_idx, query_subj_idx)
                if via_inverse
                else (query_subj_idx, query_obj_idx)
            )
            row_matches[qedge_id] = (
                subj_idx,
                edge_properties["predicate"],
                obj_idx,
                via_inverse,
                fwd_edge_idx,
            )

        if not row_valid:
            continue

        for qedge_id, match in row_matches.items():
            if match in edge_seen[qedge_id]:
                continue
            edge_seen[qedge_id].add(match)
            edge_results[qedge_id].append(match)

    for matches in edge_results.values():
        matches.sort(key=lambda item: (item[0], item[2], item[1], item[3], item[4]))

    return edge_results


def _empty_response(query_graph: dict[str, Any]) -> dict[str, Any]:
    return {
        "message": {
            "query_graph": query_graph,
            "knowledge_graph": {"nodes": {}, "edges": {}},
            "results": [],
            "auxiliary_graphs": {},
        },
        "logs": [],
    }


def answer_trapi_request(
    request: dict[str, Any],
    *,
    graph: CSRGraph,
    edge_lookup: EdgeIdLookup,
    run_qlever_query,
    resource_id: str,
    subclass: bool,
    subclass_depth: int,
    max_node_degree=None,
    min_information_content=None,
    dehydrated=None,
    bmt=None,
    predicate_expander: PredicateExpander | None = None,
    qualifier_expander: QualifierExpander | None = None,
    t_start: float | None = None,
    gc_monitor: GCMonitor | None = None,
) -> dict[str, Any]:
    if t_start is None:
        t_start = time.perf_counter()
    if gc_monitor is None:
        gc_monitor = GCMonitor()

    logger.info("Starting lookup.")
    if bmt is None:
        bmt = Toolkit()
        t_bmt = time.perf_counter()
        logger.warning("BMT initialization: %.2fs", t_bmt - t_start)
    else:
        logger.debug("Using provided BMT instance")

    predicate_expander = predicate_expander or PredicateExpander(bmt)
    qualifier_expander = qualifier_expander or QualifierExpander(bmt)

    normalized = normalize_trapi_request(
        request,
        subclass=subclass,
        subclass_depth=subclass_depth,
    )
    if not normalized["qedges"]:
        return _empty_response(normalized["original_query_graph"])

    expanded_qualifier_constraints = {
        qedge_id: qualifier_expander.expand_qualifier_constraints(
            qedge.get("qualifier_constraints", [])
        )
        for qedge_id, qedge in normalized["original_qedges"].items()
    }

    query = build_trapi_query(
        normalized,
        predicate_expander,
        expanded_qualifier_constraints,
    )
    logger.debug("Built QLever SPARQL query")
    row_column_indexes, rows = run_qlever_query(query)
    edge_results = build_edge_results(
        rows,
        row_column_indexes,
        normalized,
        graph,
        edge_lookup,
        expanded_qualifier_constraints,
        max_node_degree=max_node_degree,
        min_information_content=min_information_content,
    )

    return build_response_from_edge_results(
        graph,
        normalized["original_query_graph"],
        normalized["query_graph"],
        edge_results,
        edge_order=list(normalized["qedges"]),
        t_start=t_start,
        gc_monitor=gc_monitor,
        dehydrated=dehydrated,
        bmt=bmt,
    )


def load_manifest(directory: str | Path) -> dict[str, Any]:
    with open(Path(directory) / "backend.json", "r", encoding="utf-8") as handle:
        return json.load(handle)
