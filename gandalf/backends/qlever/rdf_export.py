"""Export the RDF needed by the QLever backend."""

import io
import shutil
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.parse import quote

from gandalf.backends.qlever.edge_lookup import resolved_qlever_edge_id
from gandalf.graph import CSRGraph
from gandalf.loader import build_graph_from_jsonl


BIOLINK_VOCAB = "https://w3id.org/biolink/vocab/"
IDENTIFIERS_ORG = "https://identifiers.org/"
RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
RDF_STATEMENT = "http://www.w3.org/1999/02/22-rdf-syntax-ns#Statement"
RDF_SUBJECT = "http://www.w3.org/1999/02/22-rdf-syntax-ns#subject"
RDF_PREDICATE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#predicate"
RDF_OBJECT = "http://www.w3.org/1999/02/22-rdf-syntax-ns#object"
XSD_NS = "http://www.w3.org/2001/XMLSchema#"


def escape_iri(value: str) -> str:
    return value.replace("\\", "%5C").replace(">", "%3E").replace("<", "%3C")


def nt_resource(value: str) -> str:
    return f"<{escape_iri(value)}>"


def write_triple(handle, subject: str, predicate: str, object_value: str) -> None:
    handle.write(f"{nt_resource(subject)} {nt_resource(predicate)} {object_value} .\n")


def nt_literal(value: Any) -> str:
    if isinstance(value, bool):
        literal = "true" if value else "false"
        return f"\"{literal}\"^^<{XSD_NS}boolean>"
    if isinstance(value, int):
        return f"\"{value}\"^^<{XSD_NS}integer>"
    if isinstance(value, float):
        return f"\"{value}\"^^<{XSD_NS}double>"
    escaped = (
        str(value)
        .replace("\\", "\\\\")
        .replace("\"", "\\\"")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    return f"\"{escaped}\""


def curie_or_iri_to_iri(value: str) -> str:
    if value.startswith(("http://", "https://", "urn:")):
        return value
    if value.startswith("biolink:"):
        return BIOLINK_VOCAB + value.split(":", 1)[1]
    return IDENTIFIERS_ORG + quote(value, safe=":/._-")


def emit_node_record(handle, node_id: str, categories: list[str]) -> None:
    node_iri = curie_or_iri_to_iri(node_id)
    for category in categories:
        write_triple(
            handle,
            node_iri,
            RDF_TYPE,
            nt_resource(curie_or_iri_to_iri(category)),
        )


def emit_edge_record(
    handle,
    *,
    edge_id: str,
    subject_id: str,
    predicate: str,
    object_id: str,
) -> None:
    edge_iri = curie_or_iri_to_iri(edge_id)
    subject_iri = curie_or_iri_to_iri(subject_id)
    predicate_iri = curie_or_iri_to_iri(predicate)
    object_iri = curie_or_iri_to_iri(object_id)

    write_triple(handle, edge_iri, RDF_TYPE, nt_resource(RDF_STATEMENT))
    write_triple(handle, edge_iri, RDF_SUBJECT, nt_resource(subject_iri))
    write_triple(handle, edge_iri, RDF_PREDICATE, nt_resource(predicate_iri))
    write_triple(handle, edge_iri, RDF_OBJECT, nt_resource(object_iri))


def emit_edge_qualifiers(handle, edge_id: str, qualifiers: list[dict[str, Any]]) -> None:
    edge_iri = curie_or_iri_to_iri(edge_id)
    for qualifier in qualifiers:
        qualifier_type_id = qualifier.get("qualifier_type_id")
        if not isinstance(qualifier_type_id, str):
            continue
        if "qualifier_value" not in qualifier:
            continue
        write_triple(
            handle,
            edge_iri,
            curie_or_iri_to_iri(qualifier_type_id),
            nt_literal(qualifier["qualifier_value"]),
        )


@contextmanager
def _open_rdf_output(output_path: Path):
    if output_path.suffix != ".zst":
        with open(output_path, "w", encoding="utf-8") as handle:
            yield handle
        return

    zstd_binary = shutil.which("zstd")
    if zstd_binary is None:
        raise RuntimeError("`zstd` is required to write compressed QLever RDF")

    process = subprocess.Popen(
        [zstd_binary, "-q", "-T0", "-f", "-o", str(output_path), "--"],
        stdin=subprocess.PIPE,
    )
    if process.stdin is None:
        process.wait()
        raise RuntimeError("Failed to open zstd compression stream")

    wrapper = io.TextIOWrapper(process.stdin, encoding="utf-8")
    try:
        yield wrapper
        wrapper.flush()
        wrapper.close()
        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, process.args)
    except Exception:
        try:
            wrapper.close()
        finally:
            process.wait()
        if output_path.exists():
            output_path.unlink()
        raise


def export_jsonl_to_rdf(
    node_jsonl_path: str | Path,
    edge_jsonl_path: str | Path,
    output_path: str | Path,
    infores: str | None = None,
) -> Path:
    """Build a Gandalf graph from KGX JSONL inputs and export QLever RDF."""
    del infores

    graph = build_graph_from_jsonl(edge_jsonl_path, node_jsonl_path)
    try:
        return export_graph_to_rdf(graph, output_path)
    finally:
        graph.close()


def export_graph_to_rdf(
    graph: CSRGraph,
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    node_ids: list[str] = ["" for _ in range(graph.num_nodes)]
    with _open_rdf_output(output_path) as handle:
        for node_idx in range(graph.num_nodes):
            node_id = graph.get_node_id(node_idx)
            node_ids[node_idx] = node_id
            emit_node_record(
                handle,
                node_id,
                list(graph.get_node_property(node_idx, "categories", [])),
            )

        for subject_idx in range(graph.num_nodes):
            subject_id = node_ids[subject_idx]
            start = int(graph.fwd_offsets[subject_idx])
            end = int(graph.fwd_offsets[subject_idx + 1])
            for fwd_edge_idx in range(start, end):
                object_idx = int(graph.fwd_targets[fwd_edge_idx])
                emit_edge_record(
                    handle,
                    edge_id=resolved_qlever_edge_id(
                        graph.get_edge_id(fwd_edge_idx),
                        fwd_edge_idx,
                    ),
                    subject_id=subject_id,
                    predicate=graph.id_to_predicate[int(graph.fwd_predicates[fwd_edge_idx])],
                    object_id=node_ids[object_idx],
                )
                emit_edge_qualifiers(
                    handle,
                    resolved_qlever_edge_id(
                        graph.get_edge_id(fwd_edge_idx),
                        fwd_edge_idx,
                    ),
                    list(graph.edge_properties.get_qualifiers(fwd_edge_idx)),
                )

    return output_path


__all__ = [
    "export_graph_to_rdf",
    "export_jsonl_to_rdf",
]
