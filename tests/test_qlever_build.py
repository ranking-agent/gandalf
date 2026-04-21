"""Tests for QLever artifact generation."""

import json
import os
import tempfile
from pathlib import Path

from gandalf.backends.qlever import (
    build_qlever_backend_from_jsonl,
    export_graph_to_rdf,
    export_jsonl_to_rdf,
)
from gandalf.backends.qlever.build import _run_qlever_index
from gandalf.backends.qlever.edge_lookup import EdgeIdLookup, synthetic_qlever_edge_id
from gandalf.graph import CSRGraph
from gandalf.loader import build_graph_from_jsonl

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
NODES_FILE = os.path.join(FIXTURES_DIR, "nodes.jsonl")
EDGES_FILE = os.path.join(FIXTURES_DIR, "edges.jsonl")


class TestQleverRdfExport:
    def test_export_jsonl_to_rdf_writes_structural_triples_and_qualifiers(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            rdf_path = export_jsonl_to_rdf(
                node_jsonl_path=NODES_FILE,
                edge_jsonl_path=EDGES_FILE,
                output_path=os.path.join(temp_dir, "fixture.nt"),
            )

            with open(rdf_path, "r", encoding="utf-8") as handle:
                payload = handle.read()

            assert "<https://identifiers.org/CHEBI:6801>" in payload
            assert "<https://w3id.org/biolink/vocab/treats>" in payload
            assert "<https://identifiers.org/MONDO:0005148>" in payload
            assert "<http://www.w3.org/1999/02/22-rdf-syntax-ns#Statement>" in payload
            assert "<http://www.w3.org/1999/02/22-rdf-syntax-ns#subject>" in payload
            assert "<http://www.w3.org/1999/02/22-rdf-syntax-ns#predicate>" in payload
            assert "<http://www.w3.org/1999/02/22-rdf-syntax-ns#object>" in payload
            assert "<https://w3id.org/biolink/vocab/object_aspect_qualifier> \"activity\"" in payload
            assert "<https://w3id.org/biolink/vocab/object_direction_qualifier> \"increased\"" in payload
            assert "<https://w3id.org/kgx/slot/sources>" not in payload
            assert "<https://w3id.org/kgx/slot/qualifiers>" not in payload
            assert "<https://w3id.org/kgx/slot/attributes>" not in payload
            assert "infores:gandalf" not in payload

    def test_export_graph_to_rdf_uses_synthetic_ids_for_edges_without_source_ids(self, tmp_path):
        nodes_path = tmp_path / "nodes.jsonl"
        edges_path = tmp_path / "edges.jsonl"
        rdf_path = tmp_path / "fixture.nt"
        lookup_path = tmp_path / "edge_id_to_idx.lmdb"

        nodes_path.write_text(
            '{"id":"CHEBI:1","name":"One","category":["biolink:ChemicalEntity"]}\n'
            '{"id":"MONDO:1","name":"Disease","category":["biolink:Disease"]}\n',
            encoding="utf-8",
        )
        edges_path.write_text(
            '{"subject":"CHEBI:1","predicate":"biolink:treats","object":"MONDO:1",'
            '"sources":[{"resource_role":"primary_knowledge_source","resource_id":"infores:test"}]}\n',
            encoding="utf-8",
        )

        graph = build_graph_from_jsonl(edges_path, nodes_path)
        try:
            EdgeIdLookup.build_from_graph(graph, lookup_path).close()
            export_graph_to_rdf(graph, rdf_path)
        finally:
            graph.close()

        synthetic_id = synthetic_qlever_edge_id(0)
        with EdgeIdLookup(lookup_path, readonly=True) as edge_lookup:
            assert edge_lookup.get(synthetic_id) == 0

        payload = rdf_path.read_text(encoding="utf-8")
        assert f"<{synthetic_id}>" in payload


class TestQleverBuildArtifacts:
    def test_run_qlever_index_uses_cat_input_files_for_native_cli(self, monkeypatch, tmp_path):
        captured = {}

        def fake_run(command, check, cwd):
            captured["command"] = command
            captured["cwd"] = cwd
            assert check is True

        monkeypatch.setattr("gandalf.backends.qlever.build.subprocess.run", fake_run)

        _run_qlever_index(
            dataset_base=tmp_path / "qlever" / "fixture" / "fixture",
            rdf_path=tmp_path / "rdf" / "fixture.nt.zst",
            workdir=tmp_path,
        )

        command = captured["command"]
        assert "--cat-input-files" in command
        assert "qlever/fixture/fixture" in command
        assert "rdf/fixture.nt.zst" in command
        assert "zstd -dc rdf/fixture.nt.zst" in command
        assert "--overwrite-existing" in command
        assert captured["cwd"] == tmp_path

    def test_build_qlever_backend_writes_manifest_metadata_and_lookup(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = build_qlever_backend_from_jsonl(
                edge_jsonl_path=EDGES_FILE,
                node_jsonl_path=NODES_FILE,
                output_dir=temp_dir,
                dataset_name="fixture",
                run_index=False,
            )

            with open(os.path.join(output_dir, "backend.json"), "r", encoding="utf-8") as handle:
                manifest = json.load(handle)

            assert manifest["backend"] == "qlever"
            assert manifest["dataset_name"] == "fixture"
            assert manifest["index_built"] is False
            assert manifest["csr_artifact_dir"] == "csr"
            assert manifest["edge_id_lookup_path"] == "edge_id_to_idx.lmdb"

            assert os.path.exists(os.path.join(output_dir, "rdf", "fixture.nt.zst"))
            assert os.path.exists(os.path.join(output_dir, "edge_id_to_idx.lmdb"))
            assert os.path.exists(os.path.join(output_dir, "csr", "node_store.lmdb"))
            assert os.path.exists(
                os.path.join(output_dir, "csr", "edge_properties.lmdb")
            )
            assert os.path.exists(os.path.join(output_dir, "csr", "edge_ids.lmdb"))
            assert os.path.exists(os.path.join(output_dir, "csr", "meta_kg.json"))
            assert os.path.exists(
                os.path.join(output_dir, "csr", "sri_testing_data.json")
            )

            graph = CSRGraph.load_mmap(Path(output_dir) / "csr")
            try:
                with EdgeIdLookup(
                    Path(output_dir) / manifest["edge_id_lookup_path"],
                    readonly=True,
                ) as edge_lookup:
                    edge_idx = edge_lookup.get(
                        "CHEBI:6801-biolink:treats-MONDO:0005148-drugcentral"
                    )
                    assert edge_idx is not None
                    edge = graph.get_edge_properties_by_index(edge_idx)
                    assert edge["sources"][0]["resource_id"] == "infores:gandalf"
            finally:
                graph.close()

    def test_build_qlever_backend_can_reuse_shared_csr_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = os.path.abspath(temp_dir)
            csr_dir = os.path.join(temp_path, "csr")
            qlever_dir = os.path.join(temp_path, "qlever")

            graph = build_graph_from_jsonl(EDGES_FILE, NODES_FILE)
            try:
                graph.save_mmap(csr_dir)
            finally:
                graph.close()

            output_dir = build_qlever_backend_from_jsonl(
                edge_jsonl_path=EDGES_FILE,
                node_jsonl_path=NODES_FILE,
                output_dir=qlever_dir,
                dataset_name="fixture",
                run_index=False,
                shared_artifact_dir=csr_dir,
            )

            with open(os.path.join(output_dir, "backend.json"), "r", encoding="utf-8") as handle:
                manifest = json.load(handle)

            assert manifest["csr_artifact_dir"] == "../csr"
            assert manifest["edge_id_lookup_path"] == "edge_id_to_idx.lmdb"

            assert os.path.exists(os.path.join(output_dir, "rdf", "fixture.nt.zst"))
            assert os.path.exists(os.path.join(output_dir, "edge_id_to_idx.lmdb"))
            assert not os.path.exists(os.path.join(output_dir, "csr"))

            graph = CSRGraph.load_mmap(csr_dir)
            try:
                with EdgeIdLookup(
                    Path(output_dir) / manifest["edge_id_lookup_path"],
                    readonly=True,
                ) as edge_lookup:
                    edge_idx = edge_lookup.get(
                        "CHEBI:6801-biolink:treats-MONDO:0005148-drugcentral"
                    )
                    assert edge_idx is not None
                    edge = graph.get_edge_properties_by_index(edge_idx)
                    assert edge["sources"][0]["resource_id"] == "infores:gandalf"
            finally:
                graph.close()
