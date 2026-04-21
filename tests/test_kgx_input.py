"""Tests for KGX directory and archive input resolution."""

import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from gandalf.kgx_input import infer_dataset_name, resolved_kgx_input

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
NODES_FILE = FIXTURES_DIR / "nodes.jsonl"
EDGES_FILE = FIXTURES_DIR / "edges.jsonl"
ZSTD_BINARY = shutil.which("zstd")


def _write_tar_zst_archive(path: Path) -> None:
    tar_path = path.with_suffix("")
    with tarfile.open(tar_path, "w") as archive:
        archive.add(NODES_FILE, arcname="nodes.jsonl")
        archive.add(EDGES_FILE, arcname="edges.jsonl")

    subprocess.run(
        [ZSTD_BINARY, "-q", "-f", "-o", str(path), str(tar_path)],
        check=True,
    )
    tar_path.unlink()


def test_infer_dataset_name_for_directory_input():
    assert infer_dataset_name(FIXTURES_DIR) == "fixtures"


@pytest.mark.skipif(ZSTD_BINARY is None, reason="zstd is required for tar.zst tests")
def test_resolved_kgx_input_extracts_tar_zst_archive(tmp_path):
    archive_path = tmp_path / "sample.tar.zst"
    _write_tar_zst_archive(archive_path)

    with resolved_kgx_input(archive_path) as (
        edges_path,
        nodes_path,
        dataset_name,
    ):
        assert dataset_name == "sample"
        assert edges_path.exists()
        assert nodes_path.exists()
        assert edges_path.read_text(encoding="utf-8") == EDGES_FILE.read_text(
            encoding="utf-8"
        )
        assert nodes_path.read_text(encoding="utf-8") == NODES_FILE.read_text(
            encoding="utf-8"
        )


@pytest.mark.skipif(ZSTD_BINARY is None, reason="zstd is required for tar.zst tests")
def test_build_graph_cli_accepts_tar_zst_input(tmp_path):
    archive_path = tmp_path / "sample.tar.zst"
    output_dir = tmp_path / "artifacts"
    _write_tar_zst_archive(archive_path)

    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "build_graph.py"),
            "--backend",
            "both",
            "--skip-qlever-index",
            "--input",
            str(archive_path),
            "--output",
            str(output_dir),
        ],
        cwd=REPO_ROOT,
        check=True,
    )

    assert (output_dir / "csr" / "node_store.lmdb").exists()
    assert (output_dir / "csr" / "edge_properties.lmdb").exists()
    assert (output_dir / "csr" / "meta_kg.json").exists()
    assert (output_dir / "qlever" / "edge_id_to_idx.lmdb").exists()

    with open(output_dir / "qlever" / "backend.json", "r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    assert manifest["dataset_name"] == "sample"
    assert manifest["csr_artifact_dir"] == "../csr"
    assert manifest["edge_id_lookup_path"] == "edge_id_to_idx.lmdb"
