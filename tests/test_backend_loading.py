"""Tests for explicit backend loading."""

import os
import tempfile
import pytest

from gandalf.backends.csr_backend import CSRBackend
from gandalf.backends.load_backend import load_backend
from gandalf.backends.qlever.backend import QLeverBackend
from gandalf.backends.qlever.build import build_qlever_backend_from_jsonl
from gandalf.loader import build_graph_from_jsonl

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
NODES_FILE = os.path.join(FIXTURES_DIR, "nodes.jsonl")
EDGES_FILE = os.path.join(FIXTURES_DIR, "edges.jsonl")


def test_load_backend_loads_csr_artifacts():
    with tempfile.TemporaryDirectory() as temp_dir:
        graph = build_graph_from_jsonl(EDGES_FILE, NODES_FILE)
        graph.save_mmap(temp_dir)
        graph.close()

        backend = load_backend(temp_dir, "csr")
        try:
            assert isinstance(backend, CSRBackend)
            assert backend.meta_kg is not None
        finally:
            backend.close()


def test_load_backend_loads_qlever_artifacts():
    with tempfile.TemporaryDirectory() as temp_dir:
        build_qlever_backend_from_jsonl(
            edge_jsonl_path=EDGES_FILE,
            node_jsonl_path=NODES_FILE,
            output_dir=temp_dir,
            dataset_name="fixture",
            run_index=False,
        )

        backend = load_backend(
            temp_dir,
            "qlever",
            qlever_host="localhost",
            qlever_port=8888,
        )
        try:
            assert isinstance(backend, QLeverBackend)
            assert backend.meta_kg is not None
            assert (
                backend.edge_lookup.get(
                    "CHEBI:6801-biolink:treats-MONDO:0005148-drugcentral"
                )
                is not None
            )
        finally:
            backend.close()


def test_load_backend_loads_qlever_artifacts_with_shared_csr_metadata():
    with tempfile.TemporaryDirectory() as temp_dir:
        csr_dir = os.path.join(temp_dir, "csr")
        qlever_dir = os.path.join(temp_dir, "qlever")

        graph = build_graph_from_jsonl(EDGES_FILE, NODES_FILE)
        try:
            graph.save_mmap(csr_dir)
        finally:
            graph.close()

        build_qlever_backend_from_jsonl(
            edge_jsonl_path=EDGES_FILE,
            node_jsonl_path=NODES_FILE,
            output_dir=qlever_dir,
            dataset_name="fixture",
            run_index=False,
            shared_artifact_dir=csr_dir,
        )

        backend = load_backend(
            qlever_dir,
            "qlever",
            qlever_host="localhost",
            qlever_port=8888,
        )
        try:
            assert isinstance(backend, QLeverBackend)
            assert backend.meta_kg is not None
            edge_idx = backend.edge_lookup.get(
                "CHEBI:6801-biolink:treats-MONDO:0005148-drugcentral"
            )
            assert edge_idx is not None
        finally:
            backend.close()


def test_load_backend_rejects_wrong_backend_type():
    with tempfile.TemporaryDirectory() as temp_dir:
        graph = build_graph_from_jsonl(EDGES_FILE, NODES_FILE)
        try:
            graph.save_mmap(temp_dir)
        finally:
            graph.close()

        with pytest.raises(ValueError, match="QLever artifacts not found"):
            load_backend(
                temp_dir,
                "qlever",
                qlever_host="localhost",
                qlever_port=8888,
            )
