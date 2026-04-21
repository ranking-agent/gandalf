"""Build QLever-compatible artifacts from Gandalf JSONL inputs."""

import os
import json
import logging
import shlex
import shutil
import subprocess
from pathlib import Path

from gandalf.backends.qlever.edge_lookup import EdgeIdLookup
from gandalf.backends.qlever.rdf_export import export_graph_to_rdf
from gandalf.graph import CSRGraph
from gandalf.loader import build_graph_from_jsonl

logger = logging.getLogger(__name__)


def _remove_artifact(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
        return
    path.unlink()


def _resolve_csr_artifact_dir(
    output_dir: Path,
    shared_artifact_dir: Path | None,
) -> str:
    if shared_artifact_dir is None:
        return "csr"

    resolved_output_dir = output_dir.resolve()
    shared_artifact_dir = shared_artifact_dir.resolve()
    if not shared_artifact_dir.exists():
        raise FileNotFoundError(
            f"Shared CSR artifact directory not found: {shared_artifact_dir}"
        )

    local_csr_dir = output_dir / "csr"
    if local_csr_dir.resolve(strict=False) != shared_artifact_dir.resolve(strict=False):
        _remove_artifact(local_csr_dir)

    return os.path.relpath(shared_artifact_dir, resolved_output_dir)


def _run_qlever_index(
    dataset_base: Path,
    rdf_path: Path,
    qlever_binary: str = "qlever",
    stxxl_memory: str = "32G",
    workdir: str | Path | None = None,
) -> None:
    dataset_base.parent.mkdir(parents=True, exist_ok=True)
    workdir = Path(workdir) if workdir is not None else Path.cwd()
    rel_dataset_base = os.path.relpath(dataset_base, workdir)
    rel_rdf_path = os.path.relpath(rdf_path, workdir)
    quoted_rdf_path = shlex.quote(rel_rdf_path)
    if rdf_path.suffix == ".zst":
        cat_input_files = f"zstd -dc {quoted_rdf_path}"
    else:
        cat_input_files = f"cat {quoted_rdf_path}"
    command = [
        qlever_binary,
        "index",
        "--system",
        "native",
        "--name",
        rel_dataset_base,
        "--format",
        "nt",
        "--input-files",
        rel_rdf_path,
        "--cat-input-files",
        cat_input_files,
        "--overwrite-existing",
        "--parallel-parsing",
        "false",
        "--text-index",
        "none",
        "--stxxl-memory",
        stxxl_memory,
    ]
    subprocess.run(command, check=True, cwd=workdir)


def build_qlever_backend_from_jsonl(
    edge_jsonl_path: str | Path,
    node_jsonl_path: str | Path,
    output_dir: str | Path,
    dataset_name: str | None = None,
    run_index: bool = True,
    qlever_binary: str = "qlever",
    stxxl_memory: str = "32G",
    shared_artifact_dir: str | Path | None = None,
) -> Path:
    """Build QLever artifacts plus the CSR artifacts they depend on."""
    edge_jsonl_path = Path(edge_jsonl_path)
    node_jsonl_path = Path(node_jsonl_path)
    output_dir = Path(output_dir)
    shared_artifact_dir = (
        Path(shared_artifact_dir) if shared_artifact_dir is not None else None
    )
    if dataset_name is None:
        dataset_name = output_dir.name

    output_dir.mkdir(parents=True, exist_ok=True)
    rdf_path = output_dir / "rdf" / f"{dataset_name}.nt.zst"
    dataset_base = output_dir / "qlever" / dataset_name / dataset_name
    edge_lookup_path = output_dir / "edge_id_to_idx.lmdb"
    graph = None

    try:
        if shared_artifact_dir is None:
            csr_artifact_dir = output_dir / "csr"
            _remove_artifact(csr_artifact_dir)
            graph = build_graph_from_jsonl(edge_jsonl_path, node_jsonl_path)
            graph.save_mmap(csr_artifact_dir)
            csr_artifact_dir_path = "csr"
        else:
            logger.info(
                "Reusing shared Gandalf artifacts from %s for QLever output %s",
                shared_artifact_dir,
                output_dir,
            )
            csr_artifact_dir = shared_artifact_dir
            csr_artifact_dir_path = _resolve_csr_artifact_dir(
                output_dir,
                shared_artifact_dir,
            )
            graph = CSRGraph.load_mmap(csr_artifact_dir)

        EdgeIdLookup.build_from_graph(
            graph,
            edge_lookup_path,
        ).close()

        export_graph_to_rdf(
            graph,
            rdf_path,
        )

        if run_index:
            _run_qlever_index(
                dataset_base=dataset_base,
                rdf_path=rdf_path,
                qlever_binary=qlever_binary,
                stxxl_memory=stxxl_memory,
                workdir=output_dir,
            )
    finally:
        if graph is not None:
            graph.close()

    manifest = {
        "backend": "qlever",
        "dataset_name": dataset_name,
        "csr_artifact_dir": csr_artifact_dir_path,
        "edge_id_lookup_path": "edge_id_to_idx.lmdb",
        "rdf_path": str(Path("rdf") / f"{dataset_name}.nt.zst"),
        "qlever_dataset_base": str(Path("qlever") / dataset_name / dataset_name),
        "index_built": run_index,
    }
    with open(output_dir / "backend.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    return output_dir
