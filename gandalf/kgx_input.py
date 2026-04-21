"""Helpers for resolving KGX build inputs."""

from contextlib import contextmanager
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Iterator


def _is_tar_zst_archive(path: Path) -> bool:
    suffixes = path.suffixes
    return suffixes[-2:] == [".tar", ".zst"] or path.suffix == ".tzst"


def infer_dataset_name(input_path: str | Path) -> str:
    path = Path(input_path)
    if path.is_dir():
        return path.resolve().name

    name = path.name
    for suffix in (".tar.zst", ".tzst", ".tar"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def _validate_kgx_directory(directory: Path) -> tuple[Path, Path]:
    nodes_path = directory / "nodes.jsonl"
    edges_path = directory / "edges.jsonl"
    if not nodes_path.exists():
        raise FileNotFoundError(f"Node file not found: {nodes_path}")
    if not edges_path.exists():
        raise FileNotFoundError(f"Edge file not found: {edges_path}")
    return edges_path, nodes_path


def extract_tar_zst_archive(archive_path: str | Path, output_dir: str | Path) -> tuple[Path, Path]:
    archive_path = Path(archive_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    zstd_binary = shutil.which("zstd")
    if zstd_binary is None:
        raise RuntimeError("`zstd` is required to read KGX tar.zst archives")

    tar_binary = shutil.which("tar")
    if tar_binary is None:
        raise RuntimeError("`tar` is required to read KGX tar.zst archives")

    zstd_command = [zstd_binary, "-dc", "--", str(archive_path)]
    tar_command = [
        tar_binary,
        "-xf",
        "-",
        "-C",
        str(output_dir),
        "nodes.jsonl",
        "edges.jsonl",
    ]

    zstd_process = subprocess.Popen(zstd_command, stdout=subprocess.PIPE)
    try:
        if zstd_process.stdout is None:
            raise RuntimeError("Failed to open zstd decompression stream")
        subprocess.run(
            tar_command,
            stdin=zstd_process.stdout,
            check=True,
        )
        zstd_process.stdout.close()
        zstd_returncode = zstd_process.wait()
        if zstd_returncode != 0:
            raise subprocess.CalledProcessError(zstd_returncode, zstd_command)
    except Exception:
        if zstd_process.stdout is not None:
            zstd_process.stdout.close()
        zstd_process.wait()
        raise

    return _validate_kgx_directory(output_dir)


@contextmanager
def resolved_kgx_input(input_path: str | Path) -> Iterator[tuple[Path, Path, str]]:
    path = Path(input_path)
    dataset_name = infer_dataset_name(path)

    if path.is_dir():
        edges_path, nodes_path = _validate_kgx_directory(path)
        yield edges_path, nodes_path, dataset_name
        return

    if not path.exists():
        raise FileNotFoundError(f"Input path not found: {path}")

    if not _is_tar_zst_archive(path):
        raise ValueError(
            "Input path must be a KGX directory or a `.tar.zst` archive"
        )

    with tempfile.TemporaryDirectory(prefix="gandalf_kgx_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        edges_path, nodes_path = extract_tar_zst_archive(path, temp_dir)
        yield edges_path, nodes_path, dataset_name
