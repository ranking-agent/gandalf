"""Backend loading."""

from pathlib import Path
from typing import Literal

from gandalf.backends.csr_backend import CSRBackend
from gandalf.backends.qlever.backend import QLeverBackend


def load_backend(
    path: str | Path,
    format: Literal["csr", "qlever"],
    *,
    qlever_host: str = "localhost",
    qlever_port: int = 8888,
    qlever_access_token: str | None = None,
):
    """Load the configured backend runtime from disk."""
    resolved_path = Path(path)
    if not resolved_path.is_dir():
        raise ValueError(f"Expected backend artifact directory: {resolved_path}")

    if format == "csr":
        if not (resolved_path / "fwd_targets.npy").exists():
            raise ValueError(
                f"CSR artifacts not found in {resolved_path}: expected fwd_targets.npy"
            )
        return CSRBackend.load_mmap(resolved_path)

    if not (resolved_path / "backend.json").exists():
        raise ValueError(
            f"QLever artifacts not found in {resolved_path}: expected backend.json"
        )

    if format == "qlever":
        return QLeverBackend.load(
            resolved_path,
            host_name=qlever_host,
            port=qlever_port,
            access_token=qlever_access_token,
        )
    raise ValueError(f"Unknown backend: {format}")
