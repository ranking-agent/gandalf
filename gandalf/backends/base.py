"""Shared backend protocol for Gandalf runtimes."""

from abc import ABC, abstractmethod
from typing import Any


class Backend(ABC):
    """Common interface for CSR and QLever runtimes."""

    meta_kg: dict[str, Any] | None
    sri_testing_data: dict[str, Any] | None
    graph_metadata: dict[str, Any] | None

    @abstractmethod
    def lookup(
        self,
        query: dict[str, Any],
        bmt=None,
        subclass: bool = True,
        subclass_depth: int = 1,
        max_node_degree=None,
        min_information_content=None,
        log_level=None,
        dehydrated=None,
    ) -> dict[str, Any]:
        """Execute a query against the backend."""

    @abstractmethod
    def close(self) -> None:
        """Release backend resources."""
