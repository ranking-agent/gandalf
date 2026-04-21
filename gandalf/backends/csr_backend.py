"""CSR runtime backend wrapper."""

from typing import Any

from gandalf.backends.base import Backend
from gandalf.graph import CSRGraph
from gandalf.search.lookup import lookup as csr_lookup


class CSRBackend(Backend):
    """Backend wrapper around the existing CSR graph runtime."""

    def __init__(self, graph: CSRGraph):
        self.graph = graph
        self.meta_kg = graph.meta_kg
        self.sri_testing_data = graph.sri_testing_data
        self.graph_metadata = graph.graph_metadata

    @classmethod
    def load_mmap(cls, path, mmap_mode="r"):
        return cls(CSRGraph.load_mmap(path, mmap_mode=mmap_mode))

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
        return csr_lookup(
            self.graph,
            query,
            bmt=bmt,
            subclass=subclass,
            subclass_depth=subclass_depth,
            max_node_degree=max_node_degree,
            min_information_content=min_information_content,
            log_level=log_level,
            dehydrated=dehydrated,
        )

    def close(self) -> None:
        self.graph.close()
