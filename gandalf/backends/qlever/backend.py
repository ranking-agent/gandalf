"""QLever backend runtime."""

import json
import logging
from pathlib import Path
from typing import Any

from gandalf.backends.base import Backend
from gandalf.backends.qlever.edge_lookup import EdgeIdLookup
from gandalf.backends.qlever.qlever_client import (
    consume_qlever_json_rows,
    run_qlever_query_json,
)
from gandalf.backends.qlever.runtime import answer_trapi_request, load_manifest
from gandalf.config import settings
from gandalf.graph import CSRGraph
from gandalf.search.lookup import execute_with_trapi_logging

logger = logging.getLogger(__name__)


class QLeverBackend(Backend):
    """Runtime for querying a prebuilt QLever backend artifact set."""

    def __init__(
        self,
        artifact_dir: str | Path,
        *,
        host_name: str,
        port: int,
        access_token: str | None = None,
    ):
        self.artifact_dir = Path(artifact_dir)
        self.host_name = host_name
        self.port = port
        self.access_token = access_token
        self.manifest = load_manifest(self.artifact_dir)

        csr_artifact_dir = self.artifact_dir / self.manifest["csr_artifact_dir"]
        self.graph = CSRGraph.load_mmap(csr_artifact_dir)
        self.edge_lookup = EdgeIdLookup(
            self.artifact_dir / self.manifest["edge_id_lookup_path"],
            readonly=True,
        )

        self.meta_kg = self.graph.meta_kg
        self.sri_testing_data = self.graph.sri_testing_data
        self.graph_metadata = self.graph.graph_metadata

    @classmethod
    def load(
        cls,
        artifact_dir: str | Path,
        *,
        host_name: str,
        port: int,
        access_token: str | None = None,
    ):
        return cls(
            artifact_dir,
            host_name=host_name,
            port=port,
            access_token=access_token,
        )

    def _run_query(self, query: str) -> tuple[dict[str, int], list[list[str]]]:
        result = run_qlever_query_json(
            self.host_name,
            self.port,
            query,
            access_token=self.access_token,
        )
        return consume_qlever_json_rows(result)

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
        return execute_with_trapi_logging(
            lambda *, t_start, gc_monitor: answer_trapi_request(
                query,
                graph=self.graph,
                edge_lookup=self.edge_lookup,
                run_qlever_query=self._run_query,
                resource_id=settings.infores,
                subclass=subclass,
                subclass_depth=subclass_depth,
                max_node_degree=max_node_degree,
                min_information_content=min_information_content,
                dehydrated=dehydrated,
                bmt=bmt,
                t_start=t_start,
                gc_monitor=gc_monitor,
            ),
            log_level=log_level,
        )

    def close(self) -> None:
        self.edge_lookup.close()
        self.graph.close()
