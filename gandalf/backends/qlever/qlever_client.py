"""Minimal QLever HTTP client helpers."""

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


def run_qlever_query_json(
    host_name: str,
    port: int,
    query: str,
    access_token: str | None = None,
) -> dict[str, Any]:
    """Execute a QLever query and return parsed qlever-results+json."""
    data = urllib.parse.urlencode({"query": query}).encode("utf-8")
    request = urllib.request.Request(
        f"http://{host_name}:{port}",
        data=data,
        headers={"Accept": "application/qlever-results+json"},
        method="POST",
    )
    if access_token:
        request.add_header("Authorization", f"Bearer {access_token}")

    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request) as response:
            payload = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace").strip()
        detail = f"QLever HTTP {exc.code}: {exc.reason}"
        if error_body:
            detail += f" - {error_body}"
        raise RuntimeError(detail) from exc
    elapsed_ms = round((time.perf_counter() - start) * 1000)

    parse_start = time.perf_counter()
    payload_json = json.loads(payload)
    json_parse_ms = round((time.perf_counter() - parse_start) * 1000)
    return {
        "format": "qlever-results+json",
        "elapsed_ms": elapsed_ms,
        "json_parse_ms": json_parse_ms,
        "json": payload_json,
    }


def normalize_iri(value: str) -> str:
    """Strip surrounding angle brackets from result values when present."""
    if value.startswith("<") and value.endswith(">"):
        return value[1:-1]
    return value


def consume_qlever_json_rows(result: dict[str, Any]) -> tuple[dict[str, int], list[list[str]]]:
    """Convert qlever-results+json payload to row arrays."""
    if result["format"] != "qlever-results+json":
        raise ValueError("Expected qlever-results+json payload")

    row_column_indexes = {
        column: index
        for index, column in enumerate(result["json"].get("selected", []))
    }
    rows: list[list[str]] = []
    for values in result["json"].get("res", []):
        if not values:
            continue
        rows.append([normalize_iri(value) for value in values])
    return row_column_indexes, rows
