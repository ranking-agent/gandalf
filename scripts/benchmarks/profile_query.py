#!/usr/bin/env python3
"""Profile a single TRAPI query end-to-end against a running Gandalf server.

Unlike the in-process profiler (which only times what happens *inside*
``lookup()``), this measures the full client-observed round trip and then
subtracts the profiler's ``lookup`` total to isolate the **response
encode + transport** phase -- the part that happens after ``lookup()``
returns (orjson serialization, ASGI send, network).

    end_to_end (client wall-clock)
      = lookup() total            [from the ProfileSummary in TRAPI logs]
      + encode + transport        [derived]

Usage:
    python scripts/benchmarks/profile_query.py \
        --url http://localhost:6429/query \
        --query my_two_hop.json \
        --repeat 2

``--repeat 2`` runs the same query twice; comparing run 1 (cold page cache)
with run 2 (warm) shows how much time is I/O-bound paging vs CPU.

The query file is a TRAPI request body (the same JSON you would POST to
``/query``).  With no ``--query`` a built-in two-hop example is used.
"""

import argparse
import json
import sys
import time

import httpx

# A generic two-hop shape; replace with --query for a real workload.
_DEFAULT_QUERY = {
    "message": {
        "query_graph": {
            "nodes": {
                "n0": {"ids": ["MONDO:0005148"]},
                "n1": {"categories": ["biolink:Gene"]},
                "n2": {"categories": ["biolink:ChemicalEntity"]},
            },
            "edges": {
                "e0": {"subject": "n0", "object": "n1"},
                "e1": {"subject": "n1", "object": "n2"},
            },
        }
    },
    "parameters": {},
}


def _extract_profile(response: dict) -> dict | None:
    """Pull the ProfileSummary tree out of the TRAPI logs, if present.

    Gandalf attaches profile entries to the top-level ``logs`` (a sibling of
    ``message``); older/other TRAPI producers use ``message.logs``, so check
    both.
    """
    logs = response.get("logs") or (response.get("message") or {}).get("logs") or []
    for entry in logs:
        if entry.get("code") == "ProfileSummary":
            try:
                return json.loads(entry["message"])
            except (KeyError, ValueError):
                return None
    return None


def _stage_rows(node: dict, depth: int = 0) -> list[tuple[str, float]]:
    """Flatten the profile tree into (indented label, duration_ms) rows."""
    rows = [("  " * depth + node.get("name", "?"), node.get("duration_ms") or 0.0)]
    for child in node.get("children", []):
        rows.extend(_stage_rows(child, depth + 1))
    return rows


def run_once(
    client: httpx.Client, url: str, body: dict, keep_raw: bool = False
) -> dict:
    t0 = time.perf_counter()
    resp = client.post(url, params={"profile": "true"}, json=body)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    resp.raise_for_status()
    payload_bytes = len(resp.content)
    data = resp.json()
    results = (data.get("message") or {}).get("results") or []
    profile = _extract_profile(data)
    return {
        "end_to_end_ms": elapsed_ms,
        "payload_bytes": payload_bytes,
        "num_results": len(results),
        "profile": profile,
        "raw": data if keep_raw else None,
    }


def report(run_idx: int, r: dict) -> None:
    e2e = r["end_to_end_ms"]
    prof = r["profile"]
    lookup_ms = (prof or {}).get("duration_ms")
    print(f"\n=== run {run_idx} ===")
    print(f"  results              : {r['num_results']:,}")
    print(f"  payload              : {r['payload_bytes'] / 1e6:.1f} MB")
    print(f"  end-to-end (client)  : {e2e:8.1f} ms")
    if lookup_ms is not None:
        encode_transport = e2e - lookup_ms
        print(f"  lookup() total       : {lookup_ms:8.1f} ms")
        print(
            f"  encode + transport   : {encode_transport:8.1f} ms"
            f"   ({encode_transport / e2e * 100:.0f}% of end-to-end)"
        )
        print("  lookup() stage breakdown:")
        for label, dur in _stage_rows(prof):
            print(f"    {label:<32} {dur:8.1f} ms")
        lmdb = prof.get("lmdb") or {}
        if lmdb.get("calls"):
            print(
                f"  LMDB: {lmdb['calls']} calls, "
                f"{lmdb['total_keys']:,} keys, {lmdb['total_ms']:.1f} ms"
            )
    else:
        print("  (no ProfileSummary in message.logs -- is the server built with the")
        print("   profiler, and did the request include ?profile=true?)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://localhost:6429/query")
    ap.add_argument("--query", help="Path to a TRAPI request body JSON file")
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--out", help="Write the last response JSON here")
    args = ap.parse_args()

    if args.query:
        with open(args.query, "r", encoding="utf-8") as f:
            body = json.load(f)
    else:
        body = _DEFAULT_QUERY

    runs = []
    with httpx.Client(timeout=args.timeout) as client:
        for i in range(1, args.repeat + 1):
            keep_raw = bool(args.out) and i == args.repeat
            r = run_once(client, args.url, body, keep_raw=keep_raw)
            report(i, r)
            runs.append(r)

    if args.repeat > 1 and all(r["profile"] for r in runs):
        cold, warm = runs[0]["end_to_end_ms"], runs[-1]["end_to_end_ms"]
        print(
            f"\nwarm vs cold end-to-end: {cold:.0f} ms -> {warm:.0f} ms "
            f"({(cold - warm) / cold * 100:+.0f}%); a large drop means I/O-bound paging."
        )

    if args.out and runs and runs[-1]["raw"] is not None:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(runs[-1]["raw"], f)
        print(f"\nwrote last response to {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
