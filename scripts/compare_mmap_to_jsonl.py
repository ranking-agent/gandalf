#!/usr/bin/env python3
"""Throwaway: check that a built mmap graph matches the JSONL it came from.

Runs the JSONL through the same normalization ``gandalf-build`` applies
(``KGXJsonlSource``, ``prune_retrieval_sources``, ``kl_at_pair``, the loader's
node-property shape), then compares against what the graph directory holds.

Edges are compared as a multiset of canonical records (subject, predicate,
object, id, sources, qualifiers, knowledge_level, agent_type, attributes), so
CSR re-ordering doesn't matter but duplicates do.  Nodes are compared by id.

Example:
    python scripts/compare_mmap_to_jsonl.py --graph graph_mmap/ \\
        --edges data/edges.jsonl --nodes data/nodes.jsonl

Exit code is 0 when everything matches, 1 otherwise.
"""

import argparse
import hashlib
import sys
from collections import defaultdict

import numpy as np
import orjson

from gandalf.graph import CSRGraph, kl_at_pair
from gandalf.biolink import NAMED_THING
from gandalf.node_annotations import BIOTHINGS_ANNOTATIONS_ATTRIBUTE_TYPE_ID
from gandalf.sources import KGXJsonlSource
from gandalf.trapi import prune_retrieval_sources

CHUNK = 100_000


def canon(obj) -> bytes:
    return orjson.dumps(obj, option=orjson.OPT_SORT_KEYS)


def sorted_quals(quals):
    # normalize._extract_qualifiers iterates a set, so qualifier order depends
    # on the hash seed of the build process; order carries no meaning.
    return sorted(quals or [], key=canon)


def digest(b: bytes) -> int:
    return int.from_bytes(hashlib.blake2b(b, digest_size=8).digest(), "little")


# ----------------------------------------------------------------------
# Edges
# ----------------------------------------------------------------------


def expected_edges(edges_path):
    """Yield canonical edge records as the builder would store them."""
    for edge in KGXJsonlSource(edges_path).iter_edges():
        kl, at = kl_at_pair(edge)
        yield {
            "subject": edge["subject"],
            "predicate": edge["predicate"],
            "object": edge["object"],
            "id": edge.get("id"),
            "sources": prune_retrieval_sources(edge["sources"]),
            "qualifiers": sorted_quals(edge["qualifiers"]),
            "knowledge_level": kl,
            "agent_type": at,
            "attributes": edge["attributes"],
        }


def graph_edges(graph, node_ids):
    """Yield canonical edge records reconstructed from the graph."""
    offsets = np.asarray(graph.fwd_offsets)
    num_edges = len(graph.fwd_targets)
    subjects = np.repeat(np.arange(graph.num_nodes), np.diff(offsets))
    for start in range(0, num_edges, CHUNK):
        positions = range(start, min(start + CHUNK, num_edges))
        edge_ids = graph.get_edge_ids_batch(positions)
        details = (
            graph.lmdb_store.get_batch(positions)
            if graph.lmdb_store is not None
            else {}
        )
        for pos in positions:
            props = graph.get_edge_properties_by_index(
                pos, lmdb_detail=details.get(pos, {})
            )
            yield {
                "subject": node_ids[int(subjects[pos])],
                "predicate": props["predicate"],
                "object": node_ids[int(graph.fwd_targets[pos])],
                "id": edge_ids.get(pos),
                "sources": props["sources"],
                "qualifiers": sorted_quals(props["qualifiers"]),
                "knowledge_level": props["knowledge_level"],
                "agent_type": props["agent_type"],
                # The builder always writes an attributes entry, even if empty.
                "attributes": props.get("attributes"),
            }


def digest_array(records) -> np.ndarray:
    return np.sort(np.fromiter((digest(canon(r)) for r in records), dtype=np.uint64))


def multiset_diff(a: np.ndarray, b: np.ndarray):
    """Return {digest: count_in_a - count_in_b} for digests whose counts differ."""
    ua, ca = np.unique(a, return_counts=True)
    ub, cb = np.unique(b, return_counts=True)
    counts = defaultdict(int)
    # Only walk the digests that aren't matched one-for-one.
    common, ia, ib = np.intersect1d(ua, ub, assume_unique=True, return_indices=True)
    for d, x, y in zip(common, ca[ia], cb[ib]):
        if x != y:
            counts[int(d)] = int(x) - int(y)
    for d in np.setdiff1d(ua, common, assume_unique=True):
        counts[int(d)] += int(ca[np.searchsorted(ua, d)])
    for d in np.setdiff1d(ub, common, assume_unique=True):
        counts[int(d)] -= int(cb[np.searchsorted(ub, d)])
    return counts


def edge_key(r):
    return (r["subject"], r["predicate"], r["object"], r["id"])


def field_diff(exp, got):
    lines = []
    for k in exp:
        if canon(exp[k]) != canon(got.get(k)):
            lines.append(f"      {k}:")
            lines.append(f"        jsonl: {canon(exp[k]).decode()[:500]}")
            lines.append(f"        graph: {canon(got.get(k)).decode()[:500]}")
    return lines


def compare_edges(graph, node_ids, edges_path, max_diffs):
    print("Hashing JSONL edges...", flush=True)
    exp = digest_array(expected_edges(edges_path))
    print(f"  {len(exp):,} edges")
    print("Hashing graph edges...", flush=True)
    got = digest_array(graph_edges(graph, node_ids))
    print(f"  {len(got):,} edges")

    if len(exp) == len(got) and np.array_equal(exp, got):
        print("EDGES: MATCH")
        return True

    diff = multiset_diff(exp, got)
    only_jsonl = sum(c for c in diff.values() if c > 0)
    only_graph = -sum(c for c in diff.values() if c < 0)
    print(
        f"EDGES: MISMATCH -- {only_jsonl:,} records only in JSONL, "
        f"{only_graph:,} only in graph"
    )

    # Second pass: pull up to max_diffs example records from each side and
    # pair them by (subject, predicate, object, id) to show field diffs.
    def collect(records, sign):
        wanted = {d for d, c in diff.items() if (c > 0) == (sign > 0)}
        found = {}
        for r in records:
            if len(found) >= max_diffs:
                break
            d = digest(canon(r))
            if d in wanted and d not in found:
                found[d] = r
        return list(found.values())

    print("Collecting examples...", flush=True)
    exp_ex = collect(expected_edges(edges_path), +1)
    got_ex = collect(graph_edges(graph, node_ids), -1)
    got_by_key = {edge_key(r): r for r in got_ex}
    for r in exp_ex:
        other = got_by_key.pop(edge_key(r), None)
        if other is not None:
            print(f"  DIFFERS {edge_key(r)}")
            print("\n".join(field_diff(r, other)))
        else:
            print(f"  ONLY IN JSONL {edge_key(r)}")
    for key in got_by_key:
        print(f"  ONLY IN GRAPH {key}")
    return False


# ----------------------------------------------------------------------
# Nodes
# ----------------------------------------------------------------------


def expected_node_props(node):
    """The loader's node-property shape (see gandalf.loader)."""
    props = {
        "categories": node.get("categories") or [NAMED_THING],
        "attributes": node.get("attributes", []),
    }
    if node.get("name") is not None:
        props["name"] = node["name"]
    return props


def strip_annotations(props):
    attrs = props.get("attributes")
    if not attrs:
        return props
    props = dict(props)
    props["attributes"] = [
        a
        for a in attrs
        if a.get("attribute_type_id") != BIOTHINGS_ANNOTATIONS_ATTRIBUTE_TYPE_ID
    ]
    return props


def compare_nodes(graph, node_ids, nodes_path, max_diffs, ignore_annotations):
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    print("Reading JSONL nodes...", flush=True)
    expected = {}  # last record wins, same as the loader
    not_on_edges = 0
    for node in KGXJsonlSource("", nodes_path).iter_nodes():
        idx = id_to_idx.get(node["id"])
        if idx is None:
            not_on_edges += 1
            continue
        expected[idx] = expected_node_props(node)
    print(f"  {len(expected):,} nodes on edges ({not_on_edges:,} not on any edge)")

    mismatches = 0
    no_record_with_props = 0
    for start in range(0, graph.num_nodes, CHUNK):
        idxs = range(start, min(start + CHUNK, graph.num_nodes))
        stored = graph.get_all_node_properties_batch(idxs)
        for idx in idxs:
            got = stored.get(idx, {})
            exp = expected.get(idx)
            if ignore_annotations:
                got = strip_annotations(got)
            if exp is None:
                # Edge endpoint with no node record: the loader stores nothing
                # (or a bare annotated shell, stripped above).
                if got and (got.get("attributes") or got.get("name")):
                    no_record_with_props += 1
                continue
            if canon(exp) != canon(got):
                mismatches += 1
                if mismatches <= max_diffs:
                    print(f"  DIFFERS {node_ids[idx]}")
                    print("\n".join(field_diff(exp, got)))

    if no_record_with_props:
        print(
            f"  note: {no_record_with_props:,} graph nodes have no JSONL record "
            f"but carry stored properties"
        )
    if mismatches or no_record_with_props:
        print(f"NODES: MISMATCH -- {mismatches:,} nodes differ")
        return False
    print("NODES: MATCH")
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--graph", required=True, help="mmap graph directory")
    p.add_argument("--edges", required=True, help="original edges JSONL")
    p.add_argument("--nodes", help="original nodes JSONL (skip node check if omitted)")
    p.add_argument("--max-diffs", type=int, default=10, help="examples to print")
    p.add_argument(
        "--ignore-annotations",
        action="store_true",
        help="drop biothings_annotations attributes (graphs built with --annotate)",
    )
    p.add_argument("--skip-edges", action="store_true")
    args = p.parse_args()

    graph = CSRGraph.load_mmap(args.graph)
    print(f"Graph: {graph.num_nodes:,} nodes, {len(graph.fwd_targets):,} edges")
    ids = graph.get_node_ids_batch(range(graph.num_nodes))
    node_ids = [ids.get(i) for i in range(graph.num_nodes)]

    ok = True
    if not args.skip_edges:
        ok &= compare_edges(graph, node_ids, args.edges, args.max_diffs)
    if args.nodes:
        ok &= compare_nodes(
            graph, node_ids, args.nodes, args.max_diffs, args.ignore_annotations
        )
    print("ALL MATCH" if ok else "DIFFERENCES FOUND")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
