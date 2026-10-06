#!/usr/bin/env python3
"""Throwaway: count (and optionally list) edges between two node categories.

A node matches a category if any entry of its ``categories`` list matches, so a
node tagged ``[biolink:Gene, biolink:GeneOrGeneProduct]`` counts as a Gene.
By default each category and predicate is expanded to its Biolink
descendants (``biolink:PhenotypicFeature`` also matches
``biolink:ClinicalFinding``); pass ``--exact`` to turn that off.

Examples:
    # All gene -> phenotype edges, broken down by predicate
    python scripts/count_edges.py --graph graph_mmap/ \\
        --subject biolink:Gene --object biolink:PhenotypicFeature

    # Only one predicate, either direction, and dump the matches
    python scripts/count_edges.py --graph graph_mmap/ \\
        --subject Gene --object PhenotypicFeature \\
        --predicate has_phenotype --either-direction --dump matches.tsv
"""

import argparse
import logging
import sys
from collections import Counter

import numpy as np

from gandalf.graph import CSRGraph

logger = logging.getLogger(__name__)

CHUNK = 100_000


def curie(name: str) -> str:
    return name if ":" in name else f"biolink:{name}"


def expand(names, exact):
    """Return the set of CURIEs plus (unless *exact*) their Biolink descendants."""
    names = {curie(n) for n in names}
    if exact:
        return names
    try:
        from gandalf.biolink import make_toolkit

        bmt = make_toolkit()
        out = set(names)
        for n in names:
            out.update(bmt.get_descendants(n, formatted=True) or [])
        return out
    except Exception as e:
        print(
            f"warning: could not load Biolink model ({e}); matching exactly",
            file=sys.stderr,
        )
        return names


def category_masks(graph, subject_cats, object_cats):
    """Boolean arrays over node indices: does the node match each category set."""
    subj = np.zeros(graph.num_nodes, dtype=np.bool_)
    obj = np.zeros(graph.num_nodes, dtype=np.bool_)
    for start in range(0, graph.num_nodes, CHUNK):
        idxs = range(start, min(start + CHUNK, graph.num_nodes))
        props = graph.get_all_node_properties_batch(idxs)
        for idx in idxs:
            cats = props.get(idx, {}).get("categories") or ["biolink:NamedThing"]
            subj[idx] = not subject_cats.isdisjoint(cats)
            obj[idx] = not object_cats.isdisjoint(cats)
    return subj, obj


def main():
    p = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        epilog=__doc__.split("Examples:")[1],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--graph", required=True, help="mmap graph directory")
    p.add_argument("--subject", required=True, nargs="+", help="subject category(ies)")
    p.add_argument("--object", required=True, nargs="+", help="object category(ies)")
    p.add_argument("--predicate", nargs="+", help="restrict to these predicate(s)")
    p.add_argument(
        "--exact",
        action="store_true",
        help="don't expand categories/predicates to Biolink descendants",
    )
    p.add_argument(
        "--either-direction",
        action="store_true",
        help="also count object -> subject edges",
    )
    p.add_argument(
        "--examples", type=int, default=5, help="example edges to print (default 5)"
    )
    p.add_argument("--dump", help="write every matching edge to this TSV file")
    args = p.parse_args()

    logging.basicConfig(level=logging.WARNING)

    subject_cats = expand(args.subject, args.exact)
    object_cats = expand(args.object, args.exact)
    print(f"Subject categories: {sorted(subject_cats)}")
    print(f"Object categories:  {sorted(object_cats)}")

    graph = CSRGraph.load_mmap(args.graph)
    print(f"Graph: {graph.num_nodes:,} nodes, {len(graph.fwd_targets):,} edges")

    subj_mask, obj_mask = category_masks(graph, subject_cats, object_cats)
    print(
        f"Matching nodes: {int(subj_mask.sum()):,} subject, "
        f"{int(obj_mask.sum()):,} object"
    )

    offsets = np.asarray(graph.fwd_offsets)
    targets = np.asarray(graph.fwd_targets)
    preds = np.asarray(graph.fwd_predicates)
    sources = np.repeat(np.arange(graph.num_nodes, dtype=np.int64), np.diff(offsets))

    hit = subj_mask[sources] & obj_mask[targets]
    if args.either_direction:
        hit |= obj_mask[sources] & subj_mask[targets]
    if args.predicate:
        wanted = expand(args.predicate, args.exact)
        present = wanted & set(graph.predicate_to_idx)
        print(f"Predicates:         {sorted(present) or '(none in graph)'}")
        hit &= graph.predicate_mask(wanted)[preds]

    positions = np.flatnonzero(hit)
    print(f"\nTOTAL MATCHING EDGES: {len(positions):,}\n")
    if not len(positions):
        return

    by_pred = Counter(preds[positions].tolist())
    width = max(len(graph.id_to_predicate[p]) for p in by_pred)
    for pred_id, n in by_pred.most_common():
        print(f"  {graph.id_to_predicate[pred_id]:<{width}}  {n:>12,}")

    def rows(pos_list):
        node_ids = graph.get_node_ids_batch(
            set(sources[pos_list].tolist()) | set(targets[pos_list].tolist())
        )
        edge_ids = graph.get_edge_ids_batch(pos_list)
        for pos in pos_list:
            yield (
                node_ids.get(int(sources[pos])),
                graph.id_to_predicate[int(preds[pos])],
                node_ids.get(int(targets[pos])),
                edge_ids.get(int(pos)) or "",
            )

    if args.examples:
        print("\nExamples:")
        for row in rows(positions[: args.examples]):
            print("  " + "  ".join(map(str, row)))

    if args.dump:
        with open(args.dump, "w") as f:
            f.write("subject\tpredicate\tobject\tedge_id\n")
            for start in range(0, len(positions), CHUNK):
                for row in rows(positions[start : start + CHUNK]):
                    f.write("\t".join(map(str, row)) + "\n")
        print(f"\nWrote {len(positions):,} edges to {args.dump}")


if __name__ == "__main__":
    main()
