#!/usr/bin/env python3
"""Throwaway: count (and optionally list) edges by infores, category and predicate.

Every filter is optional and they combine with AND.  ``--infores`` matches an
edge whose ``sources`` list carries that resource id (any role, unless
``--role`` narrows it); a bare name gets the ``infores:`` prefix.

A node matches a category if any entry of its ``categories`` list matches, so a
node tagged ``[biolink:Gene, biolink:GeneOrGeneProduct]`` counts as a Gene.
By default each category and predicate is expanded to its Biolink
descendants (``biolink:PhenotypicFeature`` also matches
``biolink:ClinicalFinding``); pass ``--exact`` to turn that off.

Examples:
    # Every edge from infores:gene2phenotype, by predicate and category pair
    python scripts/count_edges.py --graph graph_mmap/ --infores gene2phenotype

    # ... only where it is the primary knowledge source
    python scripts/count_edges.py --graph graph_mmap/ \\
        --infores gene2phenotype --role primary_knowledge_source

    # Every infores in the graph with its edge count
    python scripts/count_edges.py --graph graph_mmap/ --list-infores

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


def infores_name(name: str) -> str:
    return name if name.startswith("infores:") else f"infores:{name}"


def source_pool_mask(graph, infores, role):
    """Boolean array over the interned source lists: does each list match."""
    pool = graph.edge_properties._sources_pool
    mask = np.zeros(len(pool), dtype=np.bool_)
    for i, sources in enumerate(pool):
        mask[i] = any(
            s.get("resource_id") in infores
            and (role is None or s.get("resource_role") == role)
            for s in sources
        )
    return mask


def infores_edge_counts(graph):
    """Counter of (resource_id, resource_role) -> number of edges carrying it."""
    pool = graph.edge_properties._sources_pool
    per_list = np.bincount(
        np.asarray(graph.edge_properties._sources_idx), minlength=len(pool)
    )
    counts = Counter()
    for sources, n in zip(pool, per_list.tolist()):
        for s in {(s.get("resource_id"), s.get("resource_role")) for s in sources}:
            counts[s] += n
    return counts


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
    p.add_argument("--infores", nargs="+", help="infores id(s), e.g. gene2phenotype")
    p.add_argument(
        "--role",
        help="only match --infores in this resource_role "
        "(e.g. primary_knowledge_source)",
    )
    p.add_argument(
        "--list-infores",
        action="store_true",
        help="print every infores/role in the graph with its edge count, then exit",
    )
    p.add_argument("--subject", nargs="+", help="subject category(ies)")
    p.add_argument("--object", nargs="+", help="object category(ies)")
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

    graph = CSRGraph.load_mmap(args.graph)
    print(f"Graph: {graph.num_nodes:,} nodes, {len(graph.fwd_targets):,} edges")

    if args.list_infores:
        counts = infores_edge_counts(graph)
        width = max(len(str(r)) for r, _ in counts)
        for (rid, role), n in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"  {rid:<{width}}  {role:<30}  {n:>12,}")
        return

    offsets = np.asarray(graph.fwd_offsets)
    targets = np.asarray(graph.fwd_targets)
    preds = np.asarray(graph.fwd_predicates)
    sources = np.repeat(np.arange(graph.num_nodes, dtype=np.int64), np.diff(offsets))

    hit = np.ones(len(targets), dtype=np.bool_)
    if args.infores:
        infores = {infores_name(n) for n in args.infores}
        print(f"Infores:            {sorted(infores)} (role: {args.role or 'any'})")
        pool_mask = source_pool_mask(graph, infores, args.role)
        hit &= pool_mask[np.asarray(graph.edge_properties._sources_idx)]
        if not pool_mask.any():
            # Probably a typo or a different id; suggest close matches.
            needles = [n.split(":", 1)[-1].lower() for n in infores]
            similar = sorted(
                {
                    rid
                    for rid, _ in infores_edge_counts(graph)
                    if rid and any(x in rid.lower() for x in needles)
                }
            )
            print(
                f"  no edge carries it{' in that role' if args.role else ''}; similar infores in graph: {similar or 'none'}"
            )
    if args.subject or args.object:
        subject_cats = expand(args.subject or ["NamedThing"], args.exact)
        object_cats = expand(args.object or ["NamedThing"], args.exact)
        print(f"Subject categories: {sorted(subject_cats)}")
        print(f"Object categories:  {sorted(object_cats)}")
        subj_mask, obj_mask = category_masks(graph, subject_cats, object_cats)
        print(
            f"Matching nodes: {int(subj_mask.sum()):,} subject, "
            f"{int(obj_mask.sum()):,} object"
        )
        cat_hit = subj_mask[sources] & obj_mask[targets]
        if args.either_direction:
            cat_hit |= obj_mask[sources] & subj_mask[targets]
        hit &= cat_hit
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

    # Break down by (subject category, object category) using each node's
    # first (most specific) category, read only for the nodes involved.
    print()
    first_cat = {}
    involved = np.unique(np.concatenate([sources[positions], targets[positions]]))
    for start in range(0, len(involved), CHUNK):
        chunk = involved[start : start + CHUNK].tolist()
        for idx, props in graph.get_all_node_properties_batch(chunk).items():
            first_cat[idx] = (props.get("categories") or ["biolink:NamedThing"])[0]
    by_cats = Counter(
        (first_cat.get(int(a), "?"), first_cat.get(int(b), "?"))
        for a, b in zip(sources[positions], targets[positions])
    )
    width = max(len(a) + len(b) + 4 for a, b in by_cats)
    for (a, b), n in by_cats.most_common():
        print(f"  {a + ' -> ' + b:<{width}}  {n:>12,}")

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
