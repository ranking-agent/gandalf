"""Deterministic synthetic knowledge graph + query set for lookup benchmarks.

The graph is shaped to exercise the parts of ``lookup()`` that dominate on
the real Translator graph, at a size that builds in a minute or two:

* a skewed (Zipf-like) degree distribution, so a handful of hub genes and
  diseases have very large neighbourhoods;
* realistically sized node records (equivalent identifiers, xrefs,
  description, information content), since node-store reads unpack the
  whole record;
* qualifiers, multiple sources, and publications on edges;
* a ``subclass_of`` disease hierarchy, so subclass expansion does real work;
* symmetric (``interacts_with``) and canonical/inverse predicate pairs.

The same ``seed`` and ``scale`` always produce the same graph, so numbers
from different commits are comparable.

Examples:
    >>> spec = SCALES["tiny"]
    >>> spec.n_chemicals > 0
    True
    >>> [q["name"] for q in synthetic_queries()][:2]
    ['1hop_treats_hub_disease', '1hop_affects_hub_gene_qualified']
"""

import json
import random
from dataclasses import dataclass
from pathlib import Path

#: The most-connected disease: pinned by several queries and the root of the
#: subclass hierarchy, so subclass expansion reaches every other disease.
HUB_DISEASE = "MONDO:0000001"
#: A mid-hierarchy disease, for queries that should stay moderately sized.
MID_DISEASE = "MONDO:0000010"
#: The most-connected gene.
HUB_GENE = "NCBIGene:1"
#: A well-connected chemical, used as the pinned end of the path queries.
HUB_CHEMICAL = "CHEBI:1"

_SOURCES = [
    "infores:ctd",
    "infores:drugcentral",
    "infores:chembl",
    "infores:gwas-catalog",
    "infores:hpo-annotations",
    "infores:string",
]
_ASPECTS = ["activity", "abundance", "expression", "activity_or_abundance"]
_DIRECTIONS = ["increased", "decreased"]


@dataclass(frozen=True)
class ScaleSpec:
    """Node counts and mean out-degrees for one benchmark graph size."""

    n_chemicals: int
    n_genes: int
    n_diseases: int
    n_phenotypes: int
    chem_affects_gene: int
    chem_treats_disease: int
    gene_assoc_disease: int
    gene_interacts_gene: int
    disease_has_phenotype: int


SCALES = {
    # Seconds to build; for smoke-testing the harness itself.
    "tiny": ScaleSpec(300, 600, 200, 100, 6, 2, 3, 4, 4),
    "small": ScaleSpec(3_000, 8_000, 2_000, 1_000, 10, 3, 5, 6, 6),
    # ~1.2M edges; builds in a couple of minutes and produces queries with
    # 10^5-10^6 paths, which is where the join/response costs show.
    "medium": ScaleSpec(15_000, 30_000, 8_000, 4_000, 25, 4, 10, 8, 8),
}


def _zipf_pick(rng: random.Random, n: int, skew: float = 3.0) -> int:
    """Pick an index in ``[0, n)`` with a heavy head (index 0 is the hub).

    Uses an inverse-power transform of a uniform draw, which is cheap and
    deterministic for a seeded ``rng``.  With the default ``skew`` of 3 the
    first 1% of indices receive about a fifth of all picks.

    >>> rng = random.Random(0)
    >>> picks = [_zipf_pick(rng, 1000) for _ in range(10_000)]
    >>> 0.15 < sum(p < 10 for p in picks) / len(picks) < 0.25
    True
    """
    u = rng.random()
    return min(int(n * u**skew), n - 1)


def _node_record(curie: str, name: str, categories: list, rng: random.Random) -> dict:
    """A KGX node with roughly the property payload of a real one."""
    prefix = curie.split(":")[0]
    return {
        "id": curie,
        "name": name,
        "category": categories,
        "equivalent_identifiers": [curie]
        + [f"{prefix}X{k}:{rng.randrange(10**7)}" for k in range(5)],
        "xref": [f"XREF{k}:{rng.randrange(10**7)}" for k in range(6)],
        "description": f"Synthetic {categories[0]} {name} " + "lorem ipsum " * 6,
        "information_content": round(rng.uniform(40.0, 100.0), 1),
    }


def _edge_record(
    subj: str, pred: str, obj: str, rng: random.Random, qualified: bool = False
) -> dict:
    """A KGX edge with sources, publications, and optional qualifiers."""
    edge = {
        "id": f"{subj}-{pred}-{obj}-{rng.randrange(10**9)}",
        "subject": subj,
        "predicate": pred,
        "object": obj,
        "sources": [
            {
                "resource_id": rng.choice(_SOURCES),
                "resource_role": "primary_knowledge_source",
            },
            {
                "resource_id": "infores:synthetic-aggregator",
                "resource_role": "aggregator_knowledge_source",
            },
        ],
        "knowledge_level": "knowledge_assertion",
        "agent_type": "manual_agent",
        "publications": [
            f"PMID:{rng.randrange(10**8)}" for _ in range(rng.randrange(0, 4))
        ],
    }
    if qualified:
        edge["object_aspect_qualifier"] = rng.choice(_ASPECTS)
        edge["object_direction_qualifier"] = rng.choice(_DIRECTIONS)
    return edge


def write_kgx(out_dir: Path, scale: str, seed: int = 42) -> tuple[Path, Path]:
    """Write ``nodes.jsonl`` and ``edges.jsonl`` for *scale* into *out_dir*.

    Returns:
        ``(nodes_path, edges_path)``.
    """
    spec = SCALES[scale]
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    nodes_path = out_dir / "nodes.jsonl"
    edges_path = out_dir / "edges.jsonl"

    chems = [f"CHEBI:{i + 1}" for i in range(spec.n_chemicals)]
    genes = [f"NCBIGene:{i + 1}" for i in range(spec.n_genes)]
    diseases = [f"MONDO:{i + 1:07d}" for i in range(spec.n_diseases)]
    phenos = [f"HP:{i + 1:07d}" for i in range(spec.n_phenotypes)]

    with open(nodes_path, "w") as fh:
        groups = [
            (chems, ["biolink:SmallMolecule", "biolink:ChemicalEntity"]),
            (genes, ["biolink:Gene", "biolink:GeneOrGeneProduct"]),
            (diseases, ["biolink:Disease", "biolink:DiseaseOrPhenotypicFeature"]),
            (phenos, ["biolink:PhenotypicFeature"]),
        ]
        for ids, cats in groups:
            for i, curie in enumerate(ids):
                record = _node_record(curie, f"{cats[0][8:]} {i}", cats, rng)
                fh.write(json.dumps(record) + "\n")

    def fan_out(fh, sources, targets, pred, mean_degree, qualified=False):
        for subj in sources:
            for _ in range(rng.randrange(1, 2 * mean_degree)):
                obj = targets[_zipf_pick(rng, len(targets))]
                if obj == subj:
                    continue
                edge = _edge_record(
                    subj, pred, obj, rng, qualified=qualified and rng.random() < 0.5
                )
                fh.write(json.dumps(edge) + "\n")

    with open(edges_path, "w") as fh:
        fan_out(
            fh, chems, genes, "biolink:affects", spec.chem_affects_gene, qualified=True
        )
        fan_out(fh, chems, diseases, "biolink:treats", spec.chem_treats_disease)
        fan_out(
            fh,
            genes,
            diseases,
            "biolink:gene_associated_with_condition",
            spec.gene_assoc_disease,
        )
        fan_out(fh, genes, genes, "biolink:interacts_with", spec.gene_interacts_gene)
        fan_out(
            fh, diseases, phenos, "biolink:has_phenotype", spec.disease_has_phenotype
        )
        # Disease hierarchy: every disease after the root points at an earlier
        # one, weighted toward the top, so the root has a large subtree.
        for i in range(1, len(diseases)):
            parent = diseases[_zipf_pick(rng, i, skew=2.0)]
            fh.write(
                json.dumps(
                    _edge_record(diseases[i], "biolink:subclass_of", parent, rng)
                )
                + "\n"
            )

    return nodes_path, edges_path


def build_graph_dir(out_dir: Path, scale: str, seed: int = 42) -> Path:
    """Generate KGX for *scale* and build a gandalf graph under *out_dir*.

    Returns:
        The graph directory (loadable with ``CSRGraph.load_mmap``).
    """
    from gandalf import build_graph_from_jsonl

    kgx_dir = out_dir / "kgx"
    graph_dir = out_dir / "graph"
    nodes_path, edges_path = write_kgx(kgx_dir, scale, seed)
    graph = build_graph_from_jsonl(str(edges_path), str(nodes_path))
    graph.save_mmap(graph_dir)
    return graph_dir


def _qnode(categories=None, ids=None) -> dict:
    node: dict = {}
    if categories:
        node["categories"] = categories
    if ids:
        node["ids"] = ids
    return node


def _qedge(subject: str, obj: str, predicates: list, **extra) -> dict:
    return {"subject": subject, "object": obj, "predicates": predicates, **extra}


def _query(name: str, nodes: dict, edges: dict, **parameters) -> dict:
    return {
        "name": name,
        "message": {"query_graph": {"nodes": nodes, "edges": edges}},
        "parameters": parameters,
    }


def synthetic_queries() -> list[dict]:
    """The benchmark query set for synthetic graphs.

    Each entry is a TRAPI request body plus a ``name`` key, which the
    benchmark runner strips before calling ``lookup``.  Together they cover
    every traversal case in ``query_edge`` (forward, backward, both pinned),
    inverse/symmetric matching, qualifier constraints, subclass expansion,
    multi-hop joins, and the dehydrated response path.
    """
    affects_qualified = {
        "qualifiers": [
            {"biolink:object_aspect_qualifier": "activity_or_abundance"},
        ]
    }
    return [
        # Backward from a pinned hub, with subclass expansion over the whole
        # disease hierarchy.
        _query(
            "1hop_treats_hub_disease",
            {
                "SN": _qnode(["biolink:ChemicalEntity"]),
                "ON": _qnode(["biolink:Disease"], [HUB_DISEASE]),
            },
            {"t": _qedge("SN", "ON", ["biolink:treats"])},
        ),
        # Backward with a qualifier constraint that must be expanded.
        _query(
            "1hop_affects_hub_gene_qualified",
            {
                "SN": _qnode(["biolink:ChemicalEntity"]),
                "ON": _qnode(["biolink:Gene"], [HUB_GENE]),
            },
            {
                "e": _qedge(
                    "SN", "ON", ["biolink:affects"], constraints=affects_qualified
                )
            },
        ),
        # Symmetric predicate from a pinned hub: forward + inverse lookups.
        _query(
            "1hop_interacts_hub_gene_symmetric",
            {
                "a": _qnode(["biolink:Gene"], [HUB_GENE]),
                "b": _qnode(["biolink:Gene"]),
            },
            {"e": _qedge("a", "b", ["biolink:interacts_with"])},
        ),
        # Inverse predicate: stored as chem-treats->disease, asked as
        # disease-treated_by->chem.
        _query(
            "1hop_treated_by_inverse",
            {
                "d": _qnode(["biolink:Disease"], [MID_DISEASE]),
                "c": _qnode(["biolink:ChemicalEntity"]),
            },
            {"e": _qedge("d", "c", ["biolink:treated_by"])},
        ),
        # Two hops into a pinned disease: two traversals and one join.
        _query(
            "2hop_chem_gene_disease",
            {
                "c": _qnode(["biolink:ChemicalEntity"]),
                "g": _qnode(["biolink:Gene"]),
                "d": _qnode(["biolink:Disease"], [HUB_DISEASE]),
            },
            {
                "e0": _qedge("c", "g", ["biolink:affects"]),
                "e1": _qedge("g", "d", ["biolink:gene_associated_with_condition"]),
            },
        ),
        # The same, dehydrated: the lightweight response path.
        _query(
            "2hop_chem_gene_disease_dehydrated",
            {
                "c": _qnode(["biolink:ChemicalEntity"]),
                "g": _qnode(["biolink:Gene"]),
                "d": _qnode(["biolink:Disease"], [HUB_DISEASE]),
            },
            {
                "e0": _qedge("c", "g", ["biolink:affects"]),
                "e1": _qedge("g", "d", ["biolink:gene_associated_with_condition"]),
            },
            dehydrated=True,
        ),
        # Pathfinder shape: both ends pinned, two free intermediates.
        _query(
            "3hop_pathfinder_chem_to_disease",
            {
                "sn": _qnode(ids=[HUB_CHEMICAL]),
                "i0": _qnode(["biolink:Gene"]),
                "i1": _qnode(["biolink:Gene"]),
                "on": _qnode(ids=[HUB_DISEASE]),
            },
            {
                "e0": _qedge("sn", "i0", ["biolink:affects"]),
                "e1": _qedge("i0", "i1", ["biolink:interacts_with"]),
                "e2": _qedge("i1", "on", ["biolink:gene_associated_with_condition"]),
            },
        ),
        # Node-filter plugins on a large neighbourhood.
        _query(
            "1hop_treats_hub_disease_filtered",
            {
                "SN": _qnode(["biolink:ChemicalEntity"]),
                "ON": _qnode(["biolink:Disease"], [HUB_DISEASE]),
            },
            {"t": _qedge("SN", "ON", ["biolink:treats"])},
            filter_config={"max_node_degree": 200, "min_information_content": 50},
        ),
    ]
