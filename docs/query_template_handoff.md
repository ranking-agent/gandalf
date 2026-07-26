# Query-rule mining: state of the work and how to port it to Shepherd

Handoff for moving the query-template work into Shepherd, where queries are
built and dispatched to Gandalf.

Everything described here lives on `ranking-agent/gandalf`, branch
`claude/metagraph-census-script-grwf83`:

| Commit    | Contents                                                    |
| --------- | ----------------------------------------------------------- |
| `552f4ec` | `scripts/metagraph_census.py` + tests — the census           |
| `b2a1384` | qualifier and knowledge-source census                        |
| `200683c` | `scripts/query_templates.py` + tests — the template portfolio |

---

## 1. Why this exists

The goal is answering "what drugs may treat this disease" by fanning out a set
of templated TRAPI queries to Gandalf in parallel, inside a ~10s budget, with
downstream ranking doing the filtering. That means templates need to be chosen
for recall, but bounded so they don't bury the ranker.

Choosing them by biomedical intuition alone does not work, because intuition
does not know which shapes this particular graph can support. So the first step
was a census of the graph, and the templates were derived from it. Three of the
findings in §3 contradicted what I would otherwise have written.

---

## 2. What was built

### 2.1 `scripts/metagraph_census.py` — the metagraph census

Emits one row per `(subject_category, predicate, object_category)` triple that
occurs, with edge count, distinct subject count and distinct object count, plus
the same counts recomputed at every Biolink ancestor of the predicate — which
is what makes it a *query-granularity* census rather than a leaf census.

```bash
python scripts/metagraph_census.py --graph <mmap dir> --out census/ \
    --match-semantics query
```

Outputs (TSV unless noted):

| File                       | Contents                                              |
| -------------------------- | ----------------------------------------------------- |
| `manifest.json`            | provenance, totals, unmapped terms                     |
| `biolink_closure.json`     | ancestor chains for predicates, categories, qualifiers |
| `census_leaf.tsv`          | one row per occurring triple                           |
| `census_rollup.tsv`        | one row per `(subj_cat, ancestor_pred, obj_cat)`       |
| `census_wide.tsv`          | leaf rows + the ancestor rollup as a JSON column       |
| `predicate_summary.tsv`    | per-predicate totals, own vs. whole subtree            |
| `category_summary.tsv`     | per-category node and edge counts                      |
| `category_rollup.tsv`      | per-ancestor-category totals + member breakdown        |
| `qualifier_signatures.tsv` | per triple, each whole qualifier conjunction           |
| `qualifier_values.tsv`     | per triple, each qualifier value and value ancestor    |
| `qualifier_summary.tsv`    | graph-wide totals per qualifier value                  |
| `source_census.tsv`        | per triple, each primary knowledge source              |

Three properties matter when consuming these:

- **`--match-semantics query` mirrors `PredicateExpander` exactly**: descendants
  filtered to canonical-or-symmetric, inverses matched in reverse with subject
  and object swapped, `biolink:related_to` matching everything both ways. The
  census that produced the numbers below was run this way, so its rows are
  directly what a qedge would match. (Symmetric edges are counted once per
  matching orientation, which is why `related_to` totals 200% of the graph.)
- **Qualifier signatures are whole conjunctions**, because a TRAPI
  `qualifier_set` ANDs its members. "40% carry a direction, 60% carry an aspect"
  says nothing about how many carry both.
- **Qualifier values roll up their enum hierarchy** (`expression` → `abundance`
  → `activity_or_abundance`), mirroring `QualifierExpander`.

Cost: ~4 minutes and a few GB on the 28.7M-edge graph.

### 2.2 `scripts/query_templates.py` — the template portfolio and cost model

Renders a portfolio of TRAPI query graphs for a pinned disease and prices each
against the census.

```bash
python scripts/query_templates.py --census census/                  # price only
python scripts/query_templates.py --census census/ \
    --disease MONDO:0004979 --out queries/ [--exclude-leaky] [--budget N]
```

Writes one request per template plus an `index.json` carrying each template's
tier, mechanism claim, `filter_config`, and cost estimate.

### 2.3 Tests

`tests/test_metagraph_census.py` (42) and `tests/test_query_templates.py` (24).
Counting logic is checked against hand-built fixtures where every number is
verifiable by eye. One integration test (`@pytest.mark.integration`) sweeps the
whole Biolink model through the closure lookups.

---

## 3. What the census says about this graph

Census run: 1,670,341 nodes, 28,709,074 edges, 60 predicates, 40 categories,
3,774 occurring triples, Biolink 4.3.2, query semantics.

**Gene–disease biology lives on `biolink:Protein`, not `biolink:Gene`.**

| Triple                                | Edges  | Distinct subjects |
| ------------------------------------- | ------ | ----------------- |
| `Disease -associated_with-> Protein`  | 99,582 | 8,832 diseases    |
| `Disease -associated_with-> Gene`     | 1,329  | 436 diseases      |

A `Gene`-pinned mechanism template returns almost nothing. Every mechanism
template in the portfolio pins `Protein`.

**The disease side carries no direction qualifiers.** The only qualifier
appearing with a `Disease` subject is HPO `frequency_qualifier` on
`has_phenotype`. The textbook reversal template — drug *decreases* what disease
*increases* — is **not expressible on this graph**. The directional signal
exists as predicates instead:

| Triple                             | Edges  | Distinct diseases |
| ---------------------------------- | ------ | ----------------- |
| `Protein -contributes_to-> Disease` | 15,393 | 2,436             |
| `Protein -causes-> Disease`         | 6,053  | 1,239             |

`causal_gene_inhibition` is the closest expressible substitute: causality from
the predicate, drug direction from the qualifier.

**Qualifiers are cheaper *and* sharper on the drug side.** This is the opposite
of the usual precision/cost tradeoff:

| Drug-side hop                                              | Edges     | Chemicals per protein |
| ---------------------------------------------------------- | --------- | --------------------- |
| `Protein <-physically_interacts_with- SmallMolecule`        | 1,184,288 | ~186                  |
| `SmallMolecule -affects[direction=decreased]-> Protein`     | 699,089   | ~21                   |
| `Drug -affects[direction=decreased]-> Protein`              | 388,209   | ~14                   |
| `SmallMolecule -affects[direction=increased]-> Protein`     | 573,235   | ~18                   |
| `SmallMolecule -affects[causal_mechanism=inhibition]-> Protein` | 176,212 | ~36                |

**Other numbers worth carrying over:**

- `subclass_of` is 10.1M edges, 35% of the graph — ontology plumbing, excluded
  from every template.
- 13 qualifier types present; `object_direction_qualifier` covers 2.93M edges,
  `object_aspect_qualifier` 6.39M, `species_context_qualifier` 2.17M
  (human/mouse/rat), `causal_mechanism_qualifier` 643k.
- Direct `treats` edges exist: ~27k across chemical categories, and ~976k in the
  wider `treats_or_applied_or_studied_to_treat` family (mostly
  `studied_to_treat`, i.e. clinical trials). This is the leakage risk in §6.
- Category sizes: SmallMolecule 1,050,118; Protein 300,141; Disease 51,707;
  ChemicalEntity 51,045; Gene 43,349; Drug 5,326.
- Several high-fan-out disease entry hops have almost no coverage:
  `Disease -genetic_association-> BiologicalProcess` averages 1,116 objects per
  disease but exists for only 83 diseases. Coverage matters as much as fan-out.

---

## 4. The portfolio

Expected paths are per disease, from the cost model in §5.

| Template                  | Tier         | Est. paths | Coverage | Shape                                                       |
| ------------------------- | ------------ | ---------- | -------- | ----------------------------------------------------------- |
| `target_inhibition_sm`    | A-mechanism  | 238        | 8,832    | Disease→Protein, SmallMolecule−affects[decreased]→Protein    |
| `target_inhibition_drug`  | A-mechanism  | 160        | 8,832    | same, Drug                                                   |
| `target_activation_sm`    | A-mechanism  | 201        | 8,832    | same, direction=increased                                    |
| `causal_gene_inhibition`  | A-mechanism  | 134        | 2,436    | Protein−causes/contributes_to→Disease + decreased            |
| `inhibition_mechanism_sm` | A-mechanism  | 403        | 8,832    | same, causal_mechanism=inhibition                            |
| `target_binding_sm`       | B-broad      | 2,096      | 8,832    | Disease→Protein→SmallMolecule (physical binding)             |
| `pathway_participation`   | B-broad      | 2,192      | 8,832    | Disease→Protein→Pathway→SmallMolecule                        |
| `ppi_neighborhood`        | B-broad      | 10,986     | 8,832    | Disease→Protein→Protein, decreased by chemical               |
| `phenotype_drug_bridge`   | C-assoc      | 477        | 10,070   | Disease→Phenotype→Drug                                       |
| `direct_association`      | C-assoc      | 16         | 1,906    | Disease→Drug, one hop                                        |
| `indication_transfer`     | **D-leaky**  | 8,221      | 10,070   | Disease→Phenotype←Disease'←Drug (treats)                     |
| `two_witness_inhibition`  | **D-branch** | 2,689      | 8,832    | two proteins, same chemical decreases both                   |

Full portfolio ≈ 27.8k expected paths; Tier A alone ≈ 1.1k. `ppi_neighborhood`
and `indication_transfer` are ~70% of the total.

All twelve were executed against a purpose-built fixture graph through
`gandalf.search.lookup`, so the shapes are known to be accepted — including the
branching one.

---

## 5. The cost model

For a triple, the census gives mean fan-out directly:

```
f_forward = edge_count / distinct_subjects      # objects per subject
f_reverse = edge_count / distinct_objects       # subjects per object
```

`estimate()` walks the query graph from the pinned node, multiplying the fan-out
of each expanding hop; an edge that closes a cycle constrains rather than
expands and is not multiplied. Qualifier constraints change which census row is
read:

- **one qualifier** → `qualifier_values.tsv`, which already unions every
  signature containing that value and rolls up the value hierarchy;
- **a conjunction** → sum over `qualifier_signatures.tsv` rows whose pairs are a
  **superset** of the constraint (TRAPI matches a qualifier_set as a subset).
  This matters: aspect and direction essentially never occur alone in this graph
  — they arrive bundled with `qualified_predicate` and species context — so
  exact signature matching finds nothing. Edge counts sum exactly; distinct
  endpoints cannot, so the largest contributing signature is used, biasing
  fan-out estimates high.

**Limits, stated plainly.** These are means over a heavy-tailed distribution and
assume independence across hops. A disease with 200 associated proteins will
blow past its estimate. Each template therefore carries a `max_node_degree` in
`filter_config`, which Shepherd must pass through to Gandalf's `filter_config`
argument — it is not automatic.

Worth knowing: Gandalf's own planner (`gandalf/query_planner.py`) still uses
hardcoded `N = 1_000_000` nodes and `R = 25` edges per node for join ordering.
The census is exactly the statistics table that stub is missing, so feeding real
per-triple fan-out into the planner is an available follow-up on the Gandalf
side.

---

## 6. Porting to Shepherd

**`scripts/query_templates.py` ports as-is.** It imports only `argparse`, `csv`,
`json`, `logging`, `sys`, `dataclasses`, `pathlib`, `typing` — no Gandalf
dependency. Copy the module and its tests; it needs only a census directory on
disk.

**`scripts/metagraph_census.py` should stay in Gandalf.** It has three Gandalf
touchpoints, all lazy imports: `gandalf.node_store.NodeStore` (only for the
`--graph` mmap path), and `gandalf.biolink.make_toolkit` + `gandalf.config`
(only when `--biolink-version` / `--biolink-schema` are not given). It *can* run
in Shepherd against KGX jsonl with an explicit `--biolink-version`, but the
natural arrangement is: Gandalf produces the census when the graph is built,
Shepherd consumes the TSVs.

That implies a **contract between the repos**: the census output directory. Ship
it as a build artifact alongside the mmap graph, version it with the graph, and
have Shepherd read the same directory. `manifest.json` carries the graph
provenance, Biolink version, and match semantics, so Shepherd can refuse to
start if the census does not match the graph it is querying.

### Suggested shape in Shepherd

1. **Template registry** — the `TEMPLATES` tuple, ideally loaded from or
   validated against config so templates can be added without a deploy.
2. **Portfolio selection per question** — given a disease and a result budget,
   pick the subset to fire. Today `--budget` only warns; in Shepherd this should
   actually choose. Tier A first, add Tier B until the budget is spent.
3. **Per-disease adaptation** — the estimates are global means, but disease
   degree varies by orders of magnitude. A ~200ms one-hop probe of the pinned
   disease's neighbourhood (Gandalf has `do_one_hop`) would let Shepherd pick
   broad templates for sparse diseases and restrictive ones for dense diseases.
   This is the piece most likely to make the 10s budget hold in the tail, and it
   is not built yet.
4. **`filter_config` passthrough** — per-template, not global.
5. **Result post-processing** — see the `two_witness_inhibition` defect below.

---

## 7. Known defects and caveats

- **`two_witness_inhibition` returns degenerate results.** TRAPI cannot express
  `n_protein_a != n_protein_b`, so results include pairs where both witnesses
  are the same protein, and each genuine pair comes back twice as `(a,b)` and
  `(b,a)`. Shepherd must drop the degenerate ones and de-duplicate unordered
  pairs. Confirmed by running it: 4 results where 1 was meaningful.
- **`indication_transfer` reads `treats` edges.** It will flatter itself against
  any ground truth drawn from the same indication data. Keep it in its own
  evaluation bucket; `--exclude-leaky` drops it.
- **Adding a qualifier constraint excludes edges that lack the qualifier.**
  Constraining `species_context_qualifier=NCBITaxon:9606` does not mean "human
  only", it means "edges explicitly labelled human" — everything unlabelled is
  discarded too. Same trap for any quality constraint.
- **Estimates are means, not bounds.** See §5.
- **Non-canonical predicates.** Gandalf only expands a qedge into
  canonical-or-symmetric descendants, so edges on non-canonical predicates are
  reachable only by naming them (or their inverse) directly. The census reports
  how many edges this affects.
- **BMT defect worked around.** Two Biolink 4.3.2 slots are named with
  underscores (`gene_fusion_with`, `genetic_neighborhood_of`); BMT
  de-underscores CURIEs when resolving, so `SchemaView.inverse` raises
  `AttributeError` on them. `_model_inverse` guards it. If Shepherd does its own
  BMT inverse lookups, it needs the same guard.

---

## 8. Evaluation protocol

The ground truth is deliberately held out, so template selection so far has used
**only** census statistics and mechanism argument — no fitting. To keep that
property:

1. Freeze the template set before looking at ground truth.
2. Evaluate Tier A/B/C together; evaluate Tier D (`indication_transfer`)
   separately, and expect it to look artificially strong.
3. Use `direct_association` (16 paths, one hop, no mechanism) as the baseline. If
   the Tier A templates do not beat it, that is a finding about the ranker, not
   about the templates.
4. Report coverage alongside accuracy: `causal_gene_inhibition` can only fire for
   2,436 diseases, so its hit rate is not comparable to
   `target_inhibition_sm`'s 8,832 without conditioning on that.

A dev signal that does not touch the held-out set: mask the graph's own `treats`
edges and check whether templates recover them — but only if you are confident
the graph's `treats` edges and your ground truth do not share a source.

---

## 9. Open threads

- **Per-disease probe and adaptive portfolio selection** (§6.3) — the biggest
  remaining gap.
- **Metapath mining.** ~7,400 2-hop and ~991,000 3-hop Disease→Chemical
  metapaths occur in this graph. The current portfolio is hand-curated from the
  mechanistically defensible ones. A degree-preserving null model would let you
  rank the rest by lift without touching ground truth. Prior art:
  Himmelstein's Hetionet/Rephetio work, which used degree-weighted path counts
  over metapaths for the same problem.
- **Generalizing beyond `treats`.** The portfolio is drug→disease. The grammar
  question — node classes × edge classes at census-clean granularity — is
  unstarted, and it is what makes gene→disease and chemical→phenotype cheap to
  add later.
- **Knowledge level / publication counts** are not in the census: they live in
  Gandalf's cold-path LMDB and would need a full scan. Only worth adding if
  quality constraints on those fields turn out to matter.
- **Feeding census statistics into Gandalf's query planner** (§5).
