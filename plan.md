# Plan: TRAPI Attribute Constraints in Gandalf

## Background

TRAPI defines `AttributeConstraint` objects that can be attached to query nodes (`constraints` field) and query edges (`attribute_constraints` field). These allow callers to filter results based on attribute values (e.g. "only return edges where molecular_mass > 57.0 kDa").

### AttributeConstraint Schema (from TRAPI spec)
```
{
  "id": "CURIE",              # e.g. "EDAM:data_0844" or "biolink:p_value"
  "name": "string",           # human-readable label
  "not": false,               # negate the constraint
  "operator": "==" | ">" | "<" | "matches" | "===",
  "value": any,               # constraint value (may be a list)
  "unit_id": "CURIE | null",  # measurement units
  "unit_name": "string | null"
}
Required: id, name, operator, value
```

### Semantics
- **QNode**: field is called `constraints` — multiple constraints use AND logic
- **QEdge**: field is called `attribute_constraints` — multiple constraints use AND logic
- Constraints match against the `attributes` list on KG nodes/edges by comparing `attribute_type_id` to constraint `id`
- Operators: `==` (equality), `>` / `<` (numeric comparison), `matches` (regex), `===` (strict equality including type/order)
- `not: true` inverts the result

### Current State in Gandalf
- **No attribute constraint support exists** — these fields are silently ignored
- Node attributes are stored in-memory in `node_properties[idx]["attributes"]`
- Edge attributes are stored in LMDB (cold path) — only accessed during response enrichment
- The only attribute-based filtering today is `information_content` on nodes (hardcoded in `_passes_node_filters`)

## Implementation Plan

### Step 1: Create attribute constraint matching module
**New file:** `gandalf/search/attribute_constraints.py`

Create a function `matches_attribute_constraints(attributes, constraints) -> bool` that:
- Takes a list of TRAPI attribute dicts and a list of `AttributeConstraint` dicts
- Returns True if ALL constraints are satisfied (AND logic)
- For each constraint:
  - Find attribute(s) in the list where `attribute_type_id == constraint["id"]`
  - If no matching attribute found: constraint fails (unless `not: true`)
  - Apply the operator to compare `attribute["value"]` vs `constraint["value"]`
  - If `not: true`, invert the result
- Operator implementations:
  - `==`: standard equality (`==`)
  - `>`: numeric greater-than
  - `<`: numeric less-than
  - `matches`: regex match (`re.search`) on string values
  - `===`: strict equality (type + value, for lists also order)

### Step 2: Add node attribute constraint filtering to query_edge.py
**File:** `gandalf/search/query_edge.py`

- Add a `node_constraints` parameter (dict mapping qnode_id -> constraints list) to `query_edge()` and the three internal functions (`_query_forward`, `_query_backward`, `_query_both_pinned`)
- After the existing `_passes_node_filters()` check, add a check for node attribute constraints:
  ```python
  if node_constraints:
      node_attrs = graph.get_node_property(node_idx, "attributes", [])
      if not matches_attribute_constraints(node_attrs, node_constraints):
          continue
  ```
- Node attributes are already in-memory (`node_properties`), so this is efficient

### Step 3: Add edge attribute constraint filtering to query_edge.py
**File:** `gandalf/search/query_edge.py`

This is the trickier part because edge attributes live in LMDB (cold path), not in memory.

**Option A (Recommended): Filter during traversal using LMDB lookups**
- Add an `attribute_constraints` parameter to `query_edge()` and internal functions
- After qualifier constraint checks pass, if `attribute_constraints` is non-empty:
  - Fetch edge attributes from LMDB: `graph.lmdb_store.get(fwd_edge_idx)`
  - Apply `matches_attribute_constraints(attrs, attribute_constraints)`
- This adds per-edge LMDB reads during traversal, but only for edges that already passed all other filters (predicates, categories, qualifiers), so the number should be manageable
- The LMDB is memory-mapped, so hot pages will be cached by the OS

**Option B (Alternative): Post-filter after path reconstruction**
- Apply edge attribute constraints after `_build_response()` by filtering KG edges
- Pro: No changes to traversal code
- Con: Wasteful — builds full paths then discards; also harder to integrate cleanly

Going with **Option A** since it filters early and avoids wasted work.

### Step 4: Wire constraints through lookup.py
**File:** `gandalf/search/lookup.py`

In the edge processing loop (around line 116-186):
- Extract `constraints` from each QNode in the query graph
- Extract `attribute_constraints` from each QEdge in the query graph
- Pass node constraints for the start/end nodes to `query_edge()` as new parameters
- Pass edge attribute constraints to `query_edge()` as a new parameter

Specifically:
```python
# Get node attribute constraints
start_node_constraints = start_node.get("constraints", [])
end_node_constraints = end_node.get("constraints", [])

# Get edge attribute constraints
edge_attribute_constraints = next_edge.get("attribute_constraints", [])
```

### Step 5: Add tests
**File:** `tests/test_attribute_constraints.py`

Unit tests for the matching function:
- Test each operator (`==`, `>`, `<`, `matches`, `===`)
- Test `not: true` negation
- Test AND logic across multiple constraints
- Test missing attributes (constraint should fail)
- Test with list values for `>` / `<` (OR logic per spec)
- Test edge cases: type mismatches, empty constraints, None values

Integration tests:
- Test node constraints filtering during query
- Test edge attribute constraints filtering during query

## Key Design Decisions

1. **Node constraints are cheap** — attributes are in-memory, so filtering during traversal adds negligible cost
2. **Edge constraints require LMDB reads** — but only for edges that pass all other filters, and LMDB is memory-mapped
3. **AND semantics** for multiple constraints on same node/edge (per TRAPI spec)
4. **Filter early** — during traversal, not post-hoc, to avoid building paths that will be discarded
5. **Reuse existing `information_content` pattern** — the existing `_passes_node_filters` function already does attribute-based filtering; this generalizes that pattern

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `gandalf/search/attribute_constraints.py` | Create | Core matching logic |
| `gandalf/search/query_edge.py` | Modify | Add constraint params and filtering calls |
| `gandalf/search/lookup.py` | Modify | Extract constraints from query graph, pass to query_edge |
| `tests/test_attribute_constraints.py` | Create | Unit + integration tests |
