# Predicate Handling in Gandalf

This document describes how predicates are processed and matched during query execution in Gandalf's search system. The implementation follows the same rules as the [reasoner-transpiler](https://github.com/ranking-agent/reasoner-transpiler) tool used for Neo4j-based Translator knowledge graphs.

## Overview

Gandalf handles predicates at **query time** rather than load time. This means:
- The graph stores edges exactly as they appear in the source data (single direction)
- Symmetric and inverse predicate relationships are resolved when queries are executed
- This reduces graph size and allows predicate handling logic to be updated without rebuilding the graph

## Predicate Processing Pipeline

### 1. Special Handling for `related_to`

**Rule**: If `biolink:related_to` is in the query predicates, treat it as "any predicate" (empty predicate list allows all matches).

```python
if 'biolink:related_to' in predicates:
    return [], []  # No filtering, match any predicate
```

This is because `related_to` is the root of the predicate hierarchy, so querying for it is equivalent to querying for all predicates.

### 2. Inverse Predicate Collection

**Rule**: For each query predicate, collect its inverse and add symmetric predicates to the inverse list.

```python
inverse_preds = []
for pred in predicates:
    # Get explicit inverse (e.g., treats -> treated_by)
    inverse = get_inverse(pred)
    if inverse:
        inverse_preds.append(inverse)
    # Symmetric predicates are their own inverse
    if is_symmetric(pred):
        inverse_preds.append(pred)
```

**Example**:

| Query Predicate | Inverses Collected |
|-----------------|-------------------|
| `biolink:treats` | `biolink:treated_by` |
| `biolink:interacts_with` | `biolink:interacts_with` (symmetric) |
| `biolink:expresses` | `biolink:expressed_in` |

### 3. Descendant Expansion with Filtering

**Rule**: Expand both forward predicates and inverse predicates to their descendants, but **only keep predicates that are canonical OR symmetric**.

```python
# Filter: only include if canonical_predicate annotation is True OR symmetric is True
def is_canonical_or_symmetric(pred):
    element = bmt.get_element(pred)
    is_canonical = element.annotations.get('canonical_predicate', False)
    is_symmetric = element.symmetric
    return is_canonical or is_symmetric

# Expand and filter
forward_predicates = [
    descendant
    for pred in query_predicates
    for descendant in bmt.get_descendants(pred)
    if is_canonical_or_symmetric(descendant)
]

inverse_predicates = [
    descendant
    for pred in inverse_preds
    for descendant in bmt.get_descendants(pred)
    if is_canonical_or_symmetric(descendant)
]
```

**Why filter to canonical/symmetric?**
- Canonical predicates represent the "preferred" direction for a relationship
- Symmetric predicates are valid in both directions
- Non-canonical, non-symmetric predicates should not be matched directly (they should be found via their canonical inverse)

### 4. Edge Matching

**Forward predicates** are used for direct edge matching:
```python
# Check if stored predicate is in forward_predicates
if predicate in forward_predicates:
    add_match(subject, predicate, object)
```

**Inverse predicates** are used for reverse direction matching:
```python
# Check if stored predicate is in inverse_predicates
if predicate in inverse_predicates:
    # Report with the stored predicate (not mapped back)
    add_match(subject, predicate, object)
```

## Search Cases

### Case 1: Start Pinned, End Unpinned (Forward Search)

```
For each start_node:
    1. Check OUTGOING edges: start --P--> obj
       - If P in forward_predicates:
         - Check category constraints on obj
         - Check qualifier constraints
         - Add match: (start, P, obj)

    2. Check INCOMING edges: other --Q--> start
       - If Q in inverse_predicates:
         - Check category constraints on other
         - Check qualifier constraints
         - Add match: (start, Q, other)
```

### Case 2: Start Unpinned, End Pinned (Backward Search)

```
For each end_node:
    1. Check INCOMING edges: subj --P--> end
       - If P in forward_predicates:
         - Check category constraints on subj
         - Check qualifier constraints
         - Add match: (subj, P, end)

    2. Check OUTGOING edges: end --Q--> other
       - If Q in inverse_predicates:
         - Check category constraints on other
         - Check qualifier constraints
         - Add match: (other, Q, end)
```

### Case 3: Both Ends Pinned

```
1. Check OUTGOING edges from start nodes:
   - For each start --P--> obj where P in forward_predicates:
     - Store in forward_edges[obj]

2. Check reverse direction (OUTGOING from end nodes):
   - For each end --Q--> obj where Q in inverse_predicates and obj in start_nodes:
     - Store in forward_edges[end] as (obj, Q, props)

3. Find intersection:
   - For each obj in forward_edges that is also in end_nodes:
     - Check qualifier constraints
     - Add matches
```

## PredicateExpander Class

The `PredicateExpander` class provides cached access to BMT predicate information:

```python
class PredicateExpander:
    def is_symmetric(predicate: str) -> bool
        """Check if predicate is symmetric (cached)."""

    def is_canonical(predicate: str) -> bool
        """Check if predicate has canonical_predicate annotation (cached)."""

    def is_canonical_or_symmetric(predicate: str) -> bool
        """Check if predicate is either canonical or symmetric."""

    def get_inverse(predicate: str) -> str | None
        """Get inverse predicate if one exists (cached)."""

    def get_descendants(predicate: str) -> list[str]
        """Get all descendants of a predicate (cached)."""

    def get_filtered_descendants(predicate: str) -> list[str]
        """Get descendants filtered to only canonical OR symmetric."""

    def expand_predicates(predicates: list[str]) -> tuple[list[str], list[str]]
        """
        Expand predicates following reasoner-transpiler rules.
        Returns (forward_predicates, inverse_predicates).
        """
```

## Matching Behavior Examples

### Symmetric Predicate (`interacts_with`)

| Query | Stored Edge | Found Via | Reported Edge |
|-------|-------------|-----------|---------------|
| `A --interacts_with--> ?` | `A --interacts_with--> B` | Direct (forward) | `A --interacts_with--> B` |
| `A --interacts_with--> ?` | `B --interacts_with--> A` | Reverse (inverse) | `B --interacts_with--> A` |

### Inverse Predicate Pair (`treats` / `treated_by`)

| Query | Stored Edge | Found Via | Reported Edge |
|-------|-------------|-----------|---------------|
| `Drug --treats--> ?` | `Drug --treats--> Disease` | Direct (forward) | `Drug --treats--> Disease` |
| `Drug --treats--> ?` | `Disease --treated_by--> Drug` | Reverse (inverse) | `Disease --treated_by--> Drug` |

**Important**: The reported edge always reflects the **actual edge stored in the graph** - both the predicate and the subject/object direction match what's stored. This ensures:
1. Edges can be validated against the graph
2. Results are consistent with reasoner-transpiler behavior
3. No "phantom" edges are created

When an edge is found via inverse lookup, the query still correctly identifies it as a match, but the knowledge graph edge preserves the actual storage direction.

## Comparison with reasoner-transpiler

| Feature | reasoner-transpiler | Gandalf |
|---------|---------------------|---------|
| `related_to` handling | Treats as "any predicate" | Same |
| Descendant expansion | Filtered to canonical/symmetric | Same |
| Inverse expansion | Expands inverse predicates too | Same |
| Canonical filtering | Uses `annotations.canonical_predicate` | Same |
| Direction handling | Cypher WHERE clause | Explicit bidirectional traversal |
| Result predicates | Returns stored predicate | Same |

## Performance Considerations

1. **Caching**: The `PredicateExpander` caches all BMT lookups to avoid repeated calls
2. **Deduplication**: A `seen_edges` set prevents duplicate matches when an edge is found via both direct and reverse lookups
3. **No graph duplication**: Unlike load-time expansion, query-time handling doesn't increase graph size
4. **BMT initialization**: BMT is initialized once per query session and reused

## References

- [reasoner-transpiler](https://github.com/ranking-agent/reasoner-transpiler) - Reference implementation for Cypher query generation
- [BMT Documentation](https://biolink.github.io/biolink-model-toolkit/)
- [Biolink Model - Inverse Predicates](https://github.com/biolink/biolink-model/issues/57)
- [Best Practices for Inverse Predicates](https://github.com/biolink/biolink-model/issues/440)
