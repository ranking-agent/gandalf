# GANDALF

Graph Analysis Navigator for Discovery And Link Finding

## Features
- **Compressed Sparse Row (CSR)** graph representation for memory efficiency
- **Bidirectional search** for optimal performance
- **O(1) property lookups** via hash indexing
- **Predicate filtering** to reduce path explosion
- **Batch property enrichment** for fast results
- **Diagnostic tools** to understand path counts

## Installation

**Recommended: Use a virtual environment**

Some transitive dependencies (e.g., `stringcase`, `pytest-logging`) require modern pip/setuptools to build correctly. Using a virtual environment ensures you have updated tools.

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Upgrade pip and setuptools (important for building dependencies)
pip install --upgrade pip setuptools wheel

# Install the package
pip install -e .
```

**Alternative: Direct install (may fail on some systems)**

If you have a recent pip/setuptools already, you can try:
```bash
pip install -e .
```

## Quick Start

### Unzipping a full translator kgx
- `tar -xvf translator_kg.tar.zst`
This will output a nodes.jsonl and edges.jsonl file

### Build a graph from JSONL
```python
from gandalf import build_graph_from_jsonl

# Build with ontology filtering
graph = build_graph_from_jsonl(
    edges_path="data/raw/edges.jsonl",
    nodes_path="data/raw/nodes.jsonl",
    excluded_predicates={'biolink:subclass_of'}
)

# Save for fast loading
graph.save_mmap("data/processed/graph_filtered")
```

### Query paths
```python
from gandalf import CSRGraph, find_paths

# Load graph (takes ~1-2 seconds)
graph = CSRGraph.load_mmap("data/processed/graph")

# Find paths
paths = find_paths(
    graph,
    start_id="CHEBI:45783",
    end_id="MONDO:0004979"
)

print(f"Found {len(paths)} paths")
```

### Filter by predicates
```python
from gandalf import find_paths_filtered

# Only mechanistic relationships
paths = find_paths_filtered(
    graph,
    start_id="CHEBI:45783",
    end_id="MONDO:0004979",
    allowed_predicates={
        'biolink:treats',
        'biolink:affects',
        'biolink:has_metabolite'
    }
)
```

## Architecture

The package uses a three-stage pipeline:

1. **Topology Search** (fast) - Find all paths using indices only
2. **Filtering** (medium) - Apply business logic on necessary node or edge properties
3. **Enrichment** (batch) - Load all properties for final paths only

This separation allows filtering millions of paths before expensive property lookups.

## Releases
Run this on the mmap folder:
- `tar -czvf gandalf_mmap_<date>.tar.gz gandalf_mmap`