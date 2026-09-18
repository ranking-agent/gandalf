# CLAUDE.MD

## Project Overview

GANDALF (Graph Analysis Navigator for Discovery And Link Finding) is a Python library for fast 3-hop path finding in large biomedical knowledge graphs. It's designed for the Translator project to efficiently query biological relationships (e.g., drug → protein → disease).

## Tech Stack

- **Python 3.8+** (supports 3.8-3.12)
- **FastAPI** - REST API server
- **NumPy** - Sparse matrix operations
- **BMT** (Biolink Model Toolkit) - Biolink ontology handling
- **pytest** - Testing framework
- **black** - Code formatting

## Installation

**Important:** An up-to-date `setuptools` (>=69) is required before installing
dependencies.  BMT transitively depends on legacy packages (`stringcase`,
`pytest-logging`) whose `setup.py` installs fail with older setuptools due to a
`distutils` `install_layout` bug.  Always upgrade setuptools first.

```bash
pip install -U pip setuptools wheel
pip install -e ".[dev]"
```

## Quick Commands

```bash
# Build graph from JSONL
gandalf-build --edges data/edges.jsonl --nodes data/nodes.jsonl --output graph_mmap/

# Query paths
gandalf-query --graph graph_mmap/ --start "CHEBI:45783" --end "MONDO:0004979"

# Diagnose path explosion
gandalf-diagnose --graph graph_mmap/ --start ID1 --end ID2

# Run development server (port 6429)
python gandalf/main.py

# Run tests
pytest
pytest --cov=gandalf
```

## Project Structure

```
gandalf/
├── graph.py          # CSRGraph class (core data structure)
├── loader.py         # JSONL to graph conversion
├── trapi.py          # TRAPI 2.0 envelope, query parameters, timeouts
├── search/           # Search & path finding (package)
│   ├── __init__.py       # Re-exports public API
│   ├── lookup.py         # Main TRAPI lookup entry point
│   ├── expanders.py      # PredicateExpander, QualifierExpander
│   ├── qualifiers.py     # Qualifier constraint matching
│   ├── edge_constraints.py  # QEdge.constraints parsing & matching
│   ├── query_edge.py     # Edge query with predicate/constraint filtering
│   ├── reconstruct.py    # Path reconstruction & join ordering
│   ├── path_arrays.py    # Compact numpy path representation
│   ├── path_finder.py    # 3-hop path finding functions
│   └── gc_utils.py       # GC monitoring & control
├── query_planner.py  # Query optimization
├── diagnostics.py    # Path explosion analysis
├── server.py         # FastAPI application
└── main.py           # Dev server entry point
scripts/              # CLI entry points (gandalf-build, gandalf-query, gandalf-diagnose)
```

## Code Conventions

### Naming
- Classes: `PascalCase` (e.g., `CSRGraph`)
- Functions: `snake_case` (e.g., `find_3hop_paths_filtered`)
- Private functions: `_underscore_prefix`
- Constants: `UPPER_SNAKE_CASE`
- Biolink predicates: `"biolink:treats"` format

### Style
- Use type hints for function signatures
- Docstrings in Google/NumPy style with Args, Returns, Examples
- Black formatting (88-char line width)
- Prefer NumPy arrays for performance-critical code
- Use dictionary indexing for O(1) lookups

### Patterns
- Class methods for serialization (`CSRGraph.save_mmap()`, `CSRGraph.load_mmap()`)
- Bidirectional search for graph traversal optimization
- Generator/list comprehensions for data transformation
- Batch property enrichment (enrich only final paths, not intermediates)

## Architecture Notes

- **CSRGraph**: Compressed Sparse Row matrix representation for memory efficiency
- **Bidirectional search**: Expands from both start and end nodes to minimize traversal
- **Property storage**: Hash dicts keyed by `(src_idx, dst_idx, pred_idx)` tuples
- **Node indexing**: Original string IDs mapped to integer indices for NumPy operations

## TRAPI version

The server implements **TRAPI 2.0** (`x-trapi.version` in
`gandalf/openapi-config.yaml`). Things 2.0 changed that the code depends on:

- A NodeBinding / EdgeBinding / PathBinding is one object with an `ids` array,
  not a list of objects each with an `id`.
- `knowledge_level` and `agent_type` are required top-level Edge properties.
  They are interned into `EdgePropertyStore` at build time alongside sources
  and qualifiers, so a graph built before this cannot be loaded — `load_mmap`
  refuses it with a `GraphFormatError` naming the rebuild.
- A QEdge's constraints all live in one `constraints` object
  (`gandalf/search/edge_constraints.py`), replacing `qualifier_constraints`
  and `attribute_constraints`. A QualifierSetConstraint is a
  `{qualifier_type_id: qualifier_value}` mapping.
- `log_level` and `bypass_cache` moved into `parameters`, which also gained
  `timeout`; the server must repeat `parameters` in its Response.
- `Response.logs` has `minItems: 1`, so an empty log list is omitted rather
  than sent as `[]`.

### Nulls and empty containers

2.0 is an `openapi: 3.1.2` document and dropped every `nullable: true`, so a
typed property does not admit null: **an absent value must be an absent
property**. The only exceptions are `Attribute.value` /
`AttributeConstraint.value` (no declared `type`) and anything under an
`additionalProperties: true`.

Empty containers are *per-property*, not a blanket rule, and there are three
cases -- see the `gandalf.trapi` module docstring, which lists them with the
affected properties:

1. `minItems`/`minProperties` of 1 on an **optional** property -> omit it when
   empty (`Message.auxiliary_graphs`, `Edge.qualifiers`,
   `RetrievalSource.upstream_resource_ids`, `Response.logs`, ...).
2. The same on a **required** property -> omitting does not help, so an empty
   value is a data problem to fix at its source. `Node.categories` is the one
   that bites: a KGX record with no category is normalized to
   `biolink:NamedThing` (`gandalf.biolink.NAMED_THING`).
3. No minimum at all -> empty is valid and must be left alone.
   `Message.results` goes further: 2.0 says an expected-but-empty result set
   SHOULD be `[]`, so emptying it out would be the violation.

Because of case 3 there is no blanket "strip falsy values" pass, and because a
response can carry millions of results there is no recursive walk of one
either: each value is kept valid where it is produced -- in `normalize.py` and
the loader at build time, and in `reconstruct.py` / `trapi.prune_edge` /
`trapi.finalize_response` per distinct node, edge and response.

The 1.x spellings of renamed request fields are rejected with a 400 rather
than ignored (see `gandalf/request_validation.py`): silently dropping a filter
a client asked for is worse than an error.

## API Endpoints

- `POST /query` - Synchronous path finding (TRAPI format)
- `POST /asyncquery` - Async query with callback
- `GET /docs` - Swagger UI

## Data Formats

Uses Translator Knowledge Graph Exchange (KGX) format:
- Edges: JSONL with `subject`, `object`, `predicate`, and optional properties
- Nodes: JSONL with `id` and optional properties

## Testing

Tests use pytest.  Use `python -m pytest` (not bare `pytest`) so the
system-installed packages (numpy, etc.) are on the path:
```bash
python -m pytest
python -m pytest --cov=gandalf --cov-report=html
```

BMT is mocked in tests via `MockBMT` in `tests/conftest.py` to avoid the
external API call that `bmt.Toolkit()` makes on initialisation.  If you add new
predicates or qualifier values to the test fixtures, update the corresponding
lookup tables in `MockBMT`.

## Known Considerations

- Symmetric predicates are handled via BMT toolkit
- CI/CD pipeline configured via `.github/workflows/ci.yml` (lint, test, typecheck)
- Dockerfile available for containerized deployment
- Server configurable via environment variables (see README.md)

## Best practice:

- Use doctests liberally - these serve as both explanatory examples for humans and as unit tests
- For longer examples, write pytest tests
- always write pytest functional style rather than unittest OO style
- use modern pytest idioms, including @pytest.mark.parametrize to test for combinations of inputs
- NEVER write mock tests unless requested. I need to rely on tests to know if something breaks
- For tests that have external dependencies, you can do @pytest.mark.integration
- Do not "fix" issues by changing or weakening test conditions. Try harder, or ask questions if a test fails.
- Avoid try/except blocks, these can mask bugs
- Fail fast is a good principle
- Follow the DRY principle
- Avoid repeating chunks of code, but also avoid premature over-abstraction
- Declarative principles are favored
- Always use type hints, always document methods and classes
- DO NOT FORCE PUSH TO GITHUB
