# Mithrandir

A prototype GUI for walking the NIH Translator knowledge graph one triple at a
time, against a **local GANDALF graph** — the backend calls the gandalf library
directly, in-process, so no TRAPI server is needed. (Node-name search still uses
the SRI Name Resolver and the info cards still pull from Wikipedia.)

Search for a starting node → see its neighbours grouped by relationship
(predicate + direction) as connectors fanning out to the right → open a
relationship to list the connected nodes, **grouped by node type** and ranked by
degree → click a node to step into it. The chosen node becomes the root of the
next triple; the path you build shows in the filmstrip up top with a frame
around the triple you're viewing. **‹ Back / Forward ›**, the arrow keys, or a
click on any node in the filmstrip swipes you along the path.

## Run it

The backend imports `gandalf`, so install it and point it at an mmap graph.

```bash
# 1. install gandalf (from the repo root)
pip install -e .

# 2. build (or obtain) a graph — see the root README "Build a graph from JSONL"
#    or the gandalf-build CLI; it produces an mmap directory.

# 3. run mithrandir against that graph
cd mithrandir
GANDALF_GRAPH_PATH=/path/to/graph_mmap python3 server.py   # open http://localhost:8000
```

`GANDALF_GRAPH_PATH` defaults to `/data/graph` (the same default the gandalf
server uses), so a graph built for the server works here unchanged.

### No graph handy? Try mock mode first

To confirm the app itself works without a graph (or any network), run against
built-in synthetic data:

```bash
GANDALF_MOCK=1 python3 server.py
```

Search "metformin" / "diabetes" / "TP53", pick a result, and explore — no graph
is loaded and no network calls are made.

## Configuration (environment variables)

| Variable | Default | Purpose |
| --- | --- | --- |
| `GANDALF_PORT` | `8000` | local port |
| `GANDALF_GRAPH_PATH` | `/data/graph` | local gandalf mmap graph directory |
| `GANDALF_NAME_RESOLVER` | `https://name-resolution-sri.renci.org` | name→CURIE search |
| `GANDALF_SUBCLASS` | `false` | biolink subclass inference on lookups |
| `GANDALF_IPV4` | `true` | force IPv4 for name resolver / Wikipedia (IPv6 handshake hangs) |
| `GANDALF_TIMEOUT` | `60` | per-request timeout for name resolver / Wikipedia (seconds) |
| `GANDALF_DEGREE_WORKERS` | `8` | concurrent node-degree lookups |
| `GANDALF_MOCK` | `false` | serve synthetic data, no graph or network |

## How it works

- **Start search** → `/api/search` proxies the SRI Name Resolver autocomplete;
  pick a result, or paste a CURIE and press Enter.
- **Neighbours** → for node `X` the backend calls gandalf's `lookup()`
  in-process with two predicate-agnostic one-hop query graphs (`X --?--> n1` and
  `n1 --?--> X`, since edges are directed), reads `message.knowledge_graph`, and
  groups edges by `(predicate, direction)`. Predicate groups are sorted by
  neighbour count.
- **Connected nodes** → partitioned by primary node type into expandable groups
  (sorted by group size); inside each, nodes are ranked by node degree (read
  straight from the graph's CSR offsets, cached). The largest group (or the
  group holding your previously chosen node) opens by default.
- **Chosen node** → clicking a node records it on the source step; that node
  stays highlighted and the relationship connector links to it whenever you
  swipe back to that triple.

## Known limitations (it's a prototype)

- **Degree ranking is sampled.** A relationship can have thousands of
  neighbours; the client ranks the first `DEGREE_CAP` (300) per relationship and
  labels the list accordingly. (Degrees are now cheap in-process lookups, so this
  cap is mostly about keeping the UI list manageable.)
- **Primary type only.** Nodes are grouped by their first biolink category.
- `subclass` inference is off by default so each step shows that node's literal
  edges; set `GANDALF_IPV4`/`GANDALF_SUBCLASS` as needed.

## Files

- `server.py` — static server + in-process gandalf graph backend (with mock mode)
- `index.html`, `style.css`, `app.js` — the frontend
- `test_harness.mjs` — optional headless logic test (`npm install jsdom` first, then `node test_harness.mjs`)
