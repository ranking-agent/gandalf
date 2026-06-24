# Mithrandir

A prototype GUI for walking the NIH Translator knowledge graph one triple at a
time, via the GANDALF TRAPI endpoint at RENCI (`https://gandalf.renci.org`).

Search for a starting node → see its neighbours grouped by relationship
(predicate + direction) as connectors fanning out to the right → open a
relationship to list the connected nodes, **grouped by node type** and ranked by
degree → click a node to step into it. The chosen node becomes the root of the
next triple; the path you build shows in the filmstrip up top with a frame
around the triple you're viewing. **‹ Back / Forward ›**, the arrow keys, or a
click on any node in the filmstrip swipes you along the path.

## Run it

No dependencies — just Python 3.8+.

```bash
cd gandalf-explorer
python3 server.py            # then open http://localhost:8000
```

### Can't reach RENCI? Try mock mode first

If the live endpoint is unreachable from your network, run the UI against
built-in synthetic data so you can confirm the app itself works:

```bash
GANDALF_MOCK=1 python3 server.py
```

Search "metformin" / "diabetes" / "TP53", pick a result, and explore — no
network calls are made.

## Configuration (environment variables)

| Variable | Default | Purpose |
| --- | --- | --- |
| `GANDALF_PORT` | `8000` | local port |
| `GANDALF_BASE` | `https://gandalf.renci.org` | TRAPI server |
| `GANDALF_NAME_RESOLVER` | `https://name-resolution-sri.renci.org` | name→CURIE search |
| `GANDALF_SUBCLASS` | `false` | biolink subclass inference on lookups |
| `GANDALF_IPV4` | `true` | force IPv4 (works around IPv6 handshake hangs) |
| `GANDALF_TIMEOUT` | `60` | per-request timeout (seconds) |
| `GANDALF_DEGREE_WORKERS` | `8` | concurrent `/node_degree` lookups |
| `GANDALF_MOCK` | `false` | serve synthetic data, no network calls |

## How it works

- **Start search** → `/api/search` proxies the SRI Name Resolver autocomplete;
  pick a result, or paste a CURIE and press Enter.
- **Neighbours** → for node `X` the backend runs two predicate-agnostic one-hop
  TRAPI queries (`X --?--> n1` and `n1 --?--> X`, since edges are directed),
  reads `message.knowledge_graph`, and groups edges by `(predicate, direction)`.
  Predicate groups are sorted by neighbour count.
- **Connected nodes** → partitioned by primary node type into expandable groups
  (sorted by group size); inside each, nodes are ranked by node degree
  (`/node_degree`, cached + parallel). The largest group (or the group holding
  your previously chosen node) opens by default.
- **Chosen node** → clicking a node records it on the source step; that node
  stays highlighted and the relationship connector links to it whenever you
  swipe back to that triple.

## Known limitations (it's a prototype)

- **Degree ranking is sampled.** A relationship can have thousands of
  neighbours; fetching a degree is one request per node. The client ranks the
  first `DEGREE_CAP` (300) per relationship and labels the list accordingly.
- **Primary type only.** Nodes are grouped by their first biolink category.
- `subclass` inference is off by default so each step shows that node's literal
  edges; set `GANDALF_IPV4`/`GANDALF_SUBCLASS` as needed.

## Files

- `server.py` — dependency-free proxy + static server (with mock mode)
- `index.html`, `style.css`, `app.js` — the frontend
- `test_harness.mjs` — optional headless logic test (`npm install jsdom` first, then `node test_harness.mjs`)
