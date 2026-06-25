#!/usr/bin/env python3
"""
GANDALF Explorer — local backend.

A small server that sits between the browser frontend and a **local GANDALF
graph**. The knowledge-graph operations call the gandalf library directly,
in-process (no TRAPI HTTP server in the loop); only the node-name search (SRI
Name Resolver) and the Wikipedia info cards still go out over the network.

Why a backend at all (instead of doing everything in the browser)?
  - Loads the gandalf graph once and runs `lookup()` against it in-process.
  - Lets us fan out node-degree lookups concurrently and cache them.
  - Keeps the TRAPI request shaping in one place.
  - Avoids CORS headaches when hitting the name resolver / Wikipedia.

Run:
    GANDALF_GRAPH_PATH=/path/to/graph_mmap python3 server.py
then open http://localhost:8000

Configuration via environment variables (all optional):
    GANDALF_GRAPH_PATH    path to the mmap graph directory (default /data/graph)
    GANDALF_PORT          local port to serve on          (default 8000)
    GANDALF_SUBCLASS      "true"/"false" subclass infer    (default false — literal neighbours)
    GANDALF_TIMEOUT       per-request timeout in seconds   (default 60, name resolver / wiki)
    GANDALF_DEGREE_WORKERS  concurrent degree lookups      (default 8)
    GANDALF_MOCK          serve synthetic data, no graph or network (default false)
"""

import json
import os
import socket
import sys
import urllib.parse
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
# Path to the local gandalf mmap graph directory. Same env var the gandalf
# server uses, so a graph built for the server works here unchanged.
GRAPH_PATH = os.environ.get("GANDALF_GRAPH_PATH", "/data/graph")
NAME_RESOLVER = os.environ.get(
    "GANDALF_NAME_RESOLVER", "https://name-resolution-sri.renci.org"
).rstrip("/")
PORT = int(os.environ.get("GANDALF_PORT", "8000"))
SUBCLASS = os.environ.get("GANDALF_SUBCLASS", "false").lower() in ("1", "true", "yes")
TIMEOUT = float(os.environ.get("GANDALF_TIMEOUT", "60"))
DEGREE_WORKERS = int(os.environ.get("GANDALF_DEGREE_WORKERS", "8"))
# Force IPv4: urllib doesn't do "Happy Eyeballs", so a broken IPv6 route makes
# the TLS handshake hang until timeout even though the browser (which races v4
# and v6) connects fine. Forcing IPv4 is the usual fix; set GANDALF_IPV4=false
# if you're on an IPv6-only network. Only affects the name resolver / Wikipedia
# (the graph itself is now local).
FORCE_IPV4 = os.environ.get("GANDALF_IPV4", "true").lower() in ("1", "true", "yes")
# Offline demo mode: serve synthetic data instead of loading a graph or calling
# out. Handy for trying the UI with no graph. Run: GANDALF_MOCK=1 python3 server.py
MOCK = os.environ.get("GANDALF_MOCK", "false").lower() in ("1", "true", "yes")
WIKI_UA = os.environ.get(
    "GANDALF_WIKI_UA", "Mithrandir-Graph-Explorer/1.0 (local research prototype)"
)

# Local gandalf graph + Biolink toolkit, loaded once at startup (skipped in mock
# mode). The graph and BMT are reused across every request; `lookup()` would
# otherwise rebuild the toolkit on each call.
GRAPH = None
BMT = None
if not MOCK:
    from gandalf import CSRGraph, lookup
    from gandalf.biolink import make_toolkit

    print(f"Loading gandalf graph from {GRAPH_PATH} ...", file=sys.stderr)
    GRAPH = CSRGraph.load_mmap(GRAPH_PATH)
    BMT = make_toolkit()
    print("Graph and Biolink toolkit loaded.", file=sys.stderr)

if FORCE_IPV4:
    _orig_getaddrinfo = socket.getaddrinfo

    def _ipv4_getaddrinfo(host, port, family=0, *args, **kwargs):
        results = _orig_getaddrinfo(host, port, family, *args, **kwargs)
        v4 = [r for r in results if r[0] == socket.AF_INET]
        return v4 or results

    socket.getaddrinfo = _ipv4_getaddrinfo

STATIC = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "application/javascript; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
}

# ---------------------------------------------------------------------------
# Simple in-memory caches (process lifetime)
# ---------------------------------------------------------------------------
_degree_cache = {}
_degree_lock = Lock()
_expand_cache = {}
_expand_lock = Lock()
_wiki_cache = {}
_wiki_lock = Lock()


# ---------------------------------------------------------------------------
# Mock data (offline demo mode)
# ---------------------------------------------------------------------------
import hashlib

_MOCK_CATS = {
    "Gene": "biolink:Gene",
    "Protein": "biolink:Protein",
    "Disease": "biolink:Disease",
    "Chemical": "biolink:ChemicalEntity",
    "Pathway": "biolink:Pathway",
    "Phenotype": "biolink:PhenotypicFeature",
}


def _mock_degree(curie):
    h = int(hashlib.sha1(curie.encode()).hexdigest()[:6], 16)
    return 1 + (h % 4000)


def _mock_neighbors(prefix, n, cat_mix):
    out = []
    for k in range(n):
        cat_name = cat_mix[k % len(cat_mix)]
        nid = f"MOCK:{prefix}-{k:04d}"
        out.append({
            "id": nid,
            "name": f"{cat_name} {prefix} {k:03d}",
            "categories": [_MOCK_CATS[cat_name]],
        })
    return out


def mock_expand(curie):
    groups = [
        {"key": "out|biolink:affects", "predicate": "biolink:affects", "direction": "out",
         "neighbors": _mock_neighbors("aff", 64, ["Gene", "Protein"])},
        {"key": "out|biolink:related_to", "predicate": "biolink:related_to", "direction": "out",
         "neighbors": _mock_neighbors("rel", 320, ["Chemical", "Disease", "Gene"])},
        {"key": "out|biolink:treats", "predicate": "biolink:treats", "direction": "out",
         "neighbors": _mock_neighbors("trt", 11, ["Disease"])},
        {"key": "in|biolink:has_participant", "predicate": "biolink:has_participant", "direction": "in",
         "neighbors": _mock_neighbors("par", 18, ["Pathway", "Protein"])},
        {"key": "in|biolink:has_phenotype", "predicate": "biolink:has_phenotype", "direction": "in",
         "neighbors": _mock_neighbors("phe", 7, ["Phenotype"])},
    ]
    for g in groups:
        g["count"] = len(g["neighbors"])
    groups.sort(key=lambda g: g["count"], reverse=True)
    return {"curie": curie, "name": f"Mock node {curie}",
            "categories": ["biolink:ChemicalEntity"], "groups": groups, "errors": []}


def mock_search(q):
    base = [
        ("CHEBI:6801", "metformin", ["biolink:ChemicalEntity"]),
        ("MONDO:0005148", "type 2 diabetes mellitus", ["biolink:Disease"]),
        ("NCBIGene:7157", "TP53", ["biolink:Gene"]),
        ("HP:0001250", "seizure", ["biolink:PhenotypicFeature"]),
    ]
    ql = q.lower()
    hits = [b for b in base if ql in b[1].lower()] or base
    return [{"curie": c, "label": l, "types": t} for (c, l, t) in hits[:10]]


# ---------------------------------------------------------------------------
# Outbound helpers
# ---------------------------------------------------------------------------
def _post_trapi(query):
    """Run a TRAPI query against the local gandalf graph and return the result.

    Calls gandalf's `lookup()` in-process — the same entry point the gandalf
    server's `/query` endpoint uses — so the returned dict has the same shape
    (`message.knowledge_graph`, etc.).
    """
    return lookup(GRAPH, query, bmt=BMT, subclass=SUBCLASS)


def _get_degree(curie):
    """Total degree (in + out) of a node; cached. Returns int or None if unknown.

    Computed directly from the graph's CSR offset arrays — the same arithmetic
    the gandalf server's `/node_degree/{curie}` endpoint performs.
    """
    with _degree_lock:
        if curie in _degree_cache:
            return _degree_cache[curie]
    if MOCK:
        d = _mock_degree(curie)
        with _degree_lock:
            _degree_cache[curie] = d
        return d
    try:
        node_idx = GRAPH.get_node_idx(curie)
        if node_idx is None:
            degree = None
        else:
            out_deg = int(GRAPH.fwd_offsets[node_idx + 1] - GRAPH.fwd_offsets[node_idx])
            in_deg = int(GRAPH.rev_offsets[node_idx + 1] - GRAPH.rev_offsets[node_idx])
            degree = out_deg + in_deg
    except Exception:
        degree = None
    with _degree_lock:
        _degree_cache[curie] = degree
    return degree


def _trapi_neighbor_query(curie, root_is_subject):
    """Build a one-hop, predicate-agnostic neighbour query in one direction."""
    if root_is_subject:
        edge = {"subject": "n0", "object": "n1"}
    else:
        edge = {"subject": "n1", "object": "n0"}
    return {
        "message": {
            "query_graph": {
                "nodes": {
                    "n0": {"ids": [curie]},
                    "n1": {},  # unconstrained — match any neighbour
                },
                "edges": {"e0": edge},
            }
        },
        "parameters": {"subclass": SUBCLASS},
    }


def _node_label(kg_nodes, node_id):
    node = kg_nodes.get(node_id) or {}
    name = node.get("name") or node_id
    cats = node.get("categories") or []
    return name, cats


def expand(curie):
    """
    Fetch all neighbours of `curie` in both directions, grouped by
    (predicate, direction). Returns a dict ready for the frontend.
    """
    with _expand_lock:
        if curie in _expand_cache:
            return _expand_cache[curie]

    if MOCK:
        result = mock_expand(curie)
        with _expand_lock:
            _expand_cache[curie] = result
        return result

    root_name, root_cats = curie, []
    # key -> {predicate, direction, neighbors: {id: {id,name,categories}}}
    groups = {}
    errors = []

    for root_is_subject in (True, False):
        direction = "out" if root_is_subject else "in"
        try:
            resp = _post_trapi(_trapi_neighbor_query(curie, root_is_subject))
        except urllib.error.HTTPError as e:
            errors.append(f"{direction}: HTTP {e.code}")
            continue
        except Exception as e:  # noqa: BLE001
            errors.append(f"{direction}: {e}")
            continue

        msg = (resp or {}).get("message", {}) or {}
        kg = msg.get("knowledge_graph") or {}
        kg_nodes = kg.get("nodes") or {}
        kg_edges = kg.get("edges") or {}

        # capture the root's own label if present
        if curie in kg_nodes:
            rn, rc = _node_label(kg_nodes, curie)
            root_name, root_cats = rn, rc

        for edge in kg_edges.values():
            subj = edge.get("subject")
            obj = edge.get("object")
            pred = edge.get("predicate") or "biolink:related_to"
            # neighbour is the endpoint that isn't the root
            if root_is_subject:
                neighbor = obj
            else:
                neighbor = subj
            if neighbor is None or neighbor == curie:
                continue
            key = f"{direction}|{pred}"
            grp = groups.get(key)
            if grp is None:
                grp = {"predicate": pred, "direction": direction, "neighbors": {}}
                groups[key] = grp
            if neighbor not in grp["neighbors"]:
                nname, ncats = _node_label(kg_nodes, neighbor)
                grp["neighbors"][neighbor] = {
                    "id": neighbor,
                    "name": nname,
                    "categories": ncats,
                }

    out_groups = []
    for key, grp in groups.items():
        neighbors = list(grp["neighbors"].values())
        out_groups.append(
            {
                "key": key,
                "predicate": grp["predicate"],
                "direction": grp["direction"],
                "count": len(neighbors),
                "neighbors": neighbors,
            }
        )
    # Sort predicate groups by neighbour count (descending) for a sensible default.
    out_groups.sort(key=lambda g: g["count"], reverse=True)

    result = {
        "curie": curie,
        "name": root_name,
        "categories": root_cats,
        "groups": out_groups,
        "errors": errors,
    }
    with _expand_lock:
        _expand_cache[curie] = result
    return result


def search_names(q, limit=10):
    """
    Proxy the SRI Name Resolver autocomplete. Returns a normalised list of
    {curie, label, types}. Tolerant of the resolver's list- and dict-shaped
    responses across versions.
    """
    q = (q or "").strip()
    if len(q) < 2:
        return []
    if MOCK:
        return mock_search(q)
    params = urllib.parse.urlencode(
        {"string": q, "autocomplete": "true", "offset": "0", "limit": str(limit)}
    )
    url = f"{NAME_RESOLVER}/lookup?{params}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=min(TIMEOUT, 20)) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    out = []
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            curie = item.get("curie") or item.get("id")
            if not curie:
                continue
            syns = item.get("synonyms") or []
            label = item.get("label") or (syns[0] if syns else curie)
            out.append({"curie": curie, "label": label, "types": item.get("types") or []})
    elif isinstance(data, dict):
        # older shape: {curie: [synonyms...]}
        for curie, syns in data.items():
            label = syns[0] if isinstance(syns, list) and syns else curie
            out.append({"curie": curie, "label": label, "types": []})
    return out


def wiki_summary(name):
    """
    Resolve a node name to an English Wikipedia article by search, then return
    a short summary. Cached per name. Returns {found, title, description,
    extract, url, type} or {found: False}.
    """
    name = (name or "").strip()
    if not name:
        return {"found": False}
    with _wiki_lock:
        if name in _wiki_cache:
            return _wiki_cache[name]

    if MOCK:
        result = {
            "found": True, "title": name, "description": "synthetic entry",
            "extract": f"This is a synthetic Wikipedia-style summary for “{name}”, "
                       f"shown in offline mock mode so the card layout can be tested "
                       f"without network access.",
            "url": "https://en.wikipedia.org/", "type": "standard",
        }
        with _wiki_lock:
            _wiki_cache[name] = result
        return result

    headers = {"User-Agent": WIKI_UA, "Accept": "application/json"}
    result = {"found": False}
    try:
        # 1) find the best-matching article title
        sp = urllib.parse.urlencode(
            {"action": "query", "list": "search", "srsearch": name,
             "srlimit": "1", "format": "json"}
        )
        req = urllib.request.Request(f"https://en.wikipedia.org/w/api.php?{sp}", headers=headers)
        with urllib.request.urlopen(req, timeout=min(TIMEOUT, 15)) as resp:
            hits = json.loads(resp.read().decode("utf-8")).get("query", {}).get("search", [])
        if not hits:
            result = {"found": False}
        else:
            title = hits[0].get("title")
            # 2) fetch the page summary (clean plain-text extract + canonical URL)
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(title)}"
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=min(TIMEOUT, 15)) as resp:
                s = json.loads(resp.read().decode("utf-8"))
            page = (((s.get("content_urls") or {}).get("desktop") or {}).get("page")
                    or f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title)}")
            result = {
                "found": True,
                "title": s.get("title") or title,
                "description": s.get("description"),
                "extract": s.get("extract") or "",
                "url": page,
                "type": s.get("type") or "standard",
            }
    except Exception as e:  # noqa: BLE001
        result = {"found": False, "error": str(e)}

    with _wiki_lock:
        _wiki_cache[name] = result
    return result


def degrees(curies):
    """Fetch degrees for a list of curies concurrently. Returns {curie: degree|null}."""
    out = {}
    todo = [c for c in curies if c]
    if not todo:
        return out
    with ThreadPoolExecutor(max_workers=DEGREE_WORKERS) as pool:
        futs = {pool.submit(_get_degree, c): c for c in todo}
        for fut in as_completed(futs):
            c = futs[fut]
            try:
                out[c] = fut.result()
            except Exception:  # noqa: BLE001
                out[c] = None
    return out


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    server_version = "GandalfExplorer/1.0"

    def log_message(self, fmt, *args):  # quieter logs
        sys.stderr.write("  %s\n" % (fmt % args))

    # -- helpers ----------------------------------------------------------
    def _send_json(self, obj, status=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_static(self, filename, content_type):
        path = os.path.join(HERE, filename)
        try:
            with open(path, "rb") as f:
                body = f.read()
        except FileNotFoundError:
            self.send_error(404, "Not found")
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # -- routes -----------------------------------------------------------
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        route = parsed.path

        if route in STATIC:
            filename, ctype = STATIC[route]
            self._send_static(filename, ctype)
            return

        if route == "/api/expand":
            qs = urllib.parse.parse_qs(parsed.query)
            curie = (qs.get("curie") or [""])[0].strip()
            if not curie:
                self._send_json({"error": "missing 'curie' parameter"}, 400)
                return
            try:
                self._send_json(expand(curie))
            except Exception as e:  # noqa: BLE001
                self._send_json({"error": str(e)}, 502)
            return

        if route == "/api/config":
            self._send_json(
                {"graph_path": GRAPH_PATH, "subclass": SUBCLASS, "timeout": TIMEOUT}
            )
            return

        if route == "/api/search":
            qs = urllib.parse.parse_qs(parsed.query)
            q = (qs.get("q") or [""])[0]
            try:
                self._send_json({"results": search_names(q)})
            except Exception as e:  # noqa: BLE001
                self._send_json({"error": str(e), "results": []}, 502)
            return

        if route == "/api/wiki":
            qs = urllib.parse.parse_qs(parsed.query)
            name = (qs.get("name") or [""])[0]
            try:
                self._send_json(wiki_summary(name))
            except Exception as e:  # noqa: BLE001
                self._send_json({"found": False, "error": str(e)}, 502)
            return

        self.send_error(404, "Not found")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        route = parsed.path

        if route == "/api/degrees":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")
                curies = payload.get("curies") or []
                if not isinstance(curies, list):
                    raise ValueError("'curies' must be a list")
            except Exception as e:  # noqa: BLE001
                self._send_json({"error": f"bad request: {e}"}, 400)
                return
            try:
                self._send_json({"degrees": degrees(curies)})
            except Exception as e:  # noqa: BLE001
                self._send_json({"error": str(e)}, 502)
            return

        self.send_error(404, "Not found")


def main():
    httpd = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"Mithrandir running at  http://localhost:{PORT}")
    if not MOCK:
        print(f"  gandalf graph  -> {GRAPH_PATH} (in-process)")
    print(f"  name search    -> {NAME_RESOLVER}")
    print(f"  subclass inference: {SUBCLASS}")
    print(f"  force IPv4: {FORCE_IPV4}")
    if MOCK:
        print("  *** MOCK MODE — serving synthetic data, no graph or network ***")
    print("  press Ctrl+C to stop")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
        httpd.shutdown()


if __name__ == "__main__":
    main()
