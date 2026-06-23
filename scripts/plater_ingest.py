"""Download and create gandalf plater graphs."""

import logging

from gandalf import build_graph_from_jsonl

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

# Build graph from edges and nodes
graph = build_graph_from_jsonl(
    edge_jsonl_path="../translatorkg/edges.jsonl",
    node_jsonl_path="../translatorkg/nodes.jsonl",
)

# Save for fast reloading
graph.save_mmap("../translatorkg/gandalf_mmap")
print("Graph saved!")
