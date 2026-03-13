import json

with open("./heartburn_responses/response_gandalf_13.json", "r", encoding="utf-8") as f:
    gandalf = json.load(f)
    print("Gandalf returned results:", len(gandalf["message"]["results"]))

with open("./heartburn_responses/response_neo4j_13.json", "r", encoding="utf-8") as f:
    automat = json.load(f)
    print("Automat returned results:", len(automat["message"]["results"]))

gandalf_missed_edges = []
automat_missed_edges = []
gandalf_edges = set()
automat_edges = set()
for edge in gandalf["message"]["knowledge_graph"]["edges"].values():
    gandalf_edge = (edge["subject"], edge["predicate"], edge["object"])
    gandalf_edges.add(gandalf_edge)
for edge in automat["message"]["knowledge_graph"]["edges"].values():
    automat_edge = (edge["subject"], edge["predicate"], edge["object"])
    automat_edges.add(automat_edge)
    if automat_edge not in gandalf_edges:
        gandalf_missed_edges.append(edge)

for gandalf_edge in gandalf_edges:
    if gandalf_edge not in automat_edges:
        automat_missed_edges.append(gandalf_edge)


with open("gandalf_missed_edges.json", "w", encoding="utf-8") as f:
    json.dump(gandalf_missed_edges, f, indent=2)

with open("automat_missed_edges.json", "w", encoding="utf-8") as f:
    json.dump(automat_missed_edges, f, indent=2)
