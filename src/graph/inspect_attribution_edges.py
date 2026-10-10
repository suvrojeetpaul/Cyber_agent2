import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

NODES_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "graph"
    / "attribution"
    / "nodes.json"
)

EDGES_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "graph"
    / "attribution"
    / "edges.json"
)

with open(NODES_FILE, "r", encoding="utf-8") as file:
    nodes = json.load(file)

with open(EDGES_FILE, "r", encoding="utf-8") as file:
    edges = json.load(file)

node_lookup = {
    node["id"]: node
    for node in nodes
}

print("\nATTRIBUTION GRAPH EDGE INSPECTION")
print("=" * 60)

for edge in edges:
    source = node_lookup.get(edge["source"], {})
    target = node_lookup.get(edge["target"], {})

    print(
        f"{source.get('name', edge['source'])}"
        f" [{source.get('node_type', 'unknown')}]"
        f" --{edge.get('relation', 'unknown')}-->"
        f" {target.get('name', edge['target'])}"
        f" [{target.get('node_type', 'unknown')}]"
    )

print("\nTotal nodes:", len(nodes))
print("Total edges:", len(edges))