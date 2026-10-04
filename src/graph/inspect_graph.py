import json
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[2]

NODES_FILE = PROJECT_ROOT / "data" / "processed" / "graph" / "nodes.json"
EDGES_FILE = PROJECT_ROOT / "data" / "processed" / "graph" / "edges.json"


def inspect_graph():
    with open(NODES_FILE, "r", encoding="utf-8") as file:
        nodes = json.load(file)

    with open(EDGES_FILE, "r", encoding="utf-8") as file:
        edges = json.load(file)

    node_types = Counter()
    edge_types = Counter()

    for node in nodes:
        node_types[node.get("node_type", "unknown")] += 1

    for edge in edges:
        edge_types[edge.get("relationship", "unknown")] += 1

    print("=== GRAPH INSPECTION ===")
    print(f"Total nodes: {len(nodes)}")
    print(f"Total edges: {len(edges)}")

    print("\nNode types:")
    for node_type, count in sorted(node_types.items()):
        print(f"  {node_type}: {count}")

    print("\nEdge types:")
    for edge_type, count in sorted(edge_types.items()):
        print(f"  {edge_type}: {count}")

    print("\nSample nodes:")
    for node in nodes[:5]:
        print(
            f"  {node['id']} | "
            f"{node.get('node_type')} | "
            f"{node.get('name')}"
        )

    print("\nSample edges:")
    for edge in edges[:5]:
        print(
            f"  {edge['source']} "
            f"--[{edge['relationship']}]--> "
            f"{edge['target']}"
        )


if __name__ == "__main__":
    inspect_graph()