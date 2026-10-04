import json
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[2]

NODES_FILE = PROJECT_ROOT / "data" / "processed" / "graph" / "nodes.json"
EDGES_FILE = PROJECT_ROOT / "data" / "processed" / "graph" / "edges.json"


def load_data():
    with open(NODES_FILE, "r", encoding="utf-8") as file:
        nodes = json.load(file)

    with open(EDGES_FILE, "r", encoding="utf-8") as file:
        edges = json.load(file)

    return nodes, edges


def create_summary(nodes, edges):
    node_types = Counter()
    edge_types = Counter()

    for node in nodes:
        node_types[node.get("node_type", "unknown")] += 1

    for edge in edges:
        edge_types[edge.get("relationship", "unknown")] += 1

    summary = {
        "total_nodes": len(nodes),
        "total_edges": len(edges),
        "node_types": dict(sorted(node_types.items())),
        "edge_types": dict(sorted(edge_types.items()))
    }

    return summary


def save_summary(summary):
    output_file = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "graph"
        / "graph_summary.json"
    )

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    return output_file


def main():
    nodes, edges = load_data()
    summary = create_summary(nodes, edges)
    output_file = save_summary(summary)

    print("========================================")
    print("        GRAPH DATASET SUMMARY")
    print("========================================")

    print(f"\nTotal nodes: {summary['total_nodes']}")
    print(f"Total edges: {summary['total_edges']}")

    print("\nNode types:")
    for node_type, count in summary["node_types"].items():
        print(f"  {node_type}: {count}")

    print("\nEdge types:")
    for edge_type, count in summary["edge_types"].items():
        print(f"  {edge_type}: {count}")

    print(f"\nSummary saved to:")
    print(output_file)


if __name__ == "__main__":
    main()