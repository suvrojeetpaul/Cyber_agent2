import json
from pathlib import Path

import networkx as nx


PROJECT_ROOT = Path(__file__).resolve().parents[2]

NODES_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "graph"
    / "nodes.json"
)

EDGES_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "graph"
    / "edges.json"
)


def load_graph():
    with open(NODES_FILE, "r", encoding="utf-8") as file:
        nodes = json.load(file)

    with open(EDGES_FILE, "r", encoding="utf-8") as file:
        edges = json.load(file)

    graph = nx.MultiDiGraph()

    for node in nodes:
        graph.add_node(
            node["id"],
            name=node.get("name"),
            node_type=node.get("node_type"),
            stix_type=node.get("stix_type"),
            description=node.get("description", ""),
        )

    for edge in edges:
        graph.add_edge(
            edge["source"],
            edge["target"],
            relationship=edge["relationship"],
            edge_id=edge["id"],
        )

    return graph


if __name__ == "__main__":
    graph = load_graph()

    print("Graph loaded successfully.")
    print(f"Nodes: {graph.number_of_nodes()}")
    print(f"Edges: {graph.number_of_edges()}")