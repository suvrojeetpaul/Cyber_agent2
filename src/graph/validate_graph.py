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


def load_json(file_path):
    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def validate_nodes(nodes):
    node_ids = [node["id"] for node in nodes]

    duplicate_ids = {
        node_id
        for node_id in node_ids
        if node_ids.count(node_id) > 1
    }

    print(f"Total nodes: {len(nodes)}")
    print(f"Unique node IDs: {len(set(node_ids))}")

    if duplicate_ids:
        print(f"Duplicate node IDs: {len(duplicate_ids)}")
    else:
        print("Duplicate node IDs: 0")

    return set(node_ids)


def validate_edges(edges, node_ids):
    invalid_edges = []

    for edge in edges:
        source = edge.get("source")
        target = edge.get("target")

        if source not in node_ids or target not in node_ids:
            invalid_edges.append(edge)

    print(f"Total edges: {len(edges)}")
    print(f"Invalid edges: {len(invalid_edges)}")

    return invalid_edges


def build_graph(nodes, edges):
    graph = nx.MultiDiGraph()

    for node in nodes:
        graph.add_node(
            node["id"],
            name=node.get("name"),
            node_type=node.get("node_type"),
            stix_type=node.get("stix_type"),
        )

    for edge in edges:
        graph.add_edge(
            edge["source"],
            edge["target"],
            relationship=edge["relationship"],
            edge_id=edge["id"],
        )

    return graph


def print_graph_statistics(graph):
    print("\nGRAPH STATISTICS")
    print("=" * 50)

    print(f"Nodes: {graph.number_of_nodes()}")
    print(f"Edges: {graph.number_of_edges()}")

    print(f"Directed: {graph.is_directed()}")

    node_type_counts = {}

    for _, attributes in graph.nodes(data=True):
        node_type = attributes.get("node_type", "unknown")
        node_type_counts[node_type] = (
            node_type_counts.get(node_type, 0) + 1
        )

    print("\nNode type distribution:")

    for node_type, count in sorted(node_type_counts.items()):
        print(f"  {node_type:<20} {count}")

    relationship_counts = {}

    for _, _, attributes in graph.edges(data=True):
        relationship = attributes.get(
            "relationship",
            "unknown"
        )

        relationship_counts[relationship] = (
            relationship_counts.get(relationship, 0) + 1
        )

    print("\nRelationship distribution:")

    for relationship, count in sorted(
        relationship_counts.items()
    ):
        print(f"  {relationship:<20} {count}")


def main():
    print("Loading normalized graph dataset...")

    nodes = load_json(NODES_FILE)
    edges = load_json(EDGES_FILE)

    print("\nNODE VALIDATION")
    print("=" * 50)

    node_ids = validate_nodes(nodes)

    print("\nEDGE VALIDATION")
    print("=" * 50)

    invalid_edges = validate_edges(edges, node_ids)

    if invalid_edges:
        print("\nGraph validation failed.")
        return

    graph = build_graph(nodes, edges)

    print("\nGraph constructed successfully.")

    print_graph_statistics(graph)

    print("\nGraph validation completed successfully.")


if __name__ == "__main__":
    main()