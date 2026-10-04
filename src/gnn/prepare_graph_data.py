import json
from pathlib import Path
from collections import Counter


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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "graph"
)


def load_graph_data():
    with open(NODES_FILE, "r", encoding="utf-8") as file:
        nodes = json.load(file)

    with open(EDGES_FILE, "r", encoding="utf-8") as file:
        edges = json.load(file)

    return nodes, edges


def create_node_mapping(nodes):
    node_to_index = {}
    index_to_node = {}

    for index, node in enumerate(nodes):
        node_id = node["id"]

        node_to_index[node_id] = index
        index_to_node[str(index)] = node_id

    return node_to_index, index_to_node


def encode_node_types(nodes):
    node_types = sorted(
        set(node.get("node_type", "unknown") for node in nodes)
    )

    node_type_to_index = {
        node_type: index
        for index, node_type in enumerate(node_types)
    }

    node_type_indices = []

    for node in nodes:
        node_type = node.get("node_type", "unknown")
        node_type_indices.append(
            node_type_to_index[node_type]
        )

    return (
        node_type_to_index,
        node_type_indices
    )


def encode_edge_types(edges):
    edge_types = sorted(
        set(edge.get("relationship", "unknown") for edge in edges)
    )

    edge_type_to_index = {
        edge_type: index
        for index, edge_type in enumerate(edge_types)
    }

    edge_type_indices = []

    for edge in edges:
        relationship = edge.get(
            "relationship",
            "unknown"
        )

        edge_type_indices.append(
            edge_type_to_index[relationship]
        )

    return (
        edge_type_to_index,
        edge_type_indices
    )


def create_edge_index(edges, node_to_index):
    edge_index = []

    skipped_edges = 0

    for edge in edges:
        source = edge["source"]
        target = edge["target"]

        if source not in node_to_index:
            skipped_edges += 1
            continue

        if target not in node_to_index:
            skipped_edges += 1
            continue

        source_index = node_to_index[source]
        target_index = node_to_index[target]

        edge_index.append(
            [
                source_index,
                target_index
            ]
        )

    return edge_index, skipped_edges


def build_dataset():
    nodes, edges = load_graph_data()

    node_to_index, index_to_node = (
        create_node_mapping(nodes)
    )

    (
        node_type_to_index,
        node_type_indices
    ) = encode_node_types(nodes)

    (
        edge_type_to_index,
        edge_type_indices
    ) = encode_edge_types(edges)

    edge_index, skipped_edges = create_edge_index(
        edges,
        node_to_index
    )

    dataset = {
        "num_nodes": len(nodes),
        "num_edges": len(edge_index),

        "node_to_index": node_to_index,

        "index_to_node": index_to_node,

        "node_type_to_index": node_type_to_index,

        "node_type_indices": node_type_indices,

        "edge_type_to_index": edge_type_to_index,

        "edge_type_indices": edge_type_indices,

        "edge_index": edge_index,

        "skipped_edges": skipped_edges
    }

    return dataset


def save_dataset(dataset):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        OUTPUT_DIR
        / "gnn_graph_data.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            dataset,
            file,
            indent=2
        )

    return output_file


def print_summary(dataset):
    print()
    print("========================================")
    print("       GNN GRAPH DATA PREPARATION")
    print("========================================")

    print(
        f"\nNumber of nodes: "
        f"{dataset['num_nodes']}"
    )

    print(
        f"Number of edges: "
        f"{dataset['num_edges']}"
    )

    print(
        f"Skipped edges: "
        f"{dataset['skipped_edges']}"
    )

    print("\nNode type encoding:")

    for (
        node_type,
        index
    ) in dataset["node_type_to_index"].items():

        print(
            f"  {index} -> {node_type}"
        )

    print("\nEdge type encoding:")

    for (
        edge_type,
        index
    ) in dataset["edge_type_to_index"].items():

        print(
            f"  {index} -> {edge_type}"
        )

    print("\nSample node mapping:")

    for index in range(
        min(5, dataset["num_nodes"])
    ):

        node_id = dataset[
            "index_to_node"
        ][str(index)]

        print(
            f"  {index} -> {node_id}"
        )

    print("\nSample edges:")

    for edge in dataset[
        "edge_index"
    ][:5]:

        print(
            f"  {edge[0]} -> {edge[1]}"
        )


def main():
    dataset = build_dataset()

    output_file = save_dataset(
        dataset
    )

    print_summary(dataset)

    print(
        "\nGNN dataset saved to:"
    )

    print(output_file)


if __name__ == "__main__":
    main()