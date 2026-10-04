import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

NODES_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "graph"
    / "nodes.json"
)

GRAPH_DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "graph"
    / "gnn_graph_data.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "graph"
)


def load_nodes():
    with open(
        NODES_FILE,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def load_graph_data():
    with open(
        GRAPH_DATA_FILE,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def create_feature_vector(node):
    node_type = node.get(
        "node_type",
        "unknown"
    )

    name = node.get(
        "name",
        ""
    )

    description = node.get(
        "description",
        ""
    )

    feature_vector = [
        1 if node_type == "campaign" else 0,
        1 if node_type == "malware" else 0,
        1 if node_type == "technique" else 0,
        1 if node_type == "threat_actor" else 0,
        1 if node_type == "tool" else 0,

        min(len(name), 100) / 100.0,

        min(len(description), 1000) / 1000.0,

        1 if name else 0,

        1 if description else 0
    ]

    return feature_vector


def calculate_graph_statistics(
    graph_data
):
    num_nodes = graph_data[
        "num_nodes"
    ]

    edges = graph_data[
        "edge_index"
    ]

    in_degree = [
        0
        for _ in range(num_nodes)
    ]

    out_degree = [
        0
        for _ in range(num_nodes)
    ]

    for edge in edges:

        source = edge[0]
        target = edge[1]

        out_degree[source] += 1
        in_degree[target] += 1

    return (
        in_degree,
        out_degree
    )


def add_graph_features(
    feature_vectors,
    graph_data
):
    (
        in_degree,
        out_degree
    ) = calculate_graph_statistics(
        graph_data
    )

    num_nodes = graph_data[
        "num_nodes"
    ]

    for index in range(num_nodes):

        normalized_in_degree = (
            min(
                in_degree[index],
                100
            ) / 100.0
        )

        normalized_out_degree = (
            min(
                out_degree[index],
                100
            ) / 100.0
        )

        feature_vectors[index].extend(
            [
                normalized_in_degree,
                normalized_out_degree
            ]
        )

    return feature_vectors


def build_features():

    nodes = load_nodes()

    graph_data = load_graph_data()

    feature_vectors = []

    for node in nodes:

        vector = create_feature_vector(
            node
        )

        feature_vectors.append(
            vector
        )

    feature_vectors = add_graph_features(
        feature_vectors,
        graph_data
    )

    return feature_vectors


def save_features(
    feature_vectors
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        OUTPUT_DIR
        / "node_features.json"
    )

    data = {
        "num_nodes": len(
            feature_vectors
        ),

        "feature_dimension": len(
            feature_vectors[0]
        ),

        "features": feature_vectors
    }

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=2
        )

    return output_file


def print_summary(
    feature_vectors
):

    print()
    print(
        "========================================"
    )
    print(
        "          NODE FEATURE CREATION"
    )
    print(
        "========================================"
    )

    print(
        f"\nNumber of nodes: "
        f"{len(feature_vectors)}"
    )

    print(
        f"Feature dimension: "
        f"{len(feature_vectors[0])}"
    )

    print(
        "\nFeature structure:"
    )

    print(
        "  0-4  -> Node type indicators"
    )

    print(
        "  5    -> Name length"
    )

    print(
        "  6    -> Description length"
    )

    print(
        "  7    -> Name exists"
    )

    print(
        "  8    -> Description exists"
    )

    print(
        "  9    -> Normalized in-degree"
    )

    print(
        "  10   -> Normalized out-degree"
    )

    print(
        "\nSample feature vector:"
    )

    print(
        feature_vectors[0]
    )


def main():

    feature_vectors = build_features()

    output_file = save_features(
        feature_vectors
    )

    print_summary(
        feature_vectors
    )

    print(
        "\nNode features saved to:"
    )

    print(
        output_file
    )


if __name__ == "__main__":
    main()