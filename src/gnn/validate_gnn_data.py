import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

GRAPH_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "graph"
    / "gnn_graph_data.json"
)

FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "graph"
    / "node_features.json"
)


def load_json(file_path):

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def validate_graph_data(graph_data):

    errors = []

    num_nodes = graph_data.get(
        "num_nodes"
    )

    num_edges = graph_data.get(
        "num_edges"
    )

    edge_index = graph_data.get(
        "edge_index",
        []
    )

    edge_types = graph_data.get(
        "edge_type_indices",
        []
    )

    node_types = graph_data.get(
        "node_type_indices",
        []
    )

    if num_nodes <= 0:
        errors.append(
            "Number of nodes is invalid."
        )

    if num_edges <= 0:
        errors.append(
            "Number of edges is invalid."
        )

    if len(edge_index) != num_edges:
        errors.append(
            "Edge count does not match edge_index."
        )

    if len(edge_types) != num_edges:
        errors.append(
            "Edge type count does not match edge count."
        )

    if len(node_types) != num_nodes:
        errors.append(
            "Node type count does not match node count."
        )

    for edge in edge_index:

        if len(edge) != 2:
            errors.append(
                "Invalid edge format detected."
            )
            break

        source = edge[0]
        target = edge[1]

        if source < 0 or source >= num_nodes:
            errors.append(
                f"Invalid source node index: {source}"
            )
            break

        if target < 0 or target >= num_nodes:
            errors.append(
                f"Invalid target node index: {target}"
            )
            break

    return errors


def validate_feature_data(
    graph_data,
    feature_data
):

    errors = []

    num_nodes = graph_data[
        "num_nodes"
    ]

    features = feature_data.get(
        "features",
        []
    )

    feature_count = feature_data.get(
        "num_nodes"
    )

    feature_dimension = feature_data.get(
        "feature_dimension"
    )

    if feature_count != num_nodes:

        errors.append(
            "Feature node count does not match graph node count."
        )

    if len(features) != num_nodes:

        errors.append(
            "Number of feature vectors does not match node count."
        )

    if not features:

        errors.append(
            "Feature matrix is empty."
        )

        return errors

    for index, vector in enumerate(
        features
    ):

        if len(vector) != feature_dimension:

            errors.append(
                f"Feature dimension mismatch at node {index}."
            )

            break

        for value in vector:

            if not isinstance(
                value,
                (int, float)
            ):

                errors.append(
                    f"Non-numeric feature detected at node {index}."
                )

                break

    return errors


def print_summary(
    graph_data,
    feature_data
):

    print()
    print(
        "========================================"
    )
    print(
        "          GNN DATA VALIDATION"
    )
    print(
        "========================================"
    )

    print()
    print(
        f"Graph nodes: "
        f"{graph_data['num_nodes']}"
    )

    print(
        f"Graph edges: "
        f"{graph_data['num_edges']}"
    )

    print(
        f"Feature nodes: "
        f"{feature_data['num_nodes']}"
    )

    print(
        f"Feature dimension: "
        f"{feature_data['feature_dimension']}"
    )

    print(
        f"Node type classes: "
        f"{len(graph_data['node_type_to_index'])}"
    )

    print(
        f"Edge type classes: "
        f"{len(graph_data['edge_type_to_index'])}"
    )


def main():

    print()

    if not GRAPH_FILE.exists():

        print(
            "ERROR: GNN graph data file not found."
        )

        return

    if not FEATURE_FILE.exists():

        print(
            "ERROR: Node feature file not found."
        )

        return

    graph_data = load_json(
        GRAPH_FILE
    )

    feature_data = load_json(
        FEATURE_FILE
    )

    graph_errors = validate_graph_data(
        graph_data
    )

    feature_errors = validate_feature_data(
        graph_data,
        feature_data
    )

    all_errors = (
        graph_errors
        + feature_errors
    )

    print_summary(
        graph_data,
        feature_data
    )

    print()

    if all_errors:

        print(
            "VALIDATION STATUS: FAILED"
        )

        print()

        for error in all_errors:

            print(
                f"[ERROR] {error}"
            )

        return

    print(
        "VALIDATION STATUS: PASSED"
    )

    print()
    print(
        "All GNN inputs are internally consistent."
    )

    print(
        "The dataset is ready for GNN framework conversion."
    )


if __name__ == "__main__":
    main()