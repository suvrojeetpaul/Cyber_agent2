import json
from pathlib import Path

import torch

from relational_gnn import RelationalGNN


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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "graph"
)


def load_json(file_path):

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def prepare_tensors(
    graph_data,
    feature_data
):

    features = torch.tensor(
        feature_data["features"],
        dtype=torch.float32
    )

    edge_index = torch.tensor(
        graph_data["edge_index"],
        dtype=torch.long
    ).t().contiguous()

    edge_type = torch.tensor(
        graph_data["edge_type_indices"],
        dtype=torch.long
    )

    node_type = torch.tensor(
        graph_data["node_type_indices"],
        dtype=torch.long
    )

    return (
        features,
        edge_index,
        edge_type,
        node_type
    )


def run_model(
    features,
    edge_index,
    edge_type
):

    model = RelationalGNN(
        input_channels=11,
        hidden_channels=128,
        output_channels=64,
        num_relations=3,
        num_layers=3,
        dropout=0.2
    )

    model.eval()

    with torch.no_grad():

        embeddings = model(
            features,
            edge_index,
            edge_type
        )

    return model, embeddings


def save_embeddings(
    embeddings
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        OUTPUT_DIR
        / "gnn_embeddings.pt"
    )

    torch.save(
        embeddings,
        output_file
    )

    return output_file


def print_summary(
    features,
    edge_index,
    edge_type,
    node_type,
    embeddings
):

    print()
    print(
        "========================================"
    )
    print(
        "          GNN FORWARD PASS"
    )
    print(
        "========================================"
    )

    print()
    print(
        f"Node feature tensor: "
        f"{tuple(features.shape)}"
    )

    print(
        f"Edge index tensor: "
        f"{tuple(edge_index.shape)}"
    )

    print(
        f"Edge type tensor: "
        f"{tuple(edge_type.shape)}"
    )

    print(
        f"Node type tensor: "
        f"{tuple(node_type.shape)}"
    )

    print()
    print(
        f"Input feature dimension: "
        f"{features.shape[1]}"
    )

    print(
        f"Number of graph nodes: "
        f"{features.shape[0]}"
    )

    print(
        f"Number of graph edges: "
        f"{edge_index.shape[1]}"
    )

    print()
    print(
        f"GNN embedding tensor: "
        f"{tuple(embeddings.shape)}"
    )

    print(
        f"Embedding dimension: "
        f"{embeddings.shape[1]}"
    )

    print()
    print(
        "Sample node embedding:"
    )

    print(
        embeddings[0]
    )

    print()
    print(
        "Embedding statistics:"
    )

    print(
        f"Minimum: "
        f"{embeddings.min().item():.6f}"
    )

    print(
        f"Maximum: "
        f"{embeddings.max().item():.6f}"
    )

    print(
        f"Mean: "
        f"{embeddings.mean().item():.6f}"
    )

    print(
        f"Standard deviation: "
        f"{embeddings.std().item():.6f}"
    )


def main():

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

    (
        features,
        edge_index,
        edge_type,
        node_type
    ) = prepare_tensors(
        graph_data,
        feature_data
    )

    (
        model,
        embeddings
    ) = run_model(
        features,
        edge_index,
        edge_type
    )

    output_file = save_embeddings(
        embeddings
    )

    print_summary(
        features,
        edge_index,
        edge_type,
        node_type,
        embeddings
    )

    print()
    print(
        "GNN model:"
    )

    print(
        model
    )

    print()
    print(
        "Embeddings saved to:"
    )

    print(
        output_file
    )

    print()
    print(
        "GNN forward pass completed successfully."
    )


if __name__ == "__main__":
    main()