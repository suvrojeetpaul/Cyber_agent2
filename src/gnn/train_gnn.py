import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.nn import RGCNConv


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

MODEL_DIR = PROJECT_ROOT / "outputs" / "models"
EMBEDDING_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "graph"
    / "gnn_embeddings_trained.pt"
)

MODEL_FILE = MODEL_DIR / "rgcn_link_prediction.pt"

SEED = 42

HIDDEN_CHANNELS = 128
OUTPUT_CHANNELS = 64
NUM_LAYERS = 3
DROPOUT = 0.2

LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5

EPOCHS = 50

TRAIN_RATIO = 0.80
VALIDATION_RATIO = 0.10
TEST_RATIO = 0.10

NEGATIVE_RATIO = 1


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def extract_features(feature_data, number_of_nodes):
    possible_keys = [
        "features",
        "node_features",
        "feature_matrix",
        "vectors"
    ]

    feature_values = None

    for key in possible_keys:
        if key in feature_data:
            feature_values = feature_data[key]
            break

    if feature_values is None:
        if isinstance(feature_data, list):
            feature_values = feature_data
        else:
            raise KeyError(
                "Could not find node features in node_features.json. "
                "Expected one of: "
                + ", ".join(possible_keys)
            )

    if isinstance(feature_values, dict):
        ordered_features = []

        for index in range(number_of_nodes):
            key = str(index)

            if key in feature_values:
                ordered_features.append(feature_values[key])
            elif index in feature_values:
                ordered_features.append(feature_values[index])
            else:
                raise KeyError(
                    f"Missing feature vector for node index {index}."
                )

        feature_values = ordered_features

    features = torch.tensor(
        feature_values,
        dtype=torch.float32
    )

    if features.ndim != 2:
        raise ValueError(
            f"Node feature tensor must be 2-dimensional. "
            f"Got shape: {tuple(features.shape)}"
        )

    if features.shape[0] != number_of_nodes:
        raise ValueError(
            f"Feature node count mismatch. "
            f"Graph has {number_of_nodes} nodes, "
            f"but features contain {features.shape[0]} nodes."
        )

    return features


def load_data():
    print("Loading graph data...")

    graph_data = load_json(GRAPH_FILE)
    feature_data = load_json(FEATURE_FILE)

    number_of_nodes = int(graph_data["num_nodes"])

    edge_index = torch.tensor(
        graph_data["edge_index"],
        dtype=torch.long
    )

    if edge_index.ndim != 2:
        raise ValueError(
            f"edge_index must be 2-dimensional. "
            f"Got shape: {tuple(edge_index.shape)}"
        )

    if edge_index.shape[0] != 2:
        if edge_index.shape[1] == 2:
            edge_index = edge_index.t().contiguous()
        else:
            raise ValueError(
                "edge_index must have shape [2, number_of_edges]."
            )

    edge_type = torch.tensor(
        graph_data["edge_type_indices"],
        dtype=torch.long
    )

    node_types = torch.tensor(
        graph_data["node_type_indices"],
        dtype=torch.long
    )

    features = extract_features(
        feature_data,
        number_of_nodes
    )

    number_of_edges = edge_index.shape[1]

    if edge_type.ndim != 1:
        edge_type = edge_type.reshape(-1)

    if edge_type.shape[0] != number_of_edges:
        raise ValueError(
            f"Edge type count mismatch. "
            f"Found {edge_type.shape[0]} edge types "
            f"for {number_of_edges} edges."
        )

    if node_types.ndim != 1:
        node_types = node_types.reshape(-1)

    if node_types.shape[0] != number_of_nodes:
        raise ValueError(
            f"Node type count mismatch. "
            f"Found {node_types.shape[0]} node types "
            f"for {number_of_nodes} nodes."
        )

    number_of_relations = int(edge_type.max().item()) + 1

    feature_dimension = features.shape[1]

    print(f"Number of nodes: {number_of_nodes}")
    print(f"Number of edges: {number_of_edges}")
    print(f"Number of relations: {number_of_relations}")
    print(f"Feature dimension: {feature_dimension}")

    return (
        features,
        edge_index,
        edge_type,
        node_types,
        number_of_nodes,
        number_of_relations
    )


def split_edges(
    edge_index,
    edge_type,
    train_ratio=0.80,
    validation_ratio=0.10,
    seed=42
):
    number_of_edges = edge_index.shape[1]

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_indices = []
    validation_indices = []
    test_indices = []

    relation_values = torch.unique(edge_type).tolist()

    for relation in relation_values:
        relation_indices = torch.nonzero(
            edge_type == relation,
            as_tuple=False
        ).view(-1)

        relation_indices = relation_indices[
            torch.randperm(
                relation_indices.numel(),
                generator=generator
            )
        ]

        relation_count = relation_indices.numel()

        if relation_count < 3:
            train_indices.extend(
                relation_indices.tolist()
            )
            continue

        train_count = int(
            relation_count * train_ratio
        )

        validation_count = int(
            relation_count * validation_ratio
        )

        if train_count < 1:
            train_count = 1

        if validation_count < 1:
            validation_count = 1

        test_count = (
            relation_count
            - train_count
            - validation_count
        )

        if test_count < 1:
            test_count = 1
            validation_count = max(
                1,
                relation_count
                - train_count
                - test_count
            )

        if (
            train_count
            + validation_count
            + test_count
            > relation_count
        ):
            train_count = relation_count - 2
            validation_count = 1
            test_count = 1

        relation_train = relation_indices[
            :train_count
        ]

        relation_validation = relation_indices[
            train_count:
            train_count + validation_count
        ]

        relation_test = relation_indices[
            train_count + validation_count:
        ]

        train_indices.extend(
            relation_train.tolist()
        )

        validation_indices.extend(
            relation_validation.tolist()
        )

        test_indices.extend(
            relation_test.tolist()
        )

    train_indices = torch.tensor(
        train_indices,
        dtype=torch.long
    )

    validation_indices = torch.tensor(
        validation_indices,
        dtype=torch.long
    )

    test_indices = torch.tensor(
        test_indices,
        dtype=torch.long
    )

    train_indices = train_indices[
        torch.randperm(
            train_indices.numel(),
            generator=generator
        )
    ]

    validation_indices = validation_indices[
        torch.randperm(
            validation_indices.numel(),
            generator=generator
        )
    ]

    test_indices = test_indices[
        torch.randperm(
            test_indices.numel(),
            generator=generator
        )
    ]

    train_edge_index = edge_index[
        :,
        train_indices
    ]

    train_edge_type = edge_type[
        train_indices
    ]

    validation_edge_index = edge_index[
        :,
        validation_indices
    ]

    validation_edge_type = edge_type[
        validation_indices
    ]

    test_edge_index = edge_index[
        :,
        test_indices
    ]

    test_edge_type = edge_type[
        test_indices
    ]

    return (
        train_edge_index,
        train_edge_type,
        validation_edge_index,
        validation_edge_type,
        test_edge_index,
        test_edge_type
    )


def create_positive_lookup(
    edge_index,
    edge_type
):
    sources = edge_index[0].tolist()
    targets = edge_index[1].tolist()
    relations = edge_type.tolist()

    lookup = set()

    for source, target, relation in zip(
        sources,
        targets,
        relations
    ):
        lookup.add(
            (
                int(source),
                int(target),
                int(relation)
            )
        )

    return lookup


def create_negative_edges(
    positive_edge_index,
    positive_edge_type,
    number_of_nodes,
    number_of_relations,
    number_of_negative_edges=None,
    all_positive_edges=None,
    seed=42
):
    if number_of_negative_edges is None:
        number_of_negative_edges = int(
            positive_edge_index.shape[1]
        )
    else:
        number_of_negative_edges = int(
            number_of_negative_edges
        )

    if number_of_negative_edges <= 0:
        return (
            torch.empty(
                (2, 0),
                dtype=torch.long
            ),
            torch.empty(
                (0,),
                dtype=torch.long
            )
        )

    if all_positive_edges is None:
        positive_lookup = create_positive_lookup(
            positive_edge_index,
            positive_edge_type
        )
    else:
        positive_lookup = create_positive_lookup(
            all_positive_edges[0],
            all_positive_edges[1]
        )

    generator = torch.Generator()
    generator.manual_seed(seed)

    negative_sources = []
    negative_targets = []
    negative_relations = []

    max_attempts = max(
        number_of_negative_edges * 50,
        1000
    )

    attempts = 0

    while (
        len(negative_sources)
        < number_of_negative_edges
        and attempts < max_attempts
    ):
        attempts += 1

        source = int(
            torch.randint(
                0,
                number_of_nodes,
                (1,),
                generator=generator
            ).item()
        )

        target = int(
            torch.randint(
                0,
                number_of_nodes,
                (1,),
                generator=generator
            ).item()
        )

        relation = int(
            torch.randint(
                0,
                number_of_relations,
                (1,),
                generator=generator
            ).item()
        )

        candidate = (
            source,
            target,
            relation
        )

        if candidate in positive_lookup:
            continue

        negative_sources.append(source)
        negative_targets.append(target)
        negative_relations.append(relation)

    if len(negative_sources) < number_of_negative_edges:
        raise RuntimeError(
            "Unable to generate enough negative edges. "
            f"Requested {number_of_negative_edges}, "
            f"generated {len(negative_sources)}."
        )

    negative_edge_index = torch.tensor(
        [
            negative_sources,
            negative_targets
        ],
        dtype=torch.long
    )

    negative_edge_type = torch.tensor(
        negative_relations,
        dtype=torch.long
    )

    return (
        negative_edge_index,
        negative_edge_type
    )


class RelationalGNN(nn.Module):
    def __init__(
        self,
        input_channels,
        hidden_channels,
        output_channels,
        number_of_relations,
        num_layers=3,
        dropout=0.2
    ):
        super().__init__()

        if num_layers < 2:
            raise ValueError(
                "num_layers must be at least 2."
            )

        self.dropout = dropout

        self.convs = nn.ModuleList()

        self.convs.append(
            RGCNConv(
                input_channels,
                hidden_channels,
                number_of_relations
            )
        )

        for _ in range(num_layers - 2):
            self.convs.append(
                RGCNConv(
                    hidden_channels,
                    hidden_channels,
                    number_of_relations
                )
            )

        self.convs.append(
            RGCNConv(
                hidden_channels,
                output_channels,
                number_of_relations
            )
        )

    def forward(
        self,
        x,
        edge_index,
        edge_type
    ):
        for layer_index, conv in enumerate(
            self.convs
        ):
            x = conv(
                x,
                edge_index,
                edge_type
            )

            if layer_index != len(
                self.convs
            ) - 1:
                x = F.relu(x)

                x = F.dropout(
                    x,
                    p=self.dropout,
                    training=self.training
                )

        return x


class DistMultDecoder(nn.Module):
    def __init__(
        self,
        embedding_dimension,
        number_of_relations
    ):
        super().__init__()

        self.relation_embeddings = nn.Parameter(
            torch.empty(
                number_of_relations,
                embedding_dimension
            )
        )

        nn.init.xavier_uniform_(
            self.relation_embeddings
        )

    def forward(
        self,
        node_embeddings,
        edge_index,
        edge_type
    ):
        source_embeddings = node_embeddings[
            edge_index[0]
        ]

        target_embeddings = node_embeddings[
            edge_index[1]
        ]

        relation_embeddings = (
            self.relation_embeddings[
                edge_type
            ]
        )

        scores = (
            source_embeddings
            * relation_embeddings
            * target_embeddings
        ).sum(dim=1)

        return scores


def calculate_metrics(
    scores,
    labels
):
    probabilities = torch.sigmoid(
        scores
    ).detach().cpu().numpy()

    labels = labels.detach().cpu().numpy()

    if len(np.unique(labels)) < 2:
        return 0.0, 0.0

    roc_auc = roc_auc_score(
        labels,
        probabilities
    )

    average_precision = average_precision_score(
        labels,
        probabilities
    )

    return (
        float(roc_auc),
        float(average_precision)
    )


def evaluate(
    model,
    decoder,
    features,
    message_edge_index,
    message_edge_type,
    positive_edge_index,
    positive_edge_type,
    negative_edge_index,
    negative_edge_type
):
    model.eval()
    decoder.eval()

    with torch.no_grad():
        embeddings = model(
            features,
            message_edge_index,
            message_edge_type
        )

        positive_scores = decoder(
            embeddings,
            positive_edge_index,
            positive_edge_type
        )

        negative_scores = decoder(
            embeddings,
            negative_edge_index,
            negative_edge_type
        )

        scores = torch.cat(
            [
                positive_scores,
                negative_scores
            ]
        )

        labels = torch.cat(
            [
                torch.ones_like(
                    positive_scores
                ),
                torch.zeros_like(
                    negative_scores
                )
            ]
        )

        roc_auc, average_precision = (
            calculate_metrics(
                scores,
                labels
            )
        )

    return (
        roc_auc,
        average_precision,
        embeddings
    )


def train_one_epoch(
    model,
    decoder,
    optimizer,
    loss_function,
    features,
    train_edge_index,
    train_edge_type,
    negative_edge_index,
    negative_edge_type
):
    model.train()
    decoder.train()

    optimizer.zero_grad()

    embeddings = model(
        features,
        train_edge_index,
        train_edge_type
    )

    positive_scores = decoder(
        embeddings,
        train_edge_index,
        train_edge_type
    )

    negative_scores = decoder(
        embeddings,
        negative_edge_index,
        negative_edge_type
    )

    scores = torch.cat(
        [
            positive_scores,
            negative_scores
        ]
    )

    labels = torch.cat(
        [
            torch.ones_like(
                positive_scores
            ),
            torch.zeros_like(
                negative_scores
            )
        ]
    )

    loss = loss_function(
        scores,
        labels
    )

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        list(model.parameters())
        + list(decoder.parameters()),
        max_norm=1.0
    )

    optimizer.step()

    return float(loss.item())


def main():
    set_seed(SEED)

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 50)
    print("        RGCN LINK PREDICTION TRAINING")
    print("=" * 50)
    print()

    (
        features,
        edge_index,
        edge_type,
        node_types,
        number_of_nodes,
        number_of_relations
    ) = load_data()

    print()
    print("Splitting edges...")

    (
        train_edge_index,
        train_edge_type,
        validation_edge_index,
        validation_edge_type,
        test_edge_index,
        test_edge_type
    ) = split_edges(
        edge_index,
        edge_type,
        train_ratio=TRAIN_RATIO,
        validation_ratio=VALIDATION_RATIO,
        seed=SEED
    )

    print(
        f"Training edges: "
        f"{train_edge_index.shape[1]}"
    )

    print(
        f"Validation edges: "
        f"{validation_edge_index.shape[1]}"
    )

    print(
        f"Test edges: "
        f"{test_edge_index.shape[1]}"
    )

    print()
    print("IMPORTANT:")
    print(
        "Only training edges are used for GNN message passing."
    )
    print(
        "Validation/test edges remain hidden during training."
    )

    print()
    print("Creating negative samples...")

    all_positive_edges = (
        edge_index,
        edge_type
    )

    train_negative_edge_index, train_negative_edge_type = (
        create_negative_edges(
            train_edge_index,
            train_edge_type,
            number_of_nodes,
            number_of_relations,
            number_of_negative_edges=(
                train_edge_index.shape[1]
                * NEGATIVE_RATIO
            ),
            all_positive_edges=all_positive_edges,
            seed=SEED + 1
        )
    )

    validation_negative_edge_index, validation_negative_edge_type = (
        create_negative_edges(
            validation_edge_index,
            validation_edge_type,
            number_of_nodes,
            number_of_relations,
            number_of_negative_edges=(
                validation_edge_index.shape[1]
                * NEGATIVE_RATIO
            ),
            all_positive_edges=all_positive_edges,
            seed=SEED + 2
        )
    )

    test_negative_edge_index, test_negative_edge_type = (
        create_negative_edges(
            test_edge_index,
            test_edge_type,
            number_of_nodes,
            number_of_relations,
            number_of_negative_edges=(
                test_edge_index.shape[1]
                * NEGATIVE_RATIO
            ),
            all_positive_edges=all_positive_edges,
            seed=SEED + 3
        )
    )

    print(
        f"Training negative edges: "
        f"{train_negative_edge_index.shape[1]}"
    )

    print(
        f"Validation negative edges: "
        f"{validation_negative_edge_index.shape[1]}"
    )

    print(
        f"Test negative edges: "
        f"{test_negative_edge_index.shape[1]}"
    )

    print()
    print("Creating RGCN model...")

    model = RelationalGNN(
        input_channels=features.shape[1],
        hidden_channels=HIDDEN_CHANNELS,
        output_channels=OUTPUT_CHANNELS,
        number_of_relations=number_of_relations,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    )

    decoder = DistMultDecoder(
        embedding_dimension=OUTPUT_CHANNELS,
        number_of_relations=number_of_relations
    )

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    decoder_parameters = sum(
        parameter.numel()
        for parameter in decoder.parameters()
        if parameter.requires_grad
    )

    print()
    print(model)
    print()
    print(
        f"GNN trainable parameters: "
        f"{total_parameters:,}"
    )

    print(
        f"Decoder trainable parameters: "
        f"{decoder_parameters:,}"
    )

    optimizer = torch.optim.Adam(
        list(model.parameters())
        + list(decoder.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    loss_function = nn.BCEWithLogitsLoss()

    print()
    print("=" * 50)
    print("              TRAINING")
    print("=" * 50)

    best_validation_auc = -1.0
    best_epoch = 0

    training_history = []

    for epoch in range(
        1,
        EPOCHS + 1
    ):
        loss = train_one_epoch(
            model,
            decoder,
            optimizer,
            loss_function,
            features,
            train_edge_index,
            train_edge_type,
            train_negative_edge_index,
            train_negative_edge_type
        )

        validation_auc, validation_ap, _ = evaluate(
            model,
            decoder,
            features,
            train_edge_index,
            train_edge_type,
            validation_edge_index,
            validation_edge_type,
            validation_negative_edge_index,
            validation_negative_edge_type
        )

        training_history.append(
            {
                "epoch": epoch,
                "loss": loss,
                "validation_roc_auc": validation_auc,
                "validation_average_precision": validation_ap
            }
        )

        if validation_auc > best_validation_auc:
            best_validation_auc = validation_auc
            best_epoch = epoch

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "decoder_state_dict": decoder.state_dict(),
                    "input_channels": features.shape[1],
                    "hidden_channels": HIDDEN_CHANNELS,
                    "output_channels": OUTPUT_CHANNELS,
                    "number_of_relations": number_of_relations,
                    "num_layers": NUM_LAYERS,
                    "dropout": DROPOUT,
                    "epoch": epoch,
                    "validation_roc_auc": validation_auc,
                    "validation_average_precision": validation_ap,
                    "seed": SEED
                },
                MODEL_FILE
            )

        if (
            epoch == 1
            or epoch % 5 == 0
            or epoch == EPOCHS
        ):
            print(
                f"Epoch {epoch:03d}/{EPOCHS} | "
                f"Loss: {loss:.4f} | "
                f"Val ROC-AUC: {validation_auc:.4f} | "
                f"Val AP: {validation_ap:.4f}"
            )

    print()
    print("=" * 50)
    print("          LOADING BEST MODEL")
    print("=" * 50)

    checkpoint = torch.load(
        MODEL_FILE,
        map_location="cpu",
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    decoder.load_state_dict(
        checkpoint["decoder_state_dict"]
    )

    print(
        f"Best epoch: "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Best validation ROC-AUC: "
        f"{checkpoint['validation_roc_auc']:.4f}"
    )

    print(
        f"Best validation AP: "
        f"{checkpoint['validation_average_precision']:.4f}"
    )

    print()
    print("=" * 50)
    print("             TEST EVALUATION")
    print("=" * 50)

    test_auc, test_ap, trained_embeddings = evaluate(
        model,
        decoder,
        features,
        train_edge_index,
        train_edge_type,
        test_edge_index,
        test_edge_type,
        test_negative_edge_index,
        test_negative_edge_type
    )

    print(
        f"Test ROC-AUC: "
        f"{test_auc:.4f}"
    )

    print(
        f"Test Average Precision: "
        f"{test_ap:.4f}"
    )

    print()
    print("=" * 50)
    print("       GENERATING TRAINED EMBEDDINGS")
    print("=" * 50)

    model.eval()

    with torch.no_grad():
        trained_embeddings = model(
            features,
            train_edge_index,
            train_edge_type
        )

    embedding_payload = {
        "embeddings": trained_embeddings,
        "node_types": node_types,
        "node_to_index": load_json(
            GRAPH_FILE
        )["node_to_index"],
        "index_to_node": load_json(
            GRAPH_FILE
        )["index_to_node"],
        "embedding_dimension": OUTPUT_CHANNELS,
        "number_of_nodes": number_of_nodes,
        "training_edges": int(
            train_edge_index.shape[1]
        ),
        "best_epoch": int(
            checkpoint["epoch"]
        ),
        "validation_roc_auc": float(
            checkpoint["validation_roc_auc"]
        ),
        "validation_average_precision": float(
            checkpoint["validation_average_precision"]
        ),
        "test_roc_auc": float(
            test_auc
        ),
        "test_average_precision": float(
            test_ap
        )
    }

    torch.save(
        embedding_payload,
        EMBEDDING_FILE
    )

    print(
        f"Embedding tensor shape: "
        f"{tuple(trained_embeddings.shape)}"
    )

    print(
        f"Embedding dimension: "
        f"{trained_embeddings.shape[1]}"
    )

    print()
    print("Model saved to:")
    print(MODEL_FILE)

    print()
    print("Trained embeddings saved to:")
    print(EMBEDDING_FILE)

    history_file = (
        PROJECT_ROOT
        / "outputs"
        / "reports"
        / "gnn_training_history.json"
    )

    history_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        history_file,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            {
                "configuration": {
                    "seed": SEED,
                    "epochs": EPOCHS,
                    "learning_rate": LEARNING_RATE,
                    "weight_decay": WEIGHT_DECAY,
                    "hidden_channels": HIDDEN_CHANNELS,
                    "output_channels": OUTPUT_CHANNELS,
                    "num_layers": NUM_LAYERS,
                    "dropout": DROPOUT,
                    "train_ratio": TRAIN_RATIO,
                    "validation_ratio": VALIDATION_RATIO,
                    "test_ratio": TEST_RATIO
                },
                "best_epoch": best_epoch,
                "best_validation_roc_auc": best_validation_auc,
                "test_roc_auc": test_auc,
                "test_average_precision": test_ap,
                "history": training_history
            },
            file,
            indent=2
        )

    print()
    print("Training history saved to:")
    print(history_file)

    print()
    print("=" * 50)
    print("          TRAINING COMPLETE")
    print("=" * 50)


if __name__ == "__main__":
    main()