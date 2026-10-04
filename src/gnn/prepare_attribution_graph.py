import json
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]

GRAPH_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "graph"
    / "attribution"
)

BENCHMARK_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "evaluation"
    / "benchmark_cases.json"
)

OUTPUT_FILE = (
    GRAPH_DIR
    / "pyg_attribution_graph.pt"
)


def load_json(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def normalize_type(node_type):

    if node_type == "cve":
        return "vulnerability"

    if node_type == "attack_group":
        return "threat_actor"

    return node_type


def make_node_id(
    node_type,
    value
):

    node_type = normalize_type(
        node_type
    )

    return (
        f"{node_type}::"
        f"{value.strip().lower()}"
    )


def main():

    print("=" * 70)
    print(
        "       PREPARING ATTRIBUTION GRAPH FOR PYTORCH GEOMETRIC"
    )
    print("=" * 70)

    nodes = load_json(
        GRAPH_DIR / "nodes.json"
    )

    edges = load_json(
        GRAPH_DIR / "edges.json"
    )

    benchmark = load_json(
        BENCHMARK_FILE
    )

    print()
    print(
        f"Nodes loaded : {len(nodes)}"
    )

    print(
        f"Edges loaded : {len(edges)}"
    )

    # ---------------------------------------------------------
    # NODE MAPPING
    # ---------------------------------------------------------

    node_to_index = {}
    index_to_node = {}

    for index, node in enumerate(nodes):

        node_id = node["id"]

        node_to_index[
            node_id
        ] = index

        index_to_node[
            str(index)
        ] = node_id

    # ---------------------------------------------------------
    # NODE TYPE ENCODING
    # ---------------------------------------------------------

    node_types = sorted(
        {
            node["node_type"]
            for node in nodes
        }
    )

    node_type_to_index = {
        node_type: index
        for index, node_type
        in enumerate(node_types)
    }

    node_type_indices = [
        node_type_to_index[
            node["node_type"]
        ]
        for node in nodes
    ]

    # ---------------------------------------------------------
    # EDGE TYPE ENCODING
    # ---------------------------------------------------------

    edge_types = sorted(
        {
            edge["relation"]
            for edge in edges
        }
    )

    edge_type_to_index = {
        edge_type: index
        for index, edge_type
        in enumerate(edge_types)
    }

    # ---------------------------------------------------------
    # EDGE INDEX
    # ---------------------------------------------------------

    edge_index = []
    edge_type_indices = []

    for edge in edges:

        source = edge["source"]
        target = edge["target"]

        if source not in node_to_index:
            raise ValueError(
                f"Unknown source node: {source}"
            )

        if target not in node_to_index:
            raise ValueError(
                f"Unknown target node: {target}"
            )

        edge_index.append(
            [
                node_to_index[source],
                node_to_index[target]
            ]
        )

        edge_type_indices.append(
            edge_type_to_index[
                edge["relation"]
            ]
        )

    edge_index_tensor = torch.tensor(
        edge_index,
        dtype=torch.long
    ).t().contiguous()

    edge_type_tensor = torch.tensor(
        edge_type_indices,
        dtype=torch.long
    )

    node_type_tensor = torch.tensor(
        node_type_indices,
        dtype=torch.long
    )

    # ---------------------------------------------------------
    # NODE FEATURES
    #
    # Features:
    #   1. Node type one-hot
    #   2. Name length
    #   3. Has name
    #
    # These are deliberately simple and reproducible.
    # ---------------------------------------------------------

    num_nodes = len(nodes)
    num_node_types = len(node_types)

    node_features = []

    max_name_length = max(
        [
            len(node["name"])
            for node in nodes
        ]
        + [1]
    )

    for node in nodes:

        features = [
            0.0
        ] * num_node_types

        type_index = node_type_to_index[
            node["node_type"]
        ]

        features[type_index] = 1.0

        name_length = (
            len(node["name"])
            / max_name_length
        )

        features.append(
            float(name_length)
        )

        features.append(
            1.0
        )

        node_features.append(
            features
        )

    x = torch.tensor(
        node_features,
        dtype=torch.float32
    )

    # ---------------------------------------------------------
    # ACTOR MAPPING
    # ---------------------------------------------------------

    actor_nodes = {}

    for node in nodes:

        if node["node_type"] == "threat_actor":

            actor_nodes[
                node["name"]
            ] = node_to_index[
                node["id"]
            ]

    actor_to_index = {
        actor: index
        for index, actor
        in enumerate(
            sorted(actor_nodes.keys())
        )
    }

    index_to_actor = {
        str(index): actor
        for actor, index
        in actor_to_index.items()
    }

    # ---------------------------------------------------------
    # CASE METADATA
    # ---------------------------------------------------------

    cases = []

    for case in benchmark["cases"]:

        case_node_indices = []

        for triple in case.get(
            "triples",
            []
        ):

            source_type = normalize_type(
                triple["source_type"]
            )

            target_type = normalize_type(
                triple["target_type"]
            )

            source_id = make_node_id(
                source_type,
                triple["source"]
            )

            target_id = make_node_id(
                target_type,
                triple["target"]
            )

            if source_id in node_to_index:

                case_node_indices.append(
                    node_to_index[
                        source_id
                    ]
                )

            if target_id in node_to_index:

                case_node_indices.append(
                    node_to_index[
                        target_id
                    ]
                )

        case_node_indices = sorted(
            set(case_node_indices)
        )

        actor = case[
            "ground_truth_actor"
        ]

        if actor not in actor_to_index:

            raise ValueError(
                f"Actor not found in graph: "
                f"{actor}"
            )

        cases.append(
            {
                "case_id":
                    case["case_id"],

                "ground_truth_actor":
                    actor,

                "actor_index":
                    actor_to_index[
                        actor
                    ],

                "node_indices":
                    case_node_indices,

                "num_nodes":
                    len(case_node_indices)
            }
        )

    # ---------------------------------------------------------
    # PACKAGE
    # ---------------------------------------------------------

    graph_data = {

        "x":
            x,

        "edge_index":
            edge_index_tensor,

        "edge_type":
            edge_type_tensor,

        "node_type":
            node_type_tensor,

        "num_nodes":
            num_nodes,

        "num_edges":
            len(edges),

        "node_to_index":
            node_to_index,

        "index_to_node":
            index_to_node,

        "node_type_to_index":
            node_type_to_index,

        "edge_type_to_index":
            edge_type_to_index,

        "actor_to_index":
            actor_to_index,

        "index_to_actor":
            index_to_actor,

        "cases":
            cases
    }

    torch.save(
        graph_data,
        OUTPUT_FILE
    )

    # ---------------------------------------------------------
    # OUTPUT
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "              PYTORCH GEOMETRIC DATASET"
    )
    print("=" * 70)

    print(
        f"Number of nodes       : "
        f"{num_nodes}"
    )

    print(
        f"Number of edges       : "
        f"{len(edges)}"
    )

    print(
        f"Node feature dimension: "
        f"{x.shape[1]}"
    )

    print(
        f"Node types            : "
        f"{len(node_types)}"
    )

    print(
        f"Edge types            : "
        f"{len(edge_types)}"
    )

    print()
    print(
        "Node type encoding:"
    )

    for node_type, index in (
        node_type_to_index.items()
    ):

        print(
            f"  {index} -> {node_type}"
        )

    print()
    print(
        "Edge type encoding:"
    )

    for relation, index in (
        edge_type_to_index.items()
    ):

        print(
            f"  {index} -> {relation}"
        )

    print()
    print(
        "Actor encoding:"
    )

    for actor, index in (
        actor_to_index.items()
    ):

        print(
            f"  {index} -> {actor}"
        )

    print()
    print(
        "Case mapping:"
    )

    for case in cases:

        print(
            f"  {case['case_id']} -> "
            f"{case['ground_truth_actor']} | "
            f"nodes={case['num_nodes']}"
        )

    print()
    print(
        "PyTorch Geometric graph saved to:"
    )

    print(
        OUTPUT_FILE
    )

    print()
    print(
        "Attribution graph preparation complete."
    )


if __name__ == "__main__":
    main()