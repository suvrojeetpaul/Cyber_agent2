import json
from pathlib import Path
from collections import Counter


PROJECT_ROOT = Path(__file__).resolve().parents[2]

BENCHMARK_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "evaluation"
    / "benchmark_cases.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "graph"
    / "attribution"
)

NODES_FILE = OUTPUT_DIR / "nodes.json"
EDGES_FILE = OUTPUT_DIR / "edges.json"
SUMMARY_FILE = OUTPUT_DIR / "summary.json"


ALLOWED_NODE_TYPES = {
    "ip",
    "domain",
    "malware",
    "vulnerability",
    "technique",
    "threat_actor",
}


TYPE_NORMALIZATION = {
    "cve": "vulnerability",
    "attack_group": "threat_actor",
}


def load_json(file_path):

    if not file_path.exists():
        raise FileNotFoundError(
            f"Required file not found: {file_path}"
        )

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def normalize_node_type(node_type):

    node_type = node_type.strip().lower()

    return TYPE_NORMALIZATION.get(
        node_type,
        node_type
    )


def normalize_node_id(node_type, value):

    node_type = normalize_node_type(
        node_type
    )

    value = value.strip()

    return (
        f"{node_type}::"
        f"{value.lower()}"
    )


def create_node(
    node_type,
    value
):

    normalized_type = normalize_node_type(
        node_type
    )

    if normalized_type not in ALLOWED_NODE_TYPES:

        raise ValueError(
            f"Unsupported node type: "
            f"{node_type}"
        )

    return {
        "id": normalize_node_id(
            normalized_type,
            value
        ),
        "name": value.strip(),
        "node_type": normalized_type
    }


def main():

    print("=" * 70)
    print(
        "       BUILDING ATTRIBUTION HETEROGENEOUS GRAPH"
    )
    print("=" * 70)

    print()
    print("Loading benchmark...")

    benchmark = load_json(
        BENCHMARK_FILE
    )

    cases = benchmark.get(
        "cases",
        []
    )

    if not cases:

        raise RuntimeError(
            "No benchmark cases found."
        )

    print(
        f"Benchmark cases: {len(cases)}"
    )

    nodes = {}
    edges = {}

    case_statistics = []

    for case in cases:

        case_id = case[
            "case_id"
        ]

        triples = case.get(
            "triples",
            []
        )

        case_node_ids = set()
        case_edge_ids = set()

        for triple in triples:

            source_type = normalize_node_type(
                triple["source_type"]
            )

            target_type = normalize_node_type(
                triple["target_type"]
            )

            source = triple[
                "source"
            ].strip()

            target = triple[
                "target"
            ].strip()

            relation = triple[
                "relation"
            ].strip().lower()

            source_node = create_node(
                source_type,
                source
            )

            target_node = create_node(
                target_type,
                target
            )

            source_id = source_node[
                "id"
            ]

            target_id = target_node[
                "id"
            ]

            nodes[
                source_id
            ] = source_node

            nodes[
                target_id
            ] = target_node

            edge_id = (
                f"{source_id}"
                f"__{relation}__"
                f"{target_id}"
            )

            if edge_id not in edges:

                edges[edge_id] = {
                    "id": edge_id,
                    "source": source_id,
                    "target": target_id,
                    "relation": relation,
                    "source_type": source_type,
                    "target_type": target_type,
                    "case_ids": []
                }

            if case_id not in edges[
                edge_id
            ]["case_ids"]:

                edges[
                    edge_id
                ]["case_ids"].append(
                    case_id
                )

            case_node_ids.add(
                source_id
            )

            case_node_ids.add(
                target_id
            )

            case_edge_ids.add(
                edge_id
            )

        case_statistics.append(
            {
                "case_id": case_id,
                "ground_truth_actor":
                    case[
                        "ground_truth_actor"
                    ],
                "nodes": len(
                    case_node_ids
                ),
                "edges": len(
                    case_edge_ids
                )
            }
        )

    node_list = list(
        nodes.values()
    )

    edge_list = list(
        edges.values()
    )

    node_type_counts = Counter(
        node["node_type"]
        for node in node_list
    )

    relation_counts = Counter(
        edge["relation"]
        for edge in edge_list
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        NODES_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            node_list,
            file,
            indent=2,
            ensure_ascii=False
        )

    with open(
        EDGES_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            edge_list,
            file,
            indent=2,
            ensure_ascii=False
        )

    summary = {

        "graph_name":
            "CTI Attribution Heterogeneous Graph",

        "source":
            "Six curated CTI benchmark cases",

        "num_cases":
            len(cases),

        "num_nodes":
            len(node_list),

        "num_edges":
            len(edge_list),

        "node_type_counts":
            dict(node_type_counts),

        "relation_counts":
            dict(relation_counts),

        "cases":
            case_statistics
    }

    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False
        )

    print()
    print("=" * 70)
    print(
        "                 GRAPH SUMMARY"
    )
    print("=" * 70)

    print(
        f"Cases : {len(cases)}"
    )

    print(
        f"Nodes : {len(node_list)}"
    )

    print(
        f"Edges : {len(edge_list)}"
    )

    print()
    print(
        "Node types:"
    )

    for node_type, count in sorted(
        node_type_counts.items()
    ):

        print(
            f"  {node_type:<18} "
            f"{count}"
        )

    print()
    print(
        "Relations:"
    )

    for relation, count in sorted(
        relation_counts.items()
    ):

        print(
            f"  {relation:<18} "
            f"{count}"
        )

    print()
    print(
        "Per-case statistics:"
    )

    for case in case_statistics:

        print(
            f"  {case['case_id']} | "
            f"{case['ground_truth_actor']:<18} | "
            f"Nodes: {case['nodes']} | "
            f"Edges: {case['edges']}"
        )

    print()
    print(
        "Files saved:"
    )

    print(
        NODES_FILE
    )

    print(
        EDGES_FILE
    )

    print(
        SUMMARY_FILE
    )

    print()
    print(
        "Attribution graph construction complete."
    )


if __name__ == "__main__":
    main()