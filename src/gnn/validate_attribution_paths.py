import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

NODES_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "graph"
    / "attribution"
    / "nodes.json"
)

EDGES_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "graph"
    / "attribution"
    / "edges.json"
)

EXPECTED_RELATIONSHIPS = {
    "resolves_to": ("ip", "domain"),
    "drops": ("domain", "malware"),
    "exploited_by": ("vulnerability", "malware"),
    "attributed_to": ("malware", "threat_actor"),
    "uses": ("threat_actor", "technique")
}


def load_data():
    with open(NODES_FILE, "r", encoding="utf-8") as file:
        nodes = json.load(file)

    with open(EDGES_FILE, "r", encoding="utf-8") as file:
        edges = json.load(file)

    return nodes, edges


def build_adjacency(edges):
    adjacency = {}

    for edge in edges:
        adjacency.setdefault(edge["source"], []).append(edge)

    return adjacency


def validate_path(path_nodes, path_edges, node_lookup):
    errors = []
    warnings = []

    if len(path_nodes) != len(path_edges) + 1:
        errors.append(
            "The number of nodes does not match the number of edges."
        )

    for node_id in path_nodes:
        if node_id not in node_lookup:
            errors.append(f"Unknown node: {node_id}")

    for index, edge in enumerate(path_edges):
        if index + 1 >= len(path_nodes):
            break

        if edge.get("source") != path_nodes[index]:
            errors.append(
                f"Edge {index + 1} has an unexpected source."
            )

        if edge.get("target") != path_nodes[index + 1]:
            errors.append(
                f"Edge {index + 1} has an unexpected target."
            )

        relation = edge.get("relation")
        expected_types = EXPECTED_RELATIONSHIPS.get(relation)

        if expected_types is None:
            errors.append(
                f"Unknown relationship type: {relation}"
            )
            continue

        source_node = node_lookup.get(edge.get("source"))
        target_node = node_lookup.get(edge.get("target"))

        if source_node is None or target_node is None:
            continue

        actual_types = (
            source_node.get("node_type"),
            target_node.get("node_type")
        )

        if actual_types != expected_types:
            errors.append(
                f"Relationship {relation} has unexpected node types."
            )

        if not edge.get("case_ids"):
            warnings.append(
                f"Edge {index + 1} has no recorded case provenance."
            )

    if path_nodes:
        final_node = node_lookup.get(path_nodes[-1])

        if final_node is None:
            errors.append("The final node does not exist.")
        elif final_node.get("node_type") != "threat_actor":
            errors.append(
                "The path does not terminate at a threat actor."
            )

    return {
        "structurally_valid": len(errors) == 0,
        "provenance_complete": len(warnings) == 0,
        "errors": errors,
        "warnings": warnings
    }


def main():
    nodes, edges = load_data()

    node_lookup = {
        node["id"]: node
        for node in nodes
    }

    adjacency = build_adjacency(edges)

    paths_to_validate = [
        (
            "CASE-001",
            "185.225.17.42"
        ),
        (
            "CASE-002",
            "194.28.172.10"
        ),
        (
            "CASE-003",
            "175.45.176.8"
        ),
        (
            "CASE-004",
            "185.141.63.12"
        ),
        (
            "CASE-005",
            "198.51.100.44"
        ),
        (
            "CASE-006",
            "193.106.191.4"
        )
    ]

    print("\nATTRIBUTION PATH VALIDATION")
    print("=" * 65)

    total_valid = 0

    for case_id, start_node_name in paths_to_validate:
        start_nodes = [
            node for node in nodes
            if node.get("name") == start_node_name
        ]

        if not start_nodes:
            print(f"\n{case_id}: START NODE NOT FOUND")
            continue

        start_id = start_nodes[0]["id"]

        queue = [
            (start_id, [start_id], [])
        ]

        valid_case_paths = []

        while queue:
            current, path_nodes, path_edges = queue.pop(0)

            current_node = node_lookup[current]

            if current_node.get("node_type") == "threat_actor":
                edge_case_ids = [
                    set(edge.get("case_ids", []))
                    for edge in path_edges
                ]

                belongs_to_case = all(
                    case_id in case_ids
                    for case_ids in edge_case_ids
                )

                if belongs_to_case:
                    validation = validate_path(
                        path_nodes,
                        path_edges,
                        node_lookup
                    )

                    valid_case_paths.append(
                        (path_nodes, path_edges, validation)
                    )

                continue

            if len(path_edges) >= 5:
                continue

            for edge in adjacency.get(current, []):
                neighbor = edge["target"]

                if neighbor in path_nodes:
                    continue

                queue.append((
                    neighbor,
                    path_nodes + [neighbor],
                    path_edges + [edge]
                ))

        print(f"\n{case_id} | Starting indicator: {start_node_name}")

        if not valid_case_paths:
            print("  No actor-reaching path supported by this case's edge IDs.")
            continue

        for index, (path_nodes, path_edges, validation) in enumerate(
            valid_case_paths,
            start=1
        ):
            print(f"\n  Path {index}")

            for node_id, edge in zip(path_nodes, path_edges):
                print(
                    f"    {node_lookup[node_id]['name']} "
                    f"--{edge['relation']}-->"
                )

            print(
                f"    {node_lookup[path_nodes[-1]]['name']}"
            )

            print(
                "    Structural validation:",
                "PASS" if validation["structurally_valid"] else "FAIL"
            )

            print(
                "    Provenance recorded:",
                "YES" if validation["provenance_complete"] else "NO"
            )

            for error in validation["errors"]:
                print(f"    ERROR: {error}")

            for warning in validation["warnings"]:
                print(f"    WARNING: {warning}")

            if validation["structurally_valid"]:
                total_valid += 1

    print("\n" + "=" * 65)
    print(f"Structurally valid paths: {total_valid}")
    print(
        "This is structural path validation, not independent "
        "verification of attribution claims."
    )


if __name__ == "__main__":
    main()