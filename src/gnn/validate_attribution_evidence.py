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


def validate_edges(nodes, edges):
    node_lookup = {
        node["id"]: node
        for node in nodes
    }

    results = []

    for edge in edges:
        source_id = edge.get("source")
        target_id = edge.get("target")
        relation = edge.get("relation")

        source_node = node_lookup.get(source_id)
        target_node = node_lookup.get(target_id)

        errors = []
        warnings = []

        if source_node is None:
            errors.append("Source node does not exist")

        if target_node is None:
            errors.append("Target node does not exist")

        expected_types = EXPECTED_RELATIONSHIPS.get(relation)

        if expected_types is None:
            errors.append("Unknown relationship type")

        if source_node and target_node and expected_types:
            actual_types = (
                source_node.get("node_type"),
                target_node.get("node_type")
            )

            if actual_types != expected_types:
                errors.append(
                    "Unexpected node types: "
                    f"{actual_types[0]} -> {actual_types[1]}"
                )

        if not edge.get("case_ids"):
            warnings.append("No case provenance recorded")

        results.append({
            "edge_id": edge.get("id"),
            "source": source_id,
            "target": target_id,
            "relationship": relation,
            "errors": errors,
            "warnings": warnings,
            "valid": len(errors) == 0
        })

    return results


def main():
    nodes, edges = load_data()

    results = validate_edges(nodes, edges)

    valid_edges = sum(
        1 for result in results
        if result["valid"]
    )

    invalid_edges = len(results) - valid_edges

    provenance_warnings = sum(
        1 for result in results
        if result["warnings"]
    )

    print("\nATTRIBUTION EVIDENCE VALIDATION")
    print("=" * 65)

    print(f"Nodes checked: {len(nodes)}")
    print(f"Edges checked: {len(edges)}")
    print(f"Schema-valid edges: {valid_edges}")
    print(f"Schema-invalid edges: {invalid_edges}")
    print(f"Edges with provenance warnings: {provenance_warnings}")

    print("\nRELATIONSHIP VALIDATION")
    print("-" * 65)

    for result in results:
        status = "PASS" if result["valid"] else "FAIL"

        print(
            f"{status} | "
            f"{result['source']} "
            f"--{result['relationship']}--> "
            f"{result['target']}"
        )

        for error in result["errors"]:
            print(f"       ERROR: {error}")

        for warning in result["warnings"]:
            print(f"       WARNING: {warning}")

    print("\nVALIDATION SUMMARY")
    print("-" * 65)

    if invalid_edges == 0:
        print("All edges satisfy the configured relationship schema.")
    else:
        print("Some edges violate the configured relationship schema.")

    print(
        "Schema validity does not independently verify "
        "the factual truth of an edge."
    )


if __name__ == "__main__":
    main()