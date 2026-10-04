import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MITRE_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "mitre"
    / "enterprise-attack.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "graph"
)

NODES_FILE = OUTPUT_DIR / "nodes.json"
EDGES_FILE = OUTPUT_DIR / "edges.json"


NODE_TYPE_MAP = {
    "intrusion-set": "threat_actor",
    "malware": "malware",
    "tool": "tool",
    "attack-pattern": "technique",
    "campaign": "campaign",
}


RELATIONSHIP_TYPES = {
    "uses",
    "attributed-to",
    "subtechnique-of",
}


def load_mitre_data():
    if not MITRE_FILE.exists():
        raise FileNotFoundError(
            f"MITRE dataset not found: {MITRE_FILE}"
        )

    with open(MITRE_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def build_nodes(objects):
    nodes = []

    for obj in objects:
        object_type = obj.get("type")

        if object_type not in NODE_TYPE_MAP:
            continue

        node = {
            "id": obj.get("id"),
            "name": obj.get("name"),
            "node_type": NODE_TYPE_MAP[object_type],
            "stix_type": object_type,
            "description": obj.get("description", ""),
        }

        nodes.append(node)

    return nodes


def build_edges(objects):
    edges = []

    for obj in objects:
        if obj.get("type") != "relationship":
            continue

        relationship_type = obj.get("relationship_type")

        if relationship_type not in RELATIONSHIP_TYPES:
            continue

        source_ref = obj.get("source_ref")
        target_ref = obj.get("target_ref")

        if not source_ref or not target_ref:
            continue

        edge = {
            "id": obj.get("id"),
            "source": source_ref,
            "target": target_ref,
            "relationship": relationship_type,
        }

        edges.append(edge)

    return edges


def main():
    print("Building normalized cybersecurity graph dataset...")

    data = load_mitre_data()
    objects = data.get("objects", [])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    nodes = build_nodes(objects)
    edges = build_edges(objects)

    with open(NODES_FILE, "w", encoding="utf-8") as file:
        json.dump(nodes, file, indent=2)

    with open(EDGES_FILE, "w", encoding="utf-8") as file:
        json.dump(edges, file, indent=2)

    print("\nGraph dataset created successfully.")
    print(f"Nodes: {len(nodes)}")
    print(f"Edges: {len(edges)}")

    print("\nNode types:")

    node_counts = {}

    for node in nodes:
        node_type = node["node_type"]
        node_counts[node_type] = node_counts.get(node_type, 0) + 1

    for node_type, count in sorted(node_counts.items()):
        print(f"  {node_type}: {count}")

    print("\nEdge types:")

    edge_counts = {}

    for edge in edges:
        relationship = edge["relationship"]
        edge_counts[relationship] = (
            edge_counts.get(relationship, 0) + 1
        )

    for relationship, count in sorted(edge_counts.items()):
        print(f"  {relationship}: {count}")

    print("\nOutput files:")
    print(f"  {NODES_FILE}")
    print(f"  {EDGES_FILE}")


if __name__ == "__main__":
    main()