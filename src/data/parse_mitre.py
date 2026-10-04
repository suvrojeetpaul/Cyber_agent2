import json
from pathlib import Path
from collections import Counter


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MITRE_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "mitre"
    / "enterprise-attack.json"
)


def load_mitre_data():
    if not MITRE_FILE.exists():
        raise FileNotFoundError(
            f"MITRE ATT&CK dataset not found: {MITRE_FILE}"
        )

    with open(MITRE_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def inspect_objects(data):
    objects = data.get("objects", [])

    print("\nMITRE ATT&CK DATASET INSPECTION")
    print("=" * 50)

    print(f"Total objects: {len(objects)}")

    object_types = Counter(
        obj.get("type", "unknown")
        for obj in objects
    )

    print("\nObject counts:")
    print("-" * 50)

    for object_type, count in sorted(object_types.items()):
        print(f"{object_type:<35} {count}")

    print("\nImportant entity counts:")
    print("-" * 50)

    important_types = [
        "intrusion-set",
        "malware",
        "tool",
        "attack-pattern",
        "campaign",
        "relationship",
    ]

    for object_type in important_types:
        print(
            f"{object_type:<35} "
            f"{object_types.get(object_type, 0)}"
        )


def inspect_relationships(data):
    relationships = [
        obj
        for obj in data.get("objects", [])
        if obj.get("type") == "relationship"
    ]

    relationship_types = Counter(
        relationship.get("relationship_type", "unknown")
        for relationship in relationships
    )

    print("\nRelationship types:")
    print("-" * 50)

    for relationship_type, count in relationship_types.most_common():
        print(f"{relationship_type:<35} {count}")


def main():
    print("Loading MITRE ATT&CK Enterprise STIX data...")

    data = load_mitre_data()

    inspect_objects(data)
    inspect_relationships(data)

    print("\nInspection completed successfully.")


if __name__ == "__main__":
    main()