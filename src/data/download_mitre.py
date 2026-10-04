import json
from pathlib import Path

import requests


MITRE_URL = (
    "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/"
    "master/enterprise-attack/enterprise-attack.json"
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "mitre"
OUTPUT_FILE = OUTPUT_DIR / "enterprise-attack.json"


def download_mitre():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading MITRE ATT&CK Enterprise STIX data...")
    print(f"Source: {MITRE_URL}")

    response = requests.get(MITRE_URL, timeout=60)
    response.raise_for_status()

    try:
        data = response.json()
    except json.JSONDecodeError:
        raise ValueError("Downloaded file is not valid JSON.")

    if "objects" not in data:
        raise ValueError("Invalid MITRE STIX bundle: 'objects' field not found.")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)

    print("\nDownload completed successfully.")
    print(f"Saved to: {OUTPUT_FILE}")
    print(f"Total STIX objects: {len(data['objects'])}")

    object_types = {}

    for obj in data["objects"]:
        object_type = obj.get("type", "unknown")
        object_types[object_type] = object_types.get(object_type, 0) + 1

    print("\nSTIX object types:")

    for object_type, count in sorted(object_types.items()):
        print(f"  {object_type}: {count}")


if __name__ == "__main__":
    download_mitre()