import json
import torch
from pathlib import Path
from collections import deque

PROJECT_ROOT = Path(__file__).resolve().parents[2]

GRAPH_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "graph"
    / "attribution"
    / "pyg_attribution_graph.pt"
)

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


def load_attribution_data():
    graph_data = torch.load(
        GRAPH_FILE,
        map_location="cpu",
        weights_only=False
    )

    with open(NODES_FILE, "r", encoding="utf-8") as file:
        nodes = json.load(file)

    with open(EDGES_FILE, "r", encoding="utf-8") as file:
        edges = json.load(file)

    return graph_data, nodes, edges


def normalize(value):
    return str(value).strip().lower().replace("[.]", ".")


def find_indicator_nodes(query, nodes):
    normalized_query = normalize(query)
    matched_nodes = []

    for node in nodes:
        name = normalize(node.get("name", ""))
        node_id = normalize(node.get("id", ""))

        if not name:
            continue

        if normalized_query == name or normalized_query == node_id:
            matched_nodes.append(node)

    return matched_nodes
def find_actor_candidates(start_node_ids, nodes, edges, max_hops=4):
    adjacency = {}

    for edge in edges:
        source = edge["source"]
        target = edge["target"]
        relationship = edge.get("relation", "related_to")

        adjacency.setdefault(source, []).append(
            (target, relationship, edge)
        )

        adjacency.setdefault(target, []).append(
            ("reverse::" + source, relationship, edge)
        )

    node_lookup = {
        node["id"]: node
        for node in nodes
    }

    actor_paths = {}

    for start_id in start_node_ids:
        queue = deque()
        queue.append((start_id, 0, [start_id], []))

        best_depth = {start_id: 0}

        while queue:
            current, depth, path, relationships = queue.popleft()

            if current.startswith("reverse::"):
                continue

            current_node = node_lookup.get(current, {})

            if current_node.get("node_type") == "threat_actor":
                actor_name = current_node.get("name", current)

                actor_paths.setdefault(actor_name, []).append({
                    "start_node": start_id,
                    "path": path,
                    "relationships": relationships,
                    "hops": depth
                })

            if depth >= max_hops:
                continue

            for neighbor, relationship, edge in adjacency.get(current, []):
                if neighbor.startswith("reverse::"):
                    actual_neighbor = neighbor.split("reverse::", 1)[1]
                    next_relationship = "reverse_" + relationship
                else:
                    actual_neighbor = neighbor
                    next_relationship = relationship

                if actual_neighbor in path:
                    continue

                next_depth = depth + 1

                if next_depth > best_depth.get(actual_neighbor, float("inf")):
                    continue

                best_depth[actual_neighbor] = next_depth

                queue.append((
                    actual_neighbor,
                    next_depth,
                    path + [actual_neighbor],
                    relationships + [{
                        "relationship": next_relationship,
                        "edge": edge
                    }]
                ))

    return actor_paths, node_lookup
def attribute_indicators(indicators, max_hops=4):
    graph_data, nodes, edges = load_attribution_data()

    matched_indicators = {}
    start_node_ids = []

    for indicator in indicators:
        matches = find_indicator_nodes(indicator, nodes)

        matched_indicators[indicator] = [
            {
                "id": node["id"],
                "name": node.get("name"),
                "node_type": node.get("node_type")
            }
            for node in matches
        ]

        for node in matches:
            if node["id"] not in start_node_ids:
                start_node_ids.append(node["id"])

    if not start_node_ids:
        return {
            "matched_indicators": matched_indicators,
            "candidate_actors": [],
            "message": "No supplied indicators matched the attribution graph."
        }

    actor_paths, node_lookup = find_actor_candidates(
        start_node_ids,
        nodes,
        edges,
        max_hops
    )

    candidates = []

    for actor_name, paths in actor_paths.items():
        best_path = min(
            paths,
            key=lambda item: item["hops"]
        )

        candidates.append({
            "actor": actor_name,
            "matched_path_count": len(paths),
            "shortest_path_hops": best_path["hops"],
            "graph_paths": [
                {
                    "nodes": [
                        {
                            "id": node_id,
                            "name": node_lookup[node_id].get("name"),
                            "node_type": node_lookup[node_id].get("node_type")
                        }
                        for node_id in path["path"]
                    ],
                    "relationships": [
                        {
                            "relationship": relation["relationship"],
                            "source": relation["edge"]["source"],
                            "target": relation["edge"]["target"]
                        }
                        for relation in path["relationships"]
                    ]
                }
                for path in paths
            ]
        })

    candidates.sort(
        key=lambda item: (
            -item["matched_path_count"],
            item["shortest_path_hops"]
        )
    )

    return {
        "matched_indicators": matched_indicators,
        "candidate_actors": candidates,
        "message": "Candidates ranked using graph connectivity, not calibrated attribution probabilities."
    }


if __name__ == "__main__":
    test_indicators = [
        "185.225.17.42",
        "avsvmcloud.com",
        "SUNBURST"
    ]

    result = attribute_indicators(test_indicators)

    print("\nCYBER AGENT — GRAPH ATTRIBUTION")
    print("=" * 50)

    print("\nMatched indicators:")

    for indicator, matches in result["matched_indicators"].items():
        print(f"\n{indicator}")

        if not matches:
            print("  No matching graph node")

        for match in matches:
            print(
                f"  {match['node_type']}: "
                f"{match['name']} ({match['id']})"
            )

    print("\nCandidate threat actors:")

    for rank, candidate in enumerate(
        result["candidate_actors"],
        start=1
    ):
        print(f"\n{rank}. {candidate['actor']}")
        print(
            "   Matched paths:",
            candidate["matched_path_count"]
        )
        print(
            "   Shortest path:",
            candidate["shortest_path_hops"],
            "hops"
        )

        for path in candidate["graph_paths"]:
            print("   Graph trail:")

            for index, node in enumerate(path["nodes"]):
                print(f"     {node['name']}")

                if index < len(path["relationships"]):
                    relation = path["relationships"][index]
                    print(f"       --{relation['relationship']}-->")

    print("\n" + result["message"])