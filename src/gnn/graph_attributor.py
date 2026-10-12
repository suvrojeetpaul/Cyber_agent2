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

        if normalized_query and (
            normalized_query == name
            or normalized_query == node_id
        ):
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

    node_lookup = {
        node["id"]: node
        for node in nodes
    }

    actor_paths = {}

    for start_id in start_node_ids:
        queue = deque([
            (start_id, 0, [start_id], [])
        ])

        best_depth = {start_id: 0}

        while queue:
            current, depth, path, relationships = queue.popleft()

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

            for neighbor, relationship, edge in adjacency.get(
                current, []
            ):
                if neighbor in path:
                    continue

                next_depth = depth + 1

                if next_depth > best_depth.get(
                    neighbor, float("inf")
                ):
                    continue

                best_depth[neighbor] = next_depth

                queue.append((
                    neighbor,
                    next_depth,
                    path + [neighbor],
                    relationships + [{
                        "relationship": relationship,
                        "edge": edge
                    }]
                ))

    return actor_paths, node_lookup


def calculate_graph_score(
    matched_indicator_count,
    total_indicator_count,
    shortest_path_hops,
    max_hops
):
    if total_indicator_count == 0:
        return {
            "score": 0.0,
            "coverage": 0.0,
            "proximity": 0.0
        }

    coverage = min(
        1.0,
        matched_indicator_count / total_indicator_count
    )

    proximity = max(
        0.0,
        min(
            1.0,
            (max_hops + 1 - shortest_path_hops)
            / (max_hops + 1)
        )
    )

    score = 100 * (
        0.7 * coverage
        + 0.3 * proximity
    )

    return {
        "score": round(score, 2),
        "coverage": round(coverage, 4),
        "proximity": round(proximity, 4)
    }


def attribute_indicators(indicators, max_hops=4):
    if max_hops < 0:
        raise ValueError("max_hops must be non-negative")

    graph_data, nodes, edges = load_attribution_data()

    matched_indicators = {}
    start_node_ids = []
    matched_indicator_to_nodes = {}

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

        matched_indicator_to_nodes[indicator] = [
            node["id"] for node in matches
        ]

        for node in matches:
            if node["id"] not in start_node_ids:
                start_node_ids.append(node["id"])

    if not start_node_ids:
        return {
            "matched_indicators": matched_indicators,
            "candidate_actors": [],
            "scoring_method": (
                "70% indicator coverage and 30% graph proximity"
            ),
            "message": (
                "No supplied indicators matched the attribution graph."
            )
        }

    actor_paths, node_lookup = find_actor_candidates(
        start_node_ids,
        nodes,
        edges,
        max_hops
    )

    total_indicator_count = len(indicators)
    candidates = []

    for actor_name, paths in actor_paths.items():
        best_path = min(
            paths,
            key=lambda item: item["hops"]
        )

        supporting_start_nodes = {
            path["start_node"] for path in paths
        }

        matched_indicator_names = []

        for indicator, node_ids in matched_indicator_to_nodes.items():
            if any(
                node_id in supporting_start_nodes
                for node_id in node_ids
            ):
                matched_indicator_names.append(indicator)

        matched_indicator_count = len(matched_indicator_names)

        score_details = calculate_graph_score(
            matched_indicator_count,
            total_indicator_count,
            best_path["hops"],
            max_hops
        )

        graph_paths = []
        distinct_case_ids = set()

        for path in paths:
            graph_nodes = [
                {
                    "id": node_id,
                    "name": node_lookup[node_id].get("name"),
                    "node_type": node_lookup[node_id].get("node_type")
                }
                for node_id in path["path"]
            ]

            graph_relationships = []

            for relation in path["relationships"]:
                edge = relation["edge"]
                case_ids = edge.get("case_ids", [])

                distinct_case_ids.update(case_ids)

                graph_relationships.append({
                    "relationship": relation["relationship"],
                    "source": edge["source"],
                    "target": edge["target"],
                    "case_ids": case_ids,
                    "provenance_recorded": bool(case_ids)
                })

            graph_paths.append({
                "start_indicator_node": path["start_node"],
                "hops": path["hops"],
                "nodes": graph_nodes,
                "relationships": graph_relationships
            })

        candidates.append({
            "actor": actor_name,
            "graph_score": score_details["score"],
            "graph_score_type": "heuristic_not_probability",
            "indicator_coverage": score_details["coverage"],
            "graph_proximity_score": score_details["proximity"],
            "scoring_formula": (
                "100 * (0.70 * indicator_coverage "
                "+ 0.30 * graph_proximity_score)"
            ),
            "matched_indicator_count": matched_indicator_count,
            "total_indicator_count": total_indicator_count,
            "matched_indicators": matched_indicator_names,
            "matched_path_count": len(graph_paths),
            "shortest_path_hops": best_path["hops"],
            "graph_paths": graph_paths,
            "associated_case_ids": sorted(distinct_case_ids),
            "independent_source_count": None,
            "source_independence_status": (
                "Not assessed; case IDs are not proof of "
                "independent sources."
            )
        })

    candidates.sort(
        key=lambda item: (
            -item["graph_score"],
            -item["matched_indicator_count"],
            item["shortest_path_hops"]
        )
    )

    return {
        "matched_indicators": matched_indicators,
        "candidate_actors": candidates,
        "scoring_method": (
            "70% indicator coverage and 30% graph proximity"
        ),
        "message": (
            "Graph scores are heuristic ranking scores, "
            "not calibrated attribution probabilities. "
            "Case IDs indicate recorded graph provenance, "
            "not independent verification."
        )
    }


if __name__ == "__main__":
    test_indicators = [
        "185.225.17.42",
        "avsvmcloud.com",
        "SUNBURST"
    ]

    result = attribute_indicators(test_indicators)

    print("\nCYBER AGENT - EVIDENCE-AWARE GRAPH ATTRIBUTION")
    print("=" * 60)

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
        print(f"   Graph score: {candidate['graph_score']}/100")
        print(f"   Coverage: {candidate['indicator_coverage']}")
        print(f"   Proximity: {candidate['graph_proximity_score']}")
        print(
            f"   Indicator coverage: "
            f"{candidate['matched_indicator_count']}/"
            f"{candidate['total_indicator_count']}"
        )
        print(f"   Matched paths: {candidate['matched_path_count']}")
        print(
            f"   Shortest path: "
            f"{candidate['shortest_path_hops']} hops"
        )
        print(
            f"   Associated case IDs: "
            f"{candidate['associated_case_ids']}"
        )
        print(
            f"   Independent sources: "
            f"{candidate['independent_source_count']} "
            "(not assessed)"
        )

        for path in candidate["graph_paths"]:
            print("\n   Graph trail:")

            for index, node in enumerate(path["nodes"]):
                print(
                    f"     {node['name']} "
                    f"[{node['node_type']}]"
                )

                if index < len(path["relationships"]):
                    relation = path["relationships"][index]

                    print(
                        f"       --{relation['relationship']}-->"
                    )

                    print(
                        f"         Supporting cases: "
                        f"{', '.join(relation['case_ids']) or 'Not recorded'}"
                    )

    print("\nScoring method:", result["scoring_method"])
    print(result["message"])