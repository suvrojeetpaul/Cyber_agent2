import json
from pathlib import Path
from collections import Counter
import networkx as nx


PROJECT_ROOT = Path(__file__).resolve().parents[2]

GRAPH_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "graph"
    / "attribution"
)

NODES_FILE = GRAPH_DIR / "nodes.json"
EDGES_FILE = GRAPH_DIR / "edges.json"
BENCHMARK_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "evaluation"
    / "benchmark_cases.json"
)


def load_json(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def main():

    print("=" * 70)
    print(
        "       ATTRIBUTION GRAPH VALIDATION"
    )
    print("=" * 70)

    nodes = load_json(
        NODES_FILE
    )

    edges = load_json(
        EDGES_FILE
    )

    benchmark = load_json(
        BENCHMARK_FILE
    )

    node_ids = [
        node["id"]
        for node in nodes
    ]

    print()
    print(
        f"Nodes loaded : {len(nodes)}"
    )

    print(
        f"Edges loaded : {len(edges)}"
    )

    # ---------------------------------------------------------
    # 1. Duplicate node check
    # ---------------------------------------------------------

    duplicate_nodes = [
        node_id
        for node_id, count in Counter(
            node_ids
        ).items()
        if count > 1
    ]

    # ---------------------------------------------------------
    # 2. Node type check
    # ---------------------------------------------------------

    expected_types = {
        "ip",
        "domain",
        "malware",
        "vulnerability",
        "technique",
        "threat_actor"
    }

    actual_types = {
        node["node_type"]
        for node in nodes
    }

    missing_types = (
        expected_types
        - actual_types
    )

    unexpected_types = (
        actual_types
        - expected_types
    )

    # ---------------------------------------------------------
    # 3. Edge endpoint validation
    # ---------------------------------------------------------

    node_id_set = set(
        node_ids
    )

    invalid_edges = []

    for edge in edges:

        if edge["source"] not in node_id_set:

            invalid_edges.append(
                (
                    edge["id"],
                    "missing source"
                )
            )

        if edge["target"] not in node_id_set:

            invalid_edges.append(
                (
                    edge["id"],
                    "missing target"
                )
            )

    # ---------------------------------------------------------
    # 4. Build NetworkX graph
    # ---------------------------------------------------------

    graph = nx.MultiDiGraph()

    for node in nodes:

        graph.add_node(
            node["id"],
            node_type=node["node_type"],
            name=node["name"]
        )

    for edge in edges:

        graph.add_edge(
            edge["source"],
            edge["target"],
            relation=edge["relation"],
            edge_id=edge["id"]
        )

    # ---------------------------------------------------------
    # 5. Validate ground-truth actors
    # ---------------------------------------------------------

    actor_nodes = {
        node["name"].lower(): node["id"]
        for node in nodes
        if node["node_type"]
        == "threat_actor"
    }

    missing_actors = []

    for case in benchmark["cases"]:

        actor = case[
            "ground_truth_actor"
        ]

        if actor.lower() not in actor_nodes:

            missing_actors.append(
                actor
            )

    # ---------------------------------------------------------
    # 6. Validate case graph connectivity
    # ---------------------------------------------------------

    case_results = []

    for case in benchmark["cases"]:

        actor = case[
            "ground_truth_actor"
        ]

        actor_id = actor_nodes.get(
            actor.lower()
        )

        case_node_ids = set()

        for triple in case.get(
            "triples",
            []
        ):

            source_type = triple[
                "source_type"
            ]

            target_type = triple[
                "target_type"
            ]

            source = triple[
                "source"
            ]

            target = triple[
                "target"
            ]

            source_type = (
                "vulnerability"
                if source_type == "cve"
                else
                "threat_actor"
                if source_type == "attack_group"
                else source_type
            )

            target_type = (
                "vulnerability"
                if target_type == "cve"
                else
                "threat_actor"
                if target_type == "attack_group"
                else target_type
            )

            source_id = (
                f"{source_type}::"
                f"{source.lower()}"
            )

            target_id = (
                f"{target_type}::"
                f"{target.lower()}"
            )

            case_node_ids.add(
                source_id
            )

            case_node_ids.add(
                target_id
            )

        reachable = False

        if (
            actor_id is not None
            and case_node_ids
        ):

            for node_id in case_node_ids:

                if node_id == actor_id:

                    reachable = True
                    break

                if node_id in graph:

                    try:

                        if nx.has_path(
                            graph,
                            node_id,
                            actor_id
                        ):

                            reachable = True
                            break

                    except nx.NetworkXError:

                        pass

        case_results.append(
            {
                "case_id":
                    case["case_id"],

                "actor":
                    actor,

                "nodes":
                    len(case_node_ids),

                "actor_present":
                    actor_id is not None,

                "actor_reachable":
                    reachable
            }
        )

    # ---------------------------------------------------------
    # 7. Print results
    # ---------------------------------------------------------

    print()
    print("-" * 70)
    print("NODE VALIDATION")
    print("-" * 70)

    print(
        f"Duplicate nodes : "
        f"{len(duplicate_nodes)}"
    )

    print(
        f"Missing types   : "
        f"{len(missing_types)}"
    )

    print(
        f"Unexpected types: "
        f"{len(unexpected_types)}"
    )

    print()
    print("-" * 70)
    print("EDGE VALIDATION")
    print("-" * 70)

    print(
        f"Invalid edges   : "
        f"{len(invalid_edges)}"
    )

    print()
    print("-" * 70)
    print("ACTOR VALIDATION")
    print("-" * 70)

    print(
        f"Threat actors in graph: "
        f"{len(actor_nodes)}"
    )

    print(
        f"Missing benchmark actors: "
        f"{len(missing_actors)}"
    )

    if missing_actors:

        for actor in missing_actors:

            print(
                f"  MISSING: {actor}"
            )

    print()
    print("-" * 70)
    print("CASE CONNECTIVITY")
    print("-" * 70)

    all_cases_valid = True

    for result in case_results:

        status = (
            "PASS"
            if (
                result["actor_present"]
                and
                result["actor_reachable"]
            )
            else
            "FAIL"
        )

        if status == "FAIL":

            all_cases_valid = False

        print(
            f"{result['case_id']} | "
            f"{result['actor']:<18} | "
            f"Nodes: "
            f"{result['nodes']} | "
            f"Actor present: "
            f"{result['actor_present']} | "
            f"Reachable: "
            f"{result['actor_reachable']} | "
            f"{status}"
        )

    # ---------------------------------------------------------
    # 8. Final status
    # ---------------------------------------------------------

    validation_passed = (
        len(duplicate_nodes) == 0
        and len(missing_types) == 0
        and len(unexpected_types) == 0
        and len(invalid_edges) == 0
        and len(missing_actors) == 0
        and all_cases_valid
    )

    print()
    print("=" * 70)

    if validation_passed:

        print(
            "VALIDATION STATUS: PASSED"
        )

        print()
        print(
            "The attribution graph is "
            "internally consistent."
        )

        print(
            "All benchmark actors are "
            "present and reachable."
        )

    else:

        print(
            "VALIDATION STATUS: FAILED"
        )

        print()
        print(
            "Graph inconsistencies "
            "must be fixed before "
            "GNN attribution training."
        )

    print("=" * 70)


if __name__ == "__main__":
    main()