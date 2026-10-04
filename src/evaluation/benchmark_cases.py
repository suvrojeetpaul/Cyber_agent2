import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CTI_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "cti"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "evaluation"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "benchmark_cases.json"
)


def load_cti_reports():
    reports = []

    files = sorted(
        CTI_DIR.glob("*.json")
    )

    if not files:
        raise RuntimeError(
            f"No CTI JSON reports found in: {CTI_DIR}"
        )

    for file_path in files:
        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as file:
            report = json.load(file)

        reports.append(
            {
                "file": file_path.name,
                "data": report
            }
        )

    return reports


def build_case(report, case_number):
    data = report["data"]

    threat_actor = data.get(
        "threat_actor",
        ""
    )

    aliases = data.get(
        "aliases",
        []
    )

    triples = data.get(
        "triples",
        []
    )

    indicators = []

    for triple in triples:
        source = triple.get(
            "source",
            ""
        )

        target = triple.get(
            "target",
            ""
        )

        relation = triple.get(
            "relation",
            ""
        )

        source_type = triple.get(
            "source_type",
            ""
        )

        target_type = triple.get(
            "target_type",
            ""
        )

        if source:
            indicators.append(
                {
                    "type": source_type,
                    "value": source
                }
            )

        if target:
            indicators.append(
                {
                    "type": target_type,
                    "value": target
                })

    unique_indicators = []

    seen = set()

    for indicator in indicators:
        key = (
            indicator["type"],
            indicator["value"]
        )

        if key not in seen:
            seen.add(key)
            unique_indicators.append(
                indicator
            )

    narrative = data.get(
        "narrative",
        ""
    )

    case = {
        "case_id": (
            f"CASE-{case_number:03d}"
        ),
        "source_document": report[
            "file"
        ],
        "query": (
            f"Which threat actor is "
            f"associated with the "
            f"observed cyber activity "
            f"described in this case?"
        ),
        "ground_truth_actor": (
            threat_actor
        ),
        "aliases": aliases,
        "origin": data.get(
            "origin",
            ""
        ),
        "base_confidence": data.get(
            "confidence_base",
            None
        ),
        "indicators": unique_indicators,
        "narrative": narrative,
        "triples": triples
    }

    return case


def validate_case(case):
    required_fields = [
        "case_id",
        "source_document",
        "query",
        "ground_truth_actor",
        "indicators",
        "narrative",
        "triples"
    ]

    for field in required_fields:
        if field not in case:
            return False

    if not case["ground_truth_actor"]:
        return False

    if not case["narrative"]:
        return False

    return True


def build_benchmark():
    reports = load_cti_reports()

    cases = []

    case_number = 1

    for report in reports:
        case = build_case(
            report,
            case_number
        )

        if validate_case(case):
            cases.append(
                case
            )

        case_number += 1

    if not cases:
        raise RuntimeError(
            "No valid benchmark cases were created."
        )

    return cases


def save_benchmark(cases):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    benchmark = {
        "benchmark_name": (
            "CTI Threat Attribution "
            "Benchmark"
        ),
        "version": "0.1",
        "description": (
            "Controlled benchmark "
            "constructed from the "
            "curated CTI corpus."
        ),
        "case_count": len(cases),
        "cases": cases
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            benchmark,
            file,
            indent=2,
            ensure_ascii=False
        )


def main():
    print()
    print("=" * 60)
    print("       CTI ATTRIBUTION BENCHMARK")
    print("=" * 60)
    print()

    print("Loading CTI reports...")

    cases = build_benchmark()

    print(
        f"Valid benchmark cases: "
        f"{len(cases)}"
    )

    print()

    for case in cases:
        print(
            f"{case['case_id']} | "
            f"{case['ground_truth_actor']}"
        )

        print(
            f"Source: "
            f"{case['source_document']}"
        )

        print(
            f"Indicators: "
            f"{len(case['indicators'])}"
        )

        print(
            f"Graph triples: "
            f"{len(case['triples'])}"
        )

        print("-" * 60)

    save_benchmark(
        cases
    )

    print()
    print(
        "Benchmark saved to:"
    )
    print(OUTPUT_FILE)

    print()
    print(
        "Benchmark construction complete."
    )


if __name__ == "__main__":
    main()