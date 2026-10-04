import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi


PROJECT_ROOT = Path(__file__).resolve().parents[2]

BENCHMARK_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "evaluation"
    / "benchmark_cases.json"
)

CHUNKS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "documents"
    / "chunks.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "evaluation"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "rag_baseline_results.json"
)


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


def tokenize(text):

    text = text.lower()

    return re.findall(
        r"[a-zA-Z0-9_.:/-]+",
        text
    )


def load_benchmark():

    data = load_json(
        BENCHMARK_FILE
    )

    cases = data.get(
        "cases",
        []
    )

    if not cases:
        raise RuntimeError(
            "Benchmark contains no cases."
        )

    return cases


def load_chunks():

    chunks = load_json(
        CHUNKS_FILE
    )

    if not chunks:
        raise RuntimeError(
            "chunks.json contains no documents."
        )

    return chunks


def build_bm25(chunks):

    tokenized_documents = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    return BM25Okapi(
        tokenized_documents
    )


def normalize_document_name(name):

    name = Path(name).name

    name = re.sub(
        r"\.(json|txt)$",
        "",
        name,
        flags=re.IGNORECASE
    )

    return name.lower()


def build_actor_mapping(cases):

    mapping = {}

    for case in cases:

        document_name = case[
            "source_document"
        ]

        actor = case[
            "ground_truth_actor"
        ]

        normalized_name = (
            normalize_document_name(
                document_name
            )
        )

        mapping[
            normalized_name
        ] = actor

    return mapping


def get_actor_terms(case):

    terms = []

    actor = case.get(
        "ground_truth_actor",
        ""
    )

    if actor:
        terms.append(actor)

    aliases = case.get(
        "aliases",
        []
    )

    for alias in aliases:

        if alias:
            terms.append(alias)

    unique_terms = []

    seen = set()

    for term in terms:

        normalized = term.lower().strip()

        if (
            normalized
            and normalized not in seen
        ):

            seen.add(normalized)
            unique_terms.append(term)

    return unique_terms


def remove_actor_terms(
    text,
    actor_terms
):

    cleaned_text = text

    for actor in actor_terms:

        pattern = re.compile(
            r"\b"
            + re.escape(actor)
            + r"\b",
            re.IGNORECASE
        )

        cleaned_text = pattern.sub(
            " ",
            cleaned_text
        )

    cleaned_text = re.sub(
        r"\s+",
        " ",
        cleaned_text
    ).strip()

    return cleaned_text


def build_query(case):

    indicators = case.get(
        "indicators",
        []
    )

    query_parts = []

    for indicator in indicators:

        value = indicator.get(
            "value",
            ""
        )

        if not value:
            continue

        query_parts.append(
            value
        )

    query = " ".join(
        query_parts
    )

    actor_terms = get_actor_terms(
        case
    )

    query = remove_actor_terms(
        query,
        actor_terms
    )

    return query.strip()


def retrieve_case(
    bm25,
    chunks,
    query,
    top_k=5
):

    query_tokens = tokenize(
        query
    )

    scores = bm25.get_scores(
        query_tokens
    )

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True
    )

    results = []

    for rank, index in enumerate(
        ranked_indices[:top_k],
        start=1
    ):

        chunk = chunks[index]

        results.append(
            {
                "rank": rank,
                "chunk_id": chunk[
                    "chunk_id"
                ],
                "document_id": chunk[
                    "document_id"
                ],
                "document_name": chunk[
                    "document_name"
                ],
                "source_path": chunk[
                    "source_path"
                ],
                "score": float(
                    scores[index]
                ),
                "text": chunk[
                    "text"
                ]
            }
        )

    return results


def calculate_reciprocal_rank(
    results,
    ground_truth_document
):

    ground_truth_key = (
        normalize_document_name(
            ground_truth_document
        )
    )

    for result in results:

        result_key = (
            normalize_document_name(
                result["document_name"]
            )
        )

        if result_key == ground_truth_key:

            return 1.0 / result["rank"]

    return 0.0


def evaluate_case(
    case,
    bm25,
    chunks,
    actor_mapping
):

    case_id = case[
        "case_id"
    ]

    ground_truth_actor = case[
        "ground_truth_actor"
    ]

    ground_truth_document = case[
        "source_document"
    ]

    query = build_query(
        case
    )

    results = retrieve_case(
        bm25,
        chunks,
        query,
        top_k=5
    )

    predicted_actor = None

    if results:

        top_document = results[
            0
        ]["document_name"]

        normalized_top_document = (
            normalize_document_name(
                top_document
            )
        )

        predicted_actor = actor_mapping.get(
            normalized_top_document
        )

    ground_truth_key = (
        normalize_document_name(
            ground_truth_document
        )
    )

    retrieved_document_keys = [
        normalize_document_name(
            result["document_name"]
        )
        for result in results
    ]

    top_1_correct = (
        predicted_actor
        == ground_truth_actor
    )

    top_5_correct = (
        ground_truth_key
        in retrieved_document_keys
    )

    reciprocal_rank = (
        calculate_reciprocal_rank(
            results,
            ground_truth_document
        )
    )

    retrieved_actors = []

    for result in results:

        document_key = (
            normalize_document_name(
                result[
                    "document_name"
                ]
            )
        )

        actor = actor_mapping.get(
            document_key
        )

        if actor:

            retrieved_actors.append(
                actor
            )

    return {
        "case_id": case_id,
        "query": query,
        "ground_truth_actor":
            ground_truth_actor,
        "ground_truth_document":
            ground_truth_document,
        "predicted_actor":
            predicted_actor,
        "top_1_correct":
            top_1_correct,
        "top_5_correct":
            top_5_correct,
        "reciprocal_rank":
            reciprocal_rank,
        "retrieved_actors":
            retrieved_actors,
        "retrieval_results":
            results
    }


def calculate_metrics(
    case_results
):

    total_cases = len(
        case_results
    )

    top_1_correct = sum(
        1
        for result in case_results
        if result[
            "top_1_correct"
        ]
    )

    top_5_correct = sum(
        1
        for result in case_results
        if result[
            "top_5_correct"
        ]
    )

    mrr = sum(
        result[
            "reciprocal_rank"
        ]
        for result in case_results
    ) / total_cases

    top_1_accuracy = (
        top_1_correct
        / total_cases
    )

    top_5_accuracy = (
        top_5_correct
        / total_cases
    )

    return {
        "total_cases":
            total_cases,
        "top_1_accuracy":
            top_1_accuracy,
        "top_5_accuracy":
            top_5_accuracy,
        "mrr":
            mrr
    }


def save_results(
    case_results,
    metrics
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output = {

        "experiment":
            "RAG-only attribution baseline",

        "retrieval_method":
            "BM25",

        "query_construction":
            (
                "Indicator-based query with "
                "ground-truth actor names and "
                "aliases removed"
            ),

        "document_name_normalization":
            "json/txt extension normalization",

        "top_k":
            5,

        "metrics":
            metrics,

        "cases":
            case_results
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False
        )


def main():

    print("=" * 70)
    print(
        "             RAG-ONLY ATTRIBUTION BASELINE"
    )
    print("=" * 70)

    print()
    print("Loading benchmark...")

    cases = load_benchmark()

    print(
        f"Benchmark cases: {len(cases)}"
    )

    print()
    print("Loading CTI chunks...")

    chunks = load_chunks()

    print(
        f"CTI chunks: {len(chunks)}"
    )

    print()
    print("Building BM25 index...")

    bm25 = build_bm25(
        chunks
    )

    print(
        "BM25 index created."
    )

    actor_mapping = (
        build_actor_mapping(
            cases
        )
    )

    print()
    print("=" * 70)
    print(
        "                 CASE EVALUATION"
    )
    print("=" * 70)

    case_results = []

    for case in cases:

        result = evaluate_case(
            case,
            bm25,
            chunks,
            actor_mapping
        )

        case_results.append(
            result
        )

        print()
        print(
            result["case_id"]
        )

        print(
            f"Ground Truth : "
            f"{result['ground_truth_actor']}"
        )

        print(
            f"Prediction    : "
            f"{result['predicted_actor']}"
        )

        print(
            f"Top-1        : "
            f"{'PASS' if result['top_1_correct'] else 'FAIL'}"
        )

        print(
            f"Top-5        : "
            f"{'PASS' if result['top_5_correct'] else 'FAIL'}"
        )

        print(
            f"Reciprocal Rank: "
            f"{result['reciprocal_rank']:.4f}"
        )

        print(
            f"Query: "
            f"{result['query']}"
        )

        print()
        print(
            "Retrieved documents:"
        )

        for retrieval in result[
            "retrieval_results"
        ]:

            print(
                f"  "
                f"{retrieval['rank']}. "
                f"{retrieval['document_name']} "
                f"(score="
                f"{retrieval['score']:.4f})"
            )

    metrics = calculate_metrics(
        case_results
    )

    print()
    print("=" * 70)
    print(
        "                 RAG BASELINE RESULTS"
    )
    print("=" * 70)

    print(
        f"Total cases      : "
        f"{metrics['total_cases']}"
    )

    print(
        f"Top-1 Accuracy   : "
        f"{metrics['top_1_accuracy']:.4f}"
    )

    print(
        f"Top-5 Accuracy   : "
        f"{metrics['top_5_accuracy']:.4f}"
    )

    print(
        f"MRR              : "
        f"{metrics['mrr']:.4f}"
    )

    print()
    print(
        "Results saved to:"
    )

    save_results(
        case_results,
        metrics
    )

    print(
        OUTPUT_FILE
    )

    print()
    print(
        "RAG-only baseline evaluation complete."
    )


if __name__ == "__main__":
    main()