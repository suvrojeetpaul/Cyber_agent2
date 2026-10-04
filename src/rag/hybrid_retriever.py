import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

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
    / "documents"
)

BM25_FILE = (
    OUTPUT_DIR
    / "bm25_results.json"
)

DENSE_FILE = (
    OUTPUT_DIR
    / "dense_results.json"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "hybrid_results.json"
)

RRF_K = 60


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


def calculate_rrf_score(
    rank,
    k=RRF_K
):
    return 1.0 / (k + rank)


def build_hybrid_ranking(
    bm25_results,
    dense_results
):
    documents = {}

    for result in bm25_results:
        chunk_id = result["chunk_id"]

        if chunk_id not in documents:
            documents[chunk_id] = {
                "chunk_id": chunk_id,
                "document_id": result[
                    "document_id"
                ],
                "document_name": result[
                    "document_name"
                ],
                "source_path": result[
                    "source_path"
                ],
                "text": result["text"],
                "bm25_rank": None,
                "bm25_score": 0.0,
                "dense_rank": None,
                "dense_score": 0.0,
                "rrf_score": 0.0
            }

        documents[chunk_id][
            "bm25_rank"
        ] = result["rank"]

        documents[chunk_id][
            "bm25_score"
        ] = result["score"]

        documents[chunk_id][
            "rrf_score"
        ] += calculate_rrf_score(
            result["rank"]
        )

    for result in dense_results:
        chunk_id = result["chunk_id"]

        if chunk_id not in documents:
            documents[chunk_id] = {
                "chunk_id": chunk_id,
                "document_id": result[
                    "document_id"
                ],
                "document_name": result[
                    "document_name"
                ],
                "source_path": result[
                    "source_path"
                ],
                "text": result["text"],
                "bm25_rank": None,
                "bm25_score": 0.0,
                "dense_rank": None,
                "dense_score": 0.0,
                "rrf_score": 0.0
            }

        documents[chunk_id][
            "dense_rank"
        ] = result["rank"]

        documents[chunk_id][
            "dense_score"
        ] = result["score"]

        documents[chunk_id][
            "rrf_score"
        ] += calculate_rrf_score(
            result["rank"]
        )

    ranked_documents = sorted(
        documents.values(),
        key=lambda item: item["rrf_score"],
        reverse=True
    )

    for index, document in enumerate(
        ranked_documents
    ):
        document["hybrid_rank"] = index + 1

    return ranked_documents


def save_results(
    query,
    results
):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output = {
        "query": query,
        "retrieval_method": (
            "BM25 + dense retrieval "
            "with Reciprocal Rank Fusion"
        ),
        "rrf_k": RRF_K,
        "top_k": len(results),
        "results": results
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
    print()
    print("=" * 60)
    print("          HYBRID RAG RETRIEVAL")
    print("=" * 60)
    print()

    print("Loading BM25 results...")

    bm25_data = load_json(
        BM25_FILE
    )

    print(
        f"BM25 results loaded: "
        f"{len(bm25_data['results'])}"
    )

    print()
    print("Loading dense results...")

    dense_data = load_json(
        DENSE_FILE
    )

    print(
        f"Dense results loaded: "
        f"{len(dense_data['results'])}"
    )

    query = bm25_data["query"]

    print()
    print("Query:")
    print(query)

    print()
    print(
        "Applying Reciprocal Rank Fusion..."
    )

    results = build_hybrid_ranking(
        bm25_data["results"],
        dense_data["results"]
    )

    print()
    print("=" * 60)
    print("              HYBRID RESULTS")
    print("=" * 60)
    print()

    for result in results:
        print(
            f"Hybrid Rank: "
            f"{result['hybrid_rank']}"
        )

        print(
            f"Document: "
            f"{result['document_name']}"
        )

        print(
            f"RRF Score: "
            f"{result['rrf_score']:.6f}"
        )

        print(
            f"BM25 Rank: "
            f"{result['bm25_rank']}"
        )

        print(
            f"BM25 Score: "
            f"{result['bm25_score']:.4f}"
        )

        print(
            f"Dense Rank: "
            f"{result['dense_rank']}"
        )

        print(
            f"Dense Score: "
            f"{result['dense_score']:.4f}"
        )

        preview = (
            result["text"]
            .replace("\n", " ")
        )

        if len(preview) > 300:
            preview = (
                preview[:300]
                + "..."
            )

        print(
            f"Evidence: {preview}"
        )

        print("-" * 60)

    save_results(
        query,
        results
    )

    print()
    print("Results saved to:")
    print(OUTPUT_FILE)

    print()
    print(
        "Hybrid RAG retrieval complete."
    )


if __name__ == "__main__":
    main()