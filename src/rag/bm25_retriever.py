import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi


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


def tokenize(text):
    text = text.lower()

    tokens = re.findall(
        r"[a-zA-Z0-9_.:/-]+",
        text
    )

    return tokens


def load_chunks():
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {CHUNKS_FILE}"
        )

    with open(
        CHUNKS_FILE,
        "r",
        encoding="utf-8"
    ) as file:
        chunks = json.load(file)

    if not chunks:
        raise RuntimeError(
            "chunks.json contains no documents."
        )

    return chunks


def build_bm25(chunks):
    tokenized_documents = []

    for chunk in chunks:
        tokens = tokenize(
            chunk["text"]
        )

        tokenized_documents.append(
            tokens
        )

    return BM25Okapi(
        tokenized_documents
    )


def retrieve(
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

    for index in ranked_indices[:top_k]:
        chunk = chunks[index]

        results.append(
            {
                "rank": len(results) + 1,
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk["document_id"],
                "document_name": chunk["document_name"],
                "source_path": chunk["source_path"],
                "score": float(scores[index]),
                "text": chunk["text"]
            }
        )

    return results


def save_results(
    query,
    results
):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        OUTPUT_DIR
        / "bm25_results.json"
    )

    output = {
        "query": query,
        "top_k": len(results),
        "results": results
    }

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False
        )

    return output_file


def main():
    print()
    print("=" * 50)
    print("          BM25 SPARSE RETRIEVAL")
    print("=" * 50)
    print()

    print("Loading CTI chunks...")

    chunks = load_chunks()

    print(
        f"Chunks loaded: {len(chunks)}"
    )

    print()
    print("Building BM25 index...")

    bm25 = build_bm25(
        chunks
    )

    print("BM25 index created.")

    query = (
        "Which threat actor is associated "
        "with SolarWinds and SUNBURST malware?"
    )

    print()
    print("Test query:")
    print(query)

    results = retrieve(
        bm25,
        chunks,
        query,
        top_k=5
    )

    print()
    print("=" * 50)
    print("              RESULTS")
    print("=" * 50)
    print()

    for result in results:
        print(
            f"Rank {result['rank']}"
        )

        print(
            f"Document: "
            f"{result['document_name']}"
        )

        print(
            f"Score: "
            f"{result['score']:.4f}"
        )

        print(
            f"Chunk ID: "
            f"{result['chunk_id']}"
        )

        preview = (
            result["text"]
            .replace("\n", " ")
        )

        if len(preview) > 300:
            preview = preview[:300] + "..."

        print(
            f"Evidence: {preview}"
        )

        print("-" * 50)

    output_file = save_results(
        query,
        results
    )

    print()
    print("Results saved to:")
    print(output_file)
    print()
    print("BM25 retrieval complete.")


if __name__ == "__main__":
    main()