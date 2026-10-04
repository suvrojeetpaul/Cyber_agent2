import json
import math
import re
from collections import Counter
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

OUTPUT_FILE = (
    OUTPUT_DIR
    / "dense_results.json"
)


def tokenize(text):
    text = text.lower()

    tokens = re.findall(
        r"[a-zA-Z0-9_.:/-]+",
        text
    )

    return tokens


def build_vocabulary(documents):
    vocabulary = {}

    for document in documents:
        tokens = tokenize(
            document
        )

        for token in tokens:
            if token not in vocabulary:
                vocabulary[token] = len(vocabulary)

    return vocabulary


def text_to_vector(text, vocabulary):
    vector = [
        0.0
    ] * len(vocabulary)

    tokens = tokenize(
        text
    )

    counts = Counter(
        tokens
    )

    for token, count in counts.items():
        if token in vocabulary:
            index = vocabulary[token]
            vector[index] = float(count)

    return vector


def cosine_similarity(
    vector_a,
    vector_b
):
    dot_product = 0.0
    norm_a = 0.0
    norm_b = 0.0

    for index in range(
        len(vector_a)
    ):
        dot_product += (
            vector_a[index]
            * vector_b[index]
        )

        norm_a += (
            vector_a[index]
            * vector_a[index]
        )

        norm_b += (
            vector_b[index]
            * vector_b[index]
        )

    norm_a = math.sqrt(
        norm_a
    )

    norm_b = math.sqrt(
        norm_b
    )

    if norm_a == 0.0:
        return 0.0

    if norm_b == 0.0:
        return 0.0

    return (
        dot_product
        / (norm_a * norm_b)
    )


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


def build_dense_index(chunks):
    documents = [
        chunk["text"]
        for chunk in chunks
    ]

    vocabulary = build_vocabulary(
        documents
    )

    vectors = []

    for document in documents:
        vector = text_to_vector(
            document,
            vocabulary
        )

        vectors.append(
            vector
        )

    return vocabulary, vectors


def retrieve(
    chunks,
    vocabulary,
    vectors,
    query,
    top_k=5
):
    query_vector = text_to_vector(
        query,
        vocabulary
    )

    scores = []

    for index, vector in enumerate(
        vectors
    ):
        score = cosine_similarity(
            query_vector,
            vector
        )

        scores.append(
            (
                index,
                score
            )
        )

    scores.sort(
        key=lambda item: item[1],
        reverse=True
    )

    results = []

    for index, score in scores[:top_k]:
        chunk = chunks[index]

        results.append(
            {
                "rank": len(results) + 1,
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk["document_id"],
                "document_name": chunk["document_name"],
                "source_path": chunk["source_path"],
                "score": float(score),
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

    output = {
        "query": query,
        "top_k": len(results),
        "retrieval_method": (
            "deterministic lexical "
            "vector cosine similarity"
        ),
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
    print("=" * 50)
    print("          DENSE RETRIEVAL")
    print("=" * 50)
    print()

    print("Loading CTI chunks...")

    chunks = load_chunks()

    print(
        f"Chunks loaded: {len(chunks)}"
    )

    print()
    print("Building dense representation...")

    vocabulary, vectors = (
        build_dense_index(
            chunks
        )
    )

    print(
        f"Vocabulary size: "
        f"{len(vocabulary)}"
    )

    print(
        f"Vector dimension: "
        f"{len(vocabulary)}"
    )

    query = (
        "Which threat actor is associated "
        "with SolarWinds and SUNBURST malware?"
    )

    print()
    print("Test query:")
    print(query)

    results = retrieve(
        chunks,
        vocabulary,
        vectors,
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
            preview = (
                preview[:300]
                + "..."
            )

        print(
            f"Evidence: {preview}"
        )

        print("-" * 50)

    save_results(
        query,
        results
    )

    print()
    print("Results saved to:")
    print(OUTPUT_FILE)

    print()
    print(
        "Dense retrieval complete."
    )


if __name__ == "__main__":
    main()