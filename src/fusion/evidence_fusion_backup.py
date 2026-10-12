import json
import re
from pathlib import Path
from collections import Counter

from rank_bm25 import BM25Okapi

from src.gnn.graph_attributor import attribute_indicators


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
    / "outputs"
    / "reports"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "hybrid_attribution_report.json"
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


def tokenize(text):
    return re.findall(
        r"[a-zA-Z0-9_.:/-]+",
        str(text).lower()
    )


def calculate_bm25(chunks, query):
    tokenized_chunks = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    if not tokenized_chunks:
        return []

    model = BM25Okapi(
        tokenized_chunks
    )

    return model.get_scores(
        tokenize(query)
    )


def build_vocabulary(chunks):
    vocabulary = {}

    for chunk in chunks:
        for token in tokenize(chunk["text"]):
            if token not in vocabulary:
                vocabulary[token] = len(vocabulary)

    return vocabulary


def text_to_vector(text, vocabulary):
    vector = [0.0] * len(vocabulary)

    counts = Counter(
        tokenize(text)
    )

    for token, count in counts.items():
        if token in vocabulary:
            vector[vocabulary[token]] = float(count)

    return vector


def cosine_similarity(vector_a, vector_b):
    dot_product = sum(
        a * b
        for a, b in zip(vector_a, vector_b)
    )

    norm_a = sum(
        value * value
        for value in vector_a
    ) ** 0.5

    norm_b = sum(
        value * value
        for value in vector_b
    ) ** 0.5

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot_product / (
        norm_a * norm_b
    )


def retrieve_hybrid_evidence(
    chunks,
    query,
    top_k=5
):
    if not chunks or top_k <= 0:
        return []

    bm25_scores = calculate_bm25(
        chunks,
        query
    )

    vocabulary = build_vocabulary(
        chunks
    )

    query_vector = text_to_vector(
        query,
        vocabulary
    )

    dense_scores = []

    for chunk in chunks:
        document_vector = text_to_vector(
            chunk["text"],
            vocabulary
        )

        dense_scores.append(
            cosine_similarity(
                query_vector,
                document_vector
            )
        )

    bm25_ranking = sorted(
        range(len(chunks)),
        key=lambda index: bm25_scores[index],
        reverse=True
    )

    dense_ranking = sorted(
        range(len(chunks)),
        key=lambda index: dense_scores[index],
        reverse=True
    )

    bm25_ranks = {
        index: rank
        for rank, index in enumerate(
            bm25_ranking,
            start=1
        )
    }

    dense_ranks = {
        index: rank
        for rank, index in enumerate(
            dense_ranking,
            start=1
        )
    }

    results = []

    for index, chunk in enumerate(chunks):
        bm25_score = float(
            bm25_scores[index]
        )

        dense_score = float(
            dense_scores[index]
        )

        rrf_score = (
            1.0 / (
                RRF_K + bm25_ranks[index]
            )
            +
            1.0 / (
                RRF_K + dense_ranks[index]
            )
        )

        if bm25_score <= 0 and dense_score <= 0:
            continue

        results.append(
            {
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk["document_id"],
                "document_name": chunk["document_name"],
                "source_path": chunk["source_path"],
                "bm25_rank": bm25_ranks[index],
                "bm25_score": bm25_score,
                "dense_rank": dense_ranks[index],
                "dense_score": dense_score,
                "rrf_score": rrf_score,
                "text": chunk["text"]
            }
        )

    results.sort(
        key=lambda item: item["rrf_score"],
        reverse=True
    )

    return results[:top_k]


def build_evidence_query(indicators):
    return " ".join(
        str(indicator).strip()
        for indicator in indicators
        if str(indicator).strip()
    )


def find_actor_mentions(actor, evidence):
    actor_name = str(actor).strip()

    if not actor_name:
        return []

    matching_evidence = []

    pattern = re.compile(
        r"(?<![a-z0-9])"
        + re.escape(actor_name)
        + r"(?![a-z0-9])",
        re.IGNORECASE
    )

    for item in evidence:
        if pattern.search(item["text"]):
            matching_evidence.append(
                {
                    "chunk_id": item["chunk_id"],
                    "document_id": item["document_id"],
                    "document_name": item["document_name"],
                    "source_path": item["source_path"],
                    "rrf_score": item["rrf_score"],
                    "matched_text": item["text"],
                    "evidence_type": "actor_name_mention",
                    "independently_verified": False
                }
            )

    return matching_evidence


def find_indicator_mentions(indicators, evidence):
    matching_evidence = []

    for item in evidence:
        matched_indicators = []

        for indicator in indicators:
            indicator_value = str(indicator).strip()

            if not indicator_value:
                continue

            pattern = re.compile(
                r"(?<![a-z0-9])"
                + re.escape(indicator_value)
                + r"(?![a-z0-9])",
                re.IGNORECASE
            )

            if pattern.search(item["text"]):
                matched_indicators.append(
                    indicator_value
                )

        if matched_indicators:
            matching_evidence.append(
                {
                    "chunk_id": item["chunk_id"],
                    "document_id": item["document_id"],
                    "document_name": item["document_name"],
                    "source_path": item["source_path"],
                    "rrf_score": item["rrf_score"],
                    "matched_indicators": matched_indicators,
                    "evidence_type": "indicator_mention",
                    "independently_verified": False
                }
            )

    return matching_evidence


def create_report(
    indicators,
    max_hops=4,
    top_k=5
):
    chunks = load_json(
        CHUNKS_FILE
    )

    graph_result = attribute_indicators(
        indicators,
        max_hops=max_hops
    )

    graph_candidates = graph_result.get(
        "candidate_actors",
        []
    )

    query = build_evidence_query(
        indicators
    )

    hybrid_evidence = retrieve_hybrid_evidence(
        chunks,
        query,
        top_k=top_k
    )

    indicator_evidence = find_indicator_mentions(
        indicators,
        hybrid_evidence
    )

    report_candidates = []

    for rank, candidate in enumerate(
        graph_candidates,
        start=1
    ):
        actor = candidate["actor"]

        actor_evidence = find_actor_mentions(
            actor,
            hybrid_evidence
        )

        report_candidates.append(
            {
                "rank": rank,
                "actor": actor,
                "graph_score": candidate["graph_score"],
                "graph_score_type": (
                    "Heuristic score from indicator "
                    "coverage and graph proximity"
                ),
                "matched_indicator_count": candidate[
                    "matched_indicator_count"
                ],
                "total_indicator_count": candidate[
                    "total_indicator_count"
                ],
                "matched_indicators": candidate[
                    "matched_indicators"
                ],
                "matched_path_count": candidate[
                    "matched_path_count"
                ],
                "shortest_path_hops": candidate[
                    "shortest_path_hops"
                ],
                "graph_paths": candidate[
                    "graph_paths"
                ],
                "text_evidence": actor_evidence,
                "indicator_evidence": indicator_evidence,
                "actor_mentioned_in_retrieved_evidence": (
                    len(actor_evidence) > 0
                )
            }
        )

    report = {
        "project": "Cyber Threat Attribution Research",
        "report_type": "Graph and Hybrid RAG Evidence Fusion",
        "input_indicators": indicators,
        "graph": {
            "scoring_method": graph_result.get(
                "scoring_method"
            ),
            "message": graph_result.get(
                "message"
            ),
            "max_hops": max_hops
        },
        "retrieval": {
            "query": query,
            "method": (
                "BM25 and lexical-vector cosine similarity "
                "combined with Reciprocal Rank Fusion"
            ),
            "rrf_k": RRF_K,
            "top_k": top_k,
            "returned_evidence_count": len(
                hybrid_evidence
            ),
            "evidence": hybrid_evidence
        },
        "candidate_count": len(
            report_candidates
        ),
        "candidates": report_candidates,
        "limitations": [
            "Graph scores are heuristic ranking scores, not probabilities.",
            "Text retrieval relevance does not independently verify attribution.",
            "Actor-name mentions do not establish attribution.",
            "Indicator mentions do not establish a causal relationship with a threat actor.",
            "The current CTI corpus is small and curated.",
            "Lexical-vector cosine similarity is not a pretrained semantic embedding model.",
            "Graph edge provenance records associated case IDs but does not independently verify underlying claims.",
            "The current evidence checks do not perform semantic entailment or contradiction detection.",
            "Retrieved evidence may originate from the same curated case used to construct the graph.",
            "Indicator evidence is calculated across retrieved documents and is not candidate-specific support.",
            "The retrieval scores and graph scores have not been calibrated into attribution probabilities."
        ]
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False
        )

    return report


def main():
    indicators = [
        "185.225.17.42",
        "avsvmcloud.com",
        "SUNBURST"
    ]

    print()
    print("=" * 65)
    print("       GRAPH + HYBRID RAG EVIDENCE FUSION")
    print("=" * 65)

    print()
    print("Input indicators:")

    for indicator in indicators:
        print("-", indicator)

    print()
    print("Generating report...")

    report = create_report(
        indicators,
        max_hops=4,
        top_k=5
    )

    print()
    print("Matched graph indicators:")

    for indicator, matches in report[
        "graph"
    ].items():
        if indicator in ("max_hops", "scoring_method", "message"):
            continue

        print(
            f"{indicator}: {matches}"
        )

    print()
    print("Retrieved evidence:")

    for evidence in report["retrieval"]["evidence"]:
        print(
            f"  {evidence['document_name']} "
            f"| BM25: {evidence['bm25_score']:.4f} "
            f"| Lexical cosine: {evidence['dense_score']:.4f} "
            f"| RRF: {evidence['rrf_score']:.6f}"
        )

    print()
    print("Candidate threat actors:")

    for candidate in report["candidates"]:
        print()
        print(
            f"Rank: {candidate['rank']}"
        )

        print(
            f"Actor: {candidate['actor']}"
        )

        print(
            f"Graph score: "
            f"{candidate['graph_score']}/100"
        )

        print(
            f"Indicator coverage: "
            f"{candidate['matched_indicator_count']}/"
            f"{candidate['total_indicator_count']}"
        )

        print(
            f"Graph paths: "
            f"{candidate['matched_path_count']}"
        )

        print(
            f"Actor-name mention chunks: "
            f"{len(candidate['text_evidence'])}"
        )

        print(
            f"Indicator-evidence chunks: "
            f"{len(candidate['indicator_evidence'])}"
        )

        for evidence in candidate["text_evidence"]:
            print(
                f"  Actor mention source: "
                f"{evidence['document_name']}"
            )

    print()
    print("=" * 65)
    print("Report saved to:")
    print(OUTPUT_FILE)
    print("=" * 65)


if __name__ == "__main__":
    main()