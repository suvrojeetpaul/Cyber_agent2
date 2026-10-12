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

SCORING_FORMULA = (
    "100 * (0.7 * indicator_coverage "
    "+ 0.3 * graph_proximity_score)"
)


LIMITATIONS = [
    "Graph scores are heuristic ranking scores, not calibrated probabilities.",
    "Text retrieval relevance does not independently verify attribution.",
    "Actor-name mentions do not establish attribution.",
    "Indicator mentions do not establish a causal relationship with a threat actor.",
    "The current CTI corpus is small and curated.",
    "Lexical-vector cosine similarity is not a pretrained semantic embedding model.",
    "Graph edge provenance records associated case IDs but does not independently verify underlying claims.",
    "The current evidence checks do not perform semantic entailment or contradiction detection.",
    "Retrieved evidence may originate from the same curated case used to construct the graph.",
    "Global indicator mentions are not automatically candidate-specific attribution evidence.",
    "The retrieval scores and graph scores have not been calibrated into attribution probabilities.",
    "Graph paths may share nodes, edges, and case provenance and therefore must not be counted as independent evidence.",
    "Source independence has not been established.",
    "The current pipeline does not independently verify the factual accuracy of each CTI statement.",
    "Candidate-specific textual support is not established by actor-name or indicator co-occurrence alone."
]


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

    lexical_cosine_scores = []

    for chunk in chunks:
        document_vector = text_to_vector(
            chunk["text"],
            vocabulary
        )

        lexical_cosine_scores.append(
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

    lexical_cosine_ranking = sorted(
        range(len(chunks)),
        key=lambda index: lexical_cosine_scores[index],
        reverse=True
    )

    bm25_ranks = {
        index: rank
        for rank, index in enumerate(
            bm25_ranking,
            start=1
        )
    }

    lexical_cosine_ranks = {
        index: rank
        for rank, index in enumerate(
            lexical_cosine_ranking,
            start=1
        )
    }

    results = []

    for index, chunk in enumerate(chunks):
        bm25_score = float(
            bm25_scores[index]
        )

        lexical_cosine_score = float(
            lexical_cosine_scores[index]
        )

        rrf_score = (
            1.0 / (
                RRF_K + bm25_ranks[index]
            )
            +
            1.0 / (
                RRF_K + lexical_cosine_ranks[index]
            )
        )

        if bm25_score <= 0 and lexical_cosine_score <= 0:
            continue

        results.append(
            {
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk["document_id"],
                "document_name": chunk["document_name"],
                "source_path": chunk["source_path"],
                "bm25_rank": bm25_ranks[index],
                "bm25_score": bm25_score,
                "dense_rank": lexical_cosine_ranks[index],
                "dense_score": lexical_cosine_score,
                "lexical_cosine_rank": lexical_cosine_ranks[index],
                "lexical_cosine_score": lexical_cosine_score,
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


def build_indicator_pattern(indicator):
    value = str(indicator).strip()

    if not value:
        return None

    normalized_value = re.sub(
        r"\[\.\]",
        ".",
        value
    )

    escaped_value = re.escape(
        normalized_value
    )

    escaped_value = escaped_value.replace(
        r"\.",
        r"(?:\.|\[\.\])"
    )

    return re.compile(
        r"(?<![a-z0-9])"
        + escaped_value
        + r"(?![a-z0-9])",
        re.IGNORECASE
    )


def find_actor_mentions(actor, evidence):
    actor_name = str(actor).strip()

    if not actor_name:
        return []

    pattern = re.compile(
        r"(?<![a-z0-9])"
        + re.escape(actor_name)
        + r"(?![a-z0-9])",
        re.IGNORECASE
    )

    matching_evidence = []

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
                    "evidence_scope": "mention_only",
                    "candidate_specific_support_established": False,
                    "independently_verified": False
                }
            )

    return matching_evidence


def find_indicator_mentions(indicators, evidence):
    matching_evidence = []

    for item in evidence:
        matched_indicators = []

        for indicator in indicators:
            pattern = build_indicator_pattern(
                indicator
            )

            if pattern and pattern.search(
                item["text"]
            ):
                matched_indicators.append(
                    str(indicator).strip()
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
                    "evidence_scope": "global_retrieved_document",
                    "candidate_specific_support_established": False,
                    "independently_verified": False
                }
            )

    return matching_evidence


def collect_graph_provenance(graph_paths):
    case_ids = set()
    all_relationships_have_provenance = True
    relationship_count = 0

    for path in graph_paths:
        for relationship in path.get(
            "relationships",
            []
        ):
            relationship_count += 1

            relationship_case_ids = relationship.get(
                "case_ids",
                []
            )

            if relationship_case_ids:
                case_ids.update(
                    str(case_id)
                    for case_id in relationship_case_ids
                )
            else:
                all_relationships_have_provenance = False

    return {
        "associated_case_ids": sorted(case_ids),
        "relationship_count": relationship_count,
        "all_relationships_have_recorded_case_provenance": (
            all_relationships_have_provenance
            if relationship_count > 0
            else False
        ),
        "independent_source_count": None,
        "source_independence_status": (
            "not_assessed"
        ),
        "provenance_interpretation": (
            "Case IDs identify recorded graph provenance. "
            "They do not independently verify factual claims "
            "or establish independent sources."
        )
    }


def enrich_candidate(candidate, indicators, evidence):
    actor = candidate["actor"]

    actor_evidence = find_actor_mentions(
        actor,
        evidence
    )

    graph_paths = candidate.get(
        "graph_paths",
        []
    )

    graph_score = candidate.get(
        "graph_score"
    )

    coverage = candidate.get(
        "indicator_coverage"
    )

    proximity = candidate.get(
        "graph_proximity_score"
    )

    if coverage is None:
        matched_count = candidate.get(
            "matched_indicator_count",
            0
        )

        total_count = candidate.get(
            "total_indicator_count",
            len(indicators)
        )

        coverage = (
            matched_count / total_count
            if total_count > 0
            else 0.0
        )

    if proximity is None:
        proximity = None

    provenance = collect_graph_provenance(
        graph_paths
    )

    return {
        "rank": candidate.get(
            "rank"
        ),
        "actor": actor,
        "graph_score": graph_score,
        "graph_score_type": (
            "heuristic_not_probability"
        ),
        "graph_score_formula": SCORING_FORMULA,
        "indicator_coverage": round(
            float(coverage),
            4
        ),
        "graph_proximity_score": (
            round(float(proximity), 4)
            if proximity is not None
            else None
        ),
        "matched_indicator_count": candidate.get(
            "matched_indicator_count",
            0
        ),
        "total_indicator_count": candidate.get(
            "total_indicator_count",
            len(indicators)
        ),
        "matched_indicators": candidate.get(
            "matched_indicators",
            []
        ),
        "matched_path_count": candidate.get(
            "matched_path_count",
            len(graph_paths)
        ),
        "shortest_path_hops": candidate.get(
            "shortest_path_hops"
        ),
        "graph_paths": graph_paths,
        "graph_provenance": provenance,
        "text_evidence": actor_evidence,
        "actor_name_mentions": actor_evidence,
        "candidate_specific_evidence": [],
        "candidate_specific_evidence_status": (
            "not_established_by_current_mention_checks"
        ),
        "actor_mentioned_in_retrieved_evidence": (
            len(actor_evidence) > 0
        )
    }


def validate_report(report):
    errors = []
    warnings = []

    if not isinstance(
        report.get("input_indicators"),
        list
    ):
        errors.append(
            "input_indicators must be a list."
        )

    if report.get("candidate_count") != len(
        report.get("candidates", [])
    ):
        errors.append(
            "candidate_count does not match candidates length."
        )

    if report.get("retrieval", {}).get(
        "returned_evidence_count"
    ) != len(
        report.get("retrieval", {}).get(
            "evidence",
            []
        )
    ):
        errors.append(
            "returned_evidence_count does not match evidence length."
        )

    for candidate in report.get(
        "candidates",
        []
    ):
        if candidate.get(
            "graph_score_type"
        ) != "heuristic_not_probability":
            errors.append(
                f"{candidate.get('actor')}: invalid graph score type."
            )

        if candidate.get(
            "candidate_specific_evidence"
        ):
            warnings.append(
                f"{candidate.get('actor')}: candidate-specific evidence "
                "requires separate semantic and provenance validation."
            )

        if candidate.get(
            "graph_provenance",
            {}
        ).get(
            "independent_source_count"
        ) is not None:
            warnings.append(
                f"{candidate.get('actor')}: source independence must "
                "be justified before reporting a count."
            )

    return {
        "passed": len(errors) == 0,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings
    }


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

    global_indicator_mentions = find_indicator_mentions(
        indicators,
        hybrid_evidence
    )

    report_candidates = []

    for rank, candidate in enumerate(
        graph_candidates,
        start=1
    ):
        candidate_copy = dict(
            candidate
        )

        candidate_copy["rank"] = rank

        enriched_candidate = enrich_candidate(
            candidate_copy,
            indicators,
            hybrid_evidence
        )

        report_candidates.append(
            enriched_candidate
        )

    report = {
        "project": "Cyber Threat Attribution Research",
        "report_type": "Graph and Hybrid RAG Evidence Fusion",
        "input_indicators": indicators,
        "graph": {
            "scoring_method": graph_result.get(
                "scoring_method"
            ),
            "scoring_formula": SCORING_FORMULA,
            "score_interpretation": (
                "Heuristic ranking score; not a calibrated probability."
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
            "cosine_similarity_type": (
                "term-frequency lexical vectors; not pretrained embeddings"
            ),
            "rrf_k": RRF_K,
            "top_k": top_k,
            "returned_evidence_count": len(
                hybrid_evidence
            ),
            "evidence": hybrid_evidence
        },
        "global_indicator_mentions": global_indicator_mentions,
        "candidate_count": len(
            report_candidates
        ),
        "candidates": report_candidates,
        "validation": {
            "passed": True,
            "error_count": 0,
            "warning_count": 0,
            "errors": [],
            "warnings": []
        },
        "limitations": LIMITATIONS
    }

    report["validation"] = validate_report(
        report
    )

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
            f"Graph score: {candidate['graph_score']}/100"
        )

        print(
            f"Indicator coverage: "
            f"{candidate['matched_indicator_count']}/"
            f"{candidate['total_indicator_count']}"
        )

        print(
            f"Graph proximity: "
            f"{candidate['graph_proximity_score']}"
        )

        print(
            f"Graph paths: {candidate['matched_path_count']}"
        )

        print(
            f"Associated cases: "
            f"{candidate['graph_provenance']['associated_case_ids']}"
        )

        print(
            f"Independent source count: "
            f"{candidate['graph_provenance']['independent_source_count']}"
        )

        print(
            f"Actor-name mention chunks: "
            f"{len(candidate['actor_name_mentions'])}"
        )

        print(
            "Candidate-specific evidence: "
            f"{candidate['candidate_specific_evidence_status']}"
        )

    print()
    print(
        "Report validation: "
        f"{'PASS' if report['validation']['passed'] else 'FAIL'}"
    )

    for error in report["validation"]["errors"]:
        print("ERROR:", error)

    for warning in report["validation"]["warnings"]:
        print("WARNING:", warning)

    print()
    print("=" * 65)
    print("Report saved to:")
    print(OUTPUT_FILE)
    print("=" * 65)


if __name__ == "__main__":
    main()