import json
from pathlib import Path

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, CrossEncoder


# ============================================================
# CONFIGURATION
# ============================================================

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

RERANKER_MODEL_NAME = (
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

COLLECTION_NAME = "kapston_documents_v2"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

QDRANT_PATH = (
        PROJECT_ROOT
        / "storage"
        / "qdrant"
)

QUESTIONS_PATH = (
        PROJECT_ROOT
        / "evaluation"
        / "questions.json"
)

# Dense baseline returns 5 directly.
DENSE_TOP_K = 5

# Reranking gets a much larger candidate pool.
RERANK_CANDIDATE_K = 20

# Both systems are ultimately evaluated at Top 5.
FINAL_TOP_K = 5

MIN_RELEVANCE_SCORE = 0.20


# ============================================================
# LOAD MODELS
# ============================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_NAME
)

print("Loading cross-encoder reranker...")

reranker = CrossEncoder(
    RERANKER_MODEL_NAME
)


# ============================================================
# LOAD QUESTIONS
# ============================================================

def load_questions() -> list[dict]:

    with open(
            QUESTIONS_PATH,
            "r",
            encoding="utf-8"
    ) as file:

        data = json.load(file)

    # Support either:
    #
    # [
    #   {...},
    #   {...}
    # ]
    #
    # or:
    #
    # {
    #   "questions": [...]
    # }

    if isinstance(data, list):
        return data

    if isinstance(data, dict):

        if "questions" in data:
            return data["questions"]

    raise ValueError(
        "Unsupported questions.json structure."
    )


# ============================================================
# HELPERS FOR EVALUATION SCHEMA
# ============================================================

def get_question_text(
        item: dict
) -> str:

    for key in (
            "question",
            "query"
    ):

        if key in item:
            return item[key]

    raise KeyError(
        "Question item does not contain "
        "'question' or 'query'."
    )


def get_expected_pages(
        item: dict
) -> list[int]:

    # Our evaluation file may use slightly different naming,
    # so support the common variants.

    for key in (
            "expected_pages",
            "relevant_pages",
            "pages"
    ):

        if key in item:

            pages = item[key]

            if pages is None:
                return []

            return [
                int(page)
                for page in pages
            ]

    return []


def should_retrieve(
        item: dict
) -> bool:

    if "should_retrieve" in item:
        return bool(
            item["should_retrieve"]
        )

    # Fallback:
    # if expected evidence exists, retrieval is expected.
    return bool(
        get_expected_pages(item)
    )


def get_category(
        item: dict
) -> str:

    return item.get(
        "category",
        item.get(
            "type",
            "unknown"
        )
    )


# ============================================================
# DENSE RETRIEVAL
# ============================================================

def dense_retrieve(
        client: QdrantClient,
        query: str,
        limit: int
) -> list[dict]:

    query_embedding = embedding_model.encode(
        query,
        normalize_embeddings=True
    )

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding.tolist(),
        limit=limit,
        with_payload=True
    )

    results = []

    for point in response.points:

        dense_score = float(
            point.score
        )

        if dense_score < MIN_RELEVANCE_SCORE:
            continue

        payload = point.payload or {}

        results.append({
            "id":
                point.id,

            "dense_score":
                dense_score,

            "page":
                payload.get("page"),

            "chunk":
                payload.get("chunk"),

            "text":
                payload.get(
                    "text",
                    ""
                )
        })

    return results


# ============================================================
# CROSS-ENCODER RERANKING
# ============================================================

def rerank(
        query: str,
        candidates: list[dict]
) -> list[dict]:

    if not candidates:
        return []

    pairs = [
        [
            query,
            candidate["text"]
        ]
        for candidate in candidates
    ]

    scores = reranker.predict(
        pairs,
        show_progress_bar=False
    )

    reranked = []

    for candidate, score in zip(
            candidates,
            scores
    ):

        result = candidate.copy()

        result["reranker_score"] = float(
            score
        )

        reranked.append(
            result
        )

    reranked.sort(
        key=lambda result:
        result["reranker_score"],
        reverse=True
    )

    return reranked


# ============================================================
# FIND FIRST RELEVANT RANK
# ============================================================

def first_relevant_rank(
        results: list[dict],
        expected_pages: list[int]
):
    """
    Return the rank of the first result whose page is one of
    the expected evidence pages.

    Returns None when no expected page occurs.
    """

    if not expected_pages:
        return None

    expected = set(
        expected_pages
    )

    for rank, result in enumerate(
            results,
            start=1
    ):

        if result["page"] in expected:
            return rank

    return None


# ============================================================
# RECALL@K
# ============================================================

def evidence_found(
        results: list[dict],
        expected_pages: list[int],
        k: int = FINAL_TOP_K
) -> bool:

    if not expected_pages:
        return False

    expected = set(
        expected_pages
    )

    retrieved_pages = {
        result["page"]
        for result in results[:k]
    }

    return bool(
        expected.intersection(
            retrieved_pages
        )
    )


# ============================================================
# RECIPROCAL RANK
# ============================================================

def reciprocal_rank(
        rank
) -> float:

    if rank is None:
        return 0.0

    return 1.0 / rank


# ============================================================
# MAIN EVALUATION
# ============================================================

def evaluate():

    questions = load_questions()

    print(
        f"\nLoaded {len(questions)} questions."
    )

    client = QdrantClient(
        path=str(QDRANT_PATH)
    )

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    supported_total = 0
    negative_total = 0
    oos_total = 0

    dense_supported_hits = 0
    rerank_supported_hits = 0

    dense_negative_hits = 0
    rerank_negative_hits = 0

    dense_oos_rejections = 0
    rerank_oos_rejections = 0

    dense_rr_total = 0.0
    rerank_rr_total = 0.0

    retrieval_question_count = 0

    fixed_questions = []
    broken_questions = []

    rank_improved = []
    rank_worsened = []

    try:

        # ====================================================
        # EVALUATE EACH QUESTION
        # ====================================================

        for index, item in enumerate(
                questions,
                start=1
        ):

            query = get_question_text(
                item
            )

            expected_pages = get_expected_pages(
                item
            )

            retrieve_expected = should_retrieve(
                item
            )

            category = get_category(
                item
            )

            # ------------------------------------------------
            # Retrieve 20 candidates once.
            #
            # We can use the first 5 as our dense baseline,
            # while all 20 go to the reranker.
            # ------------------------------------------------

            candidates = dense_retrieve(
                client,
                query,
                RERANK_CANDIDATE_K
            )

            dense_top5 = candidates[
                         :DENSE_TOP_K
                         ]

            reranked_all = rerank(
                query,
                candidates
            )

            reranked_top5 = reranked_all[
                            :FINAL_TOP_K
                            ]

            # ------------------------------------------------
            # Questions where evidence SHOULD exist
            # ------------------------------------------------

            if retrieve_expected:

                retrieval_question_count += 1

                dense_hit = evidence_found(
                    dense_top5,
                    expected_pages
                )

                rerank_hit = evidence_found(
                    reranked_top5,
                    expected_pages
                )

                # --------------------------------------------
                # Rank information
                # --------------------------------------------

                dense_rank = first_relevant_rank(
                    candidates,
                    expected_pages
                )

                rerank_rank = first_relevant_rank(
                    reranked_all,
                    expected_pages
                )

                dense_rr_total += reciprocal_rank(
                    dense_rank
                )

                rerank_rr_total += reciprocal_rank(
                    rerank_rank
                )

                # --------------------------------------------
                # Category counters
                # --------------------------------------------

                if category == "supported":

                    supported_total += 1

                    if dense_hit:
                        dense_supported_hits += 1

                    if rerank_hit:
                        rerank_supported_hits += 1

                elif category == "negative_supported":

                    negative_total += 1

                    if dense_hit:
                        dense_negative_hits += 1

                    if rerank_hit:
                        rerank_negative_hits += 1

                # --------------------------------------------
                # Fixed / broken Top-5 questions
                # --------------------------------------------

                if (
                        not dense_hit
                        and rerank_hit
                ):

                    fixed_questions.append({
                        "question":
                            query,

                        "dense_rank":
                            dense_rank,

                        "rerank_rank":
                            rerank_rank,

                        "expected_pages":
                            expected_pages
                    })

                elif (
                        dense_hit
                        and not rerank_hit
                ):

                    broken_questions.append({
                        "question":
                            query,

                        "dense_rank":
                            dense_rank,

                        "rerank_rank":
                            rerank_rank,

                        "expected_pages":
                            expected_pages
                    })

                # --------------------------------------------
                # Rank movement
                # --------------------------------------------

                if (
                        dense_rank is not None
                        and rerank_rank is not None
                ):

                    if rerank_rank < dense_rank:

                        rank_improved.append({
                            "question":
                                query,

                            "before":
                                dense_rank,

                            "after":
                                rerank_rank
                        })

                    elif rerank_rank > dense_rank:

                        rank_worsened.append({
                            "question":
                                query,

                            "before":
                                dense_rank,

                            "after":
                                rerank_rank
                        })

            # ------------------------------------------------
            # OUT-OF-SCOPE QUESTIONS
            # ------------------------------------------------

            else:

                oos_total += 1

                # Our existing retrieval threshold is applied
                # BEFORE reranking.
                #
                # If no dense result survives the threshold,
                # the system correctly rejects the query.

                dense_rejected = (
                        len(dense_top5) == 0
                )

                rerank_rejected = (
                        len(reranked_top5) == 0
                )

                if dense_rejected:
                    dense_oos_rejections += 1

                if rerank_rejected:
                    rerank_oos_rejections += 1

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            print(
                f"[{index:02}/{len(questions)}] "
                f"{query}"
            )

    finally:

        client.close()

    # ========================================================
    # CALCULATE TOTAL EVIDENCE RECALL
    # ========================================================

    dense_evidence_hits = (
            dense_supported_hits
            + dense_negative_hits
    )

    rerank_evidence_hits = (
            rerank_supported_hits
            + rerank_negative_hits
    )

    evidence_total = (
            supported_total
            + negative_total
    )

    # ========================================================
    # MRR
    # ========================================================

    dense_mrr = (
        dense_rr_total
        / retrieval_question_count
        if retrieval_question_count
        else 0
    )

    rerank_mrr = (
        rerank_rr_total
        / retrieval_question_count
        if retrieval_question_count
        else 0
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print(
        "\n" + "=" * 90
    )

    print(
        "RERANKING EVALUATION"
    )

    print(
        "=" * 90
    )

    print(
        "\nDENSE BASELINE"
    )

    print(
        f"Evidence Recall@5: "
        f"{dense_evidence_hits}/{evidence_total} "
        f"({percentage(dense_evidence_hits, evidence_total)})"
    )

    print(
        f"Supported Recall@5: "
        f"{dense_supported_hits}/{supported_total} "
        f"({percentage(dense_supported_hits, supported_total)})"
    )

    print(
        f"Negative Recall@5: "
        f"{dense_negative_hits}/{negative_total} "
        f"({percentage(dense_negative_hits, negative_total)})"
    )

    print(
        f"OOS Rejection: "
        f"{dense_oos_rejections}/{oos_total} "
        f"({percentage(dense_oos_rejections, oos_total)})"
    )

    print(
        f"MRR@{RERANK_CANDIDATE_K}: "
        f"{dense_mrr:.4f}"
    )

    print(
        "\nDENSE + CROSS-ENCODER"
    )

    print(
        f"Evidence Recall@5: "
        f"{rerank_evidence_hits}/{evidence_total} "
        f"({percentage(rerank_evidence_hits, evidence_total)})"
    )

    print(
        f"Supported Recall@5: "
        f"{rerank_supported_hits}/{supported_total} "
        f"({percentage(rerank_supported_hits, supported_total)})"
    )

    print(
        f"Negative Recall@5: "
        f"{rerank_negative_hits}/{negative_total} "
        f"({percentage(rerank_negative_hits, negative_total)})"
    )

    print(
        f"OOS Rejection: "
        f"{rerank_oos_rejections}/{oos_total} "
        f"({percentage(rerank_oos_rejections, oos_total)})"
    )

    print(
        f"MRR@{RERANK_CANDIDATE_K}: "
        f"{rerank_mrr:.4f}"
    )

    # ========================================================
    # FIXED QUESTIONS
    # ========================================================

    print(
        "\n" + "=" * 90
    )

    print(
        "FIXED BY RERANKING"
    )

    print(
        "=" * 90
    )

    if not fixed_questions:

        print(
            "None."
        )

    else:

        for item in fixed_questions:

            print(
                f"\n✓ {item['question']}"
            )

            print(
                f"  Dense rank: "
                f"{item['dense_rank']}"
            )

            print(
                f"  Reranked:   "
                f"{item['rerank_rank']}"
            )

            print(
                f"  Expected pages: "
                f"{item['expected_pages']}"
            )

    # ========================================================
    # BROKEN QUESTIONS
    # ========================================================

    print(
        "\n" + "=" * 90
    )

    print(
        "BROKEN BY RERANKING"
    )

    print(
        "=" * 90
    )

    if not broken_questions:

        print(
            "None."
        )

    else:

        for item in broken_questions:

            print(
                f"\n✗ {item['question']}"
            )

            print(
                f"  Dense rank: "
                f"{item['dense_rank']}"
            )

            print(
                f"  Reranked:   "
                f"{item['rerank_rank']}"
            )

            print(
                f"  Expected pages: "
                f"{item['expected_pages']}"
            )

    # ========================================================
    # RANK MOVEMENT
    # ========================================================

    print(
        "\n" + "=" * 90
    )

    print(
        "RANK MOVEMENT"
    )

    print(
        "=" * 90
    )

    print(
        f"\nImproved: "
        f"{len(rank_improved)} questions"
    )

    for item in rank_improved:

        print(
            f"  ↑ {item['before']} → "
            f"{item['after']} | "
            f"{item['question']}"
        )

    print(
        f"\nWorsened: "
        f"{len(rank_worsened)} questions"
    )

    for item in rank_worsened:

        print(
            f"  ↓ {item['before']} → "
            f"{item['after']} | "
            f"{item['question']}"
        )


# ============================================================
# PERCENTAGE HELPER
# ============================================================

def percentage(
        value: int,
        total: int
) -> str:

    if total == 0:
        return "0.00%"

    return (
        f"{(value / total) * 100:.2f}%"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    evaluate()