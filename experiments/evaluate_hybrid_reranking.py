import json
from pathlib import Path

from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder

from hybrid_search import (
    BM25Index,
    load_documents,
    dense_search,
    reciprocal_rank_fusion,
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

QUESTIONS_PATH = (
        PROJECT_ROOT
        / "evaluation"
        / "questions.json"
)

QDRANT_PATH = (
        PROJECT_ROOT
        / "storage"
        / "qdrant"
)

RERANKER_MODEL_NAME = (
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# First-stage retrieval
DENSE_TOP_K = 20
BM25_TOP_K = 20

# RRF produces a combined candidate pool.
HYBRID_CANDIDATE_K = 20

# Cross-encoder reduces it to final context.
FINAL_TOP_K = 5


# ============================================================
# LOAD RERANKER
# ============================================================

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

    if isinstance(data, list):
        return data

    if isinstance(data, dict):

        if "questions" in data:
            return data["questions"]

    raise ValueError(
        "Unsupported questions.json structure."
    )


# ============================================================
# QUESTION HELPERS
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
        "Question does not contain "
        "'question' or 'query'."
    )


def get_expected_pages(
        item: dict
) -> list[int]:

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
# EVALUATION HELPERS
# ============================================================

def evidence_found(
        results: list[dict],
        expected_pages: list[int],
        k: int
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


def first_relevant_rank(
        results: list[dict],
        expected_pages: list[int]
):

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


def reciprocal_rank(
        rank
) -> float:

    if rank is None:
        return 0.0

    return 1.0 / rank


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
# CROSS-ENCODER
# ============================================================

def rerank_candidates(
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
# EVALUATION
# ============================================================

def evaluate():

    questions = load_questions()

    print(
        f"\nLoaded {len(questions)} questions."
    )

    client = QdrantClient(
        path=str(QDRANT_PATH)
    )

    try:

        # ----------------------------------------------------
        # Build lexical index once.
        # ----------------------------------------------------

        print(
            "Loading chunks from Qdrant..."
        )

        documents = load_documents(
            client
        )

        print(
            f"Loaded {len(documents)} chunks."
        )

        print(
            "Building BM25 index..."
        )

        bm25_index = BM25Index(
            documents
        )

        print(
            "BM25 index ready."
        )

        # ====================================================
        # COUNTERS
        # ====================================================

        supported_total = 0
        negative_total = 0
        evidence_total = 0
        oos_total = 0

        # Candidate recall
        candidate_hits = 0

        supported_candidate_hits = 0
        negative_candidate_hits = 0

        # Final Recall@5
        final_supported_hits = 0
        final_negative_hits = 0

        # Ranking
        candidate_rr_total = 0.0
        reranked_rr_total = 0.0

        # Diagnostic lists
        candidate_misses = []

        rescued_by_reranker = []
        broken_by_reranker = []

        rank_improved = []
        rank_worsened = []

        oos_with_candidates = 0

        # ====================================================
        # LOOP
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
            # STAGE 1A — DENSE
            # ------------------------------------------------

            dense_results = dense_search(
                client,
                query,
                DENSE_TOP_K
            )

            # ------------------------------------------------
            # STAGE 1B — BM25
            # ------------------------------------------------

            bm25_results = bm25_index.search(
                query,
                BM25_TOP_K
            )

            # ------------------------------------------------
            # STAGE 2 — RRF
            # ------------------------------------------------

            fused_all = reciprocal_rank_fusion(
                dense_results,
                bm25_results
            )

            hybrid_candidates = fused_all[
                                :HYBRID_CANDIDATE_K
                                ]

            # ------------------------------------------------
            # STAGE 3 — CROSS-ENCODER
            # ------------------------------------------------

            reranked_all = rerank_candidates(
                query,
                hybrid_candidates
            )

            final_top5 = reranked_all[
                         :FINAL_TOP_K
                         ]

            # =================================================
            # SHOULD RETRIEVE
            # =================================================

            if retrieve_expected:

                evidence_total += 1

                if category == "supported":
                    supported_total += 1

                elif category == "negative_supported":
                    negative_total += 1

                # ---------------------------------------------
                # Candidate recall
                # ---------------------------------------------

                candidate_hit = evidence_found(
                    hybrid_candidates,
                    expected_pages,
                    HYBRID_CANDIDATE_K
                )

                if candidate_hit:

                    candidate_hits += 1

                    if category == "supported":
                        supported_candidate_hits += 1

                    elif category == "negative_supported":
                        negative_candidate_hits += 1

                else:

                    candidate_misses.append({
                        "question":
                            query,

                        "expected_pages":
                            expected_pages
                    })

                # ---------------------------------------------
                # Before/after reranking Top-5
                # ---------------------------------------------

                hybrid_top5_hit = evidence_found(
                    hybrid_candidates,
                    expected_pages,
                    FINAL_TOP_K
                )

                reranked_top5_hit = evidence_found(
                    final_top5,
                    expected_pages,
                    FINAL_TOP_K
                )

                if (
                        not hybrid_top5_hit
                        and reranked_top5_hit
                ):

                    rescued_by_reranker.append({
                        "question":
                            query,

                        "before_rank":
                            first_relevant_rank(
                                hybrid_candidates,
                                expected_pages
                            ),

                        "after_rank":
                            first_relevant_rank(
                                reranked_all,
                                expected_pages
                            ),

                        "expected_pages":
                            expected_pages
                    })

                elif (
                        hybrid_top5_hit
                        and not reranked_top5_hit
                ):

                    broken_by_reranker.append({
                        "question":
                            query,

                        "before_rank":
                            first_relevant_rank(
                                hybrid_candidates,
                                expected_pages
                            ),

                        "after_rank":
                            first_relevant_rank(
                                reranked_all,
                                expected_pages
                            ),

                        "expected_pages":
                            expected_pages
                    })

                # ---------------------------------------------
                # Final category recall
                # ---------------------------------------------

                if reranked_top5_hit:

                    if category == "supported":
                        final_supported_hits += 1

                    elif category == "negative_supported":
                        final_negative_hits += 1

                # ---------------------------------------------
                # Rank movement
                # ---------------------------------------------

                before_rank = first_relevant_rank(
                    hybrid_candidates,
                    expected_pages
                )

                after_rank = first_relevant_rank(
                    reranked_all,
                    expected_pages
                )

                candidate_rr_total += reciprocal_rank(
                    before_rank
                )

                reranked_rr_total += reciprocal_rank(
                    after_rank
                )

                if (
                        before_rank is not None
                        and after_rank is not None
                ):

                    if after_rank < before_rank:

                        rank_improved.append({
                            "question":
                                query,

                            "before":
                                before_rank,

                            "after":
                                after_rank
                        })

                    elif after_rank > before_rank:

                        rank_worsened.append({
                            "question":
                                query,

                            "before":
                                before_rank,

                            "after":
                                after_rank
                        })

            # =================================================
            # OUT OF SCOPE
            # =================================================

            else:

                oos_total += 1

                if hybrid_candidates:
                    oos_with_candidates += 1

            print(
                f"[{index:02}/{len(questions)}] "
                f"{query}"
            )

        # ====================================================
        # FINAL METRICS
        # ====================================================

        final_evidence_hits = (
                final_supported_hits
                + final_negative_hits
        )

        candidate_mrr = (
            candidate_rr_total
            / evidence_total
            if evidence_total
            else 0.0
        )

        reranked_mrr = (
            reranked_rr_total
            / evidence_total
            if evidence_total
            else 0.0
        )

        # ====================================================
        # SUMMARY
        # ====================================================

        print(
            "\n" + "=" * 90
        )

        print(
            "HYBRID + CROSS-ENCODER EVALUATION"
        )

        print(
            "=" * 90
        )

        print(
            "\nCANDIDATE GENERATION"
        )

        print(
            f"Evidence Candidate Recall@20: "
            f"{candidate_hits}/{evidence_total} "
            f"({percentage(candidate_hits, evidence_total)})"
        )

        print(
            f"Supported Candidate Recall@20: "
            f"{supported_candidate_hits}/{supported_total} "
            f"({percentage(supported_candidate_hits, supported_total)})"
        )

        print(
            f"Negative Candidate Recall@20: "
            f"{negative_candidate_hits}/{negative_total} "
            f"({percentage(negative_candidate_hits, negative_total)})"
        )

        print(
            f"Candidate MRR@20: "
            f"{candidate_mrr:.4f}"
        )

        print(
            "\nFINAL CROSS-ENCODER TOP 5"
        )

        print(
            f"Evidence Recall@5: "
            f"{final_evidence_hits}/{evidence_total} "
            f"({percentage(final_evidence_hits, evidence_total)})"
        )

        print(
            f"Supported Recall@5: "
            f"{final_supported_hits}/{supported_total} "
            f"({percentage(final_supported_hits, supported_total)})"
        )

        print(
            f"Negative Recall@5: "
            f"{final_negative_hits}/{negative_total} "
            f"({percentage(final_negative_hits, negative_total)})"
        )

        print(
            f"Reranked MRR@20: "
            f"{reranked_mrr:.4f}"
        )

        # This is diagnostic only.
        #
        # We are NOT treating candidate presence as our
        # production answerability decision.
        print(
            "\nOUT-OF-SCOPE DIAGNOSTIC"
        )

        print(
            f"OOS queries producing hybrid candidates: "
            f"{oos_with_candidates}/{oos_total}"
        )

        # ====================================================
        # CANDIDATE MISSES
        # ====================================================

        print(
            "\n" + "=" * 90
        )

        print(
            "EVIDENCE MISSING FROM HYBRID TOP 20"
        )

        print(
            "=" * 90
        )

        if not candidate_misses:

            print(
                "None."
            )

        else:

            for item in candidate_misses:

                print(
                    f"\n✗ {item['question']}"
                )

                print(
                    f"  Expected pages: "
                    f"{item['expected_pages']}"
                )

        # ====================================================
        # RESCUED
        # ====================================================

        print(
            "\n" + "=" * 90
        )

        print(
            "RESCUED INTO TOP 5 BY CROSS-ENCODER"
        )

        print(
            "=" * 90
        )

        if not rescued_by_reranker:

            print(
                "None."
            )

        else:

            for item in rescued_by_reranker:

                print(
                    f"\n✓ {item['question']}"
                )

                print(
                    f"  Hybrid rank: "
                    f"{item['before_rank']}"
                )

                print(
                    f"  Reranked: "
                    f"{item['after_rank']}"
                )

                print(
                    f"  Expected pages: "
                    f"{item['expected_pages']}"
                )

        # ====================================================
        # BROKEN
        # ====================================================

        print(
            "\n" + "=" * 90
        )

        print(
            "PUSHED OUT OF TOP 5 BY CROSS-ENCODER"
        )

        print(
            "=" * 90
        )

        if not broken_by_reranker:

            print(
                "None."
            )

        else:

            for item in broken_by_reranker:

                print(
                    f"\n✗ {item['question']}"
                )

                print(
                    f"  Hybrid rank: "
                    f"{item['before_rank']}"
                )

                print(
                    f"  Reranked: "
                    f"{item['after_rank']}"
                )

                print(
                    f"  Expected pages: "
                    f"{item['expected_pages']}"
                )

        # ====================================================
        # RANK MOVEMENT
        # ====================================================

        print(
            "\n" + "=" * 90
        )

        print(
            "RANK MOVEMENT AFTER CROSS-ENCODER"
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

    finally:

        client.close()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    evaluate()