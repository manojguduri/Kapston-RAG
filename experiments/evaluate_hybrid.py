import json
from pathlib import Path

from qdrant_client import QdrantClient

# Import the components we already built.
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

DENSE_CANDIDATE_K = 20
BM25_CANDIDATE_K = 20

FINAL_TOP_K = 5


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
# EVIDENCE HELPERS
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
        # Build BM25 once.
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
        oos_total = 0

        dense_supported_hits = 0
        hybrid_supported_hits = 0

        dense_negative_hits = 0
        hybrid_negative_hits = 0

        dense_oos_rejections = 0
        hybrid_oos_rejections = 0

        dense_rr_total = 0.0
        hybrid_rr_total = 0.0

        retrieval_question_count = 0

        fixed_questions = []
        broken_questions = []

        rank_improved = []
        rank_worsened = []

        # ====================================================
        # LOOP THROUGH QUESTIONS
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
            # DENSE
            # ------------------------------------------------

            dense_results = dense_search(
                client,
                query,
                DENSE_CANDIDATE_K
            )

            dense_top5 = dense_results[
                         :FINAL_TOP_K
                         ]

            # ------------------------------------------------
            # BM25
            # ------------------------------------------------

            bm25_results = bm25_index.search(
                query,
                BM25_CANDIDATE_K
            )

            # ------------------------------------------------
            # RRF
            # ------------------------------------------------

            hybrid_all = reciprocal_rank_fusion(
                dense_results,
                bm25_results
            )

            hybrid_top5 = hybrid_all[
                          :FINAL_TOP_K
                          ]

            # =================================================
            # SHOULD RETRIEVE
            # =================================================

            if retrieve_expected:

                retrieval_question_count += 1

                dense_hit = evidence_found(
                    dense_top5,
                    expected_pages
                )

                hybrid_hit = evidence_found(
                    hybrid_top5,
                    expected_pages
                )

                # ---------------------------------------------
                # Relevant ranks
                # ---------------------------------------------

                dense_rank = first_relevant_rank(
                    dense_results,
                    expected_pages
                )

                hybrid_rank = first_relevant_rank(
                    hybrid_all,
                    expected_pages
                )

                dense_rr_total += reciprocal_rank(
                    dense_rank
                )

                hybrid_rr_total += reciprocal_rank(
                    hybrid_rank
                )

                # ---------------------------------------------
                # Category metrics
                # ---------------------------------------------

                if category == "supported":

                    supported_total += 1

                    if dense_hit:
                        dense_supported_hits += 1

                    if hybrid_hit:
                        hybrid_supported_hits += 1

                elif category == "negative_supported":

                    negative_total += 1

                    if dense_hit:
                        dense_negative_hits += 1

                    if hybrid_hit:
                        hybrid_negative_hits += 1

                # ---------------------------------------------
                # Fixed by hybrid
                # ---------------------------------------------

                if (
                        not dense_hit
                        and hybrid_hit
                ):

                    fixed_questions.append({
                        "question":
                            query,

                        "dense_rank":
                            dense_rank,

                        "hybrid_rank":
                            hybrid_rank,

                        "expected_pages":
                            expected_pages
                    })

                # ---------------------------------------------
                # Broken by hybrid
                # ---------------------------------------------

                elif (
                        dense_hit
                        and not hybrid_hit
                ):

                    broken_questions.append({
                        "question":
                            query,

                        "dense_rank":
                            dense_rank,

                        "hybrid_rank":
                            hybrid_rank,

                        "expected_pages":
                            expected_pages
                    })

                # ---------------------------------------------
                # Rank movement
                # ---------------------------------------------

                if (
                        dense_rank is not None
                        and hybrid_rank is not None
                ):

                    if hybrid_rank < dense_rank:

                        rank_improved.append({
                            "question":
                                query,

                            "before":
                                dense_rank,

                            "after":
                                hybrid_rank
                        })

                    elif hybrid_rank > dense_rank:

                        rank_worsened.append({
                            "question":
                                query,

                            "before":
                                dense_rank,

                            "after":
                                hybrid_rank
                        })

            # =================================================
            # OUT OF SCOPE
            # =================================================

            else:

                oos_total += 1

                # Dense rejection:
                # no result survives the existing
                # dense threshold.

                dense_rejected = (
                        len(dense_top5) == 0
                )

                # IMPORTANT:
                #
                # Hybrid has a lexical retriever.
                # BM25 can return candidates even when
                # dense retrieval rejects the query.
                #
                # For this experiment we define hybrid
                # rejection as:
                #
                # no fused candidates at all.
                #
                # This lets us observe the effect of adding
                # BM25 instead of artificially forcing the
                # dense threshold onto the hybrid system.

                hybrid_rejected = (
                        len(hybrid_top5) == 0
                )

                if dense_rejected:
                    dense_oos_rejections += 1

                if hybrid_rejected:
                    hybrid_oos_rejections += 1

            print(
                f"[{index:02}/{len(questions)}] "
                f"{query}"
            )

        # ====================================================
        # TOTALS
        # ====================================================

        dense_evidence_hits = (
                dense_supported_hits
                + dense_negative_hits
        )

        hybrid_evidence_hits = (
                hybrid_supported_hits
                + hybrid_negative_hits
        )

        evidence_total = (
                supported_total
                + negative_total
        )

        dense_mrr = (
            dense_rr_total
            / retrieval_question_count
            if retrieval_question_count
            else 0.0
        )

        hybrid_mrr = (
            hybrid_rr_total
            / retrieval_question_count
            if retrieval_question_count
            else 0.0
        )

        # ====================================================
        # SUMMARY
        # ====================================================

        print(
            "\n" + "=" * 90
        )

        print(
            "HYBRID RETRIEVAL EVALUATION"
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
            f"MRR@{DENSE_CANDIDATE_K}: "
            f"{dense_mrr:.4f}"
        )

        print(
            "\nDENSE + BM25 + RRF"
        )

        print(
            f"Evidence Recall@5: "
            f"{hybrid_evidence_hits}/{evidence_total} "
            f"({percentage(hybrid_evidence_hits, evidence_total)})"
        )

        print(
            f"Supported Recall@5: "
            f"{hybrid_supported_hits}/{supported_total} "
            f"({percentage(hybrid_supported_hits, supported_total)})"
        )

        print(
            f"Negative Recall@5: "
            f"{hybrid_negative_hits}/{negative_total} "
            f"({percentage(hybrid_negative_hits, negative_total)})"
        )

        print(
            f"OOS Rejection: "
            f"{hybrid_oos_rejections}/{oos_total} "
            f"({percentage(hybrid_oos_rejections, oos_total)})"
        )

        print(
            f"MRR@{DENSE_CANDIDATE_K}: "
            f"{hybrid_mrr:.4f}"
        )

        # ====================================================
        # FIXED
        # ====================================================

        print(
            "\n" + "=" * 90
        )

        print(
            "FIXED BY HYBRID RETRIEVAL"
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
                    f"  Hybrid rank: "
                    f"{item['hybrid_rank']}"
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
            "BROKEN BY HYBRID RETRIEVAL"
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
                    f"  Hybrid rank: "
                    f"{item['hybrid_rank']}"
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

    finally:

        client.close()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    evaluate()