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
# CONFIG
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

DENSE_TOP_K = 20
BM25_TOP_K = 20
HYBRID_TOP_K = 20


# ============================================================
# LOAD CROSS-ENCODER
# ============================================================

print("Loading cross-encoder...")

reranker = CrossEncoder(
    RERANKER_MODEL_NAME
)


# ============================================================
# QUESTION HELPERS
# ============================================================

def load_questions():

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
        "Unsupported questions.json format."
    )


def get_question_text(item):

    for key in (
            "question",
            "query"
    ):

        if key in item:
            return item[key]

    raise KeyError(
        "Question text not found."
    )


def should_retrieve(item):

    if "should_retrieve" in item:
        return bool(
            item["should_retrieve"]
        )

    return False


def get_category(item):

    return item.get(
        "category",
        item.get(
            "type",
            "unknown"
        )
    )


# ============================================================
# RERANK
# ============================================================

def rerank(
        query,
        candidates
):

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

    results = []

    for candidate, score in zip(
            candidates,
            scores
    ):

        result = candidate.copy()

        result["reranker_score"] = float(
            score
        )

        results.append(
            result
        )

    results.sort(
        key=lambda x:
        x["reranker_score"],
        reverse=True
    )

    return results


# ============================================================
# MAIN ANALYSIS
# ============================================================

def analyze():

    questions = load_questions()

    client = QdrantClient(
        path=str(QDRANT_PATH)
    )

    try:

        print(
            "Loading documents..."
        )

        documents = load_documents(
            client
        )

        bm25_index = BM25Index(
            documents
        )

        rows = []

        # ====================================================
        # PROCESS QUESTIONS
        # ====================================================

        for index, item in enumerate(
                questions,
                start=1
        ):

            query = get_question_text(
                item
            )

            answerable = should_retrieve(
                item
            )

            category = get_category(
                item
            )

            # ------------------------------------------------
            # Dense
            # ------------------------------------------------

            dense_results = dense_search(
                client,
                query,
                DENSE_TOP_K
            )

            # ------------------------------------------------
            # BM25
            # ------------------------------------------------

            bm25_results = bm25_index.search(
                query,
                BM25_TOP_K
            )

            # ------------------------------------------------
            # RRF
            # ------------------------------------------------

            fused = reciprocal_rank_fusion(
                dense_results,
                bm25_results
            )

            candidates = fused[
                         :HYBRID_TOP_K
                         ]

            # ------------------------------------------------
            # Cross-encoder
            # ------------------------------------------------

            reranked = rerank(
                query,
                candidates
            )

            # =================================================
            # SIGNALS
            # =================================================

            top_dense = (
                dense_results[0]["dense_score"]
                if dense_results
                else 0.0
            )

            top_bm25 = (
                bm25_results[0]["bm25_score"]
                if bm25_results
                else 0.0
            )

            top_reranker = (
                reranked[0]["reranker_score"]
                if reranked
                else float("-inf")
            )

            second_reranker = (
                reranked[1]["reranker_score"]
                if len(reranked) > 1
                else float("-inf")
            )

            if (
                    reranked
                    and len(reranked) > 1
            ):
                reranker_margin = (
                        top_reranker
                        - second_reranker
                )
            else:
                reranker_margin = 0.0

            top_page = (
                reranked[0]["page"]
                if reranked
                else None
            )

            rows.append({
                "question": query,
                "answerable": answerable,
                "category": category,
                "dense": top_dense,
                "bm25": top_bm25,
                "reranker": top_reranker,
                "margin": reranker_margin,
                "top_page": top_page,
            })

            label = (
                "ANSWERABLE"
                if answerable
                else "OOS"
            )

            print(
                f"[{index:02}/{len(questions)}] "
                f"{label:<10} "
                f"dense={top_dense:.4f} "
                f"bm25={top_bm25:.4f} "
                f"reranker={top_reranker:.4f} "
                f"margin={reranker_margin:.4f} "
                f"| {query}"
            )

        # ====================================================
        # SPLIT GROUPS
        # ====================================================

        answerable_rows = [
            row
            for row in rows
            if row["answerable"]
        ]

        oos_rows = [
            row
            for row in rows
            if not row["answerable"]
        ]

        # ====================================================
        # STATISTICS
        # ====================================================

        def stats(
                group,
                field
        ):

            values = [
                row[field]
                for row in group
            ]

            if not values:
                return None

            return {
                "min": min(values),
                "max": max(values),
                "avg": (
                        sum(values)
                        / len(values)
                )
            }

        # ====================================================
        # PRINT SUMMARY
        # ====================================================

        print(
            "\n" + "=" * 100
        )

        print(
            "ANSWERABILITY SIGNAL ANALYSIS"
        )

        print(
            "=" * 100
        )

        for field in (
                "dense",
                "bm25",
                "reranker",
                "margin"
        ):

            answerable_stats = stats(
                answerable_rows,
                field
            )

            oos_stats = stats(
                oos_rows,
                field
            )

            print(
                f"\n{field.upper()}"
            )

            print(
                "  Answerable:"
                f" min={answerable_stats['min']:.4f}"
                f" avg={answerable_stats['avg']:.4f}"
                f" max={answerable_stats['max']:.4f}"
            )

            print(
                "  OOS:"
                f" min={oos_stats['min']:.4f}"
                f" avg={oos_stats['avg']:.4f}"
                f" max={oos_stats['max']:.4f}"
            )

        # ====================================================
        # MOST DIFFICULT ANSWERABLE QUESTIONS
        # ====================================================

        print(
            "\n" + "=" * 100
        )

        print(
            "LOWEST RERANKER SCORES — ANSWERABLE"
        )

        print(
            "=" * 100
        )

        weakest_answerable = sorted(
            answerable_rows,
            key=lambda x:
            x["reranker"]
        )

        for row in weakest_answerable[:10]:

            print(
                f"{row['reranker']:>8.4f} | "
                f"dense={row['dense']:.4f} | "
                f"{row['question']}"
            )

        # ====================================================
        # MOST DANGEROUS OOS QUESTIONS
        # ====================================================

        print(
            "\n" + "=" * 100
        )

        print(
            "HIGHEST RERANKER SCORES — OUT OF SCOPE"
        )

        print(
            "=" * 100
        )

        dangerous_oos = sorted(
            oos_rows,
            key=lambda x:
            x["reranker"],
            reverse=True
        )

        for row in dangerous_oos:

            print(
                f"{row['reranker']:>8.4f} | "
                f"dense={row['dense']:.4f} | "
                f"{row['question']}"
            )

    finally:

        client.close()


if __name__ == "__main__":
    analyze()