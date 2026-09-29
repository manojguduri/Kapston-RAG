import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from src.data_loader import load_pdf, chunk_text, PDF_PATH
from src.chunker import chunk_text_by_tokens
from structure_chunker import structure_aware_chunks


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "all-MiniLM-L6-v2"

TOP_K = 5

# Keep the same threshold as our previous experiments so that
# the comparison between strategies remains fair.
MIN_RELEVANCE_SCORE = 0.20

QUESTIONS_PATH = Path(
    r"/evaluation/questions.json"
)


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

model = SentenceTransformer(
    MODEL_NAME
)


# ============================================================
# LOAD EVALUATION QUESTIONS
# ============================================================

def load_questions() -> list[dict]:
    """
    Load our manually labelled evaluation dataset.

    Each question contains:

        question
        type
        expected_pages
        should_retrieve

    Types:

        supported
        negative_supported
        out_of_scope
    """

    with open(
            QUESTIONS_PATH,
            "r",
            encoding="utf-8"
    ) as file:

        return json.load(file)


# ============================================================
# BUILD CHUNKS
# ============================================================

def build_chunks(
        strategy: str
) -> list[dict]:
    """
    Build chunks using the requested strategy.

    Strategies currently supported:

        char_500
        token_128
        token_192
        token_256
        structure

    Every returned chunk uses a common format:

        {
            "chunk": ...,
            "pages": [...],
            "text": ...
        }

    This common representation is important because our new
    structure-aware chunks may span multiple PDF pages.
    """

    pages = load_pdf(PDF_PATH)

    chunks = []

    # --------------------------------------------------------
    # CHARACTER CHUNKING
    # --------------------------------------------------------

    if strategy == "char_500":

        chunk_id = 0

        for page in pages:

            page_chunks = chunk_text(
                page["text"],
                chunk_size=500,
                overlap=100
            )

            for text in page_chunks:

                chunks.append({
                    "chunk": chunk_id,

                    # Character chunks belong to one page.
                    "pages": [
                        page["page"]
                    ],

                    "text": text
                })

                chunk_id += 1

    # --------------------------------------------------------
    # FIXED TOKEN CHUNKING — 128
    # --------------------------------------------------------

    elif strategy == "token_128":

        chunk_id = 0

        for page in pages:

            page_chunks = chunk_text_by_tokens(
                page["text"],
                chunk_size=128,
                overlap=24
            )

            for text in page_chunks:

                chunks.append({
                    "chunk": chunk_id,
                    "pages": [
                        page["page"]
                    ],
                    "text": text
                })

                chunk_id += 1

    # --------------------------------------------------------
    # FIXED TOKEN CHUNKING — 192
    # --------------------------------------------------------

    elif strategy == "token_192":

        chunk_id = 0

        for page in pages:

            page_chunks = chunk_text_by_tokens(
                page["text"],
                chunk_size=192,
                overlap=32
            )

            for text in page_chunks:

                chunks.append({
                    "chunk": chunk_id,
                    "pages": [
                        page["page"]
                    ],
                    "text": text
                })

                chunk_id += 1

    # --------------------------------------------------------
    # FIXED TOKEN CHUNKING — 256
    # --------------------------------------------------------

    elif strategy == "token_256":

        chunk_id = 0

        for page in pages:

            page_chunks = chunk_text_by_tokens(
                page["text"],
                chunk_size=256,
                overlap=48
            )

            for text in page_chunks:

                chunks.append({
                    "chunk": chunk_id,
                    "pages": [
                        page["page"]
                    ],
                    "text": text
                })

                chunk_id += 1

    # --------------------------------------------------------
    # STRUCTURE-AWARE CHUNKING
    # --------------------------------------------------------

    elif strategy == "structure":

        structure_chunks = structure_aware_chunks(
            pages,
            max_tokens=256
        )

        for chunk in structure_chunks:

            chunks.append({
                "chunk": chunk["chunk"],
                "pages": chunk["pages"],
                "heading": chunk["heading"],

                # Used later as source/context
                "text": chunk["text"],

                # Used for retrieval embedding
                "embedding_text": chunk["embedding_text"]
            })

    else:

        raise ValueError(
            f"Unknown chunking strategy: {strategy}"
        )

    return chunks


# ============================================================
# EVALUATE ONE STRATEGY
# ============================================================

def evaluate_strategy(
        strategy: str,
        questions: list[dict]
) -> dict:
    """
    Evaluate one chunking strategy.

    For supported / negative_supported questions:

        PASS =
        at least one expected evidence page appears in
        the top-k retrieved chunks.

    For out_of_scope questions:

        PASS =
        no retrieved chunk survives our relevance threshold.

    Note:
    This benchmark evaluates RETRIEVAL, not Gemini answer
    generation.
    """

    # --------------------------------------------------------
    # Build chunks
    # --------------------------------------------------------

    chunks = build_chunks(
        strategy
    )

    chunk_texts = [
        chunk.get(
            "embedding_text",
            chunk["text"]
        )
        for chunk in chunks
    ]

    # --------------------------------------------------------
    # Embed chunks
    # --------------------------------------------------------

    # normalize_embeddings=True means dot product becomes
    # equivalent to cosine similarity.
    chunk_embeddings = model.encode(
        chunk_texts,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    should_retrieve_total = 0
    retrieval_hits = 0

    should_reject_total = 0
    correct_rejections = 0

    # Separate metrics are useful because supported and
    # negative-supported questions test different behaviours.
    supported_total = 0
    supported_hits = 0

    negative_supported_total = 0
    negative_supported_hits = 0

    failures = []

    # --------------------------------------------------------
    # Evaluate every question
    # --------------------------------------------------------

    for item in questions:

        question = item["question"]
        question_type = item["type"]
        should_retrieve = item["should_retrieve"]
        expected_pages = item["expected_pages"]

        # ----------------------------------------------------
        # Embed query
        # ----------------------------------------------------

        query_embedding = model.encode(
            question,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        # Since embeddings are normalized:
        #
        # dot(A, B) == cosine_similarity(A, B)
        scores = np.dot(
            chunk_embeddings,
            query_embedding
        )

        # ----------------------------------------------------
        # Get top K BEFORE threshold filtering
        # ----------------------------------------------------

        top_indices = np.argsort(
            scores
        )[::-1][:TOP_K]

        results = []

        for index in top_indices:

            score = float(
                scores[index]
            )

            # Keep only chunks that satisfy our current
            # relevance threshold.
            if score >= MIN_RELEVANCE_SCORE:

                results.append({
                    "pages":
                        chunks[index]["pages"],

                    "chunk":
                        chunks[index]["chunk"],

                    "heading":
                        chunks[index].get(
                            "heading"
                        ),

                    "score":
                        score
                })

        # ----------------------------------------------------
        # Collect every page represented by retrieved chunks.
        #
        # This matters for structure-aware chunks because a
        # single section may span pages [7, 8], for example.
        # ----------------------------------------------------

        retrieved_pages = sorted({
            page
            for result in results
            for page in result["pages"]
        })

        # ====================================================
        # SHOULD RETRIEVE
        # ====================================================

        if should_retrieve:

            should_retrieve_total += 1

            hit = any(
                expected_page in retrieved_pages
                for expected_page in expected_pages
            )

            # Track overall retrieval recall.
            if hit:
                retrieval_hits += 1

            # Track supported questions separately.
            if question_type == "supported":

                supported_total += 1

                if hit:
                    supported_hits += 1

            # Track explicit negative evidence separately.
            elif question_type == "negative_supported":

                negative_supported_total += 1

                if hit:
                    negative_supported_hits += 1

            # Save failure information for debugging.
            if not hit:

                failures.append({
                    "question":
                        question,

                    "type":
                        question_type,

                    "expected_pages":
                        expected_pages,

                    "retrieved_pages":
                        retrieved_pages,

                    "top_score":
                        (
                            results[0]["score"]
                            if results
                            else None
                        ),

                    "top_heading":
                        (
                            results[0]["heading"]
                            if results
                            else None
                        )
                })

        # ====================================================
        # SHOULD REJECT
        # ====================================================

        else:

            should_reject_total += 1

            # Rejection currently means that NONE of the top
            # results survived MIN_RELEVANCE_SCORE.
            if len(results) == 0:

                correct_rejections += 1

            else:

                failures.append({
                    "question":
                        question,

                    "type":
                        question_type,

                    "expected_pages":
                        [],

                    "retrieved_pages":
                        retrieved_pages,

                    "top_score":
                        results[0]["score"],

                    "top_heading":
                        results[0]["heading"]
                })

    # ========================================================
    # CALCULATE METRICS
    # ========================================================

    retrieval_recall = (
        retrieval_hits
        / should_retrieve_total
        if should_retrieve_total
        else 0
    )

    rejection_accuracy = (
        correct_rejections
        / should_reject_total
        if should_reject_total
        else 0
    )

    supported_recall = (
        supported_hits
        / supported_total
        if supported_total
        else 0
    )

    negative_supported_recall = (
        negative_supported_hits
        / negative_supported_total
        if negative_supported_total
        else 0
    )

    return {
        "strategy":
            strategy,

        "chunks":
            len(chunks),

        "retrieval_hits":
            retrieval_hits,

        "should_retrieve_total":
            should_retrieve_total,

        "retrieval_recall":
            retrieval_recall,

        "supported_hits":
            supported_hits,

        "supported_total":
            supported_total,

        "supported_recall":
            supported_recall,

        "negative_supported_hits":
            negative_supported_hits,

        "negative_supported_total":
            negative_supported_total,

        "negative_supported_recall":
            negative_supported_recall,

        "correct_rejections":
            correct_rejections,

        "should_reject_total":
            should_reject_total,

        "rejection_accuracy":
            rejection_accuracy,

        "failures":
            failures
    }


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(
        results: list[dict]
) -> None:

    print("\n")
    print("=" * 105)
    print("RESULTS")
    print("=" * 105)

    print(
        f"{'Strategy':<15}"
        f"{'Chunks':<10}"
        f"{'Evidence Recall@5':<23}"
        f"{'Supported':<20}"
        f"{'Negative':<20}"
        f"{'OOS Rejection'}"
    )

    print("-" * 105)

    for result in results:

        retrieval = (
            f"{result['retrieval_hits']}/"
            f"{result['should_retrieve_total']} "
            f"({result['retrieval_recall']:.2%})"
        )

        supported = (
            f"{result['supported_hits']}/"
            f"{result['supported_total']} "
            f"({result['supported_recall']:.2%})"
        )

        negative = (
            f"{result['negative_supported_hits']}/"
            f"{result['negative_supported_total']} "
            f"({result['negative_supported_recall']:.2%})"
        )

        rejection = (
            f"{result['correct_rejections']}/"
            f"{result['should_reject_total']} "
            f"({result['rejection_accuracy']:.2%})"
        )

        print(
            f"{result['strategy']:<15}"
            f"{result['chunks']:<10}"
            f"{retrieval:<23}"
            f"{supported:<20}"
            f"{negative:<20}"
            f"{rejection}"
        )


# ============================================================
# PRINT FAILURES
# ============================================================

def print_failures(
        results: list[dict]
) -> None:

    print("\n")
    print("=" * 105)
    print("FAILURES")
    print("=" * 105)

    for result in results:

        print(
            f"\n--- {result['strategy']} ---"
        )

        if not result["failures"]:

            print("No failures.")
            continue

        for failure in result["failures"]:

            print(
                f"\n[{failure['type']}] "
                f"{failure['question']}"
            )

            print(
                "Expected pages: "
                f"{failure['expected_pages']}"
            )

            print(
                "Retrieved pages: "
                f"{failure['retrieved_pages']}"
            )

            top_score = failure[
                "top_score"
            ]

            if top_score is not None:

                print(
                    f"Top score: "
                    f"{top_score:.4f}"
                )

            else:

                print(
                    "Top score: None"
                )

            # This is particularly useful for our new
            # structure-aware strategy.
            top_heading = failure.get(
                "top_heading"
            )

            if top_heading:

                print(
                    "Top heading: "
                    f"{top_heading}"
                )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    questions = load_questions()

    strategies = [
        "char_500",
        "token_128",
        "token_192",
        "token_256",
        "structure"
    ]

    print("\nCHUNKING BENCHMARK")
    print("=" * 105)

    print(
        f"Questions: {len(questions)}"
    )

    print(
        f"Top K: {TOP_K}"
    )

    print(
        f"Minimum relevance score: "
        f"{MIN_RELEVANCE_SCORE}"
    )

    results = []

    for strategy in strategies:

        print(
            f"\nEvaluating "
            f"{strategy}..."
        )

        result = evaluate_strategy(
            strategy,
            questions
        )

        results.append(
            result
        )

    print_results(
        results
    )

    print_failures(
        results
    )