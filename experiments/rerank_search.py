from pathlib import Path

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, CrossEncoder


# ============================================================
# CONFIGURATION
# ============================================================

# First-stage dense retrieval model
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Second-stage reranking model
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

COLLECTION_NAME = "kapston_documents_v2"

QDRANT_PATH = Path(
    r"/storage/qdrant"
)

# Retrieve a larger candidate set first.
RETRIEVAL_TOP_K = 20

# After reranking, keep only the best results.
FINAL_TOP_K = 5

# Keep our current provisional dense-retrieval threshold.
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
# STAGE 1 — DENSE RETRIEVAL
# ============================================================

def retrieve_candidates(
        query: str
) -> list[dict]:
    """
    Retrieve a broad candidate set from Qdrant.

    This is the fast first-stage retriever.

    Query
        ↓
    MiniLM embedding
        ↓
    Qdrant cosine search
        ↓
    Top 20 candidates
    """

    query_embedding = embedding_model.encode(
        query,
        normalize_embeddings=True
    )

    client = QdrantClient(
        path=str(QDRANT_PATH)
    )

    try:

        response = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding.tolist(),
            limit=RETRIEVAL_TOP_K,
            with_payload=True
        )

        candidates = []

        for point in response.points:

            dense_score = float(
                point.score
            )

            if dense_score < MIN_RELEVANCE_SCORE:
                continue

            payload = point.payload or {}

            candidates.append({
                "id":
                    point.id,

                "dense_score":
                    dense_score,

                "source":
                    payload.get("source"),

                "page":
                    payload.get("page"),

                "chunk":
                    payload.get("chunk"),

                "page_chunk":
                    payload.get("page_chunk"),

                "text":
                    payload.get(
                        "text",
                        ""
                    )
            })

        return candidates

    finally:

        client.close()


# ============================================================
# STAGE 2 — CROSS-ENCODER RERANKING
# ============================================================

def rerank_candidates(
        query: str,
        candidates: list[dict]
) -> list[dict]:
    """
    Rerank retrieved candidates using a cross-encoder.

    Unlike dense retrieval, the cross-encoder sees the query
    and document chunk together.

    Example:

        Query:
        "What AC services does Kapston provide?"

        Passage:
        "... AC installation ... AC repair ...
        AC servicing ... gas refilling ..."

    The model produces a new relevance score for the pair.
    """

    if not candidates:
        return []

    # --------------------------------------------------------
    # Build query-passage pairs
    # --------------------------------------------------------

    pairs = [
        [
            query,
            candidate["text"]
        ]
        for candidate in candidates
    ]

    # --------------------------------------------------------
    # Cross-encoder inference
    # --------------------------------------------------------

    reranker_scores = reranker.predict(
        pairs,
        show_progress_bar=False
    )

    # --------------------------------------------------------
    # Attach reranker scores
    # --------------------------------------------------------

    for candidate, score in zip(
            candidates,
            reranker_scores
    ):

        candidate["reranker_score"] = float(
            score
        )

    # --------------------------------------------------------
    # Sort using CROSS-ENCODER score,
    # not the original dense score.
    # --------------------------------------------------------

    reranked = sorted(
        candidates,
        key=lambda result: result[
            "reranker_score"
        ],
        reverse=True
    )

    return reranked


# ============================================================
# COMPLETE SEARCH PIPELINE
# ============================================================

def search_with_reranking(
        query: str
) -> tuple[list[dict], list[dict]]:
    """
    Complete retrieval pipeline:

        Query
          ↓
        Dense retrieval
          ↓
        Top 20
          ↓
        Cross-encoder
          ↓
        Reranked Top 5

    Returns both lists so we can inspect what changed.
    """

    candidates = retrieve_candidates(
        query
    )

    reranked = rerank_candidates(
        query,
        candidates
    )

    final_results = reranked[
                    :FINAL_TOP_K
                    ]

    return candidates, final_results


# ============================================================
# DISPLAY DENSE CANDIDATES
# ============================================================

def print_dense_candidates(
        candidates: list[dict]
):
    """
    Show the original Qdrant ranking before reranking.
    """

    print(
        "\n" + "=" * 80
    )

    print(
        "STAGE 1: DENSE RETRIEVAL"
    )

    print(
        "=" * 80
    )

    if not candidates:

        print(
            "No candidates retrieved."
        )

        return

    for rank, result in enumerate(
            candidates,
            start=1
    ):

        print(
            f"#{rank:<2} "
            f"Dense={result['dense_score']:.4f} "
            f"Page={result['page']} "
            f"Chunk={result['chunk']}"
        )


# ============================================================
# DISPLAY FINAL RERANKED RESULTS
# ============================================================

def print_reranked_results(
        query: str,
        results: list[dict]
):
    """
    Display the final top results after cross-encoder
    reranking.
    """

    print(
        "\n" + "=" * 80
    )

    print(
        f"QUERY: {query}"
    )

    print(
        "=" * 80
    )

    print(
        "\nSTAGE 2: CROSS-ENCODER RERANKED RESULTS"
    )

    if not results:

        print(
            "\nNo relevant results found."
        )

        return

    for rank, result in enumerate(
            results,
            start=1
    ):

        print(
            "\n" + "-" * 80
        )

        print(
            f"RESULT #{rank}"
        )

        print(
            f"Dense score: "
            f"{result['dense_score']:.4f}"
        )

        print(
            f"Reranker score: "
            f"{result['reranker_score']:.4f}"
        )

        print(
            f"Page: "
            f"{result['page']}"
        )

        print(
            f"Chunk: "
            f"{result['chunk']}"
        )

        print(
            f"Page chunk: "
            f"{result['page_chunk']}"
        )

        print(
            f"Source: "
            f"{result['source']}"
        )

        print()

        print(
            result["text"]
        )


# ============================================================
# INTERACTIVE TEST
# ============================================================

if __name__ == "__main__":

    print(
        "\nKapston Cross-Encoder Search"
    )

    print(
        "Dense retrieval: Top 20"
    )

    print(
        "Cross-encoder: Top 5"
    )

    print(
        "Type 'exit' to stop."
    )

    while True:

        query = input(
            "\nAsk a question: "
        ).strip()

        if query.lower() in {
            "exit",
            "quit"
        }:
            break

        if not query:
            continue

        candidates, results = (
            search_with_reranking(
                query
            )
        )

        print_dense_candidates(
            candidates
        )

        print_reranked_results(
            query,
            results
        )