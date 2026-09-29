from pathlib import Path

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "all-MiniLM-L6-v2"

COLLECTION_NAME = "kapston_documents_v2"

QDRANT_PATH = Path(
    r"/storage/qdrant"
)

TOP_K = 5

MIN_RELEVANCE_SCORE = 0.20


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print(
    "Loading embedding model..."
)

model = SentenceTransformer(
    MODEL_NAME
)


# ============================================================
# SEARCH
# ============================================================

def search_documents(
        query: str,
        top_k: int = TOP_K
) -> list[dict]:
    """
    Search the Kapston vector database.

    Steps:

        User query
            ↓
        MiniLM embedding
            ↓
        Qdrant cosine search
            ↓
        Relevance threshold
            ↓
        Retrieved document chunks
    """

    # --------------------------------------------------------
    # Convert query into embedding
    # --------------------------------------------------------

    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    )

    # --------------------------------------------------------
    # Open local Qdrant
    # --------------------------------------------------------

    client = QdrantClient(
        path=str(QDRANT_PATH)
    )

    try:

        # ----------------------------------------------------
        # Search collection
        # ----------------------------------------------------

        response = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding.tolist(),
            limit=top_k,
            with_payload=True
        )

        results = []

        # ----------------------------------------------------
        # Convert Qdrant results into our own format
        # ----------------------------------------------------

        for point in response.points:

            score = float(
                point.score
            )

            # Ignore results below our provisional relevance
            # threshold.
            if score < MIN_RELEVANCE_SCORE:
                continue

            payload = (
                    point.payload or {}
            )

            results.append({
                "score":
                    score,

                "source":
                    payload.get(
                        "source"
                    ),

                "page":
                    payload.get(
                        "page"
                    ),

                "chunk":
                    payload.get(
                        "chunk"
                    ),

                "page_chunk":
                    payload.get(
                        "page_chunk"
                    ),

                "text":
                    payload.get(
                        "text",
                        ""
                    )
            })

        return results

    finally:

        # Release local Qdrant storage.
        client.close()


# ============================================================
# DISPLAY RESULTS
# ============================================================

def print_results(
        query: str,
        results: list[dict]
):
    """
    Pretty-print retrieved chunks for manual inspection.
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

    if not results:

        print(
            "\nNo relevant chunks found."
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
            f"Score: "
            f"{result['score']:.4f}"
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
# INTERACTIVE SEARCH
# ============================================================

if __name__ == "__main__":

    print(
        "\nKapston Semantic Search"
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

        results = search_documents(
            query
        )

        print_results(
            query,
            results
        )