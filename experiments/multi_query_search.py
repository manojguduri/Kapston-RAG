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

RESULTS_PER_QUERY = 20

FINAL_TOP_K = 20

MIN_RELEVANCE_SCORE = 0.20


# ============================================================
# MODEL
# ============================================================

print(
    "Loading embedding model..."
)

model = SentenceTransformer(
    MODEL_NAME
)


# ============================================================
# SEARCH ONE QUERY
# ============================================================

def search_one_query(
        client: QdrantClient,
        query: str
) -> list:
    """
    Perform dense retrieval for one query.
    """

    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    )

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding.tolist(),
        limit=RESULTS_PER_QUERY,
        with_payload=True
    )

    return response.points


# ============================================================
# MULTI-QUERY SEARCH
# ============================================================

def multi_query_search(
        queries: list[str]
) -> list[dict]:
    """
    Search Qdrant using multiple representations of the same
    information need.

    Results from all queries are merged by Qdrant point ID.

    If the same chunk appears for several queries, we retain
    its highest similarity score.
    """

    client = QdrantClient(
        path=str(QDRANT_PATH)
    )

    try:

        candidates = {}

        for query in queries:

            print(
                f"\nSearching: {query}"
            )

            points = search_one_query(
                client,
                query
            )

            for point in points:

                score = float(
                    point.score
                )

                if score < MIN_RELEVANCE_SCORE:
                    continue

                point_id = point.id

                payload = (
                        point.payload or {}
                )

                # --------------------------------------------
                # First time we've encountered this chunk.
                # --------------------------------------------

                if point_id not in candidates:

                    candidates[point_id] = {
                        "id":
                            point_id,

                        "score":
                            score,

                        "matched_query":
                            query,

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

                        "text":
                            payload.get(
                                "text",
                                ""
                            )
                    }

                # --------------------------------------------
                # Same chunk found through another query.
                #
                # Keep whichever query produced the strongest
                # similarity score.
                # --------------------------------------------

                elif (
                        score
                        > candidates[
                            point_id
                        ]["score"]
                ):

                    candidates[
                        point_id
                    ]["score"] = score

                    candidates[
                        point_id
                    ]["matched_query"] = query

        # ----------------------------------------------------
        # Sort merged candidates
        # ----------------------------------------------------

        ranked_results = sorted(
            candidates.values(),
            key=lambda result: result["score"],
            reverse=True
        )

        return ranked_results[
               :FINAL_TOP_K
               ]

    finally:

        client.close()


# ============================================================
# DISPLAY
# ============================================================

def print_results(
        results: list[dict]
):

    print(
        "\n" + "=" * 80
    )

    print(
        "FINAL MERGED RESULTS"
    )

    print(
        "=" * 80
    )

    if not results:

        print(
            "No relevant chunks found."
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
            f"Matched query: "
            f"{result['matched_query']}"
        )

        print(
            f"Page: "
            f"{result['page']}"
        )

        print(
            f"Chunk: "
            f"{result['chunk']}"
        )

        print()

        print(
            result["text"]
        )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    queries = [
        "What AC services does Kapston provide?",

        "Kapston air conditioner services",

        (
            "AC installation repair servicing "
            "gas refilling"
        )
    ]

    results = multi_query_search(
        queries
    )

    print_results(
        results
    )