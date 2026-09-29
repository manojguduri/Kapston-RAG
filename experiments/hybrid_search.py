import re
from collections import Counter
from math import log
from pathlib import Path

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "kapston_documents_v2"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
QDRANT_PATH = PROJECT_ROOT / "storage" / "qdrant"

# Number of candidates retrieved independently by each system.
DENSE_TOP_K = 20
BM25_TOP_K = 20

# Number returned after Reciprocal Rank Fusion.
FINAL_TOP_K = 5

# Standard RRF constant.
RRF_K = 60

# Existing provisional dense threshold.
MIN_RELEVANCE_SCORE = 0.20

# BM25 parameters.
BM25_K1 = 1.5
BM25_B = 0.75


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_NAME
)


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize(text: str) -> list[str]:
    """
    Simple tokenizer for BM25.

    Example:
        "What AC services?"
            ->
        ["what", "ac", "services"]

    We intentionally keep this simple for the experiment.
    """

    return re.findall(
        r"\b[a-z0-9]+\b",
        text.lower()
    )


# ============================================================
# LOAD ALL DOCUMENT CHUNKS FROM QDRANT
# ============================================================

def load_documents(
        client: QdrantClient
) -> list[dict]:

    documents = []
    offset = None

    while True:

        records, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False
        )

        for point in records:

            payload = point.payload or {}

            documents.append({
                "id": point.id,
                "page": payload.get("page"),
                "chunk": payload.get("chunk"),
                "page_chunk": payload.get("page_chunk"),
                "source": payload.get("source"),
                "text": payload.get("text", "")
            })

        if next_offset is None:
            break

        offset = next_offset

    return documents


# ============================================================
# SIMPLE BM25 INDEX
# ============================================================

class BM25Index:

    def __init__(
            self,
            documents: list[dict],
            k1: float = BM25_K1,
            b: float = BM25_B
    ):

        self.documents = documents
        self.k1 = k1
        self.b = b

        self.document_tokens = [
            tokenize(document["text"])
            for document in documents
        ]

        self.document_lengths = [
            len(tokens)
            for tokens in self.document_tokens
        ]

        self.document_count = len(
            self.documents
        )

        if self.document_count == 0:
            raise ValueError(
                "BM25 cannot be created because "
                "the collection contains no documents."
            )

        self.average_document_length = (
                sum(self.document_lengths)
                / self.document_count
        )

        # Number of documents containing each term.
        self.document_frequency = Counter()

        for tokens in self.document_tokens:

            for term in set(tokens):
                self.document_frequency[term] += 1

    def idf(
            self,
            term: str
    ) -> float:

        df = self.document_frequency.get(
            term,
            0
        )

        # Standard BM25-style smoothed IDF.
        return log(
            1
            + (
                    self.document_count
                    - df
                    + 0.5
            )
            / (
                    df
                    + 0.5
            )
        )

    def score_document(
            self,
            query_tokens: list[str],
            document_index: int
    ) -> float:

        document_tokens = (
            self.document_tokens[
                document_index
            ]
        )

        document_length = (
            self.document_lengths[
                document_index
            ]
        )

        term_frequencies = Counter(
            document_tokens
        )

        score = 0.0

        # Repeating a query word doesn't need to repeatedly
        # contribute to our simple BM25 implementation.
        for term in set(query_tokens):

            frequency = term_frequencies.get(
                term,
                0
            )

            if frequency == 0:
                continue

            numerator = (
                    frequency
                    * (self.k1 + 1)
            )

            denominator = (
                    frequency
                    + self.k1
                    * (
                            1
                            - self.b
                            + self.b
                            * (
                                    document_length
                                    / self.average_document_length
                            )
                    )
            )

            score += (
                    self.idf(term)
                    * numerator
                    / denominator
            )

        return score

    def search(
            self,
            query: str,
            limit: int = BM25_TOP_K
    ) -> list[dict]:

        query_tokens = tokenize(
            query
        )

        scored_results = []

        for index, document in enumerate(
                self.documents
        ):

            score = self.score_document(
                query_tokens,
                index
            )

            # Unlike dense search, zero means there was no
            # lexical overlap at all.
            if score <= 0:
                continue

            result = document.copy()

            result["bm25_score"] = score

            scored_results.append(
                result
            )

        scored_results.sort(
            key=lambda result:
            result["bm25_score"],
            reverse=True
        )

        return scored_results[:limit]


# ============================================================
# DENSE SEARCH
# ============================================================

def dense_search(
        client: QdrantClient,
        query: str,
        limit: int = DENSE_TOP_K
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

        score = float(
            point.score
        )

        if score < MIN_RELEVANCE_SCORE:
            continue

        payload = point.payload or {}

        results.append({
            "id": point.id,
            "dense_score": score,
            "page": payload.get("page"),
            "chunk": payload.get("chunk"),
            "page_chunk": payload.get(
                "page_chunk"
            ),
            "source": payload.get("source"),
            "text": payload.get(
                "text",
                ""
            )
        })

    return results


# ============================================================
# RECIPROCAL RANK FUSION
# ============================================================

def reciprocal_rank_fusion(
        dense_results: list[dict],
        bm25_results: list[dict]
) -> list[dict]:
    """
    Fuse rankings instead of directly combining their scores.

    RRF score:

        1 / (K + dense_rank)
            +
        1 / (K + bm25_rank)

    A result appearing in both lists receives contributions
    from both retrieval systems.

    This avoids trying to compare cosine similarity directly
    with BM25 scores.
    """

    fused = {}

    # --------------------------------------------------------
    # Dense contribution
    # --------------------------------------------------------

    for rank, result in enumerate(
            dense_results,
            start=1
    ):

        point_id = result["id"]

        if point_id not in fused:

            fused[point_id] = {
                **result,
                "dense_rank": None,
                "bm25_rank": None,
                "dense_score": None,
                "bm25_score": None,
                "rrf_score": 0.0
            }

        fused[point_id][
            "dense_rank"
        ] = rank

        fused[point_id][
            "dense_score"
        ] = result["dense_score"]

        fused[point_id][
            "rrf_score"
        ] += 1 / (
                RRF_K + rank
        )

    # --------------------------------------------------------
    # BM25 contribution
    # --------------------------------------------------------

    for rank, result in enumerate(
            bm25_results,
            start=1
    ):

        point_id = result["id"]

        if point_id not in fused:

            fused[point_id] = {
                **result,
                "dense_rank": None,
                "bm25_rank": None,
                "dense_score": None,
                "bm25_score": None,
                "rrf_score": 0.0
            }

        fused[point_id][
            "bm25_rank"
        ] = rank

        fused[point_id][
            "bm25_score"
        ] = result["bm25_score"]

        fused[point_id][
            "rrf_score"
        ] += 1 / (
                RRF_K + rank
        )

    results = list(
        fused.values()
    )

    results.sort(
        key=lambda result:
        result["rrf_score"],
        reverse=True
    )

    return results


# ============================================================
# HYBRID SEARCH
# ============================================================

def hybrid_search(
        client: QdrantClient,
        bm25_index: BM25Index,
        query: str,
        final_top_k: int = FINAL_TOP_K
):

    dense_results = dense_search(
        client,
        query,
        DENSE_TOP_K
    )

    bm25_results = bm25_index.search(
        query,
        BM25_TOP_K
    )

    fused_results = reciprocal_rank_fusion(
        dense_results,
        bm25_results
    )

    return (
        dense_results,
        bm25_results,
        fused_results[:final_top_k]
    )


# ============================================================
# DISPLAY
# ============================================================

def print_results(
        query: str,
        dense_results: list[dict],
        bm25_results: list[dict],
        hybrid_results: list[dict]
):

    print(
        "\n" + "=" * 90
    )

    print(
        f"QUERY: {query}"
    )

    print(
        "=" * 90
    )

    # --------------------------------------------------------
    # Dense
    # --------------------------------------------------------

    print(
        "\nDENSE TOP RESULTS"
    )

    for rank, result in enumerate(
            dense_results[:5],
            start=1
    ):

        print(
            f"#{rank:<2} "
            f"score={result['dense_score']:.4f} "
            f"page={result['page']} "
            f"chunk={result['chunk']}"
        )

    # --------------------------------------------------------
    # BM25
    # --------------------------------------------------------

    print(
        "\nBM25 TOP RESULTS"
    )

    for rank, result in enumerate(
            bm25_results[:5],
            start=1
    ):

        print(
            f"#{rank:<2} "
            f"score={result['bm25_score']:.4f} "
            f"page={result['page']} "
            f"chunk={result['chunk']}"
        )

    # --------------------------------------------------------
    # Hybrid
    # --------------------------------------------------------

    print(
        "\nHYBRID RRF TOP RESULTS"
    )

    for rank, result in enumerate(
            hybrid_results,
            start=1
    ):

        dense_rank = (
            result["dense_rank"]
            if result["dense_rank"]
               is not None
            else "-"
        )

        bm25_rank = (
            result["bm25_rank"]
            if result["bm25_rank"]
               is not None
            else "-"
        )

        print(
            "\n" + "-" * 90
        )

        print(
            f"RESULT #{rank}"
        )

        print(
            f"RRF score: "
            f"{result['rrf_score']:.6f}"
        )

        print(
            f"Dense rank: "
            f"{dense_rank}"
        )

        print(
            f"BM25 rank: "
            f"{bm25_rank}"
        )

        print(
            f"Dense score: "
            f"{result['dense_score']}"
        )

        print(
            f"BM25 score: "
            f"{result['bm25_score']}"
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
# MAIN
# ============================================================

if __name__ == "__main__":

    client = QdrantClient(
        path=str(QDRANT_PATH)
    )

    try:

        print(
            "Loading documents from Qdrant..."
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
            "Hybrid search ready."
        )

        print(
            "\nDense + BM25 + RRF"
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

            (
                dense_results,
                bm25_results,
                hybrid_results
            ) = hybrid_search(
                client,
                bm25_index,
                query
            )

            print_results(
                query,
                dense_results,
                bm25_results,
                hybrid_results
            )

    finally:

        client.close()