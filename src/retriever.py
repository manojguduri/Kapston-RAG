import re
from collections import Counter
from math import log
from pathlib import Path

from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder

from embedder import embed_query


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

QDRANT_PATH = (
        PROJECT_ROOT
        / "storage"
        / "qdrant"
)

COLLECTION_NAME = "kapston_documents_v2"

# Candidate generation
DENSE_TOP_K = 20
BM25_TOP_K = 20

# RRF candidate pool passed to the cross-encoder
HYBRID_TOP_K = 20

# Final context sent to the generator
FINAL_TOP_K = 5

# Reciprocal Rank Fusion
RRF_K = 60

# Existing provisional dense threshold
MIN_RELEVANCE_SCORE = 0.20

# BM25
BM25_K1 = 1.5
BM25_B = 0.75

# Cross-encoder
RERANKER_MODEL_NAME = (
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize(text: str) -> list[str]:
    """
    Simple tokenizer used by BM25.

    Example:
        "What AC services?"
        ->
        ["what", "ac", "services"]
    """

    return re.findall(
        r"\b[a-z0-9]+\b",
        text.lower()
    )


# ============================================================
# LOAD DOCUMENTS FROM QDRANT
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

            documents.append(
                {
                    "id": point.id,
                    "page": payload.get("page"),
                    "chunk": payload.get("chunk"),
                    "page_chunk": payload.get(
                        "page_chunk"
                    ),
                    "source": payload.get(
                        "source"
                    ),
                    "text": payload.get(
                        "text",
                        ""
                    )
                }
            )

        if next_offset is None:
            break

        offset = next_offset

    return documents


# ============================================================
# BM25
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

        # Preserve the behavior of our evaluated
        # hybrid_search.py: repeated query terms
        # contribute only once.
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
# DENSE RETRIEVAL
# ============================================================

def dense_search(
        client: QdrantClient,
        query: str,
        limit: int = DENSE_TOP_K
) -> list[dict]:

    query_embedding = embed_query(
        query
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

        results.append(
            {
                "id": point.id,
                "dense_score": score,
                "page": payload.get("page"),
                "chunk": payload.get("chunk"),
                "page_chunk": payload.get(
                    "page_chunk"
                ),
                "source": payload.get(
                    "source"
                ),
                "text": payload.get(
                    "text",
                    ""
                )
            }
        )

    return results


# ============================================================
# RECIPROCAL RANK FUSION
# ============================================================

def reciprocal_rank_fusion(
        dense_results: list[dict],
        bm25_results: list[dict]
) -> list[dict]:

    fused = {}

    # Dense contribution
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

    # BM25 contribution
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
# CROSS-ENCODER RERANKER
# ============================================================

class Retriever:

    def __init__(
            self,
            client: QdrantClient
    ):

        self.client = client

        print(
            "Loading documents for BM25..."
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

        self.bm25_index = BM25Index(
            documents
        )

        print(
            "Loading cross-encoder reranker..."
        )

        self.reranker = CrossEncoder(
            RERANKER_MODEL_NAME
        )

        print(
            "Retriever ready."
        )


    def rerank(
            self,
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

        scores = self.reranker.predict(
            pairs,
            show_progress_bar=False
        )

        reranked = []

        for candidate, score in zip(
                candidates,
                scores
        ):

            result = candidate.copy()

            result[
                "reranker_score"
            ] = float(score)

            reranked.append(
                result
            )

        reranked.sort(
            key=lambda result:
            result["reranker_score"],
            reverse=True
        )

        return reranked


    # ========================================================
    # COMPLETE RETRIEVAL PIPELINE
    # ========================================================

    def retrieve(
            self,
            query: str,
            final_top_k: int = FINAL_TOP_K
    ) -> list[dict]:

        # 1. Dense
        dense_results = dense_search(
            self.client,
            query,
            DENSE_TOP_K
        )

        # 2. BM25
        bm25_results = self.bm25_index.search(
            query,
            BM25_TOP_K
        )

        # 3. RRF
        fused_results = reciprocal_rank_fusion(
            dense_results,
            bm25_results
        )

        # 4. Keep Top 20 hybrid candidates
        hybrid_candidates = fused_results[
                            :HYBRID_TOP_K
                            ]

        # 5. Cross-encoder
        reranked_results = self.rerank(
            query,
            hybrid_candidates
        )

        # 6. Final Top 5
        return reranked_results[
               :final_top_k
               ]