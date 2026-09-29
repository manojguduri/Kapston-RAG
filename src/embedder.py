from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    MODEL_NAME
)


def embed_texts(
        texts: list[str]
):
    return embedding_model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def embed_query(
        query: str
):
    return embedding_model.encode(
        query,
        normalize_embeddings=True,
        show_progress_bar=False,
    )