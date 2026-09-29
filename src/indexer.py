from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from data_loader import load_pdf
from chunker import chunk_pages
from embedder import embed_texts


PROJECT_ROOT = Path(__file__).resolve().parent.parent

QDRANT_PATH = (
        PROJECT_ROOT
        / "storage"
        / "qdrant"
)

COLLECTION_NAME = "kapston_documents_v2"

VECTOR_SIZE = 384


def index_documents():

    print("Loading document...")

    pages = load_pdf()

    print(
        f"Loaded {len(pages)} pages."
    )

    print("Chunking document...")

    chunks = chunk_pages(pages)

    print(
        f"Created {len(chunks)} chunks."
    )

    print("Creating embeddings...")

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = embed_texts(
        texts
    )

    print(
        f"Embedding shape: "
        f"{embeddings.shape}"
    )

    print("Opening Qdrant...")

    client = QdrantClient(
        path=str(QDRANT_PATH)
    )

    try:

        existing_collections = {
            collection.name
            for collection
            in client.get_collections().collections
        }

        if COLLECTION_NAME in existing_collections:

            raise RuntimeError(
                f"Collection "
                f"'{COLLECTION_NAME}' "
                f"already exists. "
                f"Refusing to overwrite it."
            )

        print(
            f"Creating collection: "
            f"{COLLECTION_NAME}"
        )

        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE,
            ),
        )

        points = []

        for chunk, embedding in zip(
                chunks,
                embeddings
        ):

            points.append(
                PointStruct(
                    id=chunk["chunk"],
                    vector=embedding.tolist(),
                    payload={
                        "source": "Kapston Home Services Documentation",
                        "page": chunk["page"],
                        "chunk": chunk["chunk"],
                        "page_chunk": chunk["page_chunk"],
                        "text": chunk["text"],
                    },
                )
            )

        print(
            f"Uploading {len(points)} points..."
        )

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=True,
        )

        count = client.count(
            collection_name=COLLECTION_NAME,
            exact=True,
        ).count

        print()
        print("INDEXING COMPLETE")
        print(
            f"Collection: {COLLECTION_NAME}"
        )
        print(
            f"Chunks: {len(chunks)}"
        )
        print(
            f"Points: {count}"
        )

        if count != len(chunks):
            raise RuntimeError(
                "Index verification failed."
            )

        print(
            "Index verification passed."
        )

    finally:
        client.close()


if __name__ == "__main__":
    index_documents()