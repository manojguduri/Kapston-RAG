import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from src.data_loader import load_pdf, chunk_text, PDF_PATH


model = SentenceTransformer("all-MiniLM-L6-v2")


# -------------------------
# Build our chunks
# -------------------------

pages = load_pdf(PDF_PATH)

all_chunks = []

for page in pages:

    chunks = chunk_text(
        page["text"],
        chunk_size=500,
        overlap=100
    )

    for chunk_number, chunk in enumerate(chunks):

        all_chunks.append({
            "source": PDF_PATH.name,
            "page": page["page"],
            "chunk": chunk_number,
            "text": chunk
        })

chunk_texts = [
    chunk["text"]
    for chunk in all_chunks
]

chunk_embeddings = model.encode(chunk_texts)

print("Number of embeddings:", len(chunk_embeddings))
print("Dimensions per embedding:", len(chunk_embeddings[0]))
print("Shape:", chunk_embeddings.shape)

query = "How do I cook chicken biryani?"

query_embedding = model.encode(query)

print("\nQuery:")
print(query)

print("Query embedding shape:")
print(query_embedding.shape)

similarities = cos_sim(
    query_embedding,
    chunk_embeddings
)[0]

top_k = 5

top_indices = similarities.argsort(
    descending=True
)[:top_k]

print("\n\nTOP RESULTS")
print("=" * 60)

for rank, index in enumerate(top_indices, start=1):

    index = index.item()

    chunk = all_chunks[index]
    score = similarities[index].item()

    print(f"\nRESULT {rank}")
    print(f"Similarity: {score:.4f}")
    print(f"Page: {chunk['page']}")
    print(f"Chunk: {chunk['chunk']}")
    print("-" * 60)
    print(chunk["text"])