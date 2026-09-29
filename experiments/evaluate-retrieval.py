import json
from pathlib import Path

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


COLLECTION_NAME = "kapston_documents"

TOP_K = 5
MIN_RELEVANCE_SCORE = 0.20

BASE_DIR = Path(__file__).parent.parent

STORAGE_PATH = BASE_DIR / "storage" / "qdrant"
EVALUATION_FILE = BASE_DIR / "evaluation" / "questions.json"


# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Connect to existing vector DB
client = QdrantClient(
    path=str(STORAGE_PATH)
)


def retrieve(question):

    query_embedding = model.encode(question)

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding.tolist(),
        limit=TOP_K
    ).points

    relevant_results = [
        result
        for result in results
        if result.score >= MIN_RELEVANCE_SCORE
    ]

    return relevant_results


def main():

    with open(EVALUATION_FILE, "r", encoding="utf-8") as file:
        questions = json.load(file)

    total = len(questions)

    answerable_total = 0
    answerable_hits = 0

    unanswerable_total = 0
    correct_rejections = 0

    print("\nRAG RETRIEVAL EVALUATION")
    print("=" * 70)

    for item in questions:

        question = item["question"]
        answerable = item["answerable"]
        expected_page = item["expected_page"]

        results = retrieve(question)

        retrieved_pages = [
            result.payload["page"]
            for result in results
        ]

        top_score = (
            results[0].score
            if results
            else None
        )

        # --------------------------------
        # Answerable questions
        # --------------------------------

        if answerable:

            answerable_total += 1

            hit = expected_page in retrieved_pages

            if hit:
                answerable_hits += 1
                status = "PASS"
            else:
                status = "FAIL"

        # --------------------------------
        # Unanswerable questions
        # --------------------------------

        else:

            unanswerable_total += 1

            hit = len(results) == 0

            if hit:
                correct_rejections += 1
                status = "PASS"
            else:
                status = "FAIL"

        print(f"\n[{status}] {question}")
        print(f"Expected page: {expected_page}")
        print(f"Retrieved pages: {retrieved_pages}")

        if top_score is not None:
            print(f"Top score: {top_score:.4f}")
        else:
            print("Top score: None")

    # --------------------------------
    # Calculate metrics
    # --------------------------------

    retrieval_accuracy = (
        answerable_hits / answerable_total
        if answerable_total
        else 0
    )

    rejection_accuracy = (
        correct_rejections / unanswerable_total
        if unanswerable_total
        else 0
    )

    print("\n")
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(
        f"Answerable retrieval: "
        f"{answerable_hits}/{answerable_total} "
        f"({retrieval_accuracy:.2%})"
    )

    print(
        f"Unanswerable rejection: "
        f"{correct_rejections}/{unanswerable_total} "
        f"({rejection_accuracy:.2%})"
    )


if __name__ == "__main__":

    try:
        main()

    finally:
        client.close()