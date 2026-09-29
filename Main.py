import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"

sys.path.insert(
    0,
    str(SRC_PATH)
)

from rag import KapstonRAG


def print_sources(
        results: list[dict]
):

    print(
        "\n" + "=" * 90
    )

    print(
        "RETRIEVED CONTEXT"
    )

    print(
        "=" * 90
    )

    for index, result in enumerate(
            results,
            start=1
    ):

        print(
            f"\n#{index}"
        )

        print(
            f"Page: {result.get('page')}"
        )

        print(
            f"Chunk: {result.get('chunk')}"
        )

        print(
            f"RRF score: "
            f"{result.get('rrf_score', 0):.6f}"
        )

        print(
            f"Reranker score: "
            f"{result.get('reranker_score', 0):.4f}"
        )

        print(
            result.get(
                "text",
                ""
            )
        )


def main():

    print(
        "\nStarting Kapston RAG..."
    )

    rag = KapstonRAG()

    try:

        print(
            "\nKapston Advanced RAG CLI"
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

            try:

                answer, results = rag.ask(
                    query
                )

                print_sources(
                    results
                )

                print(
                    "\n" + "=" * 90
                )

                print(
                    "ANSWER"
                )

                print(
                    "=" * 90
                )

                print(
                    answer
                )

            except Exception as error:

                print(
                    f"\nERROR: {error}"
                )

    finally:

        rag.close()

        print(
            "\nQdrant connection closed."
        )


if __name__ == "__main__":
    main()