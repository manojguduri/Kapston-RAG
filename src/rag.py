from pathlib import Path

from qdrant_client import QdrantClient

from retriever import Retriever
from prompter import build_prompt
from generator import generate


PROJECT_ROOT = Path(__file__).resolve().parent.parent
QDRANT_PATH = PROJECT_ROOT / "storage" / "qdrant"


class KapstonRAG:

    def __init__(self):

        self.client = QdrantClient(
            path=str(QDRANT_PATH)
        )

        self.retriever = Retriever(
            self.client
        )


    def ask(
            self,
            query: str
    ) -> tuple[str, list[dict]]:

        results = self.retriever.retrieve(
            query
        )

        prompt = build_prompt(
            query,
            results
        )

        answer = generate(
            prompt
        )

        return answer, results


    def close(self):

        self.client.close()