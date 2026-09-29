def build_context(
        results: list[dict]
) -> str:

    context_parts = []

    for index, result in enumerate(
            results,
            start=1
    ):

        page = result.get(
            "page",
            "Unknown"
        )

        text = result.get(
            "text",
            ""
        )

        context_parts.append(
            f"[Source {index} | Page {page}]\n"
            f"{text}"
        )

    return "\n\n".join(
        context_parts
    )


def build_prompt(
        query: str,
        results: list[dict]
) -> str:

    context = build_context(
        results
    )

    return f"""
You are a question-answering assistant for Kapston Home Services.

Answer the user's question using ONLY the documentation provided in the
CONTEXT below.

Rules:

1. Use only information explicitly supported by the context.
2. Do not use outside knowledge.
3. Do not invent or assume missing details.
4. If the context explicitly states that the requested information is
   unavailable, unspecified, not provided, or represented by a placeholder,
   explain that clearly.
5. If the context does not contain enough information to answer the question,
   respond exactly:

   I couldn't find that information in the Kapston documentation.

6. Keep the answer concise and directly relevant.
7. Mention the relevant page number or page numbers when answering.
8. Do not mention retrieval scores, embeddings, BM25, RRF, vector search,
   cross-encoders, or internal system implementation.

CONTEXT:

{context}

USER QUESTION:

{query}

ANSWER:
""".strip()