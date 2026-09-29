from transformers import AutoTokenizer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 256
CHUNK_OVERLAP = 48


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


def chunk_text_by_tokens(
        text: str,
        chunk_size: int = CHUNK_SIZE,
        overlap: int = CHUNK_OVERLAP,
) -> list[str]:

    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
        truncation=False,
    )

    chunks = []

    step = chunk_size - overlap

    for start in range(
            0,
            len(token_ids),
            step
    ):
        chunk_ids = token_ids[
                    start:start + chunk_size
                    ]

        if not chunk_ids:
            continue

        chunk_text = tokenizer.decode(
            chunk_ids,
            skip_special_tokens=True,
        ).strip()

        if chunk_text:
            chunks.append(chunk_text)

    return chunks


def chunk_pages(
        pages: list[dict]
) -> list[dict]:

    chunks = []

    chunk_id = 0

    for page in pages:

        page_chunks = chunk_text_by_tokens(
            page["text"]
        )

        for page_chunk, text in enumerate(
                page_chunks
        ):
            chunks.append(
                {
                    "chunk": chunk_id,
                    "page": page["page"],
                    "page_chunk": page_chunk,
                    "text": text,
                }
            )

            chunk_id += 1

    return chunks