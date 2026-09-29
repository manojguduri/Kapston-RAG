import re

from transformers import AutoTokenizer


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# Maximum depth we expect in this document:
#
# 3.           -> level 1
# 3.9          -> level 2
#
# We are deliberately NOT treating arbitrary numeric-looking
# lines as headings.
HEADING_PATTERN = re.compile(
    r"^("
    r"\d+\.\s+[A-Z][^\n]*"
    r"|"
    r"\d+\.\d+\s+[A-Z][^\n]*"
    r")$",
    re.MULTILINE
)


# ============================================================
# CLEAN REPEATED PDF PAGE FURNITURE
# ============================================================

def clean_page_text(text: str) -> str:
    """
    Remove repeated PDF header/footer text before chunking.

    Repeated document branding is useful visually but creates
    unnecessary noise when embedded repeatedly.
    """

    text = re.sub(
        r"KAPSTON HOME SERVICES\s*"
        r"Company & Customer Documentation\s*"
        r"Source:\s*kapstonhomeservices\.in\s*\|\s*"
        r"Consolidated from website content\s*"
        r"Page\s+\d+",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


# ============================================================
# TOKEN HELPERS
# ============================================================

def encode_tokens(text: str) -> list[int]:

    return tokenizer.encode(
        text,
        add_special_tokens=False,
        truncation=False,
        verbose=False
    )


def token_count(text: str) -> int:

    return len(
        encode_tokens(text)
    )


# ============================================================
# HEADING HELPERS
# ============================================================

def parse_heading(heading: str):
    """
    Parse:

        3. Service Portfolio

    into:

        number = "3"
        title  = "Service Portfolio"

    and:

        3.9 AC Services

    into:

        number = "3.9"
        title  = "AC Services"
    """

    match = re.match(
        r"^(\d+(?:\.\d+)?)\.?\s+(.+)$",
        heading.strip()
    )

    if not match:
        return None

    number = match.group(1)
    title = match.group(2).strip()

    return number, title


def heading_level(number: str) -> int:
    """
    3       -> level 1
    3.9     -> level 2
    """

    return len(
        number.split(".")
    )


def is_valid_heading(
        heading: str,
        current_parent_number: str | None
) -> bool:
    """
    Validate candidate headings.

    This prevents content such as:

        14 May 2026
        11 Jun 2026

    from becoming headings.

    It also prevents decimal-looking metrics from casually
    creating a subsection under the wrong parent.

    A level-2 heading such as 3.9 is accepted only while its
    parent section is 3.
    """

    parsed = parse_heading(
        heading
    )

    if not parsed:
        return False

    number, title = parsed

    level = heading_level(
        number
    )

    # ---------------------------------------------
    # Top-level section:
    #
    # 3. Service Portfolio
    # 4. Service Delivery Commitments
    # ---------------------------------------------

    if level == 1:
        return True

    # ---------------------------------------------
    # Subsection:
    #
    # 3.9 AC Services
    #
    # Parent must currently be section 3.
    # ---------------------------------------------

    if level == 2:

        parent_number = number.split(".")[0]

        return (
                current_parent_number
                == parent_number
        )

    return False


# ============================================================
# FALLBACK SPLITTING
# ============================================================

def hard_token_split(
        text: str,
        max_tokens: int
) -> list[str]:
    """
    Last-resort token splitting.

    Normal chunks are NOT created this way.
    """

    token_ids = encode_tokens(
        text
    )

    chunks = []

    for start in range(
            0,
            len(token_ids),
            max_tokens
    ):

        piece_ids = token_ids[
                    start:start + max_tokens
                    ]

        piece = tokenizer.decode(
            piece_ids,
            skip_special_tokens=True
        ).strip()

        if piece:
            chunks.append(
                piece
            )

    return chunks


def split_into_units(
        text: str
) -> list[str]:
    """
    Split oversized structural sections using natural
    sentence/newline boundaries.
    """

    units = re.split(
        r"(?<=[.!?])\s+|\n+",
        text
    )

    return [
        unit.strip()
        for unit in units
        if unit.strip()
    ]


def split_large_section(
        heading: str,
        text: str,
        max_tokens: int
) -> list[str]:
    """
    Keep the complete structural section whenever possible.

    Only sections exceeding max_tokens are subdivided.

    Every child retains its heading.
    """

    if token_count(text) <= max_tokens:

        return [
            text.strip()
        ]

    units = split_into_units(
        text
    )

    # Avoid duplicating the heading.
    if (
            units
            and units[0].strip()
            == heading.strip()
    ):
        units = units[1:]

    heading_tokens = token_count(
        heading
    )

    available_tokens = (
            max_tokens - heading_tokens
    )

    if available_tokens <= 0:

        raise ValueError(
            "max_tokens is too small "
            "for the section heading."
        )

    chunks = []
    current_units = []

    for unit in units:

        # -----------------------------------------
        # One individual unit is enormous.
        # -----------------------------------------

        if token_count(unit) > available_tokens:

            if current_units:

                chunks.append(
                    heading
                    + "\n"
                    + "\n".join(
                        current_units
                    )
                )

                current_units = []

            pieces = hard_token_split(
                unit,
                available_tokens
            )

            for piece in pieces:

                chunks.append(
                    heading
                    + "\n"
                    + piece
                )

            continue

        # -----------------------------------------
        # Try adding unit to current child.
        # -----------------------------------------

        candidate = (
                heading
                + "\n"
                + "\n".join(
            current_units + [unit]
        )
        )

        if (
                current_units
                and token_count(candidate)
                > max_tokens
        ):

            chunks.append(
                heading
                + "\n"
                + "\n".join(
                    current_units
                )
            )

            current_units = [
                unit
            ]

        else:

            current_units.append(
                unit
            )

    if current_units:

        chunks.append(
            heading
            + "\n"
            + "\n".join(
                current_units
            )
        )

    return [
        chunk.strip()
        for chunk in chunks
        if chunk.strip()
    ]


# ============================================================
# STRUCTURE EXTRACTION
# ============================================================

def extract_sections(
        pages: list[dict]
) -> list[dict]:
    """
    Parse the document sequentially.

    Sections can span page boundaries.

    We maintain the current top-level parent so that a
    subsection such as 3.9 is only recognized under section 3.
    """

    sections = []

    current_heading = None
    current_number = None

    current_parent_number = None
    current_parent_title = None

    current_text_parts = []
    current_pages = []

    for page in pages:

        page_number = page["page"]

        text = clean_page_text(
            page["text"]
        )

        # Find syntactically possible headings first.
        candidate_matches = list(
            HEADING_PATTERN.finditer(
                text
            )
        )

        valid_matches = []

        # This temporary parent tracks structure as we move
        # through headings on the current page.
        temp_parent = (
            current_parent_number
        )

        for match in candidate_matches:

            candidate = (
                match.group().strip()
            )

            parsed = parse_heading(
                candidate
            )

            if not parsed:
                continue

            number, title = parsed

            level = heading_level(
                number
            )

            if is_valid_heading(
                    candidate,
                    temp_parent
            ):

                valid_matches.append(
                    match
                )

                # Encountering a top-level heading changes
                # the active parent for later subsections.
                if level == 1:
                    temp_parent = number

        # ---------------------------------------------
        # No structural heading on this page.
        # ---------------------------------------------

        if not valid_matches:

            if current_heading is not None:

                if text:

                    current_text_parts.append(
                        text
                    )

                    if (
                            page_number
                            not in current_pages
                    ):
                        current_pages.append(
                            page_number
                        )

            elif text:

                sections.append({
                    "number": None,
                    "heading":
                        "Document Introduction",
                    "parent_number": None,
                    "parent_title": None,
                    "pages":
                        [page_number],
                    "text":
                        text
                })

            continue

        cursor = 0

        # ---------------------------------------------
        # Process structural headings sequentially.
        # ---------------------------------------------

        for match in valid_matches:

            before_heading = text[
                             cursor:match.start()
                             ].strip()

            # Anything before the new heading belongs
            # to the previous open section.
            if before_heading:

                if current_heading is not None:

                    current_text_parts.append(
                        before_heading
                    )

                    if (
                            page_number
                            not in current_pages
                    ):
                        current_pages.append(
                            page_number
                        )

                else:

                    sections.append({
                        "number": None,
                        "heading":
                            "Document Introduction",
                        "parent_number": None,
                        "parent_title": None,
                        "pages":
                            [page_number],
                        "text":
                            before_heading
                    })

            # Save previous open section.
            if current_heading is not None:

                sections.append({
                    "number":
                        current_number,

                    "heading":
                        current_heading,

                    "parent_number":
                        current_parent_number,

                    "parent_title":
                        current_parent_title,

                    "pages":
                        current_pages.copy(),

                    "text":
                        "\n".join(
                            current_text_parts
                        ).strip()
                })

            # Start new section.
            heading = (
                match.group().strip()
            )

            number, title = parse_heading(
                heading
            )

            level = heading_level(
                number
            )

            # -----------------------------------------
            # Top-level heading
            # -----------------------------------------

            if level == 1:

                current_parent_number = (
                    number
                )

                current_parent_title = (
                    title
                )

                parent_number = None
                parent_title = None

            # -----------------------------------------
            # Child subsection
            # -----------------------------------------

            else:

                parent_number = (
                    current_parent_number
                )

                parent_title = (
                    current_parent_title
                )

            current_number = number
            current_heading = heading

            current_text_parts = [
                heading
            ]

            current_pages = [
                page_number
            ]

            # Store the parent belonging specifically to
            # this newly opened section.
            #
            # We temporarily attach these attributes through
            # local variables below.
            if level == 1:

                section_parent_number = None
                section_parent_title = None

            else:

                section_parent_number = (
                    parent_number
                )

                section_parent_title = (
                    parent_title
                )

            # Save these values for when this section closes.
            current_section_parent_number = (
                section_parent_number
            )

            current_section_parent_title = (
                section_parent_title
            )

            # Attach to function-local state used when
            # saving subsequent sections.
            current_parent_for_section_number = (
                current_section_parent_number
            )

            current_parent_for_section_title = (
                current_section_parent_title
            )

            cursor = match.end()

        # Everything after final heading belongs to
        # the currently open section.
        remaining = text[
                    cursor:
                    ].strip()

        if remaining:

            current_text_parts.append(
                remaining
            )

            if (
                    page_number
                    not in current_pages
            ):
                current_pages.append(
                    page_number
                )

    # ---------------------------------------------
    # Save final section.
    # ---------------------------------------------

    if current_heading is not None:

        parsed = parse_heading(
            current_heading
        )

        number = (
            parsed[0]
            if parsed
            else None
        )

        level = (
            heading_level(number)
            if number
            else 1
        )

        if level == 1:

            parent_number = None
            parent_title = None

        else:

            parent_number = (
                current_parent_number
            )

            parent_title = (
                current_parent_title
            )

        sections.append({
            "number":
                number,

            "heading":
                current_heading,

            "parent_number":
                parent_number,

            "parent_title":
                parent_title,

            "pages":
                current_pages.copy(),

            "text":
                "\n".join(
                    current_text_parts
                ).strip()
        })

    return sections


# ============================================================
# EMBEDDING CONTEXT
# ============================================================

def build_embedding_text(
        heading: str,
        parent_title: str | None,
        text: str
) -> str:
    """
    Build the text that will actually be embedded.

    We retain the original chunk text separately.

    Example original text:

        3.9 AC Services
        Cool Comfort...
        AC Installation...
        AC Repair...

    Embedding text:

        Service Portfolio > AC Services

        3.9 AC Services
        Cool Comfort...
        AC Installation...
        AC Repair...

    This gives the embedding useful hierarchical context
    without polluting the original source text.
    """

    parsed = parse_heading(
        heading
    )

    if parsed:
        _, title = parsed
    else:
        title = heading

    if parent_title:

        context = (
            f"{parent_title} > {title}"
        )

    else:

        context = title

    return (
            context
            + "\n\n"
            + text
    ).strip()


# ============================================================
# FINAL STRUCTURE-AWARE CHUNKER
# ============================================================

def structure_aware_chunks(
        pages: list[dict],
        max_tokens: int = 256
) -> list[dict]:

    sections = extract_sections(
        pages
    )

    chunks = []

    chunk_id = 0

    for section in sections:

        pieces = split_large_section(
            heading=section["heading"],
            text=section["text"],
            max_tokens=max_tokens
        )

        for part_number, piece in enumerate(
                pieces
        ):

            embedding_text = (
                build_embedding_text(
                    heading=section["heading"],
                    parent_title=section[
                        "parent_title"
                    ],
                    text=piece
                )
            )

            chunks.append({
                "chunk":
                    chunk_id,

                "number":
                    section["number"],

                "heading":
                    section["heading"],

                "parent_number":
                    section[
                        "parent_number"
                    ],

                "parent_title":
                    section[
                        "parent_title"
                    ],

                "pages":
                    section["pages"],

                "part":
                    part_number,

                # Original source text.
                #
                # This is what Gemini should eventually see.
                "text":
                    piece,

                # Context-enriched representation.
                #
                # This is what we should embed/search.
                "embedding_text":
                    embedding_text,

                "tokens":
                    token_count(piece)
            })

            chunk_id += 1

    return chunks


# ============================================================
# INSPECTION
# ============================================================

if __name__ == "__main__":

    from src.data_loader import load_pdf, PDF_PATH

    pages = load_pdf(
        PDF_PATH
    )

    chunks = structure_aware_chunks(
        pages,
        max_tokens=256
    )

    print(
        f"\nTotal structure-aware chunks: "
        f"{len(chunks)}"
    )

    print("\n" + "=" * 80)
    print("DETECTED STRUCTURE")
    print("=" * 80)

    for chunk in chunks:

        print("\n" + "-" * 80)

        print(
            f"Chunk: {chunk['chunk']}"
        )

        print(
            f"Number: {chunk['number']}"
        )

        print(
            f"Heading: {chunk['heading']}"
        )

        print(
            f"Parent: {chunk['parent_title']}"
        )

        print(
            f"Pages: {chunk['pages']}"
        )

        print(
            f"Part: {chunk['part']}"
        )

        print(
            f"Tokens: {chunk['tokens']}"
        )

        print("\nEMBEDDING TEXT:")

        print(
            chunk["embedding_text"]
        )