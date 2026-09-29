import sys
from pathlib import Path

import streamlit as st


# ============================================================
# IMPORT PROJECT MODULES
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"

sys.path.insert(
    0,
    str(SRC_PATH)
)

from rag import KapstonRAG


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Kapston RAG",
    page_icon="🏠",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title(
    "🏠 Kapston Home Services"
)

st.caption(
    "Advanced RAG Assistant"
)

st.markdown(
    """
Ask questions about Kapston Home Services using the
company documentation.
"""
)


# ============================================================
# INITIALIZE RAG
# ============================================================

@st.cache_resource
def load_rag():

    return KapstonRAG()


with st.spinner(
        "Loading retrieval models..."
):
    rag = load_rag()


# ============================================================
# CHAT HISTORY
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "About"
    )

    st.write(
        """
This assistant answers questions using
Kapston Home Services documentation.
"""
    )

    st.divider()

    st.subheader(
        "RAG Pipeline"
    )

    st.markdown(
        """
**Retrieval**
- MiniLM dense search
- BM25 lexical search
- Reciprocal Rank Fusion

**Reranking**
- Cross-Encoder

**Generation**
- Gemini
"""
    )

    st.divider()

    if st.button(
            "Clear conversation",
            use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
            message["role"]
    ):

        st.markdown(
            message["content"]
        )

        if (
                message["role"] == "assistant"
                and message.get("sources")
        ):

            with st.expander(
                    "View retrieved sources"
            ):

                for index, source in enumerate(
                        message["sources"],
                        start=1
                ):

                    st.markdown(
                        f"### Source {index}"
                    )

                    st.caption(
                        f"Page {source.get('page')} "
                        f"• Chunk {source.get('chunk')}"
                    )

                    st.write(
                        source.get(
                            "text",
                            ""
                        )
                    )

                    st.divider()


# ============================================================
# CHAT INPUT
# ============================================================

query = st.chat_input(
    "Ask about Kapston Home Services..."
)


if query:

    # --------------------------------------------------------
    # Display user message
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )

    with st.chat_message(
            "user"
    ):

        st.markdown(
            query
        )


    # --------------------------------------------------------
    # Run RAG
    # --------------------------------------------------------

    with st.chat_message(
            "assistant"
    ):

        with st.spinner(
                "Searching Kapston documentation..."
        ):

            try:

                answer, results = rag.ask(
                    query
                )

                st.markdown(
                    answer
                )


                # --------------------------------------------
                # Sources
                # --------------------------------------------

                with st.expander(
                        "View retrieved sources"
                ):

                    for index, result in enumerate(
                            results,
                            start=1
                    ):

                        st.markdown(
                            f"### Source {index}"
                        )

                        st.caption(
                            f"Page "
                            f"{result.get('page')} "
                            f"• Chunk "
                            f"{result.get('chunk')}"
                        )

                        st.write(
                            result.get(
                                "text",
                                ""
                            )
                        )

                        st.caption(
                            f"RRF: "
                            f"{result.get('rrf_score', 0):.6f}"
                            f" | "
                            f"Reranker: "
                            f"{result.get('reranker_score', 0):.4f}"
                        )

                        st.divider()


                # --------------------------------------------
                # Save assistant response
                # --------------------------------------------

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": results
                    }
                )

            except Exception as error:

                st.error(
                    f"Something went wrong: {error}"
                )