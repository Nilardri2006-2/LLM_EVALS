"""
src/app.py — Streamlit UI for the RAG pipeline.

Run from PROJECT ROOT:

    python -m streamlit run src/app.py
"""

import sys
import time
from pathlib import Path

# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ============================================================
# IMPORTS
# ============================================================

import streamlit as st

from src.rag_pipeline import RagPipeline


# ============================================================
# CONFIG
# ============================================================

ABSTENTION = (
    "I don't have enough information in the course material "
    "to answer that."
)

st.set_page_config(
    page_title="LLM Evals TA",
    page_icon="🎓",
    layout="centered",
)


# ============================================================
# PIPELINE
# ============================================================

@st.cache_resource(show_spinner="Loading retriever and reranker...")
def get_pipeline(fetch_k: int, top_k: int) -> RagPipeline:
    """
    Cache the RAG pipeline so the vector store and cross-encoder
    are not loaded on every Streamlit rerun.
    """
    return RagPipeline(
        fetch_k=fetch_k,
        top_k=top_k,
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Retrieval Settings")

    fetch_k = st.slider(
        "fetch_k",
        min_value=5,
        max_value=30,
        value=10,
        help=(
            "Number of chunks retrieved from the vector store "
            "before reranking."
        ),
    )

    top_k = st.slider(
        "top_k",
        min_value=1,
        max_value=10,
        value=5,
        help=(
            "Number of chunks kept by the cross-encoder "
            "and passed to the generator."
        ),
    )

    if top_k > fetch_k:
        st.warning(
            "top_k cannot effectively exceed fetch_k."
        )

    st.divider()

    show_context = st.toggle(
        "Show retrieved context",
        value=True,
    )

    if st.button(
        "Clear chat",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    st.markdown("### Pipeline")

    st.caption(
        "Vector Retriever → Cross-Encoder Reranker → "
        "Cohere Command A Generator"
    )

    st.caption(
        "Answers are grounded only in the course transcripts. "
        "If the material does not contain enough information, "
        "the assistant abstains."
    )


# ============================================================
# HEADER
# ============================================================

st.title("🎓 LLM Evals — Course TA")

st.caption(
    "Ask questions about the course transcripts. "
    "Answers are generated only from retrieved course material."
)


# ============================================================
# CHAT STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# CONTEXT RENDERER
# ============================================================

def render_context(
    context: list[str],
    latency: float | None = None,
) -> None:

    label = (
        f"📚 {len(context)} retrieved "
        f"chunk{'s' if len(context) != 1 else ''}"
    )

    if latency is not None:
        label += f" · {latency:.2f}s"

    with st.expander(label):

        for i, chunk in enumerate(context, start=1):

            st.markdown(f"**Chunk {i}**")

            st.text(chunk)

            if i != len(context):
                st.divider()


# ============================================================
# REPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if (
            message["role"] == "assistant"
            and show_context
            and message.get("context")
        ):
            render_context(
                message["context"],
                message.get("latency"),
            )


# ============================================================
# CHAT INPUT
# ============================================================

query = st.chat_input(
    "e.g. What is the difference between online and offline eval?"
)


if query:

    # --------------------------------------------------------
    # USER MESSAGE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query,
        }
    )

    with st.chat_message("user"):
        st.markdown(query)


    # --------------------------------------------------------
    # RAG RESPONSE
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        try:

            rag = get_pipeline(
                fetch_k,
                top_k,
            )

            with st.spinner(
                "Retrieving, reranking and generating..."
            ):

                started = time.perf_counter()

                result = rag.invoke(query)

                latency = time.perf_counter() - started


        except Exception as exc:

            st.error(
                f"Something went wrong: {exc}"
            )

            # Remove unanswered user message
            st.session_state.messages.pop()


        else:

            answer = result["answer"]
            context = result["context"]


            # ------------------------------------------------
            # ANSWER
            # ------------------------------------------------

            st.markdown(answer)


            # ------------------------------------------------
            # ABSTENTION
            # ------------------------------------------------

            if answer.strip() == ABSTENTION:

                st.info(
                    "The assistant abstained — "
                    "this isn't covered in the course transcripts."
                )


            # ------------------------------------------------
            # RETRIEVED CONTEXT
            # ------------------------------------------------

            if show_context:

                render_context(
                    context,
                    latency,
                )


            # ------------------------------------------------
            # SAVE ASSISTANT MESSAGE
            # ------------------------------------------------

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "context": context,
                    "latency": latency,
                }
            )