
"""
FinMate AI - Streamlit chat UI.

Run with: streamlit run app.py

A single-page chat app over the LangGraph pipeline in
`finmate.orchestrator`. The sidebar shows a read-only view of the user's
stored profile and offers a "forget this user's data" control.
"""

from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from finmate import db, rag
from finmate.casual import is_casual_message, pick_casual_response
from finmate.llm import LLMClient, LLMConfigError
from finmate.orchestrator import run_finmate


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

load_dotenv()


# ---------------------------------------------------------------------------
# Streamlit configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="FinMate AI",
    page_icon="💰",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

DB_PATH = os.environ.get(
    "FINMATE_DB_PATH",
    db.DEFAULT_DB_PATH,
)

db.init_db(DB_PATH)


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------

def get_llm_client() -> tuple[LLMClient | None, str | None]:
    """
    Return (client, error_message).

    Exactly one of the two returned values should be None.
    """

    if "llm_client" not in st.session_state:
        try:
            st.session_state["llm_client"] = LLMClient()
            st.session_state["llm_client_error"] = None

        except LLMConfigError as exc:
            st.session_state["llm_client"] = None
            st.session_state["llm_client_error"] = str(exc)

    return (
        st.session_state["llm_client"],
        st.session_state["llm_client_error"],
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("FinMate AI")

    user_id = st.text_input(
        "User ID",
        value=st.session_state.get("user_id", "demo_user"),
    )

    st.session_state["user_id"] = user_id

    # ---------------------------------------------------------------
    # Stored profile
    # ---------------------------------------------------------------

    st.subheader("Stored profile (read-only)")

    profile = db.get_user_profile(
        user_id,
        db_path=DB_PATH,
    )

    if profile is None:
        st.caption(
            "No profile stored yet. Try: "
            "'My monthly income is 90000 INR' "
            "or run scripts/seed_demo_data.py for a full demo profile."
        )
    else:
        st.json(
            profile.model_dump(),
            expanded=False,
        )

    # ---------------------------------------------------------------
    # Data control
    # ---------------------------------------------------------------

    st.divider()

    st.subheader("Data control")

    confirm = st.checkbox(
        "I understand this permanently deletes this user's stored data"
    )

    if st.button(
        "Forget this user's data",
        type="secondary",
        disabled=not confirm,
    ):
        db.delete_user_data(
            user_id,
            db_path=DB_PATH,
        )

        rag.delete_user_vector_index(user_id)

        st.session_state.pop(
            f"messages_{user_id}",
            None,
        )

        st.success(
            f"All stored data for '{user_id}' has been deleted."
        )

        st.rerun()

    # ---------------------------------------------------------------
    # Disclaimer
    # ---------------------------------------------------------------

    st.divider()

    st.caption(
        "FinMate AI provides informational assistance only, not guaranteed "
        "financial advice, and cannot move money, execute trades, or take "
        "any external action on your accounts."
    )


# ---------------------------------------------------------------------------
# Main chat
# ---------------------------------------------------------------------------

st.title("FinMate AI")

st.caption(
    "A multi-agent personal finance assistant that uses your stored "
    "profile, financial records, and deterministic calculations when "
    "relevant to your request."
)


# ---------------------------------------------------------------------------
# Initialize LLM
# ---------------------------------------------------------------------------

llm_client, llm_error = get_llm_client()

if llm_client is None:
    st.error(
        f"{llm_error}\n\n"
        "Copy .env.example to .env, add a free key from Groq "
        "(console.groq.com) or Gemini (aistudio.google.com/apikey), "
        "and restart. "
        "(Profile storage, calculations, and tests work without an API key; "
        "only the chat itself needs one.)"
    )

else:
    st.caption(
        f"Using {llm_client.provider} · {llm_client.model}"
    )


# ---------------------------------------------------------------------------
# Conversation state
# ---------------------------------------------------------------------------

messages_key = f"messages_{user_id}"

if messages_key not in st.session_state:
    st.session_state[messages_key] = []


# ---------------------------------------------------------------------------
# Display previous messages
# ---------------------------------------------------------------------------

for msg in st.session_state[messages_key]:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant" and msg.get("meta"):
            with st.expander("Pipeline details"):
                st.json(msg["meta"])


# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------

user_input = st.chat_input(
    "Ask about your budget, cash flow, goals, debts, or spending...",
    disabled=llm_client is None,
)


# ---------------------------------------------------------------------------
# Process new message
# ---------------------------------------------------------------------------

if user_input:

    # ---------------------------------------------------------------
    # Save/display user message
    # ---------------------------------------------------------------

    st.session_state[messages_key].append(
        {
            "role": "user",
            "content": user_input,
        }
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    # ---------------------------------------------------------------
    # Assistant response
    # ---------------------------------------------------------------

    with st.chat_message("assistant"):

        try:

            # =======================================================
            # FAST PATH
            # =======================================================
            #
            # Greetings and simple conversational messages DO NOT
            # invoke the financial pipeline.
            #
            # This means:
            #
            # Hi
            #   ↓
            # Direct response
            #
            # No:
            #   Router
            #   RAG
            #   Calculation
            #   Specialist
            #   Synthesis (also does final-response formatting as of the
            #              Priority-2 pipeline redesign -- see
            #              finmate/orchestrator.py's module docstring
            #              "Fewer sequential calls: synthesis + formatter
            #              merged"; there's no separate Formatter stage
            #              to list here anymore)
            #   Critic
            #
            # =======================================================

            if is_casual_message(user_input):

                response = pick_casual_response()

                st.markdown(response)

                st.session_state[messages_key].append(
                    {
                        "role": "assistant",
                        "content": response,
                    }
                )

            # =======================================================
            # FULL FINANCIAL PIPELINE
            # =======================================================

            else:

                with st.spinner(
                    "Routing, retrieving evidence, calculating, and verifying..."
                ):

                    # Short-term conversational memory (see
                    # finmate/orchestrator.py's module docstring
                    # "Conversation history"): everything already in
                    # this session except the user_input just appended
                    # above, as plain {role, content} dicts -- drop the
                    # "meta" key stored alongside assistant messages
                    # here, which the orchestrator doesn't take.
                    history = [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state[messages_key][:-1]
                    ]

                    result = run_finmate(
                        user_id,
                        user_input,
                        llm_client,
                        db_path=DB_PATH,
                        conversation_history=history,
                    )

                # ---------------------------------------------------
                # Final verified response
                # ---------------------------------------------------

                st.markdown(
                    result.final_response
                )

                # ---------------------------------------------------
                # Pipeline metadata
                # ---------------------------------------------------

                meta = {
                    "intent": result.router_output.intent,
                    "risk_level": result.router_output.risk_level,
                    "critic_passed": result.critic_passed,
                    "critic_retries_used": result.retry_count,
                }

                if result.critic_result:

                    meta["critic_errors"] = (
                        result.critic_result.errors
                    )

                    meta["critic_unsupported_claims"] = (
                        result.critic_result.unsupported_claims
                    )

                # ---------------------------------------------------
                # Pipeline details
                # ---------------------------------------------------

                with st.expander("Pipeline details"):
                    st.json(meta)

                # ---------------------------------------------------
                # Save assistant response
                # ---------------------------------------------------

                st.session_state[messages_key].append(
                    {
                        "role": "assistant",
                        "content": result.final_response,
                        "meta": meta,
                    }
                )

        # ===========================================================
        # PIPELINE ERROR
        # ===========================================================

        except Exception as exc:

            error_text = (
                "Something went wrong in the pipeline and I can't give "
                f"you a verified answer: {exc}"
            )

            st.error(error_text)

            st.session_state[messages_key].append(
                {
                    "role": "assistant",
                    "content": error_text,
                }
            )
