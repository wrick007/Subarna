"""
Casual-conversation fast path, shared by every frontend (app.py's
Streamlit UI, backend/app/routers/chat.py's API).

A handful of greetings/pleasantries are answered directly, with zero LLM
calls -- no router, no RAG, no calculation, no specialist agent, no
synthesis, no critic, no formatter. This isn't a RAG or token-cost
optimization in the same sense as `finmate/rag.py`'s "Performance"
section (those all still route through the real pipeline; this instead
recognizes when the pipeline shouldn't run at all), but it's the same
spirit applied one level up: don't spend six LLM calls' worth of latency
and tokens confirming that "hi" doesn't need a budget calculation.

Originally lived only in app.py; extracted here once a second frontend
(the FastAPI backend) needed the identical check -- one implementation,
not two that can quietly drift apart.
"""

from __future__ import annotations

CASUAL_MESSAGES = {
    "hi",
    "hello",
    "hey",
    "hi there",
    "hello there",
    "good morning",
    "good afternoon",
    "good evening",
    "thanks",
    "thank you",
    "bye",
    "goodbye",
}

CASUAL_RESPONSE = "Hello! I'm FinMate AI. How can I help you with your finances today?"


def is_casual_message(message: str) -> bool:
    """True for simple greetings/pleasantries that don't need the
    financial pipeline at all. Whitespace- and case-insensitive; anything
    with real content beyond an exact match in `CASUAL_MESSAGES` (even
    "hi, what's my savings rate?") is NOT casual -- this is deliberately
    narrow, since a false positive here would silently skip a real
    question."""
    normalized = " ".join(message.lower().strip().split())
    return normalized in CASUAL_MESSAGES
