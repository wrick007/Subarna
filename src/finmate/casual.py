"""
Casual-conversation fast path, shared by every frontend (app.py's
Streamlit UI, backend/app/routers/chat.py's API).

A handful of greetings/pleasantries are answered directly, with zero LLM
calls -- no router, no RAG, no calculation, no specialist agent, no
synthesis, no critic. This isn't a RAG or token-cost optimization in the
same sense as `finmate/rag.py`'s "Performance" section (those all still
route through the real pipeline; this instead recognizes when the
pipeline shouldn't run at all), but it's the same spirit applied one
level up: don't spend several LLM calls' worth of latency and tokens
confirming that "hi" doesn't need a budget calculation.

Originally lived only in app.py; extracted here once a second frontend
(the FastAPI backend) needed the identical check -- one implementation,
not two that can quietly drift apart.
"""

from __future__ import annotations

import random

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

# A pool, not a single fixed string, so the same greeting doesn't read as
# a canned/scripted reply on the second, third, ... time a person sees
# it in one session. Deliberately still zero-LLM-call: picking one of a
# handful of pre-written strings costs nothing, and asking a model to
# *generate* variety here would spend exactly the latency/token budget
# this fast path exists to avoid (see module docstring, and
# orchestrator.py's "Fewer sequential calls" -- the same reasoning
# applied one level up: a casual message shouldn't cost a call any more
# than an informational one should cost three).
#
# Each variation still does the two things CASUAL_RESPONSE always did:
# names FinMate AI (so it's clear who/what is answering) and invites a
# financial question (so a first-time user isn't left wondering what to
# ask) -- see `tests/test_casual.py` for both properties asserted across
# every entry, not just the old single string.
CASUAL_RESPONSES: tuple[str, ...] = (
    "Hello! I'm FinMate AI. How can I help you with your finances today?",
    "Hi there! I'm FinMate AI -- ask me anything about your spending, budget, or goals.",
    "Hey! FinMate AI here. What would you like to know about your finances?",
    "Hello! I'm FinMate AI, happy to help with your finances -- what's on your mind?",
    "Hi! I'm FinMate AI, your personal finance assistant. What can I help with?",
)


def is_casual_message(message: str) -> bool:
    """True for simple greetings/pleasantries that don't need the
    financial pipeline at all. Whitespace- and case-insensitive; anything
    with real content beyond an exact match in `CASUAL_MESSAGES` (even
    "hi, what's my savings rate?") is NOT casual -- this is deliberately
    narrow, since a false positive here would silently skip a real
    question."""
    normalized = " ".join(message.lower().strip().split())
    return normalized in CASUAL_MESSAGES


def pick_casual_response() -> str:
    """One variation from `CASUAL_RESPONSES`, chosen at random.

    Random rather than round-robin: this app can run as multiple
    independent worker processes (Render, or any multi-worker uvicorn
    deployment -- see backend/app/main.py), each with its own memory. A
    round-robin counter would need to be shared/synchronized across all
    of them to actually rotate evenly; without that, each worker would
    just cycle through its own local slice, and which worker happens to
    handle each request is not something this fast path (or its caller)
    controls or should need to care about. `random.choice` needs no
    shared state to already do the one thing that matters here: the same
    person is unlikely to see the identical string twice in a row.
    """
    return random.choice(CASUAL_RESPONSES)
