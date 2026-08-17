"""Shared FastAPI dependencies.

The `LLMClient` is constructed once, at startup (see `app.main`'s
lifespan handler), not per-request -- constructing it per-request would
mean re-resolving env config and re-creating the underlying `openai`
SDK client (its own HTTP connection pool) on every single chat message,
for a value that never changes while the process is running. Routers
that need it depend on `get_llm_client` below rather than importing
`app.main` directly, which would risk a circular import (`app.main`
already imports every router to register it).
"""

from __future__ import annotations

from fastapi import HTTPException, Request

from finmate.llm import LLMClient


def get_llm_client(request: Request) -> LLMClient:
    """Raises HTTP 503 (not 500 -- this is a configuration/availability
    problem, not a bug) with an actionable message if no provider API
    key was configured at startup. Every other endpoint (health, profile,
    transactions, seed-demo, delete) works fine without one; only actual
    chat needs it, matching app.py's Streamlit behavior."""
    client = getattr(request.app.state, "llm_client", None)
    if client is None:
        error = getattr(request.app.state, "llm_client_error", None) or "No LLM provider configured."
        raise HTTPException(
            status_code=503,
            detail=(
                f"Chat is unavailable: {error} Set GROQ_API_KEY or GEMINI_API_KEY "
                "(see backend/.env.example) and restart the server."
            ),
        )
    return client
