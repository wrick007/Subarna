"""Stage 1 - Intent Router. Owns prompts.ROUTER."""

from __future__ import annotations

from ..llm import LLMClient
from ..prompts import ROUTER
from ..schemas import RouterOutput


def run_router(
    llm_client: LLMClient,
    user_message: str,
    conversation_history: str = "",
) -> RouterOutput:
    """Convert the raw user message into a structured task plan. Does not
    answer the financial question -- see prompts.ROUTER for the contract.

    `conversation_history`: optional pre-rendered block of recent turns
    (see `finmate/orchestrator.py:_render_history_block`), prepended so
    the router can resolve a reference like "how about that instead"
    against what was actually discussed. Empty string (the default) is a
    no-op -- exactly today's behavior for any caller that doesn't pass
    it, and no extra tokens are spent when a conversation has no prior
    turns yet.
    """
    context = user_message if not conversation_history else f"{conversation_history}\n\nCurrent user message:\n{user_message}"
    return llm_client.call(
        agent_system_prompt=ROUTER,
        user_message=context,
        response_model=RouterOutput,
    )
