"""Stage 1 - Intent Router. Owns prompts.ROUTER."""

from __future__ import annotations

from ..llm import LLMClient
from ..prompts import ROUTER
from ..schemas import RouterOutput


def run_router(llm_client: LLMClient, user_message: str) -> RouterOutput:
    """Convert the raw user message into a structured task plan. Does not
    answer the financial question -- see prompts.ROUTER for the contract."""
    return llm_client.call(
        agent_system_prompt=ROUTER,
        user_message=user_message,
        response_model=RouterOutput,
    )
