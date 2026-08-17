"""Stage 5 - Budget Agent. Owns prompts.BUDGET_AGENT."""

from __future__ import annotations

from ..llm import LLMClient
from ..prompts import BUDGET_AGENT
from ..rag import RetrievalResult
from ..schemas import CalculationResult, UserProfile
from ._shared import build_context_block


def run_budget_agent(
    llm_client: LLMClient, user_message: str, profile: UserProfile,
    evidence: RetrievalResult, calc_results: list[CalculationResult], skipped: list[str],
) -> str:
    context = build_context_block(user_message, profile, evidence, calc_results, skipped)
    return llm_client.call(agent_system_prompt=BUDGET_AGENT, user_message=context, response_model=None)
