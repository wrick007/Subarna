"""Stage 9 - Investment Information Agent. Owns prompts.INVESTMENT_AGENT."""

from __future__ import annotations

from ..llm import LLMClient
from ..prompts import INVESTMENT_AGENT
from ..rag import RetrievalResult
from ..schemas import CalculationResult, UserProfile
from ._shared import build_context_block


def run_investment_agent(
    llm_client: LLMClient, user_message: str, profile: UserProfile,
    evidence: RetrievalResult, calc_results: list[CalculationResult], skipped: list[str],
) -> str:
    context = build_context_block(user_message, profile, evidence, calc_results, skipped)
    return llm_client.call(agent_system_prompt=INVESTMENT_AGENT, user_message=context, response_model=None)
