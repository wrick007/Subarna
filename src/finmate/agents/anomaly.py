"""Stage 10 - Anomaly Detection Agent. Owns prompts.ANOMALY_AGENT."""

from __future__ import annotations

from ..llm import LLMClient
from ..prompts import ANOMALY_AGENT
from ..rag import RetrievalResult
from ..schemas import AnomalyReport, CalculationResult, UserProfile
from ._shared import build_context_block


def run_anomaly_agent(
    llm_client: LLMClient, user_message: str, profile: UserProfile,
    evidence: RetrievalResult, calc_results: list[CalculationResult], skipped: list[str],
) -> AnomalyReport:
    context = build_context_block(user_message, profile, evidence, calc_results, skipped)
    return llm_client.call(agent_system_prompt=ANOMALY_AGENT, user_message=context, response_model=AnomalyReport)
