"""Stage 12 - Critic/Verification Agent. Owns prompts.CRITIC_AGENT.

Builds its own payload rather than calling `finmate/agents/_shared.py`'s
`build_context_block`: that helper's fixed opening line ("Use ONLY the
data below as evidence...") is an instruction to a response-drafting
agent, not an auditor -- the Critic's job is verifying draft_response
against this same data, not writing from it, so it gets its own framing
line below. It does reuse `_shared.evidence_for_prompt` for the evidence
list itself (same trim, same reasoning -- see that module's docstring)
and the same compact-JSON serialization, so the two heaviest prompts in
the pipeline (this one and synthesis, which sees a near-identical
payload) pay an identical, minimal token cost for the identical data.
"""

from __future__ import annotations

import json

from ..llm import LLMClient
from ..prompts import CRITIC_AGENT
from ..rag import RetrievalResult
from ..schemas import CalculationResult, CriticResult, UserProfile
from ._shared import evidence_for_prompt


def run_critic_agent(
    llm_client: LLMClient,
    draft_response: str,
    profile: UserProfile,
    evidence: RetrievalResult,
    calc_results: list[CalculationResult],
) -> CriticResult:
    """Audits a draft response against the evidence it was built from.
    The orchestrator loops back to specialists/calculation (max 2 retries)
    on any `passed=False` result, per spec section 1 step 7."""
    payload = {
        "draft_response": draft_response,
        "profile": profile.model_dump(),
        "evidence": evidence_for_prompt(evidence),
        "calculation_results": [c.model_dump() for c in calc_results],
    }
    context = (
        "Audit the draft_response below against the supporting data. "
        "Every material financial claim in draft_response must be traceable "
        "to profile, evidence, or calculation_results.\n\n"
        f"{json.dumps(payload, separators=(',', ':'), default=str)}"
    )
    return llm_client.call(agent_system_prompt=CRITIC_AGENT, user_message=context, response_model=CriticResult)
