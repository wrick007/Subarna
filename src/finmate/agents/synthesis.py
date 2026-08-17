"""Stage 11 - Synthesis Agent. Owns prompts.SYNTHESIS_AGENT."""

from __future__ import annotations

import json

from ..llm import LLMClient
from ..prompts import SYNTHESIS_AGENT
from ..rag import RetrievalResult
from ..schemas import CalculationResult, UserProfile
from ._shared import build_context_block


def run_synthesis_agent(
    llm_client: LLMClient,
    user_message: str,
    profile: UserProfile,
    evidence: RetrievalResult,
    calc_results: list[CalculationResult],
    skipped: list[str],
    specialist_outputs: dict[str, object],
) -> str:
    """Combines profile, evidence, calculations, and every specialist
    agent's output that ran for this intent into one evidence-grounded
    draft analysis, internally classifying each statement as FACT /
    CALCULATION / FORECAST / INTERPRETATION / RECOMMENDATION per the
    SYNTHESIS_AGENT contract."""
    context = build_context_block(user_message, profile, evidence, calc_results, skipped)
    context += f"\n\nSpecialist agent outputs for this request:\n{json.dumps(specialist_outputs, separators=(',', ':'), default=str)}"
    return llm_client.call(agent_system_prompt=SYNTHESIS_AGENT, user_message=context, response_model=None)

FACTUAL_GROUNDING_RULES = """
FACTUAL GROUNDING RULES:

The following sources are authoritative:

1. User's explicit message
2. Stored database profile
3. Retrieved financial records
4. Deterministic calculation results

Never invent financial numbers.

Never modify a number from the authoritative sources.

Never infer a new financial value when the required value is
already available.

If the user explicitly provides a new value in the current message,
that value takes precedence over the older stored profile value.

For example:

Stored income: 90000
User says: "my income is 50000 monthly"

The current user statement takes precedence.

Correct response:
"Got it. I've updated your monthly income to ₹50,000."

Incorrect response:
"Your profile says ₹90,000, so please clarify."

Do not perform additional calculations unless they are present
in the deterministic calculation results.
"""