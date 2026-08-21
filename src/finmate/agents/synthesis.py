"""Stage 11 - Synthesis Agent. Owns prompts.SYNTHESIS_AGENT.

As of the Priority-2 redesign this agent also does what
`agents/formatter.py`'s Stage 13 used to do -- SYNTHESIS_AGENT's prompt
now asks for the final user-facing reply directly, not an intermediate
analysis (see that prompt's header and `agents/formatter.py`'s module
docstring for where the old second step went and why). One LLM call
produces what used to take two.
"""

from __future__ import annotations

import json
from typing import Iterator

from ..llm import LLMClient
from ..prompts import SYNTHESIS_AGENT
from ..rag import RetrievalResult
from ..schemas import CalculationResult, UserProfile
from ._shared import build_context_block


def _build_synthesis_context(
    user_message: str,
    profile: UserProfile,
    evidence: RetrievalResult,
    calc_results: list[CalculationResult],
    skipped: list[str],
    specialist_outputs: dict[str, object],
    conversation_history: str,
    verification_feedback: str,
) -> str:
    """Shared by `run_synthesis_agent` and `stream_synthesis_agent` so
    the streaming and non-streaming call sites can never drift apart on
    what context the model actually sees."""
    context = build_context_block(user_message, profile, evidence, calc_results, skipped)
    context += f"\n\nSpecialist agent outputs for this request:\n{json.dumps(specialist_outputs, separators=(',', ':'), default=str)}"
    if conversation_history:
        context += f"\n\n{conversation_history}"
    if verification_feedback:
        context += f"\n\n{verification_feedback}"
    return context


def run_synthesis_agent(
    llm_client: LLMClient,
    user_message: str,
    profile: UserProfile,
    evidence: RetrievalResult,
    calc_results: list[CalculationResult],
    skipped: list[str],
    specialist_outputs: dict[str, object],
    conversation_history: str = "",
    verification_feedback: str = "",
) -> str:
    """Combines profile, evidence, calculations, and every specialist
    agent's output that ran for this intent into the final,
    evidence-grounded reply shown to the user -- internally classifying
    each statement as FACT / CALCULATION / FORECAST / INTERPRETATION /
    RECOMMENDATION per the SYNTHESIS_AGENT contract, but writing the
    reply itself, not a separate analysis (see module docstring).

    `conversation_history`: optional pre-rendered block of recent turns
    (see `finmate/orchestrator.py:_render_history_block`), appended after
    the evidence block -- deliberately last, and deliberately not folded
    into `build_context_block` (which every specialist agent also calls;
    see that module's docstring): only the router and this agent need
    conversational context per the memory redesign, and putting it after
    the "use ONLY the data below as evidence" instruction keeps that
    instruction's scope unambiguous -- history is for resolving what the
    user means, not an extra source of facts. Empty string (the default)
    adds nothing, so a turn with no prior history costs zero extra
    tokens here, same as before this parameter existed.

    `verification_feedback`: optional pre-rendered note (see
    `orchestrator.py:_render_verification_feedback`) describing what a
    previous attempt at this same turn was rejected for, so a
    Critic-triggered retry actually addresses the specific issue instead
    of re-asking the same question at temperature 0 and very likely
    getting the same answer back -- see orchestrator.py's module
    docstring "Fewer sequential calls: productive retries" for why this
    matters. Empty string (the default, and always true on a first
    attempt) adds nothing.
    """
    context = _build_synthesis_context(
        user_message, profile, evidence, calc_results, skipped, specialist_outputs,
        conversation_history, verification_feedback,
    )
    return llm_client.call(agent_system_prompt=SYNTHESIS_AGENT, user_message=context, response_model=None)


def stream_synthesis_agent(
    llm_client: LLMClient,
    user_message: str,
    profile: UserProfile,
    evidence: RetrievalResult,
    calc_results: list[CalculationResult],
    skipped: list[str],
    specialist_outputs: dict[str, object],
    conversation_history: str = "",
    verification_feedback: str = "",
) -> Iterator[str]:
    """Token-by-token counterpart of `run_synthesis_agent`, for the SSE
    streaming path (`orchestrator.run_finmate_stream` /
    `backend/app/routers/chat.py`'s `/api/chat/stream`). Identical
    context-building (`_build_synthesis_context`), so streamed and
    non-streamed turns are never prompted differently -- only how the
    response comes back differs. See `LLMClient.call_stream` for what
    this can't do that `.call()` can (structured-output validation,
    automatic retry on a transient provider error)."""
    context = _build_synthesis_context(
        user_message, profile, evidence, calc_results, skipped, specialist_outputs,
        conversation_history, verification_feedback,
    )
    yield from llm_client.call_stream(agent_system_prompt=SYNTHESIS_AGENT, user_message=context)

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