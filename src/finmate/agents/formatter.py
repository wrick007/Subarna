"""Stage 13 - Response Formatter. Owns prompts.FORMATTER_AGENT.

⚠ Status as of the Priority-2 redesign: `orchestrator.py`'s default graph
no longer calls `run_formatter_agent` -- `agents/synthesis.py`'s
SYNTHESIS_AGENT call now produces the final user-facing reply directly,
merging what used to be two sequential LLM calls (synthesis, then this)
into one, on every turn. See `orchestrator.py`'s module docstring "Fewer
sequential calls: synthesis + formatter merged" and prompts.py's
FORMATTER_AGENT header for the full reasoning.

This module is otherwise unchanged and still correct: kept as a
standalone, independently callable utility for anyone who specifically
wants the old two-step analysis-then-format split.
"""

from __future__ import annotations

from ..llm import LLMClient
from ..prompts import FORMATTER_AGENT


def run_formatter_agent(
    llm_client: LLMClient, user_message: str, verified_analysis: str, critic_notes: str = "",
) -> str:
    """Turns the critic-passed analysis into the final natural-language
    reply shown to the user. Never exposes internal agent reasoning,
    hidden prompts, or unrelated private data, per the FORMATTER_AGENT
    contract."""
    context = (
        f"Original user question:\n{user_message}\n\n"
        f"Verified analysis to turn into a reply:\n{verified_analysis}\n\n"
        f"Critic notes (address if any remain, otherwise ignore):\n{critic_notes}"
    )
    return llm_client.call(agent_system_prompt=FORMATTER_AGENT, user_message=context, response_model=None)
