"""Stage 13 - Response Formatter. Owns prompts.FORMATTER_AGENT."""

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
