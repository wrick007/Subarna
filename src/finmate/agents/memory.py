"""Stage 2 - Memory/Profile Agent. Owns prompts.MEMORY_AGENT.

Two halves, matching the CONSTITUTION's "user control" principle:
  - `run_memory_agent`: LLM call that decides *what* durable fact (if any)
    the user's message contains, returning a MemoryAction.
  - `apply_memory_action`: pure Python that actually mutates the stored
    UserProfile -- the LLM never writes to the database directly, it only
    proposes a structured action that this function then applies and
    persists (via db.upsert_profile, called by the orchestrator).
"""

from __future__ import annotations

import json

from .. import db
from ..llm import LLMClient
from ..prompts import MEMORY_AGENT
from ..schemas import MemoryAction, UserProfile


def run_memory_agent(llm_client: LLMClient, user_message: str, profile: UserProfile) -> MemoryAction:
    context = (
        f"Current stored profile (JSON):\n{profile.model_dump_json()}\n\n"
        f"User message:\n{user_message}"
    )
    return llm_client.call(
        agent_system_prompt=MEMORY_AGENT,
        user_message=context,
        response_model=MemoryAction,
    )


def apply_memory_action(profile: UserProfile, action: MemoryAction) -> UserProfile:
    """Apply a validated MemoryAction to a profile in memory (caller persists it).

    `action.field` must name a top-level UserProfile field. `create`/`update`
    set the field to `action.new_value`; `delete` resets it to that field's
    empty default; `none` is a no-op. Fields requiring confirmation are
    still applied here -- gating on `requires_confirmation` before calling
    this function is the orchestrator's job, per CONSTITUTION principle 6/10.
    """
    if action.memory_action == "none":
        return profile
    if not action.field or not hasattr(profile, action.field):
        raise ValueError(f"Memory agent referenced an unknown profile field: {action.field!r}")

    if action.memory_action == "delete":
        default_profile = UserProfile(user_id=profile.user_id)
        setattr(profile, action.field, getattr(default_profile, action.field))
        return profile

    # create / update
    new_value = action.new_value
    current_field_value = getattr(profile, action.field)
    if isinstance(current_field_value, list) and isinstance(new_value, str):
        # LLMs sometimes hand back a JSON string for list fields; parse defensively.
        try:
            new_value = json.loads(new_value)
        except (json.JSONDecodeError, TypeError):
            pass
    setattr(profile, action.field, new_value)
    # Re-validate the whole profile so a bad LLM value fails loudly here,
    # not silently later when some other agent reads a malformed field.
    return UserProfile.model_validate(profile.model_dump())


def summarize_memory(user_id: str, db_path: str = db.DEFAULT_DB_PATH) -> str:
    """Used when the user asks 'what do you remember about me' -- returns
    only authorized stored information, per the MEMORY_AGENT contract."""
    profile = db.get_user_profile(user_id, db_path=db_path)
    if profile is None:
        return "I don't have any stored financial profile for you yet."
    return profile.model_dump_json(indent=2)
