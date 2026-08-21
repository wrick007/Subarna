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
import re

from .. import db
from ..llm import LLMClient
from ..prompts import MEMORY_AGENT
from ..schemas import FixedExpense, MemoryAction, UserProfile, VariableExpense


_PROFILE_TYPO_FIXES = {
    "incone": "income",
    "incme": "income",
    "expence": "expense",
    "expences": "expenses",
    "expenes": "expenses",
    "montly": "monthly",
    "monhtly": "monthly",
    "mothly": "monthly",
    "emii": "emi",
}


def _normalise_message(message: str) -> str:
    """Make a few high-confidence spelling repairs before pattern matching.

    This is intentionally narrow: it handles common money words and leaves
    the rest of the user's wording untouched for the LLM to interpret.
    """
    cleaned = message.lower().strip()
    for typo, correction in _PROFILE_TYPO_FIXES.items():
        cleaned = re.sub(rf"\b{typo}\b", correction, cleaned)
    return cleaned


def _parse_amount(raw: str, suffix: str) -> float:
    amount = float(raw.replace(",", ""))
    return amount * 1000 if suffix.lower() in {"k", "thousand"} else amount


def detect_explicit_profile_update(user_message: str, profile: UserProfile) -> MemoryAction | None:
    """Recognise unambiguous, conversational profile updates without an LLM.

    Examples: ``incone 80k``, ``expenses 30k`` and ``update emi to 30k a
    month``. This makes common shorthand reliable and prevents an EMI from
    being treated as an invalid top-level profile field.
    """
    message = _normalise_message(user_message)
    match = re.match(
        r"^(?:please\s+)?(?:update|set|change)?\s*(?:my\s+)?(?:monthly\s+)?"
        r"(?P<field>income|salary|earnings|expenses?|spending|emi|loan\s+(?:emi|payment))"
        r"\s*(?:to|is|=)?\s*(?:₹|rs\.?|inr)?\s*"
        r"(?P<amount>\d[\d,]*(?:\.\d+)?)\s*(?P<suffix>k|thousand)?"
        r"(?:\s*(?:a|per)?\s*month(?:ly)?)?\s*$",
        message,
    )
    if not match:
        return None

    amount = _parse_amount(match.group("amount"), match.group("suffix") or "")
    field = match.group("field")
    if field in {"income", "salary", "earnings"}:
        return MemoryAction(memory_action="update", field="monthly_income", new_value=amount)

    if field in {"emi", "loan emi", "loan payment"}:
        expenses = list(profile.fixed_expenses)
        existing = next((item for item in expenses if item.name.casefold() == "emi"), None)
        if existing:
            existing.amount = amount
        else:
            expenses.append(FixedExpense(name="EMI", amount=amount))
        return MemoryAction(memory_action="update", field="fixed_expenses", new_value=expenses)

    expenses = list(profile.variable_expenses)
    existing = next((item for item in expenses if item.name.casefold() == "monthly expenses"), None)
    if existing:
        existing.typical_monthly_amount = amount
    else:
        expenses.append(VariableExpense(name="Monthly expenses", typical_monthly_amount=amount))
    return MemoryAction(memory_action="update", field="variable_expenses", new_value=expenses)


def normalise_memory_action(profile: UserProfile, action: MemoryAction) -> MemoryAction:
    """Translate common LLM aliases into schema-backed profile updates."""
    alias = _normalise_message(action.field).replace(" ", "_")
    if alias not in {"emi", "monthly_emi", "loan_emi", "loan_payment", "expense", "expenses", "spending"}:
        return action

    try:
        amount = float(str(action.new_value).replace(",", "").removesuffix("k"))
        if str(action.new_value).strip().lower().endswith("k"):
            amount *= 1000
    except (TypeError, ValueError):
        return action

    field = "emi" if "emi" in alias or alias == "loan_payment" else "expenses"
    replacement = detect_explicit_profile_update(f"{field} {amount}", profile)
    return replacement or action


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
