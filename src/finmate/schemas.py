"""
Pydantic models for every JSON contract used by FinMate AI.

These validate:
  - the canonical user financial profile (spec section 2)
  - every structured LLM response contract (spec section 3)
  - every deterministic tool's input/output shape (spec section 4)

`finmate/llm.py` validates raw LLM JSON output against these models and
retries once (feeding the validation error back to the model) before
raising. `finmate/db.py` uses `UserProfile` as the single source of truth
for what a stored profile looks like.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Section 2 - canonical user profile shape
# ---------------------------------------------------------------------------


class IncomeSource(BaseModel):
    name: str = ""
    monthly_amount: float = 0
    pay_frequency: str = ""


class FixedExpense(BaseModel):
    name: str = ""
    amount: float = 0
    due_day: int = 0


class VariableExpense(BaseModel):
    name: str = ""
    typical_monthly_amount: float = 0


class Account(BaseModel):
    name: str = ""
    type: Literal["checking", "savings"] = "checking"
    balance: float = 0


class Debt(BaseModel):
    name: str = ""
    principal: float = 0
    annual_interest_rate_pct: float = 0
    term_months: int = 0
    monthly_payment: float = 0
    remaining_term_months: int = 0


class Investment(BaseModel):
    name: str = ""
    type: str = ""
    current_value: float = 0
    asset_class: str = ""


class Subscription(BaseModel):
    name: str = ""
    monthly_amount: float = 0


class FinancialGoal(BaseModel):
    name: str = ""
    target_amount: float = 0
    current_amount: float = 0
    deadline: str = ""
    months_remaining: int = 0


class UserProfile(BaseModel):
    user_id: str
    currency: str = "INR"
    monthly_income: float = 0
    income_sources: list[IncomeSource] = Field(default_factory=list)
    fixed_expenses: list[FixedExpense] = Field(default_factory=list)
    variable_expenses: list[VariableExpense] = Field(default_factory=list)
    accounts: list[Account] = Field(default_factory=list)
    debts: list[Debt] = Field(default_factory=list)
    investments: list[Investment] = Field(default_factory=list)
    subscriptions: list[Subscription] = Field(default_factory=list)
    financial_goals: list[FinancialGoal] = Field(default_factory=list)
    monthly_budget: dict[str, float] = Field(default_factory=dict)
    risk_preferences: dict[str, Any] = Field(default_factory=dict)
    financial_preferences: dict[str, Any] = Field(default_factory=dict)
    profile_last_updated: str = ""


def empty_profile(user_id: str, currency: str = "INR") -> UserProfile:
    """Construct a blank, schema-valid profile for a brand-new user."""
    return UserProfile(user_id=user_id, currency=currency)


# ---------------------------------------------------------------------------
# Stage 1 - Router output contract
# ---------------------------------------------------------------------------


class DateRange(BaseModel):
    start: Optional[str] = None
    end: Optional[str] = None


class RouterOutput(BaseModel):
    intent: str
    date_range: DateRange = Field(default_factory=DateRange)
    memory_fields_needed: list[str] = Field(default_factory=list)
    data_sources_needed: list[str] = Field(default_factory=list)
    calculations_needed: list[str] = Field(default_factory=list)
    action_required: bool = False
    confirmation_required: bool = False
    risk_level: Literal["low", "medium", "high"] = "low"


# ---------------------------------------------------------------------------
# Stage 2 - Memory agent output contract
# ---------------------------------------------------------------------------


class MemoryAction(BaseModel):
    memory_action: Literal["create", "update", "delete", "none"] = "none"
    field: str = ""
    old_value: Optional[Any] = None
    new_value: Optional[Any] = None
    source: Literal["user_message", "document", "authorized_import"] = "user_message"
    confidence: float = 1.0
    requires_confirmation: bool = False


# ---------------------------------------------------------------------------
# Stage 3 - RAG evidence item contract
# ---------------------------------------------------------------------------


class EvidenceItem(BaseModel):
    source_id: str = ""
    date: str = ""
    description: str = ""
    amount: float = 0
    currency: str = ""
    category: str = ""
    document: str = ""
    page: Optional[int] = None
    relevance: float = 0.0
    # --- RAG hybrid-retrieval upgrade: additive audit fields (all optional,
    # all default to "not applicable" rather than 0.0, so a caller can tell
    # "this stage didn't run" apart from "this stage ran and scored zero") ---
    keyword_score: Optional[float] = None
    vector_score: Optional[float] = None
    rerank_score: Optional[float] = None
    retrieval_stage: str = ""


# ---------------------------------------------------------------------------
# Stage 4 - Calculation result contract
# ---------------------------------------------------------------------------


class CalculationResult(BaseModel):
    metric: str
    value: float
    currency: str = ""
    period: str = ""
    formula: str = ""
    inputs: dict[str, Any] = Field(default_factory=dict)
    source_ids: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Stage 10 - Anomaly item contract
# ---------------------------------------------------------------------------


class AnomalyItem(BaseModel):
    type: str
    severity: Literal["low", "medium", "high"] = "low"
    evidence: list[str] = Field(default_factory=list)
    expected_pattern: str = ""
    observed_pattern: str = ""
    recommended_next_step: str = ""


class AnomalyReport(BaseModel):
    anomalies: list[AnomalyItem] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Stage 12 - Critic output contract
# ---------------------------------------------------------------------------


class CriticResult(BaseModel):
    passed: bool
    confidence: float = 0.0
    errors: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    calculation_errors: list[str] = Field(default_factory=list)
    privacy_issues: list[str] = Field(default_factory=list)
    safety_issues: list[str] = Field(default_factory=list)
    required_research: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Section 4 - external-action tool contract (stubs only, never executed)
# ---------------------------------------------------------------------------


class ProposedAction(BaseModel):
    """Returned by prepare_transfer/prepare_payment/prepare_trade. These
    tools NEVER execute anything -- they only describe what *would* happen,
    pending the user's explicit confirmation through a real banking channel
    that does not exist in this codebase."""

    action_type: Literal["transfer", "payment", "trade"]
    status: Literal["proposed_pending_confirmation"] = "proposed_pending_confirmation"
    details: dict[str, Any] = Field(default_factory=dict)
    requires_user_confirmation: bool = True
    note: str = (
        "This is a proposed action only. FinMate AI cannot and does not "
        "execute money movement, trades, or account changes."
    )


# ---------------------------------------------------------------------------
# Transaction record (DB row shape)
# ---------------------------------------------------------------------------


class Transaction(BaseModel):
    id: Optional[int] = None
    user_id: str
    date: str  # ISO date string, e.g. "2026-06-15"
    description: str
    amount: float  # positive = inflow, negative = outflow
    currency: str = "INR"
    category: str = "uncategorized"
    account: str = ""
    type: Literal["income", "expense", "transfer"] = "expense"
    source_id: str = ""
