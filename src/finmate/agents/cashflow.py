"""Stage 6 - Cash-Flow Forecast Agent. Owns prompts.CASHFLOW_AGENT."""

from __future__ import annotations

import json

from .. import tools
from ..llm import LLMClient
from ..prompts import CASHFLOW_AGENT
from ..rag import RetrievalResult
from ..schemas import CalculationResult, UserProfile
from ._shared import build_context_block


def run_cashflow_agent(
    llm_client: LLMClient, user_message: str, profile: UserProfile,
    evidence: RetrievalResult, calc_results: list[CalculationResult], skipped: list[str],
    horizon_days: int = 30,
) -> str:
    """Runs the deterministic forecast_cash_flow for all three scenarios
    first (BASE/CONSERVATIVE/STRESS, per the CASHFLOW_AGENT contract), then
    asks the LLM to narrate the numbers -- the LLM never invents the
    scenario figures themselves.
    """
    income = profile.monthly_income or tools.monthly_income(
        [s.model_dump() for s in profile.income_sources]
    )
    fixed_total = tools.recurring_expense_totals([e.model_dump() for e in profile.fixed_expenses])
    debt_total = sum(d.monthly_payment for d in profile.debts)
    opening_balance = sum(a.balance for a in profile.accounts if a.type in ("checking", "savings"))
    variable_total = sum(v.typical_monthly_amount for v in profile.variable_expenses)

    scenarios = {
        scenario: tools.forecast_cash_flow(
            opening_balance=opening_balance, expected_income=income,
            fixed_commitments=fixed_total, avg_variable_spending=variable_total,
            debt_payments=debt_total, horizon_days=horizon_days, scenario=scenario,
        )
        for scenario in ("BASE", "CONSERVATIVE", "STRESS")
    }

    context = build_context_block(user_message, profile, evidence, calc_results, skipped)
    context += f"\n\nDeterministic forecast scenarios (already computed, do not recompute):\n{json.dumps(scenarios, separators=(',', ':'))}"
    return llm_client.call(agent_system_prompt=CASHFLOW_AGENT, user_message=context, response_model=None)
