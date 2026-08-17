"""Stage 4 - Calculation Agent. Owns prompts.CALCULATION_AGENT as a
*documented contract only* -- this module is pure Python, never an LLM
call, per spec section 3's note on this stage.

`run_calculations` takes the router's `calculations_needed` list plus the
profile and retrieved evidence, assembles the right kwargs for each known
metric, and dispatches through `tools.calculate_metric` so every returned
number carries its formula/inputs/source_ids for the Critic Agent to audit.
Metrics whose required inputs aren't available are skipped with a note
rather than guessed.
"""

from __future__ import annotations

from typing import Any

from .. import tools
from ..rag import RetrievalResult
from ..schemas import CalculationResult, Transaction, UserProfile


def _evidence_to_transactions(evidence: RetrievalResult) -> list[Transaction]:
    txs = []
    for item in evidence.evidence:
        tx_type = "income" if item.amount > 0 else "expense"
        txs.append(
            Transaction(
                user_id="", date=item.date, description=item.description,
                amount=item.amount, currency=item.currency or "INR",
                category=item.category, account=item.document, type=tx_type,
                source_id=item.source_id,
            )
        )
    return txs


def run_calculations(
    calculations_needed: list[str],
    profile: UserProfile,
    evidence: RetrievalResult,
) -> tuple[list[CalculationResult], list[str]]:
    """Returns (results, skipped_with_reason)."""
    txs = _evidence_to_transactions(evidence)
    source_ids = [t.source_id for t in txs if t.source_id]
    results: list[CalculationResult] = []
    skipped: list[str] = []

    fixed_total = tools.recurring_expense_totals([e.model_dump() for e in profile.fixed_expenses])
    debt_payments_total = sum(d.monthly_payment for d in profile.debts)
    income = profile.monthly_income or tools.monthly_income(
        [s.model_dump() for s in profile.income_sources]
    )
    spending = tools.total_spending(txs)
    liquid_balance = sum(a.balance for a in profile.accounts if a.type in ("checking", "savings"))

    for metric in calculations_needed:
        try:
            if metric == "monthly_income":
                results.append(tools.calculate_metric(
                    "monthly_income", {"income_sources": [s.model_dump() for s in profile.income_sources],
                                        "currency": profile.currency, "period": "monthly"}))
            elif metric == "total_spending":
                results.append(CalculationResult(
                    metric="total_spending", value=spending, currency=profile.currency,
                    period="retrieved_range", formula="sum(abs(amount) for expense transactions)",
                    inputs={"transaction_count": len(txs)}, source_ids=source_ids))
            elif metric == "savings_rate":
                results.append(tools.calculate_metric(
                    "savings_rate", {"income": income, "spending": spending,
                                      "currency": profile.currency, "source_ids": source_ids}))
            elif metric == "disposable_income":
                results.append(tools.calculate_metric(
                    "disposable_income",
                    {"income": income, "fixed_expenses_total": fixed_total,
                     "debt_payments_total": debt_payments_total, "currency": profile.currency}))
            elif metric == "cash_flow_surplus_deficit":
                results.append(tools.calculate_metric(
                    "cash_flow_surplus_deficit",
                    {"income": income, "total_expenses": spending, "currency": profile.currency,
                     "source_ids": source_ids}))
            elif metric == "emergency_fund_coverage":
                essential = fixed_total if fixed_total > 0 else spending
                results.append(tools.calculate_metric(
                    "emergency_fund_coverage",
                    {"liquid_balance": liquid_balance, "monthly_essential_expenses": essential,
                     "currency": profile.currency}))
            elif metric == "debt_to_income_ratio":
                results.append(tools.calculate_metric(
                    "debt_to_income_ratio",
                    {"total_monthly_debt_payments": debt_payments_total, "gross_monthly_income": income,
                     "currency": profile.currency}))
            elif metric == "subscription_totals":
                results.append(tools.calculate_metric(
                    "subscription_totals",
                    {"subscriptions": [s.model_dump() for s in profile.subscriptions],
                     "currency": profile.currency, "period": "monthly"}))
            elif metric == "recurring_expense_totals":
                results.append(tools.calculate_metric(
                    "recurring_expense_totals",
                    {"fixed_expenses": [e.model_dump() for e in profile.fixed_expenses],
                     "currency": profile.currency, "period": "monthly"}))
            elif metric == "loan_amortization":
                for d in profile.debts:
                    results.append(tools.calculate_metric(
                        "loan_amortization",
                        {"principal": d.principal, "annual_rate_pct": d.annual_interest_rate_pct,
                         "term_months": d.term_months or d.remaining_term_months,
                         "currency": profile.currency, "period": f"debt:{d.name}"}))
            elif metric == "goal_contribution_required":
                for g in profile.financial_goals:
                    results.append(tools.calculate_metric(
                        "goal_contribution_required",
                        {"target_amount": g.target_amount, "current_amount": g.current_amount,
                         "months_remaining": g.months_remaining, "currency": profile.currency,
                         "period": f"goal:{g.name}"}))
            elif metric == "category_spending":
                categories = sorted({t.category for t in txs})
                for cat in categories:
                    results.append(tools.calculate_metric(
                        "category_spending",
                        {"transactions": txs, "category": cat, "currency": profile.currency,
                         "period": "retrieved_range", "source_ids": source_ids}))
            elif metric == "budget_variance":
                actual: dict[str, Any] = {}
                for t in txs:
                    if t.type == "expense":
                        actual[t.category] = actual.get(t.category, 0.0) + abs(t.amount)
                results.append(tools.calculate_metric(
                    "budget_variance",
                    {"actual_by_category": actual, "budget_by_category": profile.monthly_budget,
                     "currency": profile.currency, "source_ids": source_ids}))
            elif metric == "cagr":
                for inv in profile.investments:
                    skipped.append(
                        f"cagr for investment {inv.name!r}: needs a historical beginning value and "
                        "a holding period, which aren't in the stored profile shape -- skipped rather "
                        "than assumed."
                    )
            else:
                skipped.append(f"{metric}: not a recognized metric name")
        except (KeyError, ValueError, ZeroDivisionError, TypeError) as exc:
            skipped.append(f"{metric}: skipped, missing/invalid inputs ({exc})")

    return results, skipped
