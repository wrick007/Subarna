"""
Deterministic financial calculation engine for FinMate AI.

Every function below is pure Python: no I/O, no LLM calls, no network
access. This is the module the CONSTITUTION refers to when it says
"never mentally calculate financial totals when a tool can calculate
them" -- every number an agent shows the user must originate here (or in
`db.py` for raw lookups), never from an LLM's own arithmetic.

`calculate_metric` is the single dispatch point Stage 4 (Calculation
Agent) and the orchestrator use: it takes a metric name plus a dict of
inputs and returns a schemas.CalculationResult with the formula and
inputs recorded, so every number shown to the user is auditable.

All functions are covered by tests/test_tools.py, including at least one
hand-verified example per formula.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Callable, Literal, Optional

from .schemas import CalculationResult, ProposedAction, Transaction

# ---------------------------------------------------------------------------
# Income / spending
# ---------------------------------------------------------------------------


def monthly_income(income_sources: list[dict[str, Any]]) -> float:
    """Sum of all income sources' monthly_amount.

    Each income_sources item is expected to already carry a *monthly*
    equivalent amount (see schemas.IncomeSource) -- pay_frequency is
    informational, not a conversion instruction, per the canonical
    profile shape in spec section 2.
    """
    return round(sum(float(s.get("monthly_amount", 0)) for s in income_sources), 2)


def total_spending(transactions: list[Transaction]) -> float:
    """Sum of the absolute value of every expense-type transaction."""
    return round(
        sum(abs(t.amount) for t in transactions if t.type == "expense"), 2
    )


def category_spending(transactions: list[Transaction], category: str) -> float:
    """Total expense spending in a single category, case-insensitive match."""
    cat = category.strip().lower()
    return round(
        sum(
            abs(t.amount)
            for t in transactions
            if t.type == "expense" and t.category.strip().lower() == cat
        ),
        2,
    )


def savings_rate(income: float, spending: float) -> float:
    """(income - spending) / income, as a fraction (0.25 = 25%).

    Returns 0.0 if income is 0 to avoid a division error rather than
    raising -- callers should treat a 0 income as a missing-data case,
    not silently trust this 0.0.
    """
    if income == 0:
        return 0.0
    return round((income - spending) / income, 4)


def disposable_income(
    income: float, fixed_expenses_total: float, debt_payments_total: float = 0.0
) -> float:
    """Income remaining after fixed/recurring obligations and scheduled debt payments."""
    return round(income - fixed_expenses_total - debt_payments_total, 2)


def cash_flow_surplus_deficit(income: float, total_expenses: float) -> float:
    """Positive = surplus, negative = deficit, for a given period."""
    return round(income - total_expenses, 2)


def budget_variance(
    actual_by_category: dict[str, float], budget_by_category: dict[str, float]
) -> dict[str, float]:
    """actual - budget per category. Positive = over budget, negative = under.

    Categories present in only one of the two dicts are still included,
    treating the missing side as 0, so nothing is silently dropped.
    """
    categories = set(actual_by_category) | set(budget_by_category)
    return {
        cat: round(actual_by_category.get(cat, 0.0) - budget_by_category.get(cat, 0.0), 2)
        for cat in categories
    }


def emergency_fund_coverage(liquid_balance: float, monthly_essential_expenses: float) -> float:
    """How many months of essential expenses the liquid balance would cover.

    Returns a large sentinel (999.0) if essential expenses are 0, since
    "infinite coverage" isn't a meaningful float; callers should special-case
    that rather than displaying it directly.
    """
    if monthly_essential_expenses <= 0:
        return 999.0
    return round(liquid_balance / monthly_essential_expenses, 2)


def debt_to_income_ratio(total_monthly_debt_payments: float, gross_monthly_income: float) -> float:
    """Standard DTI ratio as a fraction (0.36 = 36%)."""
    if gross_monthly_income == 0:
        return 0.0
    return round(total_monthly_debt_payments / gross_monthly_income, 4)


# ---------------------------------------------------------------------------
# Loans
# ---------------------------------------------------------------------------


def loan_amortization(
    principal: float, annual_rate_pct: float, term_months: int
) -> dict[str, Any]:
    """Standard fixed-rate amortization (EMI formula).

    EMI = P * r * (1+r)^n / ((1+r)^n - 1), where r is the monthly rate
    (annual_rate_pct / 100 / 12) and n = term_months. Returns the EMI, a
    full month-by-month schedule (principal/interest split), and total
    interest paid over the life of the loan.

    If annual_rate_pct is 0, falls back to a simple principal/term split
    with no interest, since the EMI formula divides by zero at r=0.
    """
    if term_months <= 0:
        raise ValueError("term_months must be positive")
    if principal < 0:
        raise ValueError("principal must be non-negative")

    if annual_rate_pct == 0:
        emi = round(principal / term_months, 2)
        schedule = []
        balance = principal
        for m in range(1, term_months + 1):
            principal_component = emi if m < term_months else round(balance, 2)
            balance = round(balance - principal_component, 2)
            schedule.append(
                {"month": m, "payment": principal_component, "principal": principal_component,
                 "interest": 0.0, "remaining_balance": max(balance, 0.0)}
            )
        return {"emi": emi, "total_interest": 0.0, "total_paid": round(principal, 2), "schedule": schedule}

    r = (annual_rate_pct / 100.0) / 12.0
    n = term_months
    factor = (1 + r) ** n
    emi = principal * r * factor / (factor - 1)
    emi = round(emi, 2)

    schedule = []
    balance = principal
    total_interest = 0.0
    for m in range(1, n + 1):
        interest_component = round(balance * r, 2)
        principal_component = round(emi - interest_component, 2)
        if m == n:
            # true up the final payment against any rounding drift
            principal_component = round(balance, 2)
        balance = round(balance - principal_component, 2)
        total_interest += interest_component
        schedule.append(
            {"month": m, "payment": emi, "principal": principal_component,
             "interest": interest_component, "remaining_balance": max(balance, 0.0)}
        )

    return {
        "emi": emi,
        "total_interest": round(total_interest, 2),
        "total_paid": round(emi * n, 2),
        "schedule": schedule,
    }


# ---------------------------------------------------------------------------
# Subscriptions / recurring
# ---------------------------------------------------------------------------


def subscription_totals(subscriptions: list[dict[str, Any]]) -> float:
    return round(sum(float(s.get("monthly_amount", 0)) for s in subscriptions), 2)


def recurring_expense_totals(fixed_expenses: list[dict[str, Any]]) -> float:
    return round(sum(float(e.get("amount", 0)) for e in fixed_expenses), 2)


# ---------------------------------------------------------------------------
# Goals / investments
# ---------------------------------------------------------------------------


def goal_contribution_required(
    target_amount: float,
    current_amount: float,
    months_remaining: int,
    assumed_annual_return_pct: float = 0.0,
) -> float:
    """Required monthly contribution to reach a goal by a deadline.

    With 0% assumed return, this is simply linear: (target - current) / months.
    With a nonzero assumed return, uses the future-value-of-an-annuity
    formula so growth on both the current balance and future contributions
    is accounted for:

        contribution = (target - current*(1+r)^n) * r / ((1+r)^n - 1)

    where r is the monthly rate and n = months_remaining. Any assumed
    return must be clearly labeled as an assumption by the calling agent,
    per the Goal Planning Agent prompt -- this function does not decide
    what's a reasonable assumption, it only computes the arithmetic.
    """
    remaining = target_amount - current_amount
    if months_remaining <= 0:
        return round(max(remaining, 0.0), 2)
    if assumed_annual_return_pct == 0:
        return round(max(remaining, 0.0) / months_remaining, 2)

    r = (assumed_annual_return_pct / 100.0) / 12.0
    n = months_remaining
    factor = (1 + r) ** n
    future_value_of_current = current_amount * factor
    contribution = (target_amount - future_value_of_current) * r / (factor - 1)
    return round(max(contribution, 0.0), 2)


def cagr(beginning_value: float, ending_value: float, years: float) -> float:
    """Compound annual growth rate, as a fraction (0.10 = 10%/yr).

    Returns 0.0 if beginning_value <= 0 or years <= 0 rather than raising,
    since a CAGR is undefined in those cases and callers should treat a
    0.0 here as "not computable", not as an actual zero-growth result.
    """
    if beginning_value <= 0 or years <= 0:
        return 0.0
    return round((ending_value / beginning_value) ** (1.0 / years) - 1.0, 4)


# ---------------------------------------------------------------------------
# Stage 4 dispatcher -- calculate_metric(metric, inputs)
# ---------------------------------------------------------------------------

_SIMPLE_METRICS: dict[str, tuple[Callable[..., Any], str]] = {
    "monthly_income": (monthly_income, "sum(income_sources[].monthly_amount)"),
    "total_spending": (total_spending, "sum(abs(amount) for expense transactions)"),
    "savings_rate": (savings_rate, "(income - spending) / income"),
    "disposable_income": (disposable_income, "income - fixed_expenses_total - debt_payments_total"),
    "cash_flow_surplus_deficit": (cash_flow_surplus_deficit, "income - total_expenses"),
    "emergency_fund_coverage": (emergency_fund_coverage, "liquid_balance / monthly_essential_expenses"),
    "debt_to_income_ratio": (debt_to_income_ratio, "total_monthly_debt_payments / gross_monthly_income"),
    "subscription_totals": (subscription_totals, "sum(subscriptions[].monthly_amount)"),
    "recurring_expense_totals": (recurring_expense_totals, "sum(fixed_expenses[].amount)"),
    "goal_contribution_required": (
        goal_contribution_required,
        "annuity formula: (target - current*(1+r)^n) * r / ((1+r)^n - 1)",
    ),
    "cagr": (cagr, "(ending/beginning)^(1/years) - 1"),
    "category_spending": (category_spending, "sum(abs(amount) for expense transactions in category)"),
}


def calculate_metric(metric: str, inputs: dict[str, Any]) -> CalculationResult:
    """Single dispatch point for Stage 4 (Calculation Agent contract).

    `metric` must be one of the supported metric names; `inputs` is a
    kwargs dict matching that metric's underlying pure function, plus the
    optional bookkeeping keys "currency", "period", and "source_ids" which
    are passed through to the result but not to the function itself.
    """
    bookkeeping_keys = {"currency", "period", "source_ids"}
    fn_kwargs = {k: v for k, v in inputs.items() if k not in bookkeeping_keys}

    if metric == "budget_variance":
        value_dict = budget_variance(
            fn_kwargs.get("actual_by_category", {}), fn_kwargs.get("budget_by_category", {})
        )
        # budget_variance returns a dict, not a scalar -- surface the total
        # variance as `value` and keep the full breakdown in `inputs`.
        total_variance = round(sum(value_dict.values()), 2)
        return CalculationResult(
            metric=metric,
            value=total_variance,
            currency=inputs.get("currency", ""),
            period=inputs.get("period", ""),
            formula="sum(actual[c] - budget[c] for c in categories)",
            inputs={**fn_kwargs, "variance_by_category": value_dict},
            source_ids=inputs.get("source_ids", []),
        )

    if metric == "loan_amortization":
        result = loan_amortization(
            fn_kwargs["principal"], fn_kwargs["annual_rate_pct"], fn_kwargs["term_months"]
        )
        return CalculationResult(
            metric=metric,
            value=result["emi"],
            currency=inputs.get("currency", ""),
            period="monthly",
            formula="EMI = P*r*(1+r)^n / ((1+r)^n - 1)",
            inputs={**fn_kwargs, "total_interest": result["total_interest"],
                    "total_paid": result["total_paid"], "schedule": result["schedule"]},
            source_ids=inputs.get("source_ids", []),
        )

    if metric not in _SIMPLE_METRICS:
        raise ValueError(
            f"Unsupported metric: {metric!r}. Supported metrics: "
            f"{sorted(list(_SIMPLE_METRICS) + ['budget_variance', 'loan_amortization'])}"
        )

    fn, formula = _SIMPLE_METRICS[metric]
    value = fn(**fn_kwargs)
    return CalculationResult(
        metric=metric,
        value=float(value),
        currency=inputs.get("currency", ""),
        period=inputs.get("period", ""),
        formula=formula,
        inputs=fn_kwargs,
        source_ids=inputs.get("source_ids", []),
    )


# ---------------------------------------------------------------------------
# Cash-flow forecast (Stage 6 contract: deterministic core; the agent adds
# natural-language narration on top of this)
# ---------------------------------------------------------------------------

SCENARIO_MULTIPLIERS: dict[str, float] = {
    # Applied to variable/discretionary spending only. BASE = historical
    # average continues; CONSERVATIVE pads spending up a bit (things run
    # slightly worse than average); STRESS assumes a materially worse month.
    "BASE": 1.0,
    "CONSERVATIVE": 1.15,
    "STRESS": 1.35,
}


def forecast_cash_flow(
    opening_balance: float,
    expected_income: float,
    fixed_commitments: float,
    avg_variable_spending: float,
    debt_payments: float,
    horizon_days: int = 30,
    scenario: Literal["BASE", "CONSERVATIVE", "STRESS"] = "BASE",
) -> dict[str, Any]:
    """Contract: forecast_cash_flow(user_id, horizon_days, scenario) -- this
    is the pure-math core; the orchestrator/agent supplies the profile- and
    history-derived inputs (opening_balance, expected_income, etc.) pulled
    via db.py and tools.py before calling this.

    This is a simple linear projection over `horizon_days`, not a claim of
    certainty -- the calling agent (CASHFLOW_AGENT) is responsible for
    labeling it as an estimate and stating these assumptions to the user.
    """
    if scenario not in SCENARIO_MULTIPLIERS:
        raise ValueError(f"Unknown scenario: {scenario!r}")
    multiplier = SCENARIO_MULTIPLIERS[scenario]
    scaled_variable_spending = round(avg_variable_spending * multiplier * (horizon_days / 30.0), 2)
    scaled_fixed = round(fixed_commitments * (horizon_days / 30.0), 2)
    scaled_income = round(expected_income * (horizon_days / 30.0), 2)
    scaled_debt = round(debt_payments * (horizon_days / 30.0), 2)

    closing_balance = round(
        opening_balance + scaled_income - scaled_fixed - scaled_variable_spending - scaled_debt, 2
    )
    # Minimum projected balance: a conservative same-period estimate assuming
    # outflows land before inflows within the horizon (worst reasonable case
    # for a simple linear model, without claiming day-by-day precision).
    minimum_projected_balance = round(
        opening_balance - scaled_fixed - scaled_variable_spending - scaled_debt, 2
    )

    return {
        "scenario": scenario,
        "horizon_days": horizon_days,
        "opening_balance": round(opening_balance, 2),
        "expected_income": scaled_income,
        "fixed_commitments": scaled_fixed,
        "expected_variable_spending": scaled_variable_spending,
        "debt_payments": scaled_debt,
        "expected_closing_balance": closing_balance,
        "minimum_projected_balance": minimum_projected_balance,
        "cash_may_become_tight": minimum_projected_balance < 0,
        "assumptions": (
            f"Variable spending scaled by historical average x{multiplier} "
            f"({scenario} scenario); linear projection over {horizon_days} days; "
            "does not account for exact bill due-dates within the horizon."
        ),
    }


# ---------------------------------------------------------------------------
# Budget creation helper (Stage 15 contract: create_budget)
# ---------------------------------------------------------------------------


def create_budget(categories: list[str], limits: list[float]) -> dict[str, float]:
    """Contract: create_budget(user_id, categories, limits).

    Pure validation + assembly into the monthly_budget dict shape used by
    UserProfile.monthly_budget. Persisting it against a user_id is the
    caller's job (via db.update_memory_field), kept separate so this stays
    a pure function.
    """
    if len(categories) != len(limits):
        raise ValueError("categories and limits must be the same length")
    if any(limit < 0 for limit in limits):
        raise ValueError("budget limits must be non-negative")
    return {cat: round(float(limit), 2) for cat, limit in zip(categories, limits)}


# ---------------------------------------------------------------------------
# External-action tools -- STUBS ONLY.
#
# These three functions are the entire surface area of "external action" in
# this codebase. Each one returns a ProposedAction and does nothing else --
# no network call, no DB write that could be mistaken for execution, no
# side effect of any kind. They are intentionally NOT wired into the
# orchestrator's automatic path; nothing in this codebase can call these
# without an explicit, separate, human-reviewed call site. This is a hard
# requirement from spec section 4, enforced by the code, not by prompting.
# ---------------------------------------------------------------------------


def prepare_transfer(
    from_account: str, to_account: str, amount: float, currency: str = "INR"
) -> ProposedAction:
    return ProposedAction(
        action_type="transfer",
        details={"from_account": from_account, "to_account": to_account, "amount": amount, "currency": currency},
    )


def prepare_payment(payee: str, amount: float, due_date: Optional[str] = None, currency: str = "INR") -> ProposedAction:
    return ProposedAction(
        action_type="payment",
        details={"payee": payee, "amount": amount, "due_date": due_date, "currency": currency},
    )


def prepare_trade(
    instrument: str, side: Literal["buy", "sell"], quantity: float, order_type: str = "market"
) -> ProposedAction:
    return ProposedAction(
        action_type="trade",
        details={"instrument": instrument, "side": side, "quantity": quantity, "order_type": order_type},
    )
