"""
Tests for finmate.tools -- the deterministic calculation engine.

Every formula has at least one test against a hand-computable or
independently-known-correct expected value, per spec section 6's
requirement to verify loan_amortization against a hand-computed EMI.
Run with zero network access and no API key: `pytest tests/test_tools.py`.
"""

from __future__ import annotations

import pytest

from finmate import tools
from finmate.schemas import ProposedAction, Transaction


def make_tx(amount: float, category: str = "Groceries", ttype: str = "expense", date: str = "2026-06-01") -> Transaction:
    return Transaction(
        user_id="u1", date=date, description="test", amount=amount, currency="INR",
        category=category, account="checking", type=ttype,
    )


# ---------------------------------------------------------------------------
# Income / spending
# ---------------------------------------------------------------------------


def test_monthly_income_sums_sources():
    sources = [{"monthly_amount": 50000}, {"monthly_amount": 12000.5}]
    assert tools.monthly_income(sources) == 62000.5


def test_monthly_income_empty_list():
    assert tools.monthly_income([]) == 0


def test_total_spending_only_counts_expenses():
    txs = [make_tx(-100, ttype="expense"), make_tx(500, ttype="income"), make_tx(-50, ttype="expense")]
    assert tools.total_spending(txs) == 150


def test_category_spending_case_insensitive():
    txs = [make_tx(-100, category="Dining Out"), make_tx(-50, category="dining out"), make_tx(-30, category="Groceries")]
    assert tools.category_spending(txs, "Dining Out") == 150


def test_savings_rate_basic():
    assert tools.savings_rate(income=100000, spending=75000) == 0.25


def test_savings_rate_zero_income_returns_zero_not_error():
    assert tools.savings_rate(income=0, spending=500) == 0.0


def test_savings_rate_negative_when_overspending():
    assert tools.savings_rate(income=1000, spending=1500) == -0.5


def test_disposable_income():
    assert tools.disposable_income(income=100000, fixed_expenses_total=40000, debt_payments_total=10000) == 50000


def test_cash_flow_surplus_deficit_surplus_and_deficit():
    assert tools.cash_flow_surplus_deficit(income=5000, total_expenses=4000) == 1000
    assert tools.cash_flow_surplus_deficit(income=3000, total_expenses=4500) == -1500


def test_budget_variance_includes_categories_present_in_only_one_side():
    actual = {"Groceries": 13500, "Dining out": 9000}
    budget = {"Groceries": 13000, "Shopping": 5000}
    variance = tools.budget_variance(actual, budget)
    assert variance["Groceries"] == 500  # over budget
    assert variance["Dining out"] == 9000  # no budget set -> full amount as "variance"
    assert variance["Shopping"] == -5000  # budgeted but nothing spent


def test_emergency_fund_coverage_basic():
    assert tools.emergency_fund_coverage(liquid_balance=60000, monthly_essential_expenses=20000) == 3.0


def test_emergency_fund_coverage_zero_expenses_returns_sentinel():
    assert tools.emergency_fund_coverage(liquid_balance=1000, monthly_essential_expenses=0) == 999.0


def test_debt_to_income_ratio():
    assert tools.debt_to_income_ratio(total_monthly_debt_payments=12000, gross_monthly_income=100000) == 0.12


def test_debt_to_income_ratio_zero_income():
    assert tools.debt_to_income_ratio(1000, 0) == 0.0


# ---------------------------------------------------------------------------
# Loan amortization -- hand-verified against the classic textbook example:
# principal=100,000, annual rate=10%, term=12 months -> EMI ~= 8791.59
# (r = 0.10/12 = 0.0083333; EMI = P*r*(1+r)^12 / ((1+r)^12 - 1))
# ---------------------------------------------------------------------------


def test_loan_amortization_matches_hand_computed_emi():
    result = tools.loan_amortization(principal=100000, annual_rate_pct=10, term_months=12)
    assert result["emi"] == pytest.approx(8791.59, abs=0.5)
    # sanity: schedule has 12 rows, ends near zero balance
    assert len(result["schedule"]) == 12
    assert result["schedule"][-1]["remaining_balance"] == pytest.approx(0, abs=1.0)
    # total interest should roughly equal total_paid - principal
    assert result["total_interest"] == pytest.approx(result["total_paid"] - principal_for_check(), abs=5)


def principal_for_check():
    return 100000


def test_loan_amortization_zero_interest_is_linear_split():
    result = tools.loan_amortization(principal=12000, annual_rate_pct=0, term_months=12)
    assert result["emi"] == 1000
    assert result["total_interest"] == 0.0
    assert result["schedule"][-1]["remaining_balance"] == 0.0


def test_loan_amortization_rejects_bad_term():
    with pytest.raises(ValueError):
        tools.loan_amortization(principal=1000, annual_rate_pct=5, term_months=0)


def test_loan_amortization_rejects_negative_principal():
    with pytest.raises(ValueError):
        tools.loan_amortization(principal=-100, annual_rate_pct=5, term_months=12)


# ---------------------------------------------------------------------------
# Subscriptions / recurring
# ---------------------------------------------------------------------------


def test_subscription_totals():
    subs = [{"monthly_amount": 649}, {"monthly_amount": 119}, {"monthly_amount": 179}]
    assert tools.subscription_totals(subs) == 947


def test_recurring_expense_totals():
    fixed = [{"amount": 32000}, {"amount": 2400}, {"amount": 9500}]
    assert tools.recurring_expense_totals(fixed) == 43900


# ---------------------------------------------------------------------------
# Goal contribution required
# ---------------------------------------------------------------------------


def test_goal_contribution_required_zero_return_is_linear():
    # (target - current) / months = (250000 - 40000) / 7 = 30000.0
    assert tools.goal_contribution_required(250000, 40000, 7) == pytest.approx(30000.0, abs=0.01)


def test_goal_contribution_required_already_met_returns_zero():
    assert tools.goal_contribution_required(1000, 1500, 6) == 0.0


def test_goal_contribution_required_zero_months_returns_full_remaining():
    assert tools.goal_contribution_required(10000, 4000, 0) == 6000.0


def test_goal_contribution_required_with_assumed_return_is_lower_than_linear():
    linear = tools.goal_contribution_required(400000, 210000, 10, assumed_annual_return_pct=0)
    with_return = tools.goal_contribution_required(400000, 210000, 10, assumed_annual_return_pct=6)
    assert with_return < linear


# ---------------------------------------------------------------------------
# CAGR -- hand-verified: 1000 -> 2000 over 3 years = 2^(1/3) - 1 ~= 0.259921
# ---------------------------------------------------------------------------


def test_cagr_matches_hand_computed_value():
    assert tools.cagr(beginning_value=1000, ending_value=2000, years=3) == pytest.approx(0.2599, abs=0.001)


def test_cagr_undefined_cases_return_zero_not_error():
    assert tools.cagr(0, 2000, 3) == 0.0
    assert tools.cagr(1000, 2000, 0) == 0.0


# ---------------------------------------------------------------------------
# calculate_metric dispatcher
# ---------------------------------------------------------------------------


def test_calculate_metric_simple_metric_records_formula_and_inputs():
    result = tools.calculate_metric("savings_rate", {"income": 1000, "spending": 750, "currency": "INR"})
    assert result.metric == "savings_rate"
    assert result.value == 0.25
    assert result.currency == "INR"
    assert "formula" in result.model_dump() and result.formula


def test_calculate_metric_loan_amortization_shape():
    result = tools.calculate_metric(
        "loan_amortization", {"principal": 100000, "annual_rate_pct": 10, "term_months": 12}
    )
    assert result.metric == "loan_amortization"
    assert result.value == pytest.approx(8791.59, abs=0.5)
    assert "schedule" in result.inputs


def test_calculate_metric_budget_variance_shape():
    result = tools.calculate_metric(
        "budget_variance",
        {"actual_by_category": {"Groceries": 100}, "budget_by_category": {"Groceries": 80}},
    )
    assert result.value == 20
    assert result.inputs["variance_by_category"]["Groceries"] == 20


def test_calculate_metric_unknown_metric_raises():
    with pytest.raises(ValueError):
        tools.calculate_metric("not_a_real_metric", {})


# ---------------------------------------------------------------------------
# Cash-flow forecast
# ---------------------------------------------------------------------------


def test_forecast_cash_flow_stress_worse_than_base():
    base = tools.forecast_cash_flow(
        opening_balance=50000, expected_income=100000, fixed_commitments=40000,
        avg_variable_spending=30000, debt_payments=10000, horizon_days=30, scenario="BASE",
    )
    stress = tools.forecast_cash_flow(
        opening_balance=50000, expected_income=100000, fixed_commitments=40000,
        avg_variable_spending=30000, debt_payments=10000, horizon_days=30, scenario="STRESS",
    )
    assert stress["expected_closing_balance"] < base["expected_closing_balance"]


def test_forecast_cash_flow_flags_tight_cash():
    result = tools.forecast_cash_flow(
        opening_balance=1000, expected_income=0, fixed_commitments=5000,
        avg_variable_spending=2000, debt_payments=0, horizon_days=30, scenario="BASE",
    )
    assert result["cash_may_become_tight"] is True


def test_forecast_cash_flow_rejects_unknown_scenario():
    with pytest.raises(ValueError):
        tools.forecast_cash_flow(0, 0, 0, 0, 0, scenario="WILD_GUESS")


# ---------------------------------------------------------------------------
# create_budget
# ---------------------------------------------------------------------------


def test_create_budget_basic():
    budget = tools.create_budget(["Groceries", "Dining out"], [13000, 7000])
    assert budget == {"Groceries": 13000.0, "Dining out": 7000.0}


def test_create_budget_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        tools.create_budget(["Groceries"], [1000, 2000])


def test_create_budget_negative_limit_raises():
    with pytest.raises(ValueError):
        tools.create_budget(["Groceries"], [-100])


# ---------------------------------------------------------------------------
# External-action stubs: must NEVER execute anything, only ever return a
# ProposedAction pending confirmation.
# ---------------------------------------------------------------------------


def test_prepare_transfer_returns_proposed_action_only():
    action = tools.prepare_transfer("Checking", "Savings", 5000)
    assert isinstance(action, ProposedAction)
    assert action.status == "proposed_pending_confirmation"
    assert action.requires_user_confirmation is True
    assert action.action_type == "transfer"


def test_prepare_payment_returns_proposed_action_only():
    action = tools.prepare_payment("Landlord", 32000, due_date="2026-08-01")
    assert action.status == "proposed_pending_confirmation"
    assert action.details["payee"] == "Landlord"


def test_prepare_trade_returns_proposed_action_only():
    action = tools.prepare_trade("NIFTYBEES", "buy", 10)
    assert action.status == "proposed_pending_confirmation"
    assert action.details["side"] == "buy"
