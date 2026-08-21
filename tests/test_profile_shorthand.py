"""Regression coverage for conversational profile updates and typo handling."""

from __future__ import annotations

from finmate import db
from finmate.agents.memory import detect_explicit_profile_update
from finmate.orchestrator import run_finmate
from finmate.schemas import UserProfile


def test_detects_common_typos_and_informal_income_amounts():
    action = detect_explicit_profile_update("update my incone to 80k monthly", UserProfile(user_id="u1"))
    assert action is not None
    assert action.field == "monthly_income"
    assert action.new_value == 80_000


def test_detects_emi_as_a_recurring_expense():
    action = detect_explicit_profile_update("update emi to 30k a month", UserProfile(user_id="u1"))
    assert action is not None
    assert action.field == "fixed_expenses"
    assert action.new_value[0].name == "EMI"
    assert action.new_value[0].amount == 30_000


def test_profile_shorthand_does_not_need_an_llm(tmp_path):
    path = str(tmp_path / "test.db")
    db.init_db(path)

    class NoCallsAllowed:
        def call(self, *args, **kwargs):
            raise AssertionError("an explicit shorthand update should not call the LLM")

    result = run_finmate("u1", "expenses 30k", NoCallsAllowed(), db_path=path)

    assert "monthly expenses" in result.final_response
    profile = db.get_user_profile("u1", db_path=path)
    assert profile is not None
    assert profile.variable_expenses[0].typical_monthly_amount == 30_000
