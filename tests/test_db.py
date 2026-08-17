"""
Tests for finmate.db. Each test gets a fresh throwaway SQLite file via the
`db_path` fixture so tests never interfere with each other or with any
real data. No network access, no API key required.
"""

from __future__ import annotations

import pytest

from finmate import db
from finmate.schemas import Transaction, UserProfile


@pytest.fixture()
def db_path(tmp_path):
    path = str(tmp_path / "test_finmate.db")
    db.init_db(path)
    return path


def test_get_user_profile_returns_none_when_missing(db_path):
    assert db.get_user_profile("nobody", db_path=db_path) is None


def test_upsert_and_get_profile_roundtrip(db_path):
    profile = UserProfile(user_id="u1", currency="INR", monthly_income=90000)
    db.upsert_profile(profile, db_path=db_path)
    fetched = db.get_user_profile("u1", db_path=db_path)
    assert fetched is not None
    assert fetched.user_id == "u1"
    assert fetched.monthly_income == 90000
    assert fetched.profile_last_updated  # stamped on upsert


def test_upsert_profile_overwrites_existing(db_path):
    db.upsert_profile(UserProfile(user_id="u1", monthly_income=1000), db_path=db_path)
    db.upsert_profile(UserProfile(user_id="u1", monthly_income=2000), db_path=db_path)
    fetched = db.get_user_profile("u1", db_path=db_path)
    assert fetched.monthly_income == 2000


def test_update_memory_field_creates_profile_if_missing(db_path):
    updated = db.update_memory_field("new_user", "monthly_income", 55000, db_path=db_path)
    assert updated.monthly_income == 55000
    fetched = db.get_user_profile("new_user", db_path=db_path)
    assert fetched.monthly_income == 55000


def test_update_memory_field_unknown_field_raises(db_path):
    with pytest.raises(ValueError):
        db.update_memory_field("u1", "not_a_real_field", 123, db_path=db_path)


def test_delete_user_data_removes_profile_and_transactions(db_path):
    db.upsert_profile(UserProfile(user_id="u1"), db_path=db_path)
    db.insert_transaction(
        Transaction(user_id="u1", date="2026-06-01", description="x", amount=-100, category="Groceries"),
        db_path=db_path,
    )
    db.delete_user_data("u1", db_path=db_path)
    assert db.get_user_profile("u1", db_path=db_path) is None
    assert db.search_transactions("u1", db_path=db_path) == []


def test_delete_user_data_does_not_touch_other_users(db_path):
    db.upsert_profile(UserProfile(user_id="u1"), db_path=db_path)
    db.upsert_profile(UserProfile(user_id="u2", monthly_income=42), db_path=db_path)
    db.delete_user_data("u1", db_path=db_path)
    assert db.get_user_profile("u1", db_path=db_path) is None
    fetched_u2 = db.get_user_profile("u2", db_path=db_path)
    assert fetched_u2 is not None and fetched_u2.monthly_income == 42


def test_get_account_balances_returns_empty_list_for_unknown_user(db_path):
    assert db.get_account_balances("ghost", db_path=db_path) == []


def test_insert_and_search_transactions_basic(db_path):
    txs = [
        Transaction(user_id="u1", date="2026-06-01", description="Salary", amount=100000, category="income", type="income"),
        Transaction(user_id="u1", date="2026-06-05", description="Rent", amount=-32000, category="Rent", type="expense"),
        Transaction(user_id="u1", date="2026-07-01", description="Salary", amount=100000, category="income", type="income"),
    ]
    db.insert_transactions(txs, db_path=db_path)
    all_tx = db.search_transactions("u1", db_path=db_path)
    assert len(all_tx) == 3
    assert all(t.id is not None for t in all_tx)


def test_search_transactions_filters_by_date_range(db_path):
    txs = [
        Transaction(user_id="u1", date="2026-06-01", description="a", amount=-1, category="c"),
        Transaction(user_id="u1", date="2026-06-15", description="b", amount=-1, category="c"),
        Transaction(user_id="u1", date="2026-07-01", description="c", amount=-1, category="c"),
    ]
    db.insert_transactions(txs, db_path=db_path)
    june_only = db.search_transactions("u1", start_date="2026-06-01", end_date="2026-06-30", db_path=db_path)
    assert len(june_only) == 2
    assert all(t.date.startswith("2026-06") for t in june_only)


def test_search_transactions_filters_by_category(db_path):
    txs = [
        Transaction(user_id="u1", date="2026-06-01", description="a", amount=-1, category="Groceries"),
        Transaction(user_id="u1", date="2026-06-02", description="b", amount=-1, category="Dining out"),
    ]
    db.insert_transactions(txs, db_path=db_path)
    groceries = db.search_transactions("u1", category="Groceries", db_path=db_path)
    assert len(groceries) == 1
    assert groceries[0].description == "a"


def test_search_transactions_scoped_to_user(db_path):
    db.insert_transactions(
        [Transaction(user_id="u1", date="2026-06-01", description="a", amount=-1, category="c"),
         Transaction(user_id="u2", date="2026-06-01", description="b", amount=-1, category="c")],
        db_path=db_path,
    )
    assert len(db.search_transactions("u1", db_path=db_path)) == 1
    assert len(db.search_transactions("u2", db_path=db_path)) == 1


def test_get_goals_and_recurring_expenses_from_profile(db_path):
    profile = UserProfile(
        user_id="u1",
        fixed_expenses=[{"name": "Rent", "amount": 32000, "due_day": 1}],
        financial_goals=[{"name": "Emergency fund", "target_amount": 100000, "current_amount": 20000,
                           "deadline": "2027-01-01", "months_remaining": 12}],
    )
    db.upsert_profile(profile, db_path=db_path)
    goals = db.get_goals("u1", db_path=db_path)
    expenses = db.get_recurring_expenses("u1", db_path=db_path)
    assert len(goals) == 1 and goals[0].name == "Emergency fund"
    assert len(expenses) == 1 and expenses[0].amount == 32000
