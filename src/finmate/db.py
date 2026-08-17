"""
SQLite persistence for FinMate AI.

Implements the storage half of the tool contracts in spec section 4:
`get_user_profile`, `get_account_balances`, `get_recurring_expenses`,
`get_goals`, `search_transactions`, plus profile create/update/delete and
transaction ingestion helpers.

Two tables:
  - profiles: one row per user, profile stored as a JSON blob validated
    against `schemas.UserProfile` on every read and write.
  - transactions: normalized rows, one per transaction.

No ORM, no network calls, no API key required -- this module is fully
covered by `tests/test_db.py` using a throwaway on-disk or in-memory
SQLite file.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

from .schemas import Account, FinancialGoal, FixedExpense, Transaction, UserProfile

DEFAULT_DB_PATH = "data/finmate.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS profiles (
    user_id TEXT PRIMARY KEY,
    profile_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    date TEXT NOT NULL,
    description TEXT NOT NULL,
    amount REAL NOT NULL,
    currency TEXT NOT NULL DEFAULT 'INR',
    category TEXT NOT NULL DEFAULT 'uncategorized',
    account TEXT NOT NULL DEFAULT '',
    type TEXT NOT NULL DEFAULT 'expense',
    source_id TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_tx_user_date ON transactions(user_id, date);
CREATE INDEX IF NOT EXISTS idx_tx_user_category ON transactions(user_id, category);
"""


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """Create the database file and tables if they don't already exist."""
    path = Path(db_path)
    if path.parent and str(path.parent) not in ("", "."):
        path.parent.mkdir(parents=True, exist_ok=True)
    with _connect(db_path) as conn:
        conn.executescript(_SCHEMA)
        conn.commit()


@contextmanager
def _connect(db_path: str = DEFAULT_DB_PATH) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Profile operations
# ---------------------------------------------------------------------------


def upsert_profile(profile: UserProfile, db_path: str = DEFAULT_DB_PATH) -> UserProfile:
    """Insert or fully replace a user's profile. Stamps `profile_last_updated`."""
    profile.profile_last_updated = datetime.now(timezone.utc).isoformat()
    with _connect(db_path) as conn:
        conn.execute(
            """INSERT INTO profiles (user_id, profile_json, updated_at)
               VALUES (?, ?, ?)
               ON CONFLICT(user_id) DO UPDATE SET
                 profile_json = excluded.profile_json,
                 updated_at = excluded.updated_at""",
            (profile.user_id, profile.model_dump_json(), profile.profile_last_updated),
        )
        conn.commit()
    return profile


def get_user_profile(user_id: str, db_path: str = DEFAULT_DB_PATH) -> Optional[UserProfile]:
    """Contract: get_user_profile(user_id). Returns None if the user has no stored profile."""
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT profile_json FROM profiles WHERE user_id = ?", (user_id,)
        ).fetchone()
    if row is None:
        return None
    return UserProfile.model_validate(json.loads(row["profile_json"]))


def update_memory_field(
    user_id: str, field: str, value: object, db_path: str = DEFAULT_DB_PATH
) -> UserProfile:
    """Contract: update_memory(user_id, field, value).

    `field` is a top-level UserProfile attribute name (e.g. "monthly_income",
    "financial_goals"). Creates a blank profile first if the user has none.
    """
    profile = get_user_profile(user_id, db_path)
    if profile is None:
        profile = UserProfile(user_id=user_id)
    if not hasattr(profile, field):
        raise ValueError(f"Unknown profile field: {field!r}")
    setattr(profile, field, value)
    return upsert_profile(profile, db_path)


def delete_user_data(user_id: str, db_path: str = DEFAULT_DB_PATH) -> None:
    """Implements the sidebar 'forget this user's data' control: wipes the
    profile row and every transaction row for this user. Irreversible."""
    with _connect(db_path) as conn:
        conn.execute("DELETE FROM profiles WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM transactions WHERE user_id = ?", (user_id,))
        conn.commit()


def get_account_balances(user_id: str, db_path: str = DEFAULT_DB_PATH) -> list[Account]:
    """Contract: get_account_balances(user_id)."""
    profile = get_user_profile(user_id, db_path)
    return profile.accounts if profile else []


def get_recurring_expenses(user_id: str, db_path: str = DEFAULT_DB_PATH) -> list[FixedExpense]:
    """Contract: get_recurring_expenses(user_id)."""
    profile = get_user_profile(user_id, db_path)
    return profile.fixed_expenses if profile else []


def get_goals(user_id: str, db_path: str = DEFAULT_DB_PATH) -> list[FinancialGoal]:
    """Contract: get_goals(user_id)."""
    profile = get_user_profile(user_id, db_path)
    return profile.financial_goals if profile else []


# ---------------------------------------------------------------------------
# Transaction operations
# ---------------------------------------------------------------------------


def insert_transaction(tx: Transaction, db_path: str = DEFAULT_DB_PATH) -> Transaction:
    with _connect(db_path) as conn:
        cur = conn.execute(
            """INSERT INTO transactions
                 (user_id, date, description, amount, currency, category, account, type, source_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                tx.user_id, tx.date, tx.description, tx.amount, tx.currency,
                tx.category, tx.account, tx.type, tx.source_id,
            ),
        )
        conn.commit()
        tx.id = cur.lastrowid
    return tx


def insert_transactions(txs: list[Transaction], db_path: str = DEFAULT_DB_PATH) -> int:
    """Bulk insert; returns the number of rows inserted."""
    with _connect(db_path) as conn:
        conn.executemany(
            """INSERT INTO transactions
                 (user_id, date, description, amount, currency, category, account, type, source_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (t.user_id, t.date, t.description, t.amount, t.currency,
                 t.category, t.account, t.type, t.source_id)
                for t in txs
            ],
        )
        conn.commit()
    return len(txs)


def search_transactions(
    user_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    category: Optional[str] = None,
    account: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH,
) -> list[Transaction]:
    """Contract: search_transactions(user_id, start_date, end_date, category=None, account=None).

    This is the metadata-filter half of hybrid retrieval used by
    `finmate/rag.py` before any vector search happens.
    """
    query = "SELECT * FROM transactions WHERE user_id = ?"
    params: list[object] = [user_id]
    if start_date:
        query += " AND date >= ?"
        params.append(start_date)
    if end_date:
        query += " AND date <= ?"
        params.append(end_date)
    if category:
        query += " AND category = ?"
        params.append(category)
    if account:
        query += " AND account = ?"
        params.append(account)
    query += " ORDER BY date ASC"

    with _connect(db_path) as conn:
        rows = conn.execute(query, params).fetchall()
    return [
        Transaction(
            id=r["id"], user_id=r["user_id"], date=r["date"], description=r["description"],
            amount=r["amount"], currency=r["currency"], category=r["category"],
            account=r["account"], type=r["type"], source_id=r["source_id"],
        )
        for r in rows
    ]
