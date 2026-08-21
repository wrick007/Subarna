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
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

from .schemas import Account, FinancialGoal, FixedExpense, Transaction, UserProfile

DEFAULT_DB_PATH = "data/finmate.db"


def _supabase_client():
    """Return the server-only Supabase client when persistence is configured.

    SQLite remains the zero-configuration local/test backend. Production uses
    Supabase only when both server credentials are present, so an accidentally
    missing deployment secret cannot silently fall back to temporary storage.
    """
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url and not key:
        return None
    if not url or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must both be configured.")
    try:
        from supabase import create_client
    except ImportError as exc:  # pragma: no cover - deployment dependency guard
        raise RuntimeError("Supabase persistence is configured but the supabase package is unavailable.") from exc
    return create_client(url, key)

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
    if _supabase_client() is not None:
        # Tables are created from supabase/migrations, not at runtime.
        return
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
    client = _supabase_client()
    if client is not None:
        client.table("profiles").upsert(
            {"user_id": profile.user_id, "profile": profile.model_dump(), "updated_at": profile.profile_last_updated}
        ).execute()
        return profile
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
    client = _supabase_client()
    if client is not None:
        rows = client.table("profiles").select("profile").eq("user_id", user_id).limit(1).execute().data
        return UserProfile.model_validate(rows[0]["profile"]) if rows else None
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
    client = _supabase_client()
    if client is not None:
        # Delete dependent records explicitly; the auth account itself is
        # intentionally retained so a user can start over after "Forget".
        client.table("chat_messages").delete().eq("user_id", user_id).execute()
        client.table("transactions").delete().eq("user_id", user_id).execute()
        client.table("profiles").delete().eq("user_id", user_id).execute()
        return
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
    client = _supabase_client()
    if client is not None:
        row = client.table("transactions").insert({
            "user_id": tx.user_id, "date": tx.date, "description": tx.description,
            "amount": tx.amount, "currency": tx.currency, "category": tx.category,
            "account": tx.account, "type": tx.type, "source_id": tx.source_id,
        }).execute().data[0]
        tx.id = row["id"]
        return tx
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
    client = _supabase_client()
    if client is not None:
        if txs:
            client.table("transactions").insert([
                {"user_id": t.user_id, "date": t.date, "description": t.description, "amount": t.amount,
                 "currency": t.currency, "category": t.category, "account": t.account,
                 "type": t.type, "source_id": t.source_id}
                for t in txs
            ]).execute()
        return len(txs)
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
    client = _supabase_client()
    if client is not None:
        query = client.table("transactions").select("*").eq("user_id", user_id)
        if start_date:
            query = query.gte("date", start_date)
        if end_date:
            query = query.lte("date", end_date)
        if category:
            query = query.eq("category", category)
        if account:
            query = query.eq("account", account)
        rows = query.order("date").execute().data
        return [Transaction(**row) for row in rows]

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


def append_chat_message(user_id: str, role: str, content: str, db_path: str = DEFAULT_DB_PATH) -> None:
    """Persist a settled chat turn when Supabase storage is configured."""
    client = _supabase_client()
    if client is not None:
        client.table("chat_messages").insert({"user_id": user_id, "role": role, "content": content}).execute()


def get_chat_messages(user_id: str, limit: int = 100, db_path: str = DEFAULT_DB_PATH) -> list[dict[str, str]]:
    """Return a user's saved conversation, oldest first."""
    client = _supabase_client()
    if client is None:
        return []
    rows = client.table("chat_messages").select("role,content,created_at").eq("user_id", user_id).order(
        "created_at", desc=True
    ).limit(max(1, min(limit, 500))).execute().data
    return list(reversed(rows))
