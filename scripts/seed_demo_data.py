"""
Seed FinMate AI's local database with synthetic demo data.

Usage:
    python scripts/seed_demo_data.py [--db-path data/finmate.db]

Loads data/synthetic_profile.json and data/synthetic_transactions.csv for
user_id "demo_user" -- enough to demo every intent in the Stage 14 routing
table, including two planted anomalies for the Anomaly Agent to find:
  1. A subscription price jump (Netflix: 499 -> 649 between June and July).
  2. A duplicate-looking charge (two identical "Zomato - dinner order"
     charges for 650 INR on 2026-07-06).
  3. One unusually large one-off transaction (a MacBook repair) to
     exercise "unusually large transaction" detection too.

Requires no API key -- this only touches finmate.db (and, best-effort,
finmate.rag's vector index if the ML dependencies are installed).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from finmate import db, rag  # noqa: E402
from finmate.schemas import Transaction, UserProfile  # noqa: E402

DATA_DIR = REPO_ROOT / "data"


@dataclass
class SeedResult:
    user_id: str
    transactions_seeded: int
    vector_indexed: int


def seed(
    db_path: str, build_vector_index: bool = True, verbose: bool = True, chroma_path: str = rag.DEFAULT_CHROMA_PATH,
) -> SeedResult:
    """Load the synthetic demo profile + transactions into `db_path`
    for user_id "demo_user". Returns a `SeedResult` with real counts --
    used both by this script's own `__main__` block (printed below) and
    by `backend/app/routers/users.py`'s `/api/users/seed-demo-data`
    endpoint, so the API reports exactly what happened rather than
    inferring it after the fact from a before/after row-count diff.

    `chroma_path` matters whenever the caller isn't using the default
    on-disk location -- the backend, for instance, may have
    `FINMATE_CHROMA_PATH` pointed elsewhere (see backend/app/config.py).
    Passing it through to `rag.index_transactions_for_user` here (rather
    than silently indexing at `rag.DEFAULT_CHROMA_PATH` regardless of
    what the caller configured) is what makes the two agree on where the
    index actually lives -- otherwise seeding would appear to succeed
    while real retrieval, reading from the *configured* path, found
    nothing there.

    Idempotent: `db.insert_transactions` is a plain INSERT with no
    dedup (correctly so -- a real user's transactions can legitimately
    repeat, and this project doesn't assume `source_id` is populated or
    unique for user-entered data), so calling this function twice back
    to back would otherwise leave "demo_user" with two copies of every
    transaction. `seed()` avoids that itself, narrowly, by clearing
    *only* "demo_user"'s existing rows (`db.delete_user_data` -- the
    same call the "forget my data" button uses) immediately before
    inserting the fresh set, rather than changing `insert_transactions`'
    general semantics. The vector index doesn't need the same treatment:
    `rag.index_transactions_for_user` upserts by each transaction's own
    `source_id` (see `rag._tx_key`), which is stable across runs, so
    re-indexing the same CSV just overwrites the same points.

    `verbose=False` (the API's case -- an HTTP response already reports
    the outcome; stdout in a server process is just noise) skips the
    print statements.
    """
    db.init_db(db_path)

    profile_path = DATA_DIR / "synthetic_profile.json"
    profile_data = json.loads(profile_path.read_text())
    profile = UserProfile.model_validate(profile_data)

    db.delete_user_data(profile.user_id, db_path=db_path)  # see "Idempotent" above
    db.upsert_profile(profile, db_path=db_path)
    if verbose:
        print(f"Seeded profile for user_id={profile.user_id!r}")

    tx_path = DATA_DIR / "synthetic_transactions.csv"
    with tx_path.open(newline="") as f:
        reader = csv.DictReader(f)
        transactions = [
            Transaction(
                user_id=row["user_id"], date=row["date"], description=row["description"],
                amount=float(row["amount"]), currency=row["currency"], category=row["category"],
                account=row["account"], type=row["type"], source_id=row["source_id"],
            )
            for row in reader
        ]
    inserted = db.insert_transactions(transactions, db_path=db_path)
    if verbose:
        print(f"Seeded {inserted} transactions for user_id={profile.user_id!r}")

    indexed = 0
    if build_vector_index:
        indexed = rag.index_transactions_for_user(profile.user_id, transactions, db_path=db_path, chroma_path=chroma_path)
        if verbose:
            if indexed:
                print(f"Indexed {indexed} transactions into the vector store for semantic retrieval.")
            else:
                print(
                    "Skipped vector indexing (chromadb / sentence-transformers not installed, or "
                    "already unavailable). Metadata-filtered retrieval still works fully -- see "
                    "finmate/rag.py's documented fallback."
                )

    if verbose:
        print("\nDone. Try asking things like:")
        print("  - 'What's my savings rate this month?'")
        print("  - 'Am I overspending on dining out?'")
        print("  - 'Do I have any duplicate or unusual charges?'")
        print("  - 'How long until I hit my Japan trip goal?'")

    return SeedResult(user_id=profile.user_id, transactions_seeded=inserted, vector_indexed=indexed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", default=db.DEFAULT_DB_PATH)
    parser.add_argument("--chroma-path", default=rag.DEFAULT_CHROMA_PATH)
    parser.add_argument("--no-vector-index", action="store_true", help="Skip building the Chroma vector index.")
    args = parser.parse_args()
    seed(args.db_path, build_vector_index=not args.no_vector_index, chroma_path=args.chroma_path)
