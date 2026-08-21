"""
Profile / transactions / data-control / demo-seed endpoints.

Every function here is a plain (synchronous) `def`, not `async def`, on
purpose: each one does blocking SQLite I/O (`finmate.db`), and FastAPI
automatically runs synchronous path-operation functions in an external
thread pool rather than the event loop -- the correct, idiomatic way to
mix blocking calls into an async app without every route needing to
manually wrap things in `run_in_threadpool`. See `app.routers.chat` for
why the same is true there, for a much heavier reason.
"""

from __future__ import annotations

import sys

from fastapi import APIRouter, Depends, HTTPException, status

from .. import config
from ..auth import CurrentUser, require_current_user
from ..api_schemas import (
    DeleteUserResponse,
    ProfileResponse,
    SeedDemoResponse,
    SavedChatHistoryResponse,
    TransactionOut,
    TransactionsResponse,
)

from finmate import db, rag  # noqa: E402

router = APIRouter(prefix="/users", tags=["users"])


def _authorise_user(requested_user_id: str, current_user: CurrentUser) -> str:
    """Use the verified identity in production; preserve local dev ergonomics."""
    if current_user.user_id is None:
        return requested_user_id
    if requested_user_id != current_user.user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You cannot access another user's data.")
    return current_user.user_id


@router.get("/{user_id}/profile", response_model=ProfileResponse)
def get_profile(user_id: str, current_user: CurrentUser = Depends(require_current_user)) -> ProfileResponse:
    user_id = _authorise_user(user_id, current_user)
    profile = db.get_user_profile(user_id, db_path=config.DB_PATH)
    if profile is None:
        return ProfileResponse(user_id=user_id, has_profile=False, profile=None)
    return ProfileResponse(user_id=user_id, has_profile=True, profile=profile.model_dump())


@router.get("/{user_id}/transactions", response_model=TransactionsResponse)
def get_transactions(
    user_id: str,
    start_date: str | None = None,
    end_date: str | None = None,
    category: str | None = None,
    account: str | None = None,
    limit: int = 200,
    current_user: CurrentUser = Depends(require_current_user),
) -> TransactionsResponse:
    """Powers the frontend's spending-snapshot sidebar chart. Filters
    mirror `finmate.rag.retrieve`'s own metadata filter exactly (same
    underlying `db.search_transactions` call) so "what the chart shows"
    and "what the chat evidence panel drew from" can use identical query
    parameters when a caller wants that."""
    user_id = _authorise_user(user_id, current_user)
    limit = max(1, min(limit, 1000))
    txs = db.search_transactions(
        user_id, start_date=start_date, end_date=end_date, category=category, account=account, db_path=config.DB_PATH,
    )
    txs = sorted(txs, key=lambda t: t.date, reverse=True)[:limit]
    return TransactionsResponse(
        user_id=user_id,
        count=len(txs),
        transactions=[
            TransactionOut(
                date=t.date, description=t.description, amount=t.amount, currency=t.currency,
                category=t.category, account=t.account or "", type=t.type or "",
                source_id=t.source_id or (str(t.id) if t.id is not None else ""),
            )
            for t in txs
        ],
    )


@router.delete("/{user_id}", response_model=DeleteUserResponse)
def delete_user(user_id: str, current_user: CurrentUser = Depends(require_current_user)) -> DeleteUserResponse:
    """"Forget this user's data": removes both the SQLite rows
    (`db.delete_user_data`, same call app.py's Streamlit sidebar button
    makes) and, if one was ever built, this user's Chroma vector
    collection (`rag.delete_user_vector_index`) -- deleting only the
    former and leaving the latter behind would be an incomplete "forget
    my data" in anything but name.
    """
    user_id = _authorise_user(user_id, current_user)
    db.delete_user_data(user_id, db_path=config.DB_PATH)
    rag.delete_user_vector_index(user_id, chroma_path=config.CHROMA_PATH)
    return DeleteUserResponse(user_id=user_id, deleted=True)


@router.get("/{user_id}/messages", response_model=SavedChatHistoryResponse)
def get_saved_messages(
    user_id: str, limit: int = 100, current_user: CurrentUser = Depends(require_current_user)
) -> SavedChatHistoryResponse:
    user_id = _authorise_user(user_id, current_user)
    return SavedChatHistoryResponse(messages=db.get_chat_messages(user_id, limit=limit, db_path=config.DB_PATH))


@router.post("/seed-demo-data", response_model=SeedDemoResponse)
def seed_demo_data() -> SeedDemoResponse:
    """Seeds the fixed synthetic demo dataset (`data/synthetic_profile.json`
    + `data/synthetic_transactions.csv`, always user_id="demo_user" --
    same fixed target `scripts/seed_demo_data.py` uses) so a fresh
    deployment has something to ask about immediately. Reuses that
    script's own `seed()` function rather than reimplementing it (one
    place this logic can drift, not two), including its idempotency
    fix -- see that function's docstring -- so calling this endpoint
    more than once (a person double-clicking "Load demo data" in the
    frontend) re-seeds cleanly instead of duplicating every transaction.
    """
    if str(config.REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(config.REPO_ROOT))
    try:
        from scripts.seed_demo_data import seed  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - repo layout invariant, not a runtime condition
        raise HTTPException(status_code=500, detail=f"Could not load the demo-data seeder: {exc}") from exc

    result = seed(config.DB_PATH, build_vector_index=True, verbose=False, chroma_path=config.CHROMA_PATH)
    return SeedDemoResponse(
        user_id=result.user_id,
        transactions_seeded=result.transactions_seeded,
        vector_indexed=result.vector_indexed,
    )
