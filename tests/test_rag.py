"""
Tests for finmate.rag's metadata-filter path and documented fallback
behavior. Deliberately does NOT test the Chroma/sentence-transformers
vector path, since those are lazily-imported optional dependencies (see
rag.py's module docstring) -- this suite only exercises the parts that
must work with zero network access and no extra ML install.
"""

from __future__ import annotations

import pytest

from finmate import db, rag
from finmate.schemas import Transaction


@pytest.fixture()
def db_path(tmp_path):
    path = str(tmp_path / "test_finmate.db")
    db.init_db(path)
    return path


def test_retrieve_with_no_query_returns_recency_ordered_metadata_results(db_path):
    txs = [
        Transaction(user_id="u1", date="2026-06-01", description="a", amount=-1, category="c"),
        Transaction(user_id="u1", date="2026-06-15", description="b", amount=-1, category="c"),
    ]
    db.insert_transactions(txs, db_path=db_path)
    result = rag.retrieve("u1", query="", db_path=db_path)
    assert result.vector_search_used is False
    assert [e.date for e in result.evidence] == ["2026-06-15", "2026-06-01"]


def test_retrieve_with_no_candidates_returns_empty(db_path):
    result = rag.retrieve("ghost_user", query="anything", db_path=db_path)
    assert result.evidence == []
    assert result.vector_search_used is False


def test_retrieve_falls_back_when_vector_index_unavailable(db_path, tmp_path):
    txs = [Transaction(user_id="u1", date="2026-06-01", description="Rent", amount=-32000, category="Rent")]
    db.insert_transactions(txs, db_path=db_path)
    # No vector index has been built for this user/chroma_path, so even with
    # a query string this must fall back to metadata-only results rather
    # than erroring.
    result = rag.retrieve(
        "u1", query="rent payment", db_path=db_path, chroma_path=str(tmp_path / "unused_chroma"),
    )
    assert result.vector_search_used is False
    assert len(result.evidence) == 1
    assert result.evidence[0].description == "Rent"


def test_retrieve_respects_date_range_filter(db_path):
    txs = [
        Transaction(user_id="u1", date="2026-06-01", description="june", amount=-1, category="c"),
        Transaction(user_id="u1", date="2026-07-01", description="july", amount=-1, category="c"),
    ]
    db.insert_transactions(txs, db_path=db_path)
    result = rag.retrieve("u1", start_date="2026-07-01", end_date="2026-07-31", db_path=db_path)
    assert len(result.evidence) == 1
    assert result.evidence[0].description == "july"
