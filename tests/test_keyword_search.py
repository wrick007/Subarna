"""
Tests for finmate.keyword_search's three-tier fallback chain (FTS5 ->
rank_bm25 -> substring). Fully offline, zero network, no API key -- these
are pure in-memory ranking functions over fabricated Transaction objects,
matching the module's own "independently testable" design goal (it never
touches SQLite's transactions table, unlike finmate/db.py).

Each tier's ranking function is tested directly so these tests are
deterministic regardless of which tiers happen to be available in the
environment running them (skipped via `pytest.mark.skipif` when a tier's
real dependency isn't present). `keyword_rank`'s dispatch/precedence is
tested separately, monkeypatching the availability probes only where
doing so doesn't require calling code for a genuinely-absent dependency.
"""

from __future__ import annotations

import pytest

from finmate import keyword_search
from finmate.schemas import Transaction


def _tx(description: str, category: str = "uncategorized", source_id: str = "") -> Transaction:
    return Transaction(
        user_id="u1", date="2026-06-01", description=description, amount=-1,
        category=category, source_id=source_id or description.replace(" ", "_"),
    )


# ---------------------------------------------------------------------------
# searchable_text / _tx_key
# ---------------------------------------------------------------------------


def test_searchable_text_includes_description_and_category():
    tx = _tx("Zomato - dinner order", category="Dining out")
    text = keyword_search.searchable_text(tx)
    assert "Zomato" in text
    assert "Dining out" in text


def test_tx_key_prefers_source_id_over_db_id():
    tx = Transaction(id=42, user_id="u1", date="2026-06-01", description="x", amount=-1, source_id="txn_abc")
    assert keyword_search._tx_key(tx) == "txn_abc"


def test_tx_key_falls_back_to_db_id_when_no_source_id():
    tx = Transaction(id=42, user_id="u1", date="2026-06-01", description="x", amount=-1, source_id="")
    assert keyword_search._tx_key(tx) == "42"


# ---------------------------------------------------------------------------
# Availability probes
# ---------------------------------------------------------------------------


def test_fts5_probe_returns_a_bool():
    assert isinstance(keyword_search.fts5_available(), bool)


def test_rank_bm25_probe_returns_a_bool():
    assert isinstance(keyword_search.rank_bm25_available(), bool)


# ---------------------------------------------------------------------------
# Tier 1: FTS5
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not keyword_search.fts5_available(), reason="sqlite3 build has no FTS5 extension")
def test_fts5_tier_ranks_exact_keyword_match_first():
    candidates = [
        _tx("Rent payment - landlord NEFT", category="Rent", source_id="t1"),
        _tx("BigBasket grocery order", category="Groceries", source_id="t2"),
    ]
    hits = keyword_search._keyword_rank_fts5(candidates, "rent payment")
    assert hits
    assert hits[0].source_id == "t1"


@pytest.mark.skipif(not keyword_search.fts5_available(), reason="sqlite3 build has no FTS5 extension")
def test_fts5_tier_matches_via_category_field_too():
    candidates = [
        _tx("Health insurance premium - Star Health", category="Insurance", source_id="t1"),
        _tx("Uber rides", category="Transport / fuel", source_id="t2"),
    ]
    hits = keyword_search._keyword_rank_fts5(candidates, "insurance")
    assert [h.source_id for h in hits] == ["t1"]


@pytest.mark.skipif(not keyword_search.fts5_available(), reason="sqlite3 build has no FTS5 extension")
def test_fts5_tier_handles_special_characters_without_raising():
    # FTS5 query syntax treats quotes/asterisks/hyphens/NEAR specially;
    # raw user text containing them must never raise a MATCH syntax error.
    candidates = [_tx('Weird "quoted" -- text*', source_id="t1")]
    hits = keyword_search._keyword_rank_fts5(candidates, 'some "quoted" query* -with-hyphens NEAR')
    assert isinstance(hits, list)


@pytest.mark.skipif(not keyword_search.fts5_available(), reason="sqlite3 build has no FTS5 extension")
def test_fts5_tier_returns_empty_for_no_match():
    candidates = [_tx("BigBasket grocery order", source_id="t1")]
    assert keyword_search._keyword_rank_fts5(candidates, "zzzznonexistentword") == []


# ---------------------------------------------------------------------------
# Tier 2: rank_bm25
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not keyword_search.rank_bm25_available(), reason="rank_bm25 not installed")
def test_rank_bm25_tier_ranks_exact_keyword_match_first():
    candidates = [
        _tx("Rent payment landlord NEFT", category="Rent", source_id="t1"),
        _tx("BigBasket grocery order", category="Groceries", source_id="t2"),
    ]
    hits = keyword_search._keyword_rank_bm25_pkg(candidates, "rent payment")
    assert hits
    assert hits[0].source_id == "t1"


@pytest.mark.skipif(not keyword_search.rank_bm25_available(), reason="rank_bm25 not installed")
def test_rank_bm25_tier_returns_empty_for_no_match():
    candidates = [_tx("BigBasket grocery order", source_id="t1")]
    assert keyword_search._keyword_rank_bm25_pkg(candidates, "zzzznonexistentword") == []


# ---------------------------------------------------------------------------
# Tier 3: substring -- zero dependencies, always testable
# ---------------------------------------------------------------------------


def test_substring_tier_ranks_by_match_count():
    candidates = [
        _tx("grocery grocery grocery run", source_id="t1"),  # 3 hits
        _tx("one grocery trip", source_id="t2"),  # 1 hit
        _tx("Rent payment", source_id="t3"),  # 0 hits
    ]
    hits = keyword_search._keyword_rank_substring(candidates, "grocery")
    assert [h.source_id for h in hits] == ["t1", "t2"]


def test_substring_tier_is_case_insensitive():
    candidates = [_tx("RENT PAYMENT", source_id="t1")]
    assert [h.source_id for h in keyword_search._keyword_rank_substring(candidates, "rent")] == ["t1"]


def test_substring_tier_returns_empty_for_no_match():
    candidates = [_tx("BigBasket grocery order", source_id="t1")]
    assert keyword_search._keyword_rank_substring(candidates, "zzzznonexistentword") == []


# ---------------------------------------------------------------------------
# keyword_rank(): dispatch / precedence, and universal edge cases
# ---------------------------------------------------------------------------


def test_keyword_rank_empty_query_returns_none_tier():
    hits, tier = keyword_search.keyword_rank([_tx("BigBasket grocery order")], "")
    assert hits == []
    assert tier == "none"


def test_keyword_rank_empty_candidates_returns_none_tier():
    hits, tier = keyword_search.keyword_rank([], "groceries")
    assert hits == []
    assert tier == "none"


@pytest.mark.skipif(not keyword_search.fts5_available(), reason="sqlite3 build has no FTS5 extension")
def test_keyword_rank_prefers_fts5_when_available():
    hits, tier = keyword_search.keyword_rank([_tx("Rent payment", source_id="t1")], "rent")
    assert tier == "fts5"


@pytest.mark.skipif(not keyword_search.rank_bm25_available(), reason="rank_bm25 not installed")
def test_keyword_rank_falls_back_to_rank_bm25_when_fts5_missing(monkeypatch):
    monkeypatch.setattr(keyword_search, "fts5_available", lambda: False)
    hits, tier = keyword_search.keyword_rank([_tx("Rent payment", source_id="t1")], "rent")
    assert tier == "rank_bm25"
    assert hits and hits[0].source_id == "t1"


def test_keyword_rank_falls_back_to_substring_when_neither_available(monkeypatch):
    # Safe to fully monkeypatch both probes to False regardless of what's
    # actually installed: the substring tier has zero dependencies, so
    # forcing dispatch to it never calls code for an absent package.
    monkeypatch.setattr(keyword_search, "fts5_available", lambda: False)
    monkeypatch.setattr(keyword_search, "rank_bm25_available", lambda: False)
    hits, tier = keyword_search.keyword_rank([_tx("Rent payment", source_id="t1")], "rent")
    assert tier == "substring"
    assert hits and hits[0].source_id == "t1"


# ---------------------------------------------------------------------------
# keyword_rank_multi(): batched ranking shares one index build across
# several phrasings -- see finmate/keyword_search.py "Batched multi-query
# ranking". These tests exist because finmate/rag.py's hybrid pipeline
# calls this, not keyword_rank(), for every retrieval that has more than
# one phrasing (the original query plus query-rewrite variants).
# ---------------------------------------------------------------------------


def test_keyword_rank_multi_empty_queries_returns_none_tier():
    hits_by_query, tier = keyword_search.keyword_rank_multi([_tx("Rent payment")], [])
    assert hits_by_query == {}
    assert tier == "none"


def test_keyword_rank_multi_empty_candidates_returns_none_tier():
    hits_by_query, tier = keyword_search.keyword_rank_multi([], ["rent"])
    assert hits_by_query == {}
    assert tier == "none"


@pytest.mark.skipif(not keyword_search.fts5_available(), reason="sqlite3 build has no FTS5 extension")
def test_keyword_rank_multi_fts5_matches_calling_keyword_rank_per_query():
    candidates = [
        _tx("Rent payment - landlord NEFT", category="Rent", source_id="t1"),
        _tx("BigBasket grocery order", category="Groceries", source_id="t2"),
        _tx("Netflix subscription", category="Subscriptions", source_id="t3"),
    ]
    queries = ["rent payment", "groceries", "zzzznomatch"]

    batched, batched_tier = keyword_search.keyword_rank_multi(candidates, queries)

    for q in queries:
        single_hits, single_tier = keyword_search.keyword_rank(candidates, q)
        assert batched_tier == single_tier
        assert [(h.source_id, h.score) for h in batched[q]] == [(h.source_id, h.score) for h in single_hits]


@pytest.mark.skipif(not keyword_search.rank_bm25_available(), reason="rank_bm25 not installed")
def test_keyword_rank_multi_bm25_matches_calling_keyword_rank_per_query(monkeypatch):
    monkeypatch.setattr(keyword_search, "fts5_available", lambda: False)
    candidates = [
        _tx("Rent payment landlord NEFT", category="Rent", source_id="t1"),
        _tx("BigBasket grocery order", category="Groceries", source_id="t2"),
    ]
    queries = ["rent payment", "grocery order"]

    batched, batched_tier = keyword_search.keyword_rank_multi(candidates, queries)
    assert batched_tier == "rank_bm25"

    for q in queries:
        single_hits, _ = keyword_search.keyword_rank(candidates, q)
        assert [(h.source_id, h.score) for h in batched[q]] == [(h.source_id, h.score) for h in single_hits]


def test_keyword_rank_multi_substring_matches_calling_keyword_rank_per_query(monkeypatch):
    monkeypatch.setattr(keyword_search, "fts5_available", lambda: False)
    monkeypatch.setattr(keyword_search, "rank_bm25_available", lambda: False)
    candidates = [_tx("grocery grocery run", source_id="t1"), _tx("Rent payment", source_id="t2")]
    queries = ["grocery", "rent"]

    batched, batched_tier = keyword_search.keyword_rank_multi(candidates, queries)
    assert batched_tier == "substring"
    for q in queries:
        single_hits, _ = keyword_search.keyword_rank(candidates, q)
        assert [(h.source_id, h.score) for h in batched[q]] == [(h.source_id, h.score) for h in single_hits]


@pytest.mark.skipif(not keyword_search.fts5_available(), reason="sqlite3 build has no FTS5 extension")
def test_keyword_rank_multi_fts5_builds_the_index_only_once(monkeypatch):
    # The whole point of the batched path: N queries against the same
    # candidates should build the in-memory FTS5 table once, not N times.
    call_count = {"n": 0}
    real_build = keyword_search._build_fts5_index

    def _counting_build(conn, candidates):
        call_count["n"] += 1
        return real_build(conn, candidates)

    monkeypatch.setattr(keyword_search, "_build_fts5_index", _counting_build)
    candidates = [_tx("Rent payment", source_id="t1"), _tx("BigBasket grocery order", source_id="t2")]
    keyword_search.keyword_rank_multi(candidates, ["rent", "groceries", "nothing", "rent again"])
    assert call_count["n"] == 1
