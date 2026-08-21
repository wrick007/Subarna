"""
Tests for the performance work described in finmate/rag.py's module
docstring "Performance" section: process-lifetime model/client caching,
running vector search concurrently with query rewrite, and optional
request-scoped retrieval memoization. Deliberately separate from
tests/test_rag.py (metadata-floor behavior) and tests/test_rag_hybrid.py
(hybrid pipeline correctness) -- this file is only about "same output,
cheaper to compute", and every test here should fail if a future change
silently regresses one of those properties back to the pre-upgrade
per-call reload / fully-sequential / always-recompute behavior.

Every test here runs with zero network access and no real model
download, matching the rest of this test suite's offline guarantee.
"""

from __future__ import annotations

import sys
import threading
import types

import pytest

from finmate import db, keyword_search, query_rewrite, rag
from finmate.schemas import Transaction


def _seed(db_path: str, txs: list[Transaction]) -> None:
    db.init_db(db_path)
    db.insert_transactions(txs, db_path=db_path)


@pytest.fixture(autouse=True)
def _clear_rag_caches():
    # Every test below either exercises a cache directly or must not be
    # affected by another test's (or another test *file's*) cached
    # state -- see rag.clear_cache()'s own docstring for why a stale
    # cache silently defeats what a test thinks it's testing. Clears
    # query_rewrite's cache too: several tests below reuse common query
    # strings ("rent", "food expenses") that other test files also use
    # to exercise query rewrite, and that cache is keyed only by
    # (user_id, normalized query) -- see test_rag_hybrid.py's identical
    # fixture for the fuller explanation.
    rag.clear_cache()
    query_rewrite.clear_cache()
    yield
    rag.clear_cache()
    query_rewrite.clear_cache()


# ---------------------------------------------------------------------------
# _get_embedder / _get_chroma_client: process-lifetime singleton caching
# ---------------------------------------------------------------------------


def test_get_embedder_constructs_once_and_reuses_the_instance(monkeypatch):
    construct_count = {"n": 0}

    class _FakeSentenceTransformer:
        def __init__(self, model_name):
            construct_count["n"] += 1
            self.model_name = model_name

    fake_module = types.ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = _FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    first = rag._get_embedder("fake-model")
    second = rag._get_embedder("fake-model")

    assert first is second
    assert construct_count["n"] == 1


def test_get_embedder_caches_a_failed_load_without_retrying(monkeypatch):
    construct_count = {"n": 0}

    class _AlwaysFails:
        def __init__(self, model_name):
            construct_count["n"] += 1
            raise OSError("simulated: couldn't connect to huggingface.co")

    fake_module = types.ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = _AlwaysFails
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    first = rag._get_embedder("fake-model-2")
    second = rag._get_embedder("fake-model-2")

    assert first is None
    assert second is None
    assert construct_count["n"] == 1  # not re-attempted on the second call


def test_get_embedder_clear_cache_forces_a_fresh_attempt(monkeypatch):
    construct_count = {"n": 0}

    class _FakeSentenceTransformer:
        def __init__(self, model_name):
            construct_count["n"] += 1

    fake_module = types.ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = _FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    rag._get_embedder("fake-model-3")
    rag.clear_cache()
    rag._get_embedder("fake-model-3")

    assert construct_count["n"] == 2


def test_get_chroma_client_is_cached_per_path_not_globally(tmp_path):
    pytest.importorskip("chromadb")
    path_a = str(tmp_path / "chroma_a")
    path_b = str(tmp_path / "chroma_b")

    client_a1 = rag._get_chroma_client(path_a)
    client_a2 = rag._get_chroma_client(path_a)
    client_b = rag._get_chroma_client(path_b)

    assert client_a1 is not None
    assert client_a1 is client_a2  # same path -> cache hit, same instance
    assert client_b is not None
    assert client_b is not client_a1  # different path -> not conflated


# ---------------------------------------------------------------------------
# retrieve(): vector search runs concurrently with query rewrite, not
# sequentially after it
# ---------------------------------------------------------------------------


def test_vector_search_is_kicked_off_before_query_rewrite_completes(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    _seed(db_path, [
        Transaction(user_id="u1", date="2026-06-01", description="Rent payment",
                    amount=-32000, category="Rent", source_id="a"),
    ])

    rewrite_started = threading.Event()
    release_rewrite = threading.Event()
    vector_search_started = threading.Event()

    class _BlockingRewriteClient:
        def call(self, *args, **kwargs):
            rewrite_started.set()
            # Held open deliberately: if retrieve() sequenced vector
            # search *after* this call returns (the pre-upgrade
            # behavior), vector_search_started could not be set while
            # we're blocked here -- the assertion below would time out.
            release_rewrite.wait(timeout=5)
            raise RuntimeError("simulated failure -- retrieve() must still degrade cleanly")

    def _fake_vector_search(*args, **kwargs):
        vector_search_started.set()
        return [], {}

    monkeypatch.setattr(rag, "_vector_search", _fake_vector_search)

    outcome: dict = {}

    def _run():
        outcome["result"] = rag.retrieve(
            "u1", query="rent", db_path=db_path, llm_client=_BlockingRewriteClient(),
        )

    t = threading.Thread(target=_run)
    t.start()
    try:
        assert rewrite_started.wait(timeout=5), "query rewrite never started"
        assert vector_search_started.wait(timeout=5), (
            "vector search did not start while query rewrite was still blocked -- "
            "it is being sequenced after stage 2 instead of running concurrently"
        )
    finally:
        release_rewrite.set()
        t.join(timeout=5)

    assert "result" in outcome  # retrieve() completed despite the rewrite failure


def test_retrieve_result_unaffected_by_concurrency_change(tmp_path):
    # Same fixture and assertions as
    # tests/test_rag.py::test_retrieve_with_no_query_returns_recency_ordered_metadata_results
    # -- included here specifically as a "the output didn't change, only
    # the schedule did" regression check colocated with the rest of this
    # file's performance-only assertions.
    db_path = str(tmp_path / "test.db")
    _seed(db_path, [
        Transaction(user_id="u1", date="2026-06-01", description="a", amount=-1, category="c"),
        Transaction(user_id="u1", date="2026-06-15", description="b", amount=-1, category="c"),
    ])
    result = rag.retrieve("u1", query="", db_path=db_path)
    assert result.vector_search_used is False
    assert [e.date for e in result.evidence] == ["2026-06-15", "2026-06-01"]


# ---------------------------------------------------------------------------
# retrieve(): keyword search shares one index build across every
# phrasing, even inside the full pipeline (not just at the
# keyword_search.keyword_rank_multi unit-test level)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not keyword_search.fts5_available(), reason="sqlite3 build has no FTS5 extension")
def test_retrieve_builds_the_keyword_index_only_once_across_all_phrasings(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    _seed(db_path, [
        Transaction(user_id="u1", date="2026-06-06", description="Zomato - dinner order",
                    amount=-650, category="Dining out", source_id="a"),
        Transaction(user_id="u1", date="2026-06-11", description="BigBasket grocery order",
                    amount=-3400, category="Groceries", source_id="b"),
    ])

    class _FakeRewriteClient:
        def call(self, agent_system_prompt, user_message, response_model=None, **kwargs):
            return response_model(phrasings=["dining", "zomato", "groceries"])

    build_count = {"n": 0}
    real_build = keyword_search._build_fts5_index

    def _counting_build(conn, candidates):
        build_count["n"] += 1
        return real_build(conn, candidates)

    monkeypatch.setattr(keyword_search, "_build_fts5_index", _counting_build)

    result = rag.retrieve(
        "u1", query="food expenses", db_path=db_path,
        enable_query_rewrite=True, enable_vector_search=False,
        llm_client=_FakeRewriteClient(),
    )

    # 4 phrasings searched (original + 3 rewrites) but the index -- a
    # small, fixed candidate set -- was only built once.
    assert build_count["n"] == 1
    assert result.query_rewrite_used is True
    assert {e.source_id for e in result.evidence} == {"a", "b"}


# ---------------------------------------------------------------------------
# retrieve(cache=...): request-scoped memoization
# ---------------------------------------------------------------------------


def test_retrieve_without_cache_recomputes_every_call(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    _seed(db_path, [
        Transaction(user_id="u1", date="2026-06-01", description="Rent payment",
                    amount=-32000, category="Rent", source_id="a"),
    ])
    call_count = {"n": 0}
    real_search = db.search_transactions

    def _counting_search(*args, **kwargs):
        call_count["n"] += 1
        return real_search(*args, **kwargs)

    monkeypatch.setattr(rag.db, "search_transactions", _counting_search)

    rag.retrieve("u1", query="rent", db_path=db_path, enable_query_rewrite=False)
    rag.retrieve("u1", query="rent", db_path=db_path, enable_query_rewrite=False)

    assert call_count["n"] == 2  # cache=None (the default): no memoization, unchanged from before


def test_retrieve_with_cache_reuses_result_for_an_identical_call(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    _seed(db_path, [
        Transaction(user_id="u1", date="2026-06-01", description="Rent payment",
                    amount=-32000, category="Rent", source_id="a"),
    ])
    call_count = {"n": 0}
    real_search = db.search_transactions

    def _counting_search(*args, **kwargs):
        call_count["n"] += 1
        return real_search(*args, **kwargs)

    monkeypatch.setattr(rag.db, "search_transactions", _counting_search)

    cache: dict = {}
    result1 = rag.retrieve("u1", query="rent", db_path=db_path, enable_query_rewrite=False, cache=cache)
    result2 = rag.retrieve("u1", query="rent", db_path=db_path, enable_query_rewrite=False, cache=cache)

    assert result1 is result2
    assert call_count["n"] == 1  # second call was a cache hit -- never touched the DB


def test_retrieve_cache_is_keyed_by_query_not_shared_across_different_queries(tmp_path):
    db_path = str(tmp_path / "test.db")
    _seed(db_path, [
        Transaction(user_id="u1", date="2026-06-01", description="Rent payment",
                    amount=-32000, category="Rent", source_id="a"),
        Transaction(user_id="u1", date="2026-06-14", description="Netflix subscription",
                    amount=-499, category="Subscriptions", source_id="b"),
    ])
    cache: dict = {}
    rent_result = rag.retrieve("u1", query="rent", db_path=db_path, enable_query_rewrite=False, cache=cache)
    netflix_result = rag.retrieve("u1", query="netflix", db_path=db_path, enable_query_rewrite=False, cache=cache)

    assert rent_result is not netflix_result
    assert [e.source_id for e in rent_result.evidence] == ["a"]
    assert [e.source_id for e in netflix_result.evidence] == ["b"]


def test_retrieve_cache_is_keyed_by_user_not_shared_across_users(tmp_path):
    db_path = str(tmp_path / "test.db")
    _seed(db_path, [
        Transaction(user_id="user_a", date="2026-06-01", description="Rent payment",
                    amount=-32000, category="Rent", source_id="a"),
        Transaction(user_id="user_b", date="2026-06-01", description="Rent payment",
                    amount=-15000, category="Rent", source_id="b"),
    ])
    cache: dict = {}
    result_a = rag.retrieve("user_a", query="rent", db_path=db_path, enable_query_rewrite=False, cache=cache)
    result_b = rag.retrieve("user_b", query="rent", db_path=db_path, enable_query_rewrite=False, cache=cache)

    assert [e.source_id for e in result_a.evidence] == ["a"]
    assert [e.source_id for e in result_b.evidence] == ["b"]
