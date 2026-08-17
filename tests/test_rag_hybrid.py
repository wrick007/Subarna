"""
Integration tests for the rewritten finmate.rag.retrieve() hybrid
pipeline: stage toggles, the fallback-ladder `stage`/`note` reporting,
query-rewrite -> keyword-search integration (mocked LLM), rerank
integration (mocked cross-encoder), and the Qdrant candidate-set-scoping
fix (skipped automatically where qdrant-client isn't installed).

tests/test_rag.py already covers the pre-existing metadata-filter/
recency-floor behavior and is left as-is (still passing unchanged, see
README) -- this file focuses on what's new: the keyword/fusion/rerank/
query-rewrite stages and how they compose.

Every test here runs with zero network access. Tests that build a real
(embedded, on-disk) Qdrant index use a fake, dependency-free embedder
(see `_ConstantFakeEmbedder` below) rather than downloading a real model
-- this sandbox's build environment could not reach Hugging Face Hub to
download `all-MiniLM-L6-v2` (verified; see README "Deviations"), and
these tests must be able to run anywhere `qdrant-client` is installed,
not only where a model download would additionally succeed.
"""

from __future__ import annotations

import pytest

from finmate import db, query_rewrite, rag
from finmate.schemas import Transaction


@pytest.fixture(autouse=True)
def _clear_module_caches():
    # query_rewrite.rewrite_query's cache is keyed by (user_id, normalized
    # query) and lives for the process lifetime (see
    # finmate/query_rewrite.py's own module docstring) -- several tests
    # below deliberately populate it (e.g. user_id="u1", query="food
    # expenses"/"rent") to exercise stage 2 -> stage 3 integration.
    # Without clearing it, those entries would otherwise leak into ANY
    # later test, in this file or another, that happens to reuse the
    # same (user_id, query) pair against a *different* fake LLM client --
    # it would silently get this file's cached phrasings instead of ever
    # calling its own client. rag.clear_cache() is cheap insurance for
    # the same reason, even though this file's own tests inject a fake
    # embedder rather than relying on that cache.
    query_rewrite.clear_cache()
    rag.clear_cache()
    yield
    query_rewrite.clear_cache()
    rag.clear_cache()


def _seed(db_path: str, txs: list[Transaction]) -> None:
    db.init_db(db_path)
    db.insert_transactions(txs, db_path=db_path)


# ---------------------------------------------------------------------------
# Stage toggles / fallback floor
# ---------------------------------------------------------------------------


def test_both_keyword_and_vector_disabled_falls_back_to_metadata_only(tmp_path):
    db_path = str(tmp_path / "test.db")
    _seed(db_path, [
        Transaction(user_id="u1", date="2026-06-01", description="a", amount=-1, category="c", source_id="t1"),
        Transaction(user_id="u1", date="2026-06-02", description="b", amount=-1, category="c", source_id="t2"),
    ])
    result = rag.retrieve(
        "u1", query="anything", db_path=db_path,
        enable_keyword_search=False, enable_vector_search=False, enable_query_rewrite=False,
    )
    assert result.stage == "metadata-only, recency-ordered"
    assert result.keyword_search_used is False
    assert result.vector_search_used is False
    assert [e.source_id for e in result.evidence] == ["t2", "t1"]  # recency order


def test_keyword_only_stage_label_in_offline_environment(tmp_path):
    # In this sandbox, vector search is genuinely unavailable (no
    # downloadable embedding model -- see README), so a plain keyword
    # match should land at exactly the "keyword only" tier without any
    # mocking required.
    db_path = str(tmp_path / "test.db")
    _seed(db_path, [
        Transaction(user_id="u1", date="2026-06-01", description="Rent payment - landlord NEFT",
                    amount=-32000, category="Rent", source_id="t1"),
        Transaction(user_id="u1", date="2026-06-04", description="BigBasket grocery order",
                    amount=-3400, category="Groceries", source_id="t2"),
    ])
    result = rag.retrieve("u1", query="rent payment", db_path=db_path, enable_query_rewrite=False)
    assert result.keyword_search_used is True
    assert result.vector_search_used is False
    assert result.stage == "keyword only"
    assert result.evidence[0].source_id == "t1"
    assert result.evidence[0].retrieval_stage in ("fusion", "rerank")


def test_retrieve_respects_top_k(tmp_path):
    db_path = str(tmp_path / "test.db")
    _seed(db_path, [
        Transaction(user_id="u1", date=f"2026-06-{i:02d}", description="Zomato dinner order",
                    amount=-1, category="Dining out", source_id=f"t{i}")
        for i in range(1, 8)
    ])
    result = rag.retrieve("u1", query="zomato", db_path=db_path, top_k=3, enable_query_rewrite=False)
    assert len(result.evidence) == 3


def test_every_evidence_source_id_traces_back_to_a_real_seeded_transaction(tmp_path):
    db_path = str(tmp_path / "test.db")
    txs = [
        Transaction(user_id="u1", date="2026-06-01", description="Rent payment - landlord NEFT",
                    amount=-32000, category="Rent", source_id="t1"),
        Transaction(user_id="u1", date="2026-06-04", description="BigBasket grocery order",
                    amount=-3400, category="Groceries", source_id="t2"),
        Transaction(user_id="u1", date="2026-06-06", description="Zomato - dinner order",
                    amount=-650, category="Dining out", source_id="t3"),
    ]
    _seed(db_path, txs)
    seeded_ids = {t.source_id for t in txs}
    result = rag.retrieve("u1", query="order", db_path=db_path, enable_query_rewrite=False)
    assert result.evidence  # sanity: this query should actually match something
    assert {e.source_id for e in result.evidence} <= seeded_ids


# ---------------------------------------------------------------------------
# Query rewrite (stage 2) -> keyword search (stage 3) integration
# ---------------------------------------------------------------------------


class _FakeRewriteClient:
    def __init__(self, phrasings):
        self._phrasings = phrasings

    def call(
        self, agent_system_prompt, user_message, response_model=None, max_tokens=200,
        temperature=0.0, include_constitution=True,
    ):
        return response_model(phrasings=self._phrasings)


def test_query_rewrite_phrasing_enables_keyword_match_with_no_literal_overlap(tmp_path):
    db_path = str(tmp_path / "test.db")
    _seed(db_path, [
        Transaction(user_id="u1", date="2026-06-06", description="Zomato - dinner order",
                    amount=-650, category="Dining out", source_id="a"),
    ])

    # "food expenses" shares zero tokens with "Zomato - dinner order" /
    # "Dining out" -- without rewrite, this must fall all the way back.
    baseline = rag.retrieve("u1", query="food expenses", db_path=db_path, enable_query_rewrite=False)
    assert baseline.keyword_search_used is False
    assert baseline.stage == "metadata-only, recency-ordered"

    # With rewrite (mocked -- no real LLM call), the expanded phrasings
    # give keyword search something to match against.
    rewritten = rag.retrieve(
        "u1", query="food expenses", db_path=db_path,
        enable_query_rewrite=True, llm_client=_FakeRewriteClient(["dining", "zomato"]),
    )
    assert rewritten.query_rewrite_used is True
    assert rewritten.keyword_search_used is True
    assert [e.source_id for e in rewritten.evidence] == ["a"]


def test_finmate_rag_mode_no_llm_skips_rewrite_even_with_a_client_available(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    _seed(db_path, [
        Transaction(user_id="u1", date="2026-06-01", description="Rent payment",
                    amount=-1, category="Rent", source_id="a"),
    ])
    monkeypatch.setenv("FINMATE_RAG_MODE", "no_llm")

    class _FailIfCalled:
        def call(self, *a, **kw):
            raise AssertionError("query rewrite must not be attempted when FINMATE_RAG_MODE=no_llm")

    result = rag.retrieve("u1", query="rent", db_path=db_path, llm_client=_FailIfCalled())
    assert result.query_rewrite_used is False


def test_query_rewrite_failure_does_not_block_retrieval(tmp_path):
    db_path = str(tmp_path / "test.db")
    _seed(db_path, [
        Transaction(user_id="u1", date="2026-06-01", description="Rent payment",
                    amount=-1, category="Rent", source_id="a"),
    ])

    class _BrokenClient:
        def call(self, *a, **kw):
            raise RuntimeError("simulated 429")

    result = rag.retrieve("u1", query="rent", db_path=db_path, llm_client=_BrokenClient())
    assert result.query_rewrite_used is False
    assert result.keyword_search_used is True  # original query still searched
    assert [e.source_id for e in result.evidence] == ["a"]


# ---------------------------------------------------------------------------
# Cross-encoder rerank (stage 6) integration -- mocked, no model download
# ---------------------------------------------------------------------------


def test_rerank_stage_reorders_final_evidence_and_sets_flags(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    _seed(db_path, [
        Transaction(user_id="u1", date="2026-06-01", description="BigBasket grocery order",
                    amount=-1, category="Groceries", source_id="a"),
        Transaction(user_id="u1", date="2026-06-02", description="Swiggy lunch order",
                    amount=-1, category="Dining out", source_id="b"),
    ])
    # Both match "order" via keyword search; force the cross-encoder to
    # prefer "b" over "a" -- the opposite of insertion/keyword order --
    # so a passing test proves rerank actually took effect.
    monkeypatch.setattr(
        rag.reranker, "cross_encoder_rerank",
        lambda query, documents, model_name=rag.DEFAULT_CROSS_ENCODER_MODEL: [("b", 9.0), ("a", 1.0)],
    )
    result = rag.retrieve("u1", query="order", db_path=db_path, enable_query_rewrite=False)
    assert result.rerank_used is True
    assert [e.source_id for e in result.evidence] == ["b", "a"]
    assert result.stage == "keyword+rerank (no vector)"
    assert result.evidence[0].rerank_score == 9.0
    assert result.evidence[0].retrieval_stage == "rerank"


def test_rerank_unavailable_falls_back_to_fused_order(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    _seed(db_path, [
        Transaction(user_id="u1", date="2026-06-01", description="Zomato dinner order",
                    amount=-1, category="Dining out", source_id="a"),
    ])
    monkeypatch.setattr(
        rag.reranker, "cross_encoder_rerank",
        lambda query, documents, model_name=rag.DEFAULT_CROSS_ENCODER_MODEL: None,
    )
    result = rag.retrieve("u1", query="zomato", db_path=db_path, enable_query_rewrite=False)
    assert result.rerank_used is False
    assert result.stage == "keyword only"
    assert [e.source_id for e in result.evidence] == ["a"]


def test_enable_rerank_false_skips_rerank_even_if_available(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    _seed(db_path, [
        Transaction(user_id="u1", date="2026-06-01", description="Zomato dinner order",
                    amount=-1, category="Dining out", source_id="a"),
    ])
    called = []
    monkeypatch.setattr(
        rag.reranker, "cross_encoder_rerank",
        lambda *a, **kw: called.append(1) or [("a", 1.0)],
    )
    result = rag.retrieve("u1", query="zomato", db_path=db_path, enable_query_rewrite=False, enable_rerank=False)
    assert result.rerank_used is False
    assert called == []


# ---------------------------------------------------------------------------
# Dense vector search (stage 4) -- real embedded Qdrant, fake embeddings
# ---------------------------------------------------------------------------


class _FakeArray(list):
    """Minimal stand-in for a numpy array: only implements what
    finmate/rag.py actually calls (`.tolist()`), so these tests exercise
    real Qdrant wiring without needing numpy or a real model."""

    def tolist(self):
        return [x.tolist() if isinstance(x, _FakeArray) else x for x in self]


class _ConstantFakeEmbedder:
    """Every text embeds to the same fixed vector. These tests are about
    finmate.rag's Qdrant *wiring* (the query_points()/payload-filter fix
    -- see README "Deviations"), not about semantic ranking quality,
    which needs a real downloaded model this build's sandbox couldn't
    reach (see README)."""

    _DIM = 3

    def get_sentence_embedding_dimension(self) -> int:
        return self._DIM

    def encode(self, texts):
        return _FakeArray([_FakeArray([1.0, 0.0, 0.0]) for _ in texts])


def test_vector_search_never_returns_a_transaction_outside_the_candidate_set(tmp_path, monkeypatch):
    pytest.importorskip("qdrant_client")
    db_path = str(tmp_path / "test.db")
    qdrant_path = str(tmp_path / "qdrant")
    txs = [
        Transaction(user_id="u1", date="2026-06-01", description="Rent payment",
                    amount=-32000, category="Rent", source_id="in_filter"),
        Transaction(user_id="u1", date="2026-07-01", description="Rent payment",
                    amount=-32000, category="Rent", source_id="outside_filter"),
    ]
    _seed(db_path, txs)
    monkeypatch.setattr(rag, "_get_embedder", lambda model_name=rag.DEFAULT_EMBEDDING_MODEL: _ConstantFakeEmbedder())
    indexed = rag.index_transactions_for_user("u1", txs, db_path=db_path, qdrant_path=qdrant_path)
    assert indexed == 2

    result = rag.retrieve(
        "u1", query="rent", start_date="2026-06-01", end_date="2026-06-30",
        db_path=db_path, qdrant_path=qdrant_path,
        enable_keyword_search=False, enable_rerank=False, enable_query_rewrite=False,
    )
    ids = {e.source_id for e in result.evidence}
    assert ids == {"in_filter"}
    assert result.vector_search_used is True
    assert result.stage == "vector only"


def test_vector_search_finds_target_even_crowded_by_a_tiny_global_limit(tmp_path, monkeypatch):
    """Regression test for the pre-upgrade bug this build fixed: fetching
    only `limit` globally-ranked hits and filtering to the candidate set
    *after* the fact could miss a relevant item whenever enough
    out-of-candidate-set transactions score at least as high globally.
    Scoping the filter into the Qdrant query itself doesn't have that
    blind spot -- proven here with `limit=1`, deliberately smaller than
    the 5 identically-scored, out-of-candidate-set "noise" points."""
    pytest.importorskip("qdrant_client")
    db_path = str(tmp_path / "test.db")
    qdrant_path = str(tmp_path / "qdrant")
    noise = [
        Transaction(user_id="u1", date="2026-06-01", description="Rent noise", amount=-1,
                    category="Other", source_id=f"noise_{i}")
        for i in range(5)
    ]
    target = Transaction(user_id="u1", date="2026-06-01", description="Rent payment",
                          amount=-32000, category="Rent", source_id="target")
    all_txs = [*noise, target]
    _seed(db_path, all_txs)
    monkeypatch.setattr(rag, "_get_embedder", lambda model_name=rag.DEFAULT_EMBEDDING_MODEL: _ConstantFakeEmbedder())
    rag.index_transactions_for_user("u1", all_txs, db_path=db_path, qdrant_path=qdrant_path)

    ranked_ids, _scores = rag._vector_search(
        user_id="u1", query="rent", candidate_ids={"target"},
        qdrant_path=qdrant_path, embedding_model=rag.DEFAULT_EMBEDDING_MODEL, limit=1,
    )
    assert ranked_ids == ["target"]


def test_vector_search_returns_empty_when_embedder_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "_get_embedder", lambda model_name=rag.DEFAULT_EMBEDDING_MODEL: None)
    ranked_ids, scores = rag._vector_search(
        user_id="u1", query="rent", candidate_ids={"a"},
        qdrant_path=str(tmp_path / "unused"), embedding_model=rag.DEFAULT_EMBEDDING_MODEL, limit=10,
    )
    assert ranked_ids == []
    assert scores == {}


def test_vector_search_returns_empty_for_empty_candidate_set(tmp_path):
    ranked_ids, scores = rag._vector_search(
        user_id="u1", query="rent", candidate_ids=set(),
        qdrant_path=str(tmp_path / "unused"), embedding_model=rag.DEFAULT_EMBEDDING_MODEL, limit=10,
    )
    assert ranked_ids == []
    assert scores == {}


# ---------------------------------------------------------------------------
# delete_user_vector_index -- "forget my data" needs to actually forget the
# vector collection too, not just the SQLite rows (see
# backend/app/routers/users.py:delete_user and app.py's sidebar button).
# ---------------------------------------------------------------------------


def test_delete_user_vector_index_removes_a_real_collection(tmp_path, monkeypatch):
    pytest.importorskip("qdrant_client")
    db_path = str(tmp_path / "test.db")
    qdrant_path = str(tmp_path / "qdrant")
    txs = [
        Transaction(user_id="u1", date="2026-06-01", description="Rent payment",
                    amount=-32000, category="Rent", source_id="tx_1"),
    ]
    _seed(db_path, txs)
    monkeypatch.setattr(rag, "_get_embedder", lambda model_name=rag.DEFAULT_EMBEDDING_MODEL: _ConstantFakeEmbedder())
    rag.index_transactions_for_user("u1", txs, db_path=db_path, qdrant_path=qdrant_path)

    client = rag._get_qdrant_client(qdrant_path)
    assert client.collection_exists(f"{rag.COLLECTION_PREFIX}u1") is True

    deleted = rag.delete_user_vector_index("u1", qdrant_path=qdrant_path)
    assert deleted is True
    assert client.collection_exists(f"{rag.COLLECTION_PREFIX}u1") is False

    # A user who never had a collection (or calling it twice) is a no-op,
    # not an error -- "already forgotten" is a success state, not a failure.
    assert rag.delete_user_vector_index("u1", qdrant_path=qdrant_path) is False
    assert rag.delete_user_vector_index("someone_else_entirely", qdrant_path=qdrant_path) is False


def test_delete_user_vector_index_returns_false_when_client_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "_get_qdrant_client", lambda path=rag.DEFAULT_QDRANT_PATH: None)
    assert rag.delete_user_vector_index("u1", qdrant_path=str(tmp_path / "unused")) is False
