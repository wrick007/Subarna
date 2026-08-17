"""
Tests for finmate.reranker: Reciprocal Rank Fusion (pure math, zero
dependencies, fully offline) and cross-encoder reranking's graceful
fallback behavior. No real cross-encoder model is ever downloaded here --
`_get_cross_encoder`/`cross_encoder_rerank` are tested with a fake
encoder and with simulated import/construction/scoring failures, per the
spec's "independently testable with fake score inputs (no model download
needed)" requirement. Real-model integration is exercised only via
finmate/rag.py's pipeline when sentence-transformers happens to be
installed and the model is actually reachable.
"""

from __future__ import annotations

import builtins

import pytest

from finmate import reranker


@pytest.fixture(autouse=True)
def _clear_cross_encoder_cache():
    # _get_cross_encoder now caches its result per model_name (see
    # finmate/reranker.py "Performance"). Without clearing it between
    # tests, whichever of test_get_cross_encoder_returns_none_when_*
    # happens to run first would cache model_name -> None, and the other
    # would then get a cache hit instead of genuinely exercising its own
    # (different) failure path -- same assertion, but not actually
    # testing what it claims to.
    reranker.clear_cache()
    yield
    reranker.clear_cache()


# ---------------------------------------------------------------------------
# reciprocal_rank_fusion
# ---------------------------------------------------------------------------


def test_rrf_item_in_both_lists_outranks_item_in_only_one():
    fused = reranker.reciprocal_rank_fusion(keyword_ranked_ids=["a", "b"], vector_ranked_ids=["a", "c"])
    ids = [h.source_id for h in fused]
    assert ids[0] == "a"
    assert set(ids[1:]) == {"b", "c"}


def test_rrf_degrades_to_single_list_order_when_other_is_empty():
    fused = reranker.reciprocal_rank_fusion(keyword_ranked_ids=["x", "y", "z"], vector_ranked_ids=[])
    assert [h.source_id for h in fused] == ["x", "y", "z"]
    assert all(h.vector_rank is None for h in fused)
    assert fused[0].keyword_rank == 1


def test_rrf_both_lists_empty_returns_empty():
    assert reranker.reciprocal_rank_fusion([], []) == []


def test_rrf_raw_scores_are_carried_for_audit_but_dont_affect_ranking():
    fused_low = reranker.reciprocal_rank_fusion(
        keyword_ranked_ids=["a"], vector_ranked_ids=["a"],
        keyword_scores={"a": 999.0}, vector_scores={"a": 0.01},
    )
    fused_high = reranker.reciprocal_rank_fusion(
        keyword_ranked_ids=["a"], vector_ranked_ids=["a"],
        keyword_scores={"a": 3.5}, vector_scores={"a": 0.87},
    )
    assert fused_high[0].keyword_score == 3.5
    assert fused_high[0].vector_score == 0.87
    # same rank positions in both calls -> same rrf_score, regardless of
    # how different the raw scores are. Proves fusion is rank-based only.
    assert fused_low[0].rrf_score == fused_high[0].rrf_score


def test_rrf_ties_break_deterministically():
    fused_1 = reranker.reciprocal_rank_fusion(keyword_ranked_ids=["b", "a"], vector_ranked_ids=["a", "b"])
    fused_2 = reranker.reciprocal_rank_fusion(keyword_ranked_ids=["b", "a"], vector_ranked_ids=["a", "b"])
    assert [h.source_id for h in fused_1] == [h.source_id for h in fused_2]


def test_rrf_result_sorted_best_first():
    fused = reranker.reciprocal_rank_fusion(keyword_ranked_ids=["a", "b", "c"], vector_ranked_ids=[])
    scores = [h.rrf_score for h in fused]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# cross_encoder_rerank / _get_cross_encoder -- graceful fallback only
# ---------------------------------------------------------------------------


def test_cross_encoder_rerank_empty_documents_returns_empty_list_not_none():
    assert reranker.cross_encoder_rerank("query", []) == []


def test_cross_encoder_rerank_returns_none_when_model_unavailable(monkeypatch):
    monkeypatch.setattr(reranker, "_get_cross_encoder", lambda model_name=reranker.DEFAULT_CROSS_ENCODER_MODEL: None)
    assert reranker.cross_encoder_rerank("query", [("t1", "some text")]) is None


def test_cross_encoder_rerank_orders_by_fake_model_score_descending(monkeypatch):
    class _FakeEncoder:
        def predict(self, pairs):
            # deterministic, checkable "score" with no real model needed
            return [len(text) for _, text in pairs]

    monkeypatch.setattr(reranker, "_get_cross_encoder", lambda model_name=reranker.DEFAULT_CROSS_ENCODER_MODEL: _FakeEncoder())
    docs = [("short", "hi"), ("long", "a much longer piece of text here")]
    result = reranker.cross_encoder_rerank("query", docs)
    assert [source_id for source_id, _ in result] == ["long", "short"]


def test_cross_encoder_rerank_returns_none_if_scoring_raises(monkeypatch):
    class _BrokenEncoder:
        def predict(self, pairs):
            raise RuntimeError("simulated model failure")

    monkeypatch.setattr(reranker, "_get_cross_encoder", lambda model_name=reranker.DEFAULT_CROSS_ENCODER_MODEL: _BrokenEncoder())
    assert reranker.cross_encoder_rerank("query", [("t1", "text")]) is None


def test_get_cross_encoder_returns_none_when_package_not_installed(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "sentence_transformers":
            raise ImportError("simulated: not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert reranker._get_cross_encoder() is None


def test_get_cross_encoder_returns_none_when_construction_fails(monkeypatch):
    sentence_transformers = pytest.importorskip("sentence_transformers")

    def raise_on_init(*args, **kwargs):
        raise OSError("simulated: couldn't connect to huggingface.co")

    monkeypatch.setattr(sentence_transformers, "CrossEncoder", raise_on_init)
    assert reranker._get_cross_encoder() is None
