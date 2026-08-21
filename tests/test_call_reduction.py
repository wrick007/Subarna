"""
Tests for Priority 2's "fewer sequential calls" changes -- see
finmate/orchestrator.py's module docstring sections "Fewer sequential
calls" and "Critic: conditional, not always-on" for the design each
test here is checking.

Covers, in order:
  - `_turn_needs_verification` in isolation
  - the critic is actually skipped end-to-end when nothing needs
    verifying, and actually runs when something does
  - the router+query-rewrite merge (rag.retrieve's precomputed_phrasings)
  - the synthesis+formatter merge (one raw-text call, not two, per turn)
  - the overall call count for a representative turn, both to document
    the new number and so a future regression back toward the old ~7
    gets caught
  - a critic-triggered retry actually receives the previous attempt's
    feedback, rather than repeating the identical prompt
"""

from __future__ import annotations

from unittest.mock import patch

from finmate import db
from finmate.orchestrator import _turn_needs_verification, run_finmate
from finmate.schemas import CalculationResult, CriticResult, RouterOutput

from tests._support import RecordingLLMClient


def _db(tmp_path) -> str:
    path = str(tmp_path / "test.db")
    db.init_db(path)
    return path


# ---------------------------------------------------------------------------
# _turn_needs_verification in isolation
# ---------------------------------------------------------------------------


def test_needs_verification_false_when_nothing_produced():
    assert _turn_needs_verification({}) is False
    assert _turn_needs_verification({"calc_results": [], "specialist_outputs": {}}) is False


def test_needs_verification_true_with_calc_results():
    calc = CalculationResult(metric="savings_rate", value=12.5, currency="INR", period="monthly", formula="x", inputs={}, source_ids=[])
    assert _turn_needs_verification({"calc_results": [calc]}) is True


def test_needs_verification_true_with_specialist_outputs():
    assert _turn_needs_verification({"specialist_outputs": {"budget": {"over_budget": True}}}) is True


def test_needs_verification_true_with_evidence():
    from finmate.rag import EvidenceItem, RetrievalResult
    evidence = RetrievalResult(evidence=[
        EvidenceItem(source_id="1", date="2026-01-01", description="rent", amount=1000, currency="INR", category="housing"),
    ])
    assert _turn_needs_verification({"evidence": evidence}) is True


def test_needs_verification_false_with_empty_evidence():
    from finmate.rag import RetrievalResult
    assert _turn_needs_verification({"evidence": RetrievalResult(evidence=[])}) is False


# ---------------------------------------------------------------------------
# End-to-end: critic actually skipped / actually runs
# ---------------------------------------------------------------------------


def test_general_finance_turn_never_calls_critic(tmp_path):
    """general_finance's ROUTING_TABLE stages are [] -- rag/calculation/
    specialists never run -- so this turn has nothing for a critic to
    check, and _turn_needs_verification should skip it entirely."""
    db_path = _db(tmp_path)
    client = RecordingLLMClient(router_intent="general_finance")

    result = run_finmate("u1", "what's a good rule of thumb for an emergency fund?", client, db_path=db_path)

    assert client.calls_with(response_model=CriticResult) == []
    assert result.critic_passed is True  # synthesized pass, not a lie about what ran
    assert result.retry_count == 0
    assert result.verification_ran is False  # honest: nothing was actually checked


def test_turn_needing_verification_reports_verification_ran_true(tmp_path, monkeypatch):
    import finmate.orchestrator as orch

    db_path = _db(tmp_path)
    client = RecordingLLMClient(router_intent="general_finance")

    real_pipeline = orch._node_pipeline

    def _pipeline_with_forced_calc(state, llm_client):
        new_state = real_pipeline(state, llm_client)
        calc = CalculationResult(metric="x", value=1, currency="INR", period="monthly", formula="x", inputs={}, source_ids=[])
        return {**new_state, "calc_results": [calc]}

    monkeypatch.setattr(orch, "_node_pipeline", _pipeline_with_forced_calc)

    result = run_finmate("u1", "what did I spend?", client, db_path=db_path)
    assert result.verification_ran is True


def test_profile_update_reports_verification_ran_true(tmp_path):
    """The fast path never reaches the critic at all -- distinct from a
    *skipped* critic (nothing to verify), this is a different kind of
    turn entirely, and verification_ran=True is the inert/correct value
    here (matches critic_passed=True for the same fast path)."""
    db_path = _db(tmp_path)
    client = RecordingLLMClient(router_intent="profile_update")
    result = run_finmate("u1", "my income is 50000 monthly", client, db_path=db_path)
    assert result.verification_ran is True


def test_critic_runs_when_pipeline_state_has_calc_results(tmp_path, monkeypatch):
    """Directly forces a non-empty calc_results into the pipeline node's
    output (rather than depending on RouterOutput.calculations_needed
    driving a real calculation through a mocked LLM) to isolate exactly
    the condition _turn_needs_verification checks. Uses general_finance
    (empty ROUTING_TABLE stages) so the *only* reason critic engages is
    the forced calc_results, not an incidentally-triggered specialist."""
    import finmate.orchestrator as orch

    db_path = _db(tmp_path)
    client = RecordingLLMClient(router_intent="general_finance")

    real_pipeline = orch._node_pipeline

    def _pipeline_with_forced_calc(state, llm_client):
        new_state = real_pipeline(state, llm_client)
        calc = CalculationResult(metric="total_spend", value=500, currency="INR", period="monthly", formula="sum", inputs={}, source_ids=[])
        return {**new_state, "calc_results": [calc]}

    monkeypatch.setattr(orch, "_node_pipeline", _pipeline_with_forced_calc)
    # build_graph closes over _node_pipeline by reference at call time via
    # the lambda in build_graph, so patching the module attribute before
    # run_finmate (which calls build_graph fresh every time) is sufficient.

    result = run_finmate("u1", "what did I spend this month?", client, db_path=db_path)

    assert len(client.calls_with(response_model=CriticResult)) == 1
    assert result.critic_passed is True


# ---------------------------------------------------------------------------
# Router + query-rewrite merge
# ---------------------------------------------------------------------------


def test_router_search_phrasings_skip_a_separate_rewrite_call(tmp_path):
    """When the router supplies search_phrasings, finmate.query_rewrite
    must never be invoked for this turn -- see rag.retrieve's
    `precomputed_phrasings` docstring."""
    from finmate.schemas import Transaction

    db_path = _db(tmp_path)
    db.insert_transaction(
        Transaction(user_id="u1", date="2026-06-01", description="Zomato order", amount=450, currency="INR", category="dining"),
        db_path=db_path,
    )
    client = RecordingLLMClient(
        router_intent="transaction_question",
        router_output_overrides={"search_phrasings": ["food delivery", "restaurant"]},
    )

    with patch("finmate.query_rewrite.rewrite_query") as mocked_rewrite:
        run_finmate("u1", "how much did I spend on food delivery?", client, db_path=db_path)
        mocked_rewrite.assert_not_called()


def test_no_router_phrasings_falls_back_to_direct_rag_retrieve_call(tmp_path):
    """A direct rag.retrieve() caller that never goes through the router
    (scripts/eval_rag.py, or this test) keeps making its own rewrite
    call exactly as before this feature existed -- precomputed_phrasings
    defaults to None, which is NOT the same as an empty list. Needs at
    least one real transaction for "u1" so stage 1 (metadata filter)
    actually produces candidates -- retrieve() legitimately never
    reaches stage 2 at all when there's nothing to search over, which
    would otherwise make this test pass for the wrong reason."""
    from finmate import rag
    from finmate.query_rewrite import QueryRewriteResult
    from finmate.schemas import Transaction

    db_path = _db(tmp_path)
    db.insert_transaction(
        Transaction(user_id="u1", date="2026-06-01", description="dining out", amount=450, currency="INR", category="dining"),
        db_path=db_path,
    )

    with patch("finmate.query_rewrite.rewrite_query", return_value=QueryRewriteResult()) as mocked_rewrite:
        rag.retrieve("u1", query="dining", db_path=db_path, enable_query_rewrite=True, enable_vector_search=False)
        mocked_rewrite.assert_called_once()


# ---------------------------------------------------------------------------
# Synthesis + formatter merge: exactly one raw-text call per attempt
# ---------------------------------------------------------------------------


def test_exactly_one_raw_text_call_per_synthesis_attempt(tmp_path):
    """Before Priority 2: synthesis + formatter = 2 raw-text calls per
    attempt. After: 1. This is what "merged into one call" actually
    means at the call-count level."""
    db_path = _db(tmp_path)
    client = RecordingLLMClient(router_intent="general_finance")

    run_finmate("u1", "what's a Roth-IRA-equivalent in India?", client, db_path=db_path)

    raw_text_calls = [c for c in client.calls if c["response_model"] is None]
    assert len(raw_text_calls) == 1


# ---------------------------------------------------------------------------
# Representative-turn call count (documents the new number; a future
# regression back toward the old ~7 should fail this).
# ---------------------------------------------------------------------------


def test_representative_turn_call_count(tmp_path):
    """A general_finance turn (no rag/calc/specialist stages, and now no
    critic either) should cost exactly 2 LLM calls: router, synthesis.
    Contrast the pre-redesign baseline of up to ~7 for a turn that *did*
    use rag+specialist+critic -- see orchestrator.py's module docstring.
    This is deliberately the cheapest representative turn (the most
    common real-world case: casual-adjacent/informational questions);
    see test_turn_with_evidence_and_calc_call_count below for the
    still-reduced-but-higher count on a turn that legitimately needs
    full verification.
    """
    db_path = _db(tmp_path)
    client = RecordingLLMClient(router_intent="general_finance")

    run_finmate("u1", "what's the difference between a Roth and traditional IRA?", client, db_path=db_path)

    assert len(client.calls) == 2  # router, synthesis -- was router + synthesis + critic + formatter = 4 for this same turn shape pre-redesign


def test_turn_needing_full_verification_call_count(tmp_path, monkeypatch):
    """A turn that legitimately has evidence/calc to verify still costs
    router + synthesis + critic = 3 (was router + rewrite + synthesis +
    critic + formatter = 5 pre-redesign, for the same shape turn, before
    even counting specialists or retries). general_finance keeps the
    real pipeline's own stages at [], so the forced calc_results below
    is the *only* thing driving verification -- no specialist call to
    also account for."""
    import finmate.orchestrator as orch

    db_path = _db(tmp_path)
    client = RecordingLLMClient(router_intent="general_finance")

    real_pipeline = orch._node_pipeline

    def _pipeline_with_forced_calc(state, llm_client):
        new_state = real_pipeline(state, llm_client)
        calc = CalculationResult(metric="total_spend", value=500, currency="INR", period="monthly", formula="sum", inputs={}, source_ids=[])
        return {**new_state, "calc_results": [calc]}

    monkeypatch.setattr(orch, "_node_pipeline", _pipeline_with_forced_calc)

    run_finmate("u1", "what did I spend this month?", client, db_path=db_path)

    assert len(client.calls) == 3  # router, synthesis, critic


# ---------------------------------------------------------------------------
# Productive retries: a rejected attempt's feedback reaches the retry
# ---------------------------------------------------------------------------


def test_retry_synthesis_call_includes_prior_verification_feedback(tmp_path, monkeypatch):
    """Without this, a retry re-runs synthesis with identical inputs at
    temperature 0 and is very likely to just reproduce the same rejected
    answer -- see orchestrator.py's "Fewer sequential calls: productive
    retries" docstring section. Forces calc_results (as above) so the
    critic actually engages instead of being skipped."""
    import finmate.orchestrator as orch

    db_path = _db(tmp_path)
    # critic_fail_times=1: first critic call fails, second (the retry) passes.
    client = RecordingLLMClient(router_intent="general_finance", critic_fail_times=1)

    real_pipeline = orch._node_pipeline

    def _pipeline_with_forced_calc(state, llm_client):
        new_state = real_pipeline(state, llm_client)
        calc = CalculationResult(metric="total_spend", value=500, currency="INR", period="monthly", formula="sum", inputs={}, source_ids=[])
        return {**new_state, "calc_results": [calc]}

    monkeypatch.setattr(orch, "_node_pipeline", _pipeline_with_forced_calc)

    result = run_finmate("u1", "what did I spend this month?", client, db_path=db_path)

    synthesis_calls = [c for c in client.calls if c["response_model"] is None]
    assert len(synthesis_calls) == 2, "first attempt + one productive retry"
    assert "mocked verification failure #1" in synthesis_calls[1]["user_message"]
    assert "rejected by verification" in synthesis_calls[1]["user_message"]
    # First attempt had no feedback yet (nothing to report).
    assert "rejected by verification" not in synthesis_calls[0]["user_message"]
    assert result.critic_passed is True
    assert result.retry_count == 1


def test_exhausted_retries_get_a_disclaimer_with_no_extra_llm_call(tmp_path, monkeypatch):
    """critic_fail_times is large enough that every attempt (including
    every retry) fails -- _node_finalize should append the fixed
    disclaimer with zero additional LLM calls beyond the retry ceiling
    itself (MAX_CRITIC_RETRIES=2 retries -> 3 total critic attempts)."""
    import finmate.orchestrator as orch

    db_path = _db(tmp_path)
    client = RecordingLLMClient(router_intent="general_finance", critic_fail_times=99)

    real_pipeline = orch._node_pipeline

    def _pipeline_with_forced_calc(state, llm_client):
        new_state = real_pipeline(state, llm_client)
        calc = CalculationResult(metric="total_spend", value=500, currency="INR", period="monthly", formula="sum", inputs={}, source_ids=[])
        return {**new_state, "calc_results": [calc]}

    monkeypatch.setattr(orch, "_node_pipeline", _pipeline_with_forced_calc)

    result = run_finmate("u1", "what did I spend this month?", client, db_path=db_path)

    critic_calls = client.calls_with(response_model=CriticResult)
    assert len(critic_calls) == orch.MAX_CRITIC_RETRIES + 1
    assert result.critic_passed is False
    assert "wasn't able to fully verify" in result.final_response
    # The disclaimer itself must be templated, not a 4th LLM call.
    synthesis_calls = [c for c in client.calls if c["response_model"] is None]
    assert len(synthesis_calls) == orch.MAX_CRITIC_RETRIES + 1
