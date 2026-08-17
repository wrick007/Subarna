"""
Tests for finmate.query_rewrite: caching by (user_id, normalized query),
skip-on-any-failure, and the FINMATE_RAG_MODE=no_llm switch. Every LLM
call is mocked via a duck-typed stand-in for finmate.llm.LLMClient --
this module must never make a real network call or require an API key to
test, per the RAG upgrade spec's offline-testability requirement.
"""

from __future__ import annotations

import pytest

from finmate import query_rewrite
from finmate.query_rewrite import rewrite_query


class _FakeLLMClient:
    """query_rewrite only ever calls `.call(...)`, so this is all that's
    needed to exercise it without touching real config, `openai`, or the
    network."""

    def __init__(self, phrasings=None, raise_exc=None):
        self._phrasings = phrasings if phrasings is not None else ["dining", "groceries"]
        self._raise_exc = raise_exc
        self.calls: list[str] = []
        self.include_constitution_values: list[bool] = []

    def call(
        self, agent_system_prompt, user_message, response_model=None, max_tokens=200,
        temperature=0.0, include_constitution=True,
    ):
        self.calls.append(user_message)
        self.include_constitution_values.append(include_constitution)
        if self._raise_exc is not None:
            raise self._raise_exc
        return response_model(phrasings=self._phrasings)


@pytest.fixture(autouse=True)
def _clear_cache():
    query_rewrite.clear_cache()
    yield
    query_rewrite.clear_cache()


def test_successful_rewrite_returns_phrasings_and_used_true():
    client = _FakeLLMClient(phrasings=["dining", "restaurant", "groceries"])
    result = rewrite_query("u1", "food expenses", llm_client=client)
    assert result.used is True
    assert result.phrasings == ["dining", "restaurant", "groceries"]
    assert len(client.calls) == 1


def test_rewrite_caps_at_max_phrasings():
    client = _FakeLLMClient(phrasings=["a", "b", "c", "d", "e"])
    result = rewrite_query("u1", "food expenses", llm_client=client)
    assert len(result.phrasings) == query_rewrite.MAX_PHRASINGS


def test_repeat_query_uses_cache_not_a_second_call():
    client = _FakeLLMClient(phrasings=["dining"])
    rewrite_query("u1", "food expenses", llm_client=client)
    rewrite_query("u1", "  Food Expenses  ", llm_client=client)  # normalizes the same
    assert len(client.calls) == 1


def test_cache_is_scoped_per_user():
    client = _FakeLLMClient(phrasings=["dining"])
    rewrite_query("user_a", "food expenses", llm_client=client)
    rewrite_query("user_b", "food expenses", llm_client=client)
    assert len(client.calls) == 2


def test_different_query_is_not_cached_together():
    client = _FakeLLMClient(phrasings=["dining"])
    rewrite_query("u1", "food expenses", llm_client=client)
    rewrite_query("u1", "rent payment", llm_client=client)
    assert len(client.calls) == 2


def test_call_failure_degrades_to_used_false_never_raises():
    client = _FakeLLMClient(raise_exc=RuntimeError("simulated rate limit"))
    result = rewrite_query("u1", "food expenses", llm_client=client)
    assert result.used is False
    assert result.phrasings == []
    assert "simulated rate limit" in result.note


def test_no_llm_mode_skips_without_calling_client():
    client = _FakeLLMClient()
    result = rewrite_query("u1", "food expenses", llm_client=client, mode="no_llm")
    assert result.used is False
    assert client.calls == []


def test_no_llm_mode_from_environment(monkeypatch):
    monkeypatch.setenv("FINMATE_RAG_MODE", "no_llm")
    client = _FakeLLMClient()
    result = rewrite_query("u1", "food expenses", llm_client=client)
    assert result.used is False
    assert client.calls == []


def test_empty_query_skips_without_calling_client():
    client = _FakeLLMClient()
    result = rewrite_query("u1", "   ", llm_client=client)
    assert result.used is False
    assert client.calls == []


def test_no_client_and_no_config_degrades_cleanly(monkeypatch):
    # No llm_client injected, and nothing configured -- LLMClient()
    # construction itself fails (LLMConfigError: no key found), and that
    # must be caught here, never raised.
    for var in ("GROQ_API_KEY", "GEMINI_API_KEY", "FINMATE_API_KEY", "FINMATE_LLM_PROVIDER", "FINMATE_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    result = rewrite_query("u1", "food expenses", llm_client=None)
    assert result.used is False
    assert result.phrasings == []


def test_response_with_no_phrasings_reports_used_false():
    client = _FakeLLMClient(phrasings=[])
    result = rewrite_query("u1", "food expenses", llm_client=client)
    assert result.used is False
    assert result.phrasings == []


def test_phrasings_are_stripped_and_blanks_dropped():
    client = _FakeLLMClient(phrasings=["  dining  ", "", "   ", "groceries"])
    result = rewrite_query("u1", "food expenses", llm_client=client)
    assert result.phrasings == ["dining", "groceries"]


def test_rewrite_call_opts_out_of_the_constitution():
    # This call never produces user-facing text and never touches stored
    # data, so it has no use for the CONSTITUTION -- see
    # LLMClient.call's docstring. Skipping it saves real, repeated tokens
    # on a call this pipeline can make on every RAG-needing turn.
    client = _FakeLLMClient(phrasings=["dining"])
    rewrite_query("u1", "food expenses", llm_client=client)
    assert client.include_constitution_values == [False]
