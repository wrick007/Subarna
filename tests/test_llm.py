"""
Tests for finmate.llm.resolve_provider_config -- pure string/dict logic
with no SDK import and no network access, so these run with zero extra
dependencies beyond pydantic (same guarantee as tools.py/db.py).
"""

from __future__ import annotations

import pytest

from finmate.llm import LLMConfigError, resolve_provider_config


def test_defaults_to_groq_with_no_provider_set():
    provider, base_url, api_key, model = resolve_provider_config(
        api_key="gsk_test", env={},
    )
    assert provider == "groq"
    assert base_url == "https://api.groq.com/openai/v1"
    assert model == "llama-3.3-70b-versatile"
    assert api_key == "gsk_test"


def test_reads_groq_key_from_env():
    provider, base_url, api_key, model = resolve_provider_config(
        env={"GROQ_API_KEY": "gsk_from_env"},
    )
    assert api_key == "gsk_from_env"
    assert provider == "groq"


def test_gemini_provider_resolves_correct_defaults():
    provider, base_url, api_key, model = resolve_provider_config(
        provider="gemini", api_key="AIza_test", env={},
    )
    assert provider == "gemini"
    assert base_url == "https://generativelanguage.googleapis.com/v1beta/openai/"
    assert model == "gemini-2.5-flash"


def test_gemini_reads_key_from_env():
    provider, base_url, api_key, model = resolve_provider_config(
        provider="gemini", env={"GEMINI_API_KEY": "AIza_from_env"},
    )
    assert api_key == "AIza_from_env"


def test_env_provider_selection():
    provider, base_url, api_key, model = resolve_provider_config(
        env={"FINMATE_LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "AIza_x"},
    )
    assert provider == "gemini"
    assert base_url == "https://generativelanguage.googleapis.com/v1beta/openai/"


def test_explicit_model_override_wins_over_provider_default():
    _, _, _, model = resolve_provider_config(
        provider="groq", api_key="gsk_test", model="llama-3.1-8b-instant", env={},
    )
    assert model == "llama-3.1-8b-instant"


def test_env_model_override_wins_over_provider_default():
    _, _, _, model = resolve_provider_config(
        provider="groq", api_key="gsk_test", env={"FINMATE_MODEL": "openai/gpt-oss-120b"},
    )
    assert model == "openai/gpt-oss-120b"


def test_missing_api_key_raises_actionable_error():
    with pytest.raises(LLMConfigError, match="GROQ_API_KEY"):
        resolve_provider_config(provider="groq", env={})


def test_missing_gemini_api_key_raises_actionable_error():
    with pytest.raises(LLMConfigError, match="GEMINI_API_KEY"):
        resolve_provider_config(provider="gemini", env={})


def test_unknown_provider_raises():
    with pytest.raises(LLMConfigError, match="Unknown FINMATE_LLM_PROVIDER"):
        resolve_provider_config(provider="not_a_real_provider", api_key="x", env={})


def test_custom_provider_requires_base_url():
    with pytest.raises(LLMConfigError, match="FINMATE_BASE_URL"):
        resolve_provider_config(provider="custom", api_key="x", env={})


def test_custom_provider_requires_model():
    with pytest.raises(LLMConfigError, match="FINMATE_MODEL"):
        resolve_provider_config(
            provider="custom", api_key="x", base_url="https://example.com/v1", env={},
        )


def test_custom_provider_fully_specified():
    provider, base_url, api_key, model = resolve_provider_config(
        provider="custom", api_key="x", base_url="https://example.com/v1", model="some-model", env={},
    )
    assert provider == "custom"
    assert base_url == "https://example.com/v1"
    assert model == "some-model"


def test_explicit_args_win_over_env():
    provider, base_url, api_key, model = resolve_provider_config(
        provider="groq", api_key="explicit_key",
        env={"GROQ_API_KEY": "env_key", "FINMATE_LLM_PROVIDER": "gemini"},
    )
    # explicit provider="groq" wins over env FINMATE_LLM_PROVIDER=gemini
    assert provider == "groq"
    # explicit api_key wins over env GROQ_API_KEY
    assert api_key == "explicit_key"


def test_generic_finmate_api_key_works_for_any_provider():
    provider, base_url, api_key, model = resolve_provider_config(
        provider="groq", env={"FINMATE_API_KEY": "generic_key"},
    )
    assert api_key == "generic_key"
