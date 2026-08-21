"""
Shared LLM wrapper for FinMate AI -- built on the OpenAI Python SDK pointed
at an OpenAI-*compatible* endpoint, so it works with free-tier providers
that don't require a credit card: Groq or Google Gemini, out of the box.

Every agent call in `finmate/agents/*.py` goes through `LLMClient.call`
(or, for the one call site that streams its output to the user in
real time, `LLMClient.call_stream` -- see that method's docstring for
how it differs), so:
  - the CONSTITUTION is prepended to every system prompt exactly once, here,
    never copy-pasted into individual agent modules (opt out per-call with
    `include_constitution=False` for the rare non-user-facing utility call
    that has no use for it -- see `call`'s docstring);
  - JSON-mode responses are validated against a pydantic model and retried
    once (with the validation error fed back to the model) before raising;
  - the provider and model are read from config/env so either can be
    swapped without a code change (spec section 1).

Supported providers out of the box (set FINMATE_LLM_PROVIDER):
  - "groq"   (default): https://console.groq.com -- free, no credit card,
             generous rate limits, very fast. Get a key, set GROQ_API_KEY.
  - "gemini": https://aistudio.google.com/apikey -- free, no credit card.
             Get a key, set GEMINI_API_KEY.
  - "custom": any other OpenAI-compatible endpoint -- set FINMATE_BASE_URL,
             FINMATE_API_KEY, and FINMATE_MODEL yourself.

This module imports the `openai` SDK lazily inside `LLMClient.__init__` so
that importing `finmate.llm` (e.g. transitively, from the orchestrator)
never requires an API key or network access -- only *instantiating* and
*calling* the client does. Provider/model *config resolution* itself
(`resolve_provider_config`) is plain string logic with no import at all, so
it's unit-tested with zero dependencies in tests/test_llm.py, matching the
"testable without an API key" requirement in spec section 5/7.

Free-tier rate limits are real (e.g. Groq: ~30 requests/minute; Gemini
2.5 Flash: ~15 requests/minute) and a single FinMate user turn can make
several agent calls in a row, so `_raw_call` retries on HTTP 429 with
exponential backoff before giving up.
"""

from __future__ import annotations

import json
import os
import time
from typing import Iterator, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

from .prompts import CONSTITUTION

T = TypeVar("T", bound=BaseModel)

DEFAULT_MAX_TOKENS = 2000
MAX_RATE_LIMIT_RETRIES = 4
RATE_LIMIT_BACKOFF_BASE_SECONDS = 2.0

# base_url / default model / expected env var per built-in provider. Model
# lineups on free tiers shift over time -- override with FINMATE_MODEL if
# a default here has been retired; check the provider's current docs.
PROVIDER_DEFAULTS: dict[str, dict[str, str]] = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": "llama-3.3-70b-versatile",
        "api_key_env": "GROQ_API_KEY",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "default_model": "gemini-2.5-flash",
        "api_key_env": "GEMINI_API_KEY",
    },
}


class LLMCallError(RuntimeError):
    """Raised when the model's JSON output fails schema validation twice."""


class LLMConfigError(ValueError):
    """Raised when provider/model/API-key configuration can't be resolved."""


def resolve_provider_config(
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    env: Optional[dict[str, str]] = None,
) -> tuple[str, str, str, str]:
    """Pure config resolution, no network / no SDK import -- unit-testable
    on its own. Returns (provider, base_url, api_key, model) or raises
    LLMConfigError with an actionable message.

    Resolution order for each field: explicit argument > environment > a
    built-in provider default (base_url/model only -- there is no sane
    default for a secret).
    """
    env = os.environ if env is None else env
    provider = provider or env.get("FINMATE_LLM_PROVIDER", "groq")
    defaults = PROVIDER_DEFAULTS.get(provider)

    if defaults is None and provider != "custom":
        known = ", ".join(sorted(list(PROVIDER_DEFAULTS) + ["custom"]))
        raise LLMConfigError(f"Unknown FINMATE_LLM_PROVIDER {provider!r}. Known providers: {known}.")

    resolved_base_url = base_url or env.get("FINMATE_BASE_URL") or (defaults or {}).get("base_url")
    if not resolved_base_url:
        raise LLMConfigError(
            f"No base_url resolved for provider {provider!r}. For provider='custom', "
            "set FINMATE_BASE_URL to an OpenAI-compatible endpoint."
        )

    api_key_env_name = (defaults or {}).get("api_key_env", "FINMATE_API_KEY")
    resolved_api_key = api_key or env.get("FINMATE_API_KEY") or env.get(api_key_env_name)
    if not resolved_api_key:
        raise LLMConfigError(
            f"No API key found for provider {provider!r}. Set {api_key_env_name} "
            "in your environment or .env file (see .env.example). Both Groq "
            "(console.groq.com) and Gemini (aistudio.google.com/apikey) issue "
            "free API keys with no credit card required."
        )

    resolved_model = model or env.get("FINMATE_MODEL") or (defaults or {}).get("default_model")
    if not resolved_model:
        raise LLMConfigError(
            f"No model resolved for provider {provider!r}. For provider='custom', set FINMATE_MODEL."
        )

    return provider, resolved_base_url, resolved_api_key, resolved_model


class LLMClient:
    """Thin wrapper around an OpenAI-compatible chat-completions endpoint
    (Groq, Gemini, or any other OpenAI-compatible provider) for all
    FinMate agents."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.provider, self.base_url, self._api_key, self.model = resolve_provider_config(
            provider=provider, api_key=api_key, model=model, base_url=base_url,
        )
        try:
            from openai import OpenAI  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - exercised only without the dep installed
            raise ImportError(
                "The 'openai' package is required to make live LLM calls (it's used as an "
                "OpenAI-*compatible* client for Groq/Gemini, not to call OpenAI itself). "
                "Install it with `pip install openai`."
            ) from exc
        self._client = OpenAI(api_key=self._api_key, base_url=self.base_url)

    def call(
        self,
        agent_system_prompt: str,
        user_message: str,
        response_model: Optional[Type[T]] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.0,
        include_constitution: bool = True,
    ) -> T | str:
        """Make one agent call.

        If `response_model` is given, the raw response text is parsed as
        JSON and validated against it. On a parse/validation failure, the
        error is fed back to the model in a single retry; a second failure
        raises LLMCallError rather than returning an unvalidated guess.
        If `response_model` is None, the raw response text is returned.

        `include_constitution`: prepend the shared CONSTITUTION (see
        finmate/prompts.py) to `agent_system_prompt`. Defaults to True,
        matching every existing call site -- the CONSTITUTION carries
        FinMate's behavioral rules for talking to *the user* and handling
        *their* financial data, so every user-facing or profile-mutating
        agent needs it. Pass False only for a narrow, non-user-facing
        utility call that does neither: today, only
        finmate/query_rewrite.py's query-expansion call, which just turns
        the user's own question into a few search phrasings and never
        produces anything the user sees or anything that touches their
        stored data. That call happens on most RAG-needing turns, so
        skipping a ~400-token constitution it has no use for is a real,
        repeated token saving with no behavior change.
        """
        system_prompt = f"{CONSTITUTION}\n\n---\n\n{agent_system_prompt}" if include_constitution else agent_system_prompt
        json_mode = response_model is not None

        raw_text = self._raw_call(system_prompt, user_message, max_tokens, temperature, json_mode)
        if response_model is None:
            return raw_text

        try:
            return self._parse_and_validate(raw_text, response_model)
        except (json.JSONDecodeError, ValidationError) as first_error:
            retry_message = (
                f"{user_message}\n\n---\n"
                f"Your previous response failed validation with this error:\n{first_error}\n"
                f"Your previous response was:\n{raw_text}\n\n"
                "Return ONLY valid JSON matching the required schema, with no other text."
            )
            raw_text_2 = self._raw_call(system_prompt, retry_message, max_tokens, temperature, json_mode)
            try:
                return self._parse_and_validate(raw_text_2, response_model)
            except (json.JSONDecodeError, ValidationError) as second_error:
                raise LLMCallError(
                    f"LLM response failed schema validation twice for {response_model.__name__}. "
                    f"Last error: {second_error}. Last raw response: {raw_text_2!r}"
                ) from second_error

    def call_stream(
        self,
        agent_system_prompt: str,
        user_message: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.0,
        include_constitution: bool = True,
    ) -> Iterator[str]:
        """Stream a raw-text response token-by-token, for Priority 2's
        SSE streaming path (`orchestrator.run_finmate_stream`). Yields
        each text delta as it arrives from the provider; join them for
        the complete text, same as what `.call()` with
        `response_model=None` would have returned.

        Deliberately narrower than `.call()`:
          - raw text only -- no `response_model`/JSON mode. A partial
            JSON object isn't meaningfully renderable token-by-token the
            way prose is, and every call site that wants streaming
            (today: only agents/synthesis.py's `stream_synthesis_agent`,
            the one call whose output the user actually reads token by
            token) already wants raw text.
          - no rate-limit retry (contrast `_create_with_retry`, which
            `.call()` uses via `_raw_call`): a retry after some tokens
            have already been yielded to a caller has no clean semantics
            here -- the caller has already forwarded those tokens
            downstream (e.g. as SSE events to a browser), so silently
            restarting the underlying provider call would either
            duplicate or discard visible output. A rate-limit error here
            surfaces to the caller as a normal exception; the SSE
            endpoint (`backend/app/routers/chat.py`) catches it and emits
            an `error` event rather than leaving the connection hanging.
          - no `response_format={"type": "json_object"}` fallback dance
            that `_raw_call` does for JSON mode, for the same reason
            (nothing here is ever JSON mode).
        """
        system_prompt = f"{CONSTITUTION}\n\n---\n\n{agent_system_prompt}" if include_constitution else agent_system_prompt
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        stream = self._client.chat.completions.create(
            model=self.model, messages=messages, max_tokens=max_tokens, temperature=temperature, stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    def _raw_call(
        self, system_prompt: str, user_message: str, max_tokens: int, temperature: float, json_mode: bool,
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        kwargs: dict[str, object] = dict(
            model=self.model, messages=messages, max_tokens=max_tokens, temperature=temperature,
        )
        if json_mode:
            # Both Groq and Gemini's OpenAI-compat layer support this for
            # most current models; every agent prompt that wants JSON also
            # already says so in plain text (required by Groq's json_object
            # mode: the word "JSON" must appear somewhere in the messages).
            kwargs["response_format"] = {"type": "json_object"}

        try:
            return self._create_with_retry(kwargs)
        except Exception:  # noqa: BLE001
            if json_mode:
                # Some models/providers reject response_format entirely --
                # fall back to a plain call rather than failing outright;
                # _parse_and_validate's retry-with-error-feedback loop in
                # `call()` still catches a malformed result from here.
                kwargs.pop("response_format", None)
                return self._create_with_retry(kwargs)
            raise

    def _create_with_retry(self, kwargs: dict[str, object]) -> str:
        from openai import RateLimitError  # noqa: PLC0415

        attempt = 0
        while True:
            try:
                response = self._client.chat.completions.create(**kwargs)
                return response.choices[0].message.content or ""
            except RateLimitError:
                attempt += 1
                if attempt > MAX_RATE_LIMIT_RETRIES:
                    raise
                time.sleep(RATE_LIMIT_BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)))

    @staticmethod
    def _parse_and_validate(raw_text: str, response_model: Type[T]) -> T:
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:]
        cleaned = cleaned.strip()
        data = json.loads(cleaned)
        return response_model.model_validate(data)
