"""
Stage 2 of hybrid retrieval (optional): expand the user's natural-language
query into 2-3 short search-oriented phrasings via at most one Groq/Gemini
call through the existing `finmate.llm.LLMClient`.

This exists to help stage 3 (keyword search), which -- unlike dense vector
search -- can only match text that's actually present in a transaction's
description/category. "food expenses" has no lexical overlap with "Zomato
- dinner order", but "dining", "restaurant", "groceries" do; this module's
whole job is producing those few extra phrasings.

Hard constraints this module exists to satisfy (RAG upgrade spec section
1 and section 2 stage 2), enforced here rather than left to callers:
  - at most one LLM call per (user_id, normalized query) -- enforced by
    the in-process cache below, which is consulted before any call and
    written to on every outcome, including failures;
  - fully skippable and never blocking: every failure mode (no key, no
    network, rate limit, malformed JSON, schema validation failure) is
    caught here and reported back as `used=False` with an explanatory
    `note` -- this module never raises;
  - respects `FINMATE_RAG_MODE=no_llm` (env var, or an explicit `mode=`
    argument) as a hard "don't even try" switch.

The cache is process-lifetime, in-memory (a plain dict), not persisted to
SQLite. This app has no existing general-purpose KV store -- `finmate/db.py`
is schema'd specifically for profiles and transactions -- and adding one
solely for this cache would be a bigger footprint than the requirement
("repeat questions don't re-spend a call" within a running process) calls
for. A restart clears it; that's an acceptable, documented trade-off, not
an oversight.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from pydantic import BaseModel, Field

from .llm import LLMClient

logger = logging.getLogger("finmate.query_rewrite")

MAX_PHRASINGS = 3

QUERY_REWRITE_PROMPT = """You expand a personal-finance question into short search phrasings.

Given the user's question below, return 2-3 short search-oriented phrasings \
(a few words each, like search-engine queries, not full sentences) that \
would help find relevant transactions by keyword or category match. Stay \
grounded in the user's actual wording and evident intent -- do not invent \
unrelated topics or financial details not implied by the question.

Return ONLY JSON matching this schema, with no other text:
{"phrasings": ["...", "..."]}"""


class _QueryRewriteOutput(BaseModel):
    phrasings: list[str] = Field(default_factory=list)


@dataclass
class QueryRewriteResult:
    phrasings: list[str] = field(default_factory=list)
    used: bool = False
    note: str = ""


# Process-lifetime cache: (user_id, normalized_query) -> QueryRewriteResult.
# See module docstring for why this is a plain dict and not persisted.
_cache: dict[tuple[str, str], QueryRewriteResult] = {}


def _normalize(query: str) -> str:
    return " ".join(query.strip().lower().split())


def clear_cache() -> None:
    """Test hook. Also useful if a caller wants to force a fresh rewrite
    after, say, rotating an API key mid-process."""
    _cache.clear()


def rewrite_query(
    user_id: str,
    query: str,
    llm_client: Optional[LLMClient] = None,
    mode: Optional[str] = None,
) -> QueryRewriteResult:
    """Best-effort query expansion. Never raises.

    Args:
        user_id: used only for cache scoping (so one user's rewrite of a
            query never leaks into the cache lookup for another user's
            identical question).
        query: the raw user question.
        llm_client: inject an existing (or mocked) `LLMClient` to reuse
            one, or to test without touching real config/network. If
            None, a client is constructed lazily via `LLMClient()`
            (env-resolved provider/model/key) on first real use --
            construction failure (no key configured, etc.) is caught
            below like any other failure mode, not raised.
        mode: "no_llm" disables rewriting outright without constructing a
            client at all. Defaults to `FINMATE_RAG_MODE` from the
            environment if not given explicitly.
    """
    mode = mode if mode is not None else os.environ.get("FINMATE_RAG_MODE", "")
    if mode.strip().lower() == "no_llm":
        return QueryRewriteResult(used=False, note="Query rewrite disabled via FINMATE_RAG_MODE=no_llm.")
    if not query or not query.strip():
        return QueryRewriteResult(used=False, note="Empty query; nothing to rewrite.")

    cache_key = (user_id, _normalize(query))
    cached = _cache.get(cache_key)
    if cached is not None:
        return QueryRewriteResult(phrasings=list(cached.phrasings), used=cached.used, note=cached.note)

    try:
        client = llm_client if llm_client is not None else LLMClient()
    except Exception as exc:  # noqa: BLE001 -- e.g. LLMConfigError (no key) or ImportError (no `openai` pkg)
        result = QueryRewriteResult(used=False, note=f"Query rewrite skipped (client unavailable): {exc}")
        _cache[cache_key] = result
        return result

    try:
        parsed = client.call(
            agent_system_prompt=QUERY_REWRITE_PROMPT,
            user_message=f'User question: "{query}"',
            response_model=_QueryRewriteOutput,
            max_tokens=200,
            # This call never produces anything the user sees and never
            # touches their stored data -- it only turns their own
            # question into a few search phrasings -- so it has no use
            # for the CONSTITUTION's user-facing behavioral rules. See
            # LLMClient.call's docstring for why that makes this the one
            # call site that opts out.
            include_constitution=False,
        )
    except Exception as exc:  # noqa: BLE001 -- network error, rate limit, LLMCallError, etc. all degrade the same way
        logger.info("Query rewrite call failed, continuing with original query only: %s", exc)
        result = QueryRewriteResult(used=False, note=f"Query rewrite call failed, continuing with original query: {exc}")
        _cache[cache_key] = result
        return result

    phrasings = [p.strip() for p in parsed.phrasings if p and p.strip()][:MAX_PHRASINGS]
    result = QueryRewriteResult(
        phrasings=phrasings,
        used=bool(phrasings),
        note="" if phrasings else "Query rewrite call returned no usable phrasings; continuing with original query.",
    )
    _cache[cache_key] = result
    return result
