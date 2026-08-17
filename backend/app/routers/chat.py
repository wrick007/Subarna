"""
POST /api/chat -- wraps `finmate.orchestrator.run_finmate`, the same
pipeline entry point app.py's Streamlit UI calls.

A plain (synchronous) `def`, not `async def`, on purpose, and for a
heavier reason than `app.routers.users`: this endpoint can run the
entire multi-agent pipeline (router -> memory -> rag -> calculation ->
specialist -> synthesis -> critic -> formatter), which means several
sequential LLM calls plus local retrieval/computation -- all
fundamentally blocking (the `openai` SDK's sync client, SQLite, local
model inference; none of this project's dependencies are async).
FastAPI runs synchronous path operations in an external thread pool, so
one user's multi-second pipeline run doesn't block the event loop or
any other concurrent request. See `finmate/rag.py`'s "Performance"
section for what's already been done to shrink how long that run takes.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Request

from .. import config
from ..api_schemas import CalculationOut, ChatRequest, ChatResponse, EvidenceItemOut, RetrievalOut
from ..deps import get_llm_client

from finmate.casual import CASUAL_RESPONSE, is_casual_message  # noqa: E402
from finmate.orchestrator import run_finmate  # noqa: E402

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    started = time.monotonic()

    # Checked *before* requiring an LLM client, not after: this is the
    # one path that is genuinely allowed to need zero LLM calls (see
    # finmate/casual.py's docstring), so it should work even against a
    # freshly-deployed backend that has no provider key configured yet
    # -- useful in exactly that moment as a no-key-needed smoke test that
    # the frontend and backend are wired together correctly. Gating it
    # behind an LLM-availability check (as an earlier version of this
    # endpoint did, via a FastAPI `Depends` that runs unconditionally
    # before the route body) would silently break that property.
    if is_casual_message(payload.message):
        return ChatResponse(
            response=CASUAL_RESPONSE, is_casual=True, latency_ms=int((time.monotonic() - started) * 1000),
        )

    llm_client = get_llm_client(request)
    result = run_finmate(payload.user_id, payload.message, llm_client, db_path=config.DB_PATH)
    latency_ms = int((time.monotonic() - started) * 1000)

    retrieval_out = None
    if result.evidence is not None:
        retrieval_out = RetrievalOut(
            stage=result.evidence.stage,
            note=result.evidence.note,
            vector_search_used=result.evidence.vector_search_used,
            keyword_search_used=result.evidence.keyword_search_used,
            rerank_used=result.evidence.rerank_used,
            query_rewrite_used=result.evidence.query_rewrite_used,
            evidence=[
                EvidenceItemOut(
                    source_id=e.source_id, date=e.date, description=e.description, amount=e.amount,
                    currency=e.currency, category=e.category, document=e.document or "", relevance=e.relevance,
                    retrieval_stage=e.retrieval_stage, keyword_score=e.keyword_score,
                    vector_score=e.vector_score, rerank_score=e.rerank_score,
                )
                for e in result.evidence.evidence
            ],
        )

    critic_errors: list[str] = []
    critic_unsupported: list[str] = []
    if result.critic_result is not None:
        critic_errors = list(result.critic_result.errors)
        critic_unsupported = list(result.critic_result.unsupported_claims)

    return ChatResponse(
        response=result.final_response,
        is_casual=False,
        intent=result.router_output.intent,
        risk_level=result.router_output.risk_level,
        critic_passed=result.critic_passed,
        critic_retries_used=result.retry_count,
        critic_errors=critic_errors,
        critic_unsupported_claims=critic_unsupported,
        retrieval=retrieval_out,
        calculations=[
            CalculationOut(
                metric=c.metric, value=c.value, currency=c.currency, period=c.period,
                formula=c.formula, inputs=c.inputs, source_ids=c.source_ids,
            )
            for c in result.calc_results
        ],
        skipped_calculations=result.skipped_calculations,
        specialists_used=list(result.specialist_outputs.keys()),
        specialist_outputs=result.specialist_outputs,
        latency_ms=latency_ms,
    )
