"""
POST /api/chat -- wraps `finmate.orchestrator.run_finmate`, the same
pipeline entry point app.py's Streamlit UI calls.

POST /api/chat/stream -- Priority 2 streaming counterpart, wraps
`finmate.orchestrator.run_finmate_stream` and forwards its events as
Server-Sent Events. See that function's docstring, and this module's
`_event_stream`, for the wire format.

A plain (synchronous) `def`, not `async def`, on purpose, and for a
heavier reason than `app.routers.users`: this endpoint can run the
entire multi-agent pipeline (router -> memory -> rag -> calculation ->
specialist -> synthesis -> critic), which means several
sequential LLM calls plus local retrieval/computation -- all
fundamentally blocking (the `openai` SDK's sync client, SQLite, local
model inference; none of this project's dependencies are async).
FastAPI runs synchronous path operations in an external thread pool, so
one user's multi-second pipeline run doesn't block the event loop or
any other concurrent request. See `finmate/rag.py`'s "Performance"
section and `finmate/orchestrator.py`'s "Fewer sequential calls" section
for what's already been done to shrink how long that run takes -- and
`/chat/stream` below for making the wait itself feel shorter even where
it can't be cut further.
"""

from __future__ import annotations

import json
import time
from typing import Iterator

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from .. import config
from ..api_schemas import CalculationOut, ChatRequest, ChatResponse, EvidenceItemOut, RetrievalOut
from ..deps import get_llm_client

from finmate.casual import is_casual_message, pick_casual_response  # noqa: E402
from finmate.orchestrator import FinMateResult, run_finmate, run_finmate_stream  # noqa: E402

router = APIRouter(tags=["chat"])


def _build_chat_response(result: FinMateResult, latency_ms: int) -> ChatResponse:
    """Shared by `chat` and `chat_stream`'s final `"done"` event so a
    streamed and non-streamed call for the same turn produce identical
    response bodies -- the only difference is how the text arrived."""
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
        verification_ran=result.verification_ran,
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
            response=pick_casual_response(), is_casual=True, latency_ms=int((time.monotonic() - started) * 1000),
        )

    llm_client = get_llm_client(request)
    # Wire shape (ChatMessageIn) -> orchestrator shape (plain dict): see
    # finmate/orchestrator.py's module docstring "Conversation history"
    # for why this is a plain dict, not a pydantic model, once it's past
    # the API boundary.
    history = [{"role": h.role, "content": h.content} for h in payload.history]
    result = run_finmate(
        payload.user_id, payload.message, llm_client, db_path=config.DB_PATH, conversation_history=history,
    )
    latency_ms = int((time.monotonic() - started) * 1000)
    return _build_chat_response(result, latency_ms)


def _sse(event: str, data: dict) -> str:
    """One Server-Sent Event: an `event:` line naming it, a `data:` line
    with a single-line JSON payload, and the blank line that terminates
    it. `data` is always a JSON *object* (never a bare string/number),
    so the frontend never has to special-case parsing per event type."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _event_stream(payload: ChatRequest, llm_client, history: list[dict]) -> Iterator[str]:
    started = time.monotonic()
    for event in run_finmate_stream(
        payload.user_id, payload.message, llm_client, db_path=config.DB_PATH, conversation_history=history,
    ):
        if event.type == "token":
            yield _sse("token", {"text": event.text})
        elif event.type == "restart":
            # See finmate.orchestrator.StreamEvent's docstring: every
            # "token" since the last "restart" (or the start of the
            # stream) belongs to a now-discarded attempt. The frontend
            # (MessageBubble.tsx) clears its accumulated text on this
            # event rather than silently mixing old and new tokens.
            yield _sse("restart", {})
        elif event.type == "error":
            yield _sse("error", {"message": event.error})
        else:  # "done"
            latency_ms = int((time.monotonic() - started) * 1000)
            body = _build_chat_response(event.result, latency_ms)
            yield _sse("done", body.model_dump())


@router.post("/chat/stream")
def chat_stream(payload: ChatRequest, request: Request) -> StreamingResponse:
    """SSE counterpart of `chat` -- see `finmate.orchestrator.
    run_finmate_stream` and `StreamEvent` for the pipeline-level design,
    and `_event_stream`/`_sse` above for the wire format each event type
    maps to. `media_type="text/event-stream"` plus returning a generator
    (not a pre-built string) is what makes FastAPI actually flush each
    `yield` to the client as it happens rather than buffering the whole
    response -- the entire point of this endpoint over plain `/chat`.

    Casual messages (see `finmate.casual`) still take the zero-LLM-call
    fast path -- streamed as a single immediate "token" + "done" pair,
    so the frontend can consume both endpoints through the one same
    event protocol without needing to know in advance whether a given
    message will turn out to be casual.
    """
    started = time.monotonic()

    if is_casual_message(payload.message):

        def _casual_stream() -> Iterator[str]:
            text = pick_casual_response()
            yield _sse("token", {"text": text})
            latency_ms = int((time.monotonic() - started) * 1000)
            body = ChatResponse(response=text, is_casual=True, latency_ms=latency_ms)
            yield _sse("done", body.model_dump())

        return StreamingResponse(_casual_stream(), media_type="text/event-stream")

    llm_client = get_llm_client(request)
    history = [{"role": h.role, "content": h.content} for h in payload.history]
    return StreamingResponse(_event_stream(payload, llm_client, history), media_type="text/event-stream")
