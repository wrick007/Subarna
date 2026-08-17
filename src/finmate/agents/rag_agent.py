"""Stage 3 - Transaction/RAG Agent. Owns prompts.RAG_AGENT.

Deliberately implemented as a deterministic wrapper around `finmate.rag`
rather than an LLM call: the RAG_AGENT contract's own hardest rule is
"Never invent transactions", and the surest way to guarantee that is to
never let the model generate transaction data at all -- it only ever sees
transactions this module actually looked up. See README "deviations" for
the rationale.

This module still makes zero LLM calls itself after the RAG upgrade: the
one optional LLM call in the hybrid pipeline (query rewrite) lives in
`finmate/query_rewrite.py`, invoked from `finmate/rag.py:retrieve`, not
here. The `llm_client` parameter below only ever gets forwarded.
"""

from __future__ import annotations

from typing import Optional

from .. import db, rag
from ..llm import LLMClient
from ..schemas import RouterOutput


def run_retrieval(
    user_id: str,
    router_output: RouterOutput,
    user_message: str,
    db_path: str = db.DEFAULT_DB_PATH,
    llm_client: Optional[LLMClient] = None,
    cache: Optional[dict] = None,
) -> rag.RetrievalResult:
    """Retrieve evidence relevant to the router's task plan.

    Uses router_output.date_range for the metadata filter and the raw
    user_message as the query for the hybrid pipeline (query rewrite,
    keyword search, vector search, fusion, rerank -- all best-effort, see
    rag.retrieve's documented fallback ladder).

    `llm_client`: optional. `finmate/orchestrator.py` passes the same
    client every other agent in the pipeline already uses, so query
    rewrite reuses it instead of `finmate.query_rewrite` constructing a
    second one from scratch. Left as None, `rag.retrieve` still
    constructs its own lazily from environment config -- useful for
    calling this directly (or injecting a mock in tests) without an
    orchestrator run.

    `cache`: optional request-scoped memoization dict, forwarded as-is to
    `rag.retrieve` (see its docstring). `finmate/orchestrator.py` passes
    one fresh dict per user turn so a Critic-triggered retry of this
    stage doesn't repeat the same retrieval work. None (the default)
    disables it, unchanged from before this parameter existed.
    """
    return rag.retrieve(
        user_id=user_id,
        query=user_message,
        start_date=router_output.date_range.start,
        end_date=router_output.date_range.end,
        db_path=db_path,
        llm_client=llm_client,
        cache=cache,
    )
