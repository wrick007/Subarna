"""Shared helper for specialist agents: assembles the evidence + calculation
context every specialist agent needs into one JSON block, so each agent
module only has to format its own prompt-specific instructions.

Token cost: every specialist agent (finmate/agents/budget.py, cashflow.py,
goal.py, debt.py, investment.py, anomaly.py) and finmate/agents/synthesis.py
calls `build_context_block`, so it's the single highest-leverage place in
the whole pipeline to cut prompt tokens -- one change here helps every
agent that embeds retrieved evidence, not just RAG-labeled code. Two
changes, both lossless to what the agent actually needs:
  1. `evidence_for_prompt` drops per-EvidenceItem fields that exist for
     our own retrieval-quality auditing (finmate/rag.py's fallback-ladder
     `.stage`/`.note`, scripts/eval_rag.py) -- keyword_score, vector_score,
     rerank_score, retrieval_stage, and `page` (this app's evidence is
     always a transaction, never a document page, so `page` is always
     None: zero information, on every item, every call). None of that is
     something an agent needs to reason about or cite in its answer. The
     full RetrievalResult -- every audit field included -- is still
     returned to the orchestrator/API/UI unchanged; only what's billed as
     prompt tokens to Groq/Gemini is trimmed.
  2. Compact (not pretty-printed) JSON. `json.dumps(..., indent=2)` was
     spending real tokens on whitespace an LLM parses exactly as well
     without. This mirrors finmate/agents/critic.py's payload (which
     doesn't call this function -- see its own docstring for why -- but
     applies the identical two changes for the identical reason) and
     finmate/agents/synthesis.py's specialist_outputs dump.
"""

from __future__ import annotations

import json

from ..rag import RetrievalResult
from ..schemas import CalculationResult, UserProfile

# Fields from EvidenceItem (finmate/schemas.py) that an agent actually
# needs to reason about or cite a transaction. See module docstring.
_PROMPT_EVIDENCE_FIELDS = ("source_id", "date", "description", "amount", "currency", "category", "document", "relevance")


def evidence_for_prompt(evidence: RetrievalResult) -> list[dict]:
    """`evidence.evidence`, each item trimmed to `_PROMPT_EVIDENCE_FIELDS`.
    Shared by `build_context_block` below and `finmate/agents/critic.py`."""
    return [{k: v for k, v in item.model_dump().items() if k in _PROMPT_EVIDENCE_FIELDS} for item in evidence.evidence]


def build_context_block(
    user_message: str,
    profile: UserProfile,
    evidence: RetrievalResult,
    calc_results: list[CalculationResult],
    skipped_calculations: list[str],
) -> str:
    payload = {
        "user_message": user_message,
        "profile": profile.model_dump(),
        "evidence": evidence_for_prompt(evidence),
        "vector_search_used": evidence.vector_search_used,
        "retrieval_note": evidence.note,
        "calculation_results": [c.model_dump() for c in calc_results],
        "skipped_calculations": skipped_calculations,
    }
    return (
        "Use ONLY the data below as evidence. Do not invent transactions, "
        "balances, or numbers not present here or derivable from a listed "
        "calculation_result. If something needed is missing, say so.\n\n"
        f"{json.dumps(payload, separators=(',', ':'), default=str)}"
    )
