"""
Request/response models for the HTTP API.

Deliberately separate from `finmate/schemas.py`: those pydantic models
are the pipeline's *internal* contracts (what one agent hands the next),
built for that purpose first -- `EvidenceItem` carries retrieval-audit
fields no frontend needs (see `finmate/agents/_shared.py`), and
`UserProfile`/`CalculationResult` are reused as-is by several agents in
ways that don't always match what's convenient or stable to expose over
HTTP. Rather than let internal pipeline changes become implicit wire
contract changes, this module defines what the API actually promises,
and each router does the (thin) translation.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=200)
    message: str = Field(..., min_length=1, max_length=4000)


class EvidenceItemOut(BaseModel):
    source_id: str
    date: str
    description: str
    amount: float
    currency: str
    category: str
    document: str = ""
    relevance: float = 0.0
    retrieval_stage: str = ""
    # Included for the frontend's evidence panel (see frontend's
    # VerifiedStrip/EvidenceDrawer components) even though the LLM
    # prompt itself no longer receives these -- see
    # finmate/agents/_shared.py's docstring for why that's fine: this is
    # a UI transparency feature, unrelated to what's sent to Groq/Gemini.
    keyword_score: Optional[float] = None
    vector_score: Optional[float] = None
    rerank_score: Optional[float] = None


class RetrievalOut(BaseModel):
    stage: str
    note: str
    vector_search_used: bool
    keyword_search_used: bool
    rerank_used: bool
    query_rewrite_used: bool
    evidence: list[EvidenceItemOut] = Field(default_factory=list)


class CalculationOut(BaseModel):
    metric: str
    value: float
    currency: str
    period: str
    formula: str
    inputs: dict[str, Any] = Field(default_factory=dict)
    source_ids: list[str] = Field(default_factory=list)


class ChatResponse(BaseModel):
    response: str
    is_casual: bool = False
    intent: str = ""
    risk_level: str = "low"
    critic_passed: bool = True
    critic_retries_used: int = 0
    critic_errors: list[str] = Field(default_factory=list)
    critic_unsupported_claims: list[str] = Field(default_factory=list)
    retrieval: Optional[RetrievalOut] = None
    calculations: list[CalculationOut] = Field(default_factory=list)
    skipped_calculations: list[str] = Field(default_factory=list)
    specialists_used: list[str] = Field(default_factory=list)
    specialist_outputs: dict[str, Any] = Field(default_factory=dict)
    latency_ms: int = 0


class ProfileResponse(BaseModel):
    user_id: str
    has_profile: bool
    profile: Optional[dict[str, Any]] = None


class TransactionOut(BaseModel):
    date: str
    description: str
    amount: float
    currency: str
    category: str
    account: str = ""
    type: str = ""
    source_id: str = ""


class TransactionsResponse(BaseModel):
    user_id: str
    count: int
    transactions: list[TransactionOut]


class DeleteUserResponse(BaseModel):
    user_id: str
    deleted: bool = True


class SeedDemoResponse(BaseModel):
    user_id: str
    transactions_seeded: int
    vector_indexed: int


class HealthResponse(BaseModel):
    status: str
    llm_configured: bool
    provider: Optional[str] = None
    model: Optional[str] = None
    llm_error: Optional[str] = None
    warm_up: dict[str, bool] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    detail: str
