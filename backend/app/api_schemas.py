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

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class ChatMessageIn(BaseModel):
    """One prior turn, as the frontend already has it in its own message
    list (`frontend/components/ChatShell.tsx`'s `messages` state) --
    this is the wire shape for `ChatRequest.history` below, not an
    internal pipeline contract (compare `finmate/orchestrator.py`'s
    plain-dict `GraphState["conversation_history"]`, which this gets
    converted to in `routers/chat.py`)."""

    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    # Deprecated compatibility field. When Supabase is configured, the server
    # ignores it and uses the verified user ID from the bearer token instead.
    user_id: str = Field("", max_length=200)
    message: str = Field(..., min_length=1, max_length=4000)
    # Short-term conversational memory (see finmate/orchestrator.py's
    # module docstring "Conversation history" -- distinct from the
    # profile-fact memory finmate/agents/memory.py persists to SQLite).
    # Oldest first, should NOT include `message` itself. Optional and
    # defaults to empty so any existing caller of this API (an older
    # frontend build, a direct API integration) keeps working exactly as
    # before -- no history in, no history-aware behavior, same as today.
    # max_length=60 is a generous outer guard against an abusive/buggy
    # payload; the orchestrator applies the real, much smaller trim (see
    # `finmate.orchestrator.MAX_HISTORY_MESSAGES`) regardless of how much
    # is sent here, so a client doesn't need to match that number exactly
    # -- see ChatShell.tsx, which sends a further-trimmed slice anyway to
    # keep the request body small.
    history: list[ChatMessageIn] = Field(default_factory=list, max_length=60)


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
    # True whenever the answer needed no numeric/financial-claim
    # verification pass at all (see finmate/orchestrator.py's
    # `_turn_needs_verification` and its "Critic: conditional, not
    # always-on" module docstring section) -- distinct from
    # `critic_passed`, which is also True in that case. A frontend
    # showing a single "Verified" badge for both `verification_ran=False`
    # (nothing to check) and `verification_ran=True, critic_passed=True`
    # (checked and approved) would overstate the first case -- see
    # frontend/components/VerifiedStrip.tsx.
    verification_ran: bool = True
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


class SavedChatMessageOut(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    created_at: str


class SavedChatHistoryResponse(BaseModel):
    messages: list[SavedChatMessageOut] = Field(default_factory=list)


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
