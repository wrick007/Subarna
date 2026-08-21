
"""
LangGraph orchestrator for FinMate AI.

Pipeline:

    router  (also produces RAG search phrasings -- see "Fewer sequential
             calls" below; this is the only call for a casual message,
             which never reaches this module at all -- see finmate.casual)
        ↓
    pipeline  (memory fast path / rag / calculation / specialist agents)
        ↓
    synthesis  (produces the FINAL user-facing reply directly -- see
                "Fewer sequential calls: synthesis + formatter merged")
        ↓
    critic  (SKIPPED entirely when there's nothing concrete to verify --
             see "Critic: conditional, not always-on")
        ↓
    ┌────────────────────┐
    │ pass / skipped      │ → finalize (no LLM call) → END
    │ fail, retry left    │ → pipeline → synthesis → critic
    │ fail, retries used  │ → finalize (appends a fixed disclaimer) → END
    └────────────────────┘

Simple profile updates use a fast path and do not invoke
RAG, calculations, specialist agents, synthesis, or critic.

The pipeline supports up to two critic retries.

Conversation history (short-term memory)
-----------------------------------------
Two different, non-overlapping kinds of "memory" exist in this codebase:

  - `finmate/agents/memory.py`: *profile-fact* memory. Durable facts
    ("my income went up to X") persisted to SQLite via `db.upsert_profile`,
    read back on every future turn regardless of conversation. Unchanged
    by anything below.
  - This module: *conversational* memory. A handful of recent turns
    (`GraphState["conversation_history"]`, populated from `run_finmate`'s
    `conversation_history` argument, trimmed by `_trim_history`) threaded
    into the router (to resolve "it"/"that" when classifying intent) and
    the synthesis agent (to answer as a coherent continuation of the
    conversation) -- see `_render_history_block`. Request-scoped only:
    this module never stores it, looks it up, or persists it anywhere:
    the caller (an HTTP request, a Streamlit session) supplies it fresh
    on every call, the same way it supplies `user_message` itself.

Deliberately NOT threaded: `finmate/agents/memory.py` (profile-fact
detection must key off the current message's explicit statement, not an
older turn -- see that module's own docstring), RAG/retrieval (the
query text sent to `finmate/rag.py:retrieve` is still exactly
`user_message`, unresolved -- see `_node_pipeline` below), and every
specialist agent (budget/cashflow/goal/debt/investment/anomaly) -- only
the router and the agent that drafts the final answer need
conversational context per this design; giving every specialist its own
copy would be several more prompts paying for context most of them
don't need.

Fewer sequential calls (Priority 2)
-------------------------------------
The pre-redesign pipeline made up to ~7 sequential LLM calls for one
non-casual, non-profile-update turn: router, query rewrite, 0-2
specialists, synthesis, critic (up to 3 attempts), formatter. Three
changes cut that, each a deliberate, separately-justified choice rather
than one big rewrite:

  1. Router + query rewrite merged. `schemas.RouterOutput.search_phrasings`
     asks the SAME router call that already classifies intent to also
     produce the 2-3 search phrasings `finmate/query_rewrite.py` used to
     make a second call for. `finmate/rag.py:retrieve`'s
     `precomputed_phrasings` parameter uses these when given, and
     falls back to calling `query_rewrite.rewrite_query` itself when
     they're not (e.g. `scripts/eval_rag.py`, or any test that calls
     `rag.retrieve` directly without going through this orchestrator) --
     see that parameter's docstring. `FINMATE_RAG_MODE=no_llm` still
     disables query augmentation entirely, not just the extra call.

  2. Synthesis + formatter merged (see `agents/synthesis.py` and
     prompts.py's SYNTHESIS_AGENT/FORMATTER_AGENT headers). One call now
     produces the final user-facing reply directly. `_node_finalize`
     (which replaced `_node_formatter` in the graph) does the very last
     step -- appending a verification disclaimer, only when needed --
     with NO LLM call at all: see that function's docstring for why a
     fixed template beats asking a model to "mention this might be
     wrong" for the one piece of text that must never be silently
     softened or dropped.

  3. Critic made conditional, not always-on (see `_turn_needs_verification`
     and "Critic: conditional, not always-on" below) -- this is the one
     of the two changes worth being explicit about what it trades away,
     so it gets its own section rather than a bullet here.

  Bonus, not called for explicitly but directly in service of "fewer
  *wasted* sequential calls": a critic-triggered retry used to re-run
  synthesis with IDENTICAL inputs at temperature 0 -- i.e. very likely
  reproducing the same rejected answer and failing the same way again,
  burning up to 2 full retries for nothing. `_render_verification_feedback`
  threads the specific issues the critic found into the retry's synthesis
  call (`verification_feedback`, both agents/synthesis.py functions) so a
  retry can actually address them instead of blindly repeating.

Critic: conditional, not always-on
-------------------------------------
This is one of the two judgment calls this redesign asked to be called
out explicitly (the other is the evidence panel, a frontend decision --
see frontend/app/globals.css's design-direction comment).

`_turn_needs_verification` skips the critic call (and therefore the
retry loop it can trigger) entirely when this turn's pipeline produced
no calculation results, no specialist agent output, and no retrieved
transaction evidence -- i.e. nothing concrete for a numeric/financial
fact-checker to check. In practice this is every `general_finance` turn
(that intent's ROUTING_TABLE stages are `[]`, so it can never produce
any of the three) plus any other intent where retrieval/calculation
happened to come up empty for this specific question.

What this trades away: the critic is FinMate's actual differentiator --
"nothing it says is untraceable" -- and this change means a chunk of
turns (general financial education, casual-adjacent questions) no
longer get that verification pass. The judgment call is that general
financial education carries materially lower risk of a false, harmful
claim than a specific personalized number ("you can afford this in 14
months") does, AND the CRITIC_AGENT prompt already explicitly instructs
"general financial education by itself is not a reason to fail" -- so
in practice these turns were passing unanimously anyway, at the cost of
a full extra round trip to find that out. Skipping is keeping the same
outcome for less cost, not lowering the bar; a false claim buried in a
personalized, numeric answer is the failure mode the critic exists to
catch, and every one of those still goes through it, unconditionally.
The signal is state-derived (did anything concrete get produced this
turn), not a static per-intent allowlist, so it stays correct even for
an intent that CAN produce evidence but happens not to on a specific
question (e.g. "transaction_question" about a category with no data).

Streaming (Priority 2)
-------------------------------------
`run_finmate_stream` is the streaming counterpart of `run_finmate`,
used by `backend/app/routers/chat.py`'s `/api/chat/stream` SSE endpoint.
Same pipeline, same node functions, called directly instead of through
the compiled LangGraph (a generator can't `yield` from inside a
compiled graph's own execution) -- see that function's docstring.

Only the final synthesis call's tokens are streamed, not router/
pipeline/critic -- those produce structured data a partial stream can't
usefully show anyway (see the redesign's own framing: "add streaming for
at least the final response text"). This is also the one real,
explicitly-flagged tradeoff in this design: since a critic-triggered
retry can't be known until AFTER a full synthesis attempt has already
been streamed to the caller, a retry means tokens the caller already
forwarded downstream (e.g. as SSE events to a browser) belong to a
now-discarded attempt. `StreamEvent(type="restart")` marks this
explicitly so a consumer visibly restarts rather than silently mixing
old and new text -- see `StreamEvent`'s docstring. This is rare in
practice (most turns skip the critic entirely per the section above,
and the ones that don't are expected to pass more often than not), and
the alternative -- withholding every token until full verification
completes -- would give up most of streaming's actual latency benefit
for a failure mode that's already uncommon and now visibly, not
silently, handled.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterator, Literal, Optional, TypedDict

from langgraph.graph import END, StateGraph

from . import db
from .agents.anomaly import run_anomaly_agent
from .agents.budget import run_budget_agent
from .agents.calculation import run_calculations
from .agents.cashflow import run_cashflow_agent
from .agents.critic import run_critic_agent
from .agents.debt import run_debt_agent
from .agents.goal import run_goal_agent
from .agents.investment import run_investment_agent
from .agents.memory import (
    apply_memory_action,
    run_memory_agent,
)
from .agents.rag_agent import run_retrieval
from .agents.router import run_router
from .agents.synthesis import run_synthesis_agent, stream_synthesis_agent
from .llm import LLMClient
from .prompts import ROUTING_TABLE
from .rag import RetrievalResult
from .schemas import (
    CalculationResult,
    CriticResult,
    RouterOutput,
    UserProfile,
)

logger = logging.getLogger("finmate.orchestrator")

MAX_CRITIC_RETRIES = 2

# ---------------------------------------------------------------------------
# Short-term conversational memory (distinct from finmate/agents/memory.py's
# *profile-fact* memory -- see that module's docstring and this module's
# "Conversation history" section below for the two-kinds-of-memory split).
# ---------------------------------------------------------------------------

# How many of the most recent messages (user + assistant turns combined,
# so this is ~3 exchanges) to consider at all, before the character
# budget below even applies. Deliberately small: this block is read by
# an LLM on every non-casual turn (router) and every non-profile-update
# turn (synthesis), so it's recurring token cost, not a one-off -- and
# in practice a follow-up like "how can I reduce it" only ever needs the
# last exchange or two, not a full session transcript.
MAX_HISTORY_MESSAGES = 6

# Rough token-budget cap on the *rendered* history block (see
# _render_history_block), applied after the message-count cap above.
# ~4 characters/token is a standard rough estimate for English text when
# no real tokenizer is available (both Groq/Llama and Gemini models use
# different tokenizers than any we'd want to import just for this) --
# 2000 chars is a deliberately conservative ~500-token ceiling, small
# next to DEFAULT_MAX_TOKENS's 2000-token response budget. Guards
# against one unusually long prior message (a big pasted number, a long
# assistant answer) blowing the budget even though it's within
# MAX_HISTORY_MESSAGES.
MAX_HISTORY_CHARS = 2000


def _trim_history(history: list[dict]) -> list[dict]:
    """Keep at most the last MAX_HISTORY_MESSAGES entries, then drop
    further from the *oldest* end (never the most recent -- the most
    recent turn is the one most likely to be what a follow-up refers to)
    until the rendered block fits MAX_HISTORY_CHARS.

    Never raises on malformed input: entries missing "role"/"content",
    or with a non-string "content", are dropped rather than crashing a
    chat turn over a frontend bug -- this is a UX nicety, not something
    that should be able to take down the pipeline.

    A stale/very-long session (many turns, or turns from a while ago) is
    handled by this same trimming, not by any separate expiry logic --
    the caller decides what "recent" means by what it sends (see
    `run_finmate`'s docstring); this function just bounds it defensively
    regardless of how much a caller sends.
    """
    cleaned: list[dict] = []
    for turn in history:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role")
        content = turn.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str) or not content.strip():
            continue
        cleaned.append({"role": role, "content": content.strip()})

    trimmed = cleaned[-MAX_HISTORY_MESSAGES:] if len(cleaned) > MAX_HISTORY_MESSAGES else cleaned
    while trimmed and len(_render_history_block(trimmed)) > MAX_HISTORY_CHARS:
        trimmed = trimmed[1:]
    return trimmed


def _render_history_block(history: list[dict]) -> str:
    """Render trimmed history as a compact text block for an agent
    prompt. Returns "" for empty history -- callers append this
    unconditionally, so a turn with no prior context costs zero extra
    prompt tokens rather than sending an empty-but-present header.

    Shared by `_node_router` and `_node_synthesis` (the only two
    consumers, per the memory redesign's scope -- see prompts.py's
    ROUTER/SYNTHESIS_AGENT docstring additions) so the two agents see
    the identical rendering of the identical history.
    """
    if not history:
        return ""
    lines = [
        "Recent conversation (most recent last, for context only -- use it "
        "to resolve references like 'it'/'that', not as a source of facts):"
    ]
    for turn in history:
        speaker = "User" if turn["role"] == "user" else "Assistant"
        lines.append(f"{speaker}: {turn['content']}")
    return "\n".join(lines)


def _turn_needs_verification(state: "GraphState") -> bool:
    """True if this turn's draft has something concrete for the critic
    to check: a deterministic calculation, a specialist agent's output
    (debt payoff timelines, cash-flow forecasts, budget variances -- all
    numeric/financial claims), or retrieved transaction evidence. False
    means the draft is general/informational with nothing personalized
    or numeric behind it for a fact-checker to verify. See module
    docstring "Critic: conditional, not always-on" for the full
    reasoning and what this trades away.
    """
    if state.get("calc_results"):
        return True
    if state.get("specialist_outputs"):
        return True
    evidence = state.get("evidence")
    if evidence is not None and evidence.evidence:
        return True
    return False


def _render_verification_feedback(critic_result: Optional[CriticResult]) -> str:
    """Turn a failed CriticResult into a compact note for the next
    synthesis attempt to address directly -- see
    `agents/synthesis.py`'s `verification_feedback` parameter and this
    module's docstring "Fewer sequential calls: productive retries" for
    why a retry needs this (without it, a retry re-asks the identical
    question at temperature 0 and is very likely to just reproduce the
    same rejected answer). "" when there's nothing to report --
    `critic_result` is None (nothing has run yet) or already passed --
    so a first attempt costs nothing extra.
    """
    if critic_result is None or critic_result.passed:
        return ""
    issues: list[str] = [
        *(critic_result.errors or []),
        *(critic_result.unsupported_claims or []),
        *(critic_result.calculation_errors or []),
    ]
    issue_text = "; ".join(str(i) for i in issues) if issues else "The verification check did not approve the previous answer."
    return (
        "A previous attempt at this same request was rejected by verification.\n"
        f"Known issues: {issue_text}\n"
        "Do not present unsupported financial information as verified fact. "
        "Address these issues directly in this attempt rather than repeating "
        "the same answer."
    )


# ---------------------------------------------------------------------------
# Specialist agents
# ---------------------------------------------------------------------------

_SPECIALIST_RUNNERS = {
    "budget": run_budget_agent,
    "cashflow": run_cashflow_agent,
    "goal": run_goal_agent,
    "debt": run_debt_agent,
    "investment": run_investment_agent,
    "anomaly": run_anomaly_agent,
}


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------

class GraphState(TypedDict, total=False):
    user_id: str
    user_message: str
    db_path: str

    # Short-term conversational memory: already trimmed (see
    # _trim_history) by the time it lands in initial_state -- every node
    # below reads it via _render_history_block, never re-trims. Plain
    # dicts ({"role": "user"|"assistant", "content": str}), not a
    # pydantic model: this is request-scoped, never persisted, and never
    # validated as a stored contract the way UserProfile/RouterOutput
    # are -- see module docstring "Conversation history".
    conversation_history: list[dict]

    router_output: RouterOutput
    profile: UserProfile

    memory_applied: bool

    evidence: RetrievalResult

    calc_results: list[CalculationResult]
    skipped_calculations: list[str]

    specialist_outputs: dict[str, Any]

    draft_response: str

    critic_result: CriticResult

    pipeline_runs: int

    final_response: str

    # Request-scoped RAG memoization (see finmate/rag.py "Performance").
    # Created fresh per run_finmate() call, threaded through every node
    # via **state, so a same-turn Critic retry reuses the first pipeline
    # pass's retrieval instead of recomputing it. Never shared across
    # different run_finmate() calls -- see run_finmate below.
    retrieval_cache: dict[tuple, RetrievalResult]


# ---------------------------------------------------------------------------
# Public result
# ---------------------------------------------------------------------------

class FinMateResult:
    """Top-level result returned to callers (app.py, and
    backend/app/routers/chat.py's API wrapper around this same function).

    `evidence`, `calc_results`, `specialist_outputs`, and
    `skipped_calculations` are additive: they surface data the pipeline
    already computed internally (`GraphState`'s `evidence`/`calc_results`/
    etc. -- see below) but that the pre-API version of this class never
    exposed, since app.py's Streamlit UI only ever displayed
    `final_response` plus a small hand-picked metadata dict. A consumer
    that wants to show *why* an answer says what it says -- which
    transactions it's grounded in, which deterministic calculation
    produced a number, which specialist agent(s) ran -- reads them from
    here instead of re-deriving them (which would mean a second,
    redundant retrieval/calculation pass). Every field defaults to a safe
    empty value, so this stays backward compatible with any code
    constructing a `FinMateResult` without them.

    `verification_ran`: added alongside Priority 2's conditional critic
    (see `_turn_needs_verification`) specifically so `critic_passed=True`
    stops being ambiguous between two different situations: "the critic
    actually checked this and approved it" vs. "there was nothing
    concrete enough to check, so verification didn't run at all."
    Before that change, `critic_passed` alone was unambiguous -- the
    critic always ran, so True always meant a real pass. A consumer that
    only checks `critic_passed` still gets a safe answer either way
    (True in both cases -- neither is "unverified and shown as fact"),
    but `frontend/components/VerifiedStrip.tsx` uses this to show a
    genuinely different, honest badge for the two ("Verified" vs.
    "General information", not "Verified" for both) rather than
    silently overstating what a skipped check didn't actually confirm.
    """

    def __init__(
        self,
        final_response: str,
        critic_passed: bool,
        critic_result: Optional[CriticResult],
        router_output: RouterOutput,
        retry_count: int,
        evidence: Optional[RetrievalResult] = None,
        calc_results: Optional[list[CalculationResult]] = None,
        specialist_outputs: Optional[dict[str, Any]] = None,
        skipped_calculations: Optional[list[str]] = None,
        verification_ran: bool = True,
    ):
        self.final_response = final_response
        self.critic_passed = critic_passed
        self.critic_result = critic_result
        self.router_output = router_output
        self.retry_count = retry_count
        self.evidence = evidence
        self.calc_results = calc_results or []
        self.specialist_outputs = specialist_outputs or {}
        self.skipped_calculations = skipped_calculations or []
        self.verification_ran = verification_ran


# ---------------------------------------------------------------------------
# Router node
# ---------------------------------------------------------------------------

def _node_router(
    state: GraphState,
    llm_client: LLMClient,
) -> GraphState:

    router_output = run_router(
        llm_client,
        state["user_message"],
        conversation_history=_render_history_block(state.get("conversation_history", [])),
    )

    if router_output.intent not in ROUTING_TABLE:

        logger.warning(
            "Router returned unknown intent %r; "
            "falling back to general_finance.",
            router_output.intent,
        )

        router_output.intent = "general_finance"

    return {
        **state,
        "router_output": router_output,
    }


# ---------------------------------------------------------------------------
# Main pipeline node
# ---------------------------------------------------------------------------

def _node_pipeline(
    state: GraphState,
    llm_client: LLMClient,
) -> GraphState:

    router_output = state["router_output"]

    stages = ROUTING_TABLE.get(
        router_output.intent,
        [],
    )

    user_id = state["user_id"]

    db_path = state.get(
        "db_path",
        db.DEFAULT_DB_PATH,
    )

    pipeline_runs = (
        state.get("pipeline_runs", 0) + 1
    )

    # ---------------------------------------------------------------
    # Profile
    # ---------------------------------------------------------------

    profile = (
        state.get("profile")
        or db.get_user_profile(
            user_id,
            db_path=db_path,
        )
        or UserProfile(
            user_id=user_id,
        )
    )

    # ---------------------------------------------------------------
    # FAST PATH: explicit profile updates
    # ---------------------------------------------------------------
    #
    # Example:
    #
    #   User:
    #       "my income is 50000 INR monthly"
    #
    #   Router:
    #       profile_update
    #
    #   Memory:
    #       monthly_income = 50000
    #
    #   Database:
    #       save immediately
    #
    #   Response:
    #       "Got it. I've updated your monthly income to 50000."
    #
    # No RAG.
    # No calculation.
    # No specialist.
    # No synthesis.
    # No critic.
    #
    # This also prevents the LLM from comparing the new explicit
    # value against an older stored value and generating an
    # incorrect reconciliation response.
    # ---------------------------------------------------------------

    if router_output.intent == "profile_update":

        # Never repeat a profile update during critic retries.
        if not state.get("memory_applied"):

            memory_action = run_memory_agent(
                llm_client,
                state["user_message"],
                profile,
            )

            # -------------------------------------------------------
            # Valid explicit update
            # -------------------------------------------------------

            if (
                memory_action.memory_action != "none"
                and not memory_action.requires_confirmation
                and memory_action.field
                and memory_action.new_value is not None
            ):

                profile = apply_memory_action(
                    profile,
                    memory_action,
                )

                db.upsert_profile(
                    profile,
                    db_path=db_path,
                )

                field_name = (
                    memory_action.field
                    .replace("_", " ")
                )

                response = (
                    f"Got it. I've updated your "
                    f"{field_name} to "
                    f"{memory_action.new_value}."
                )

                return {
                    **state,
                    "profile": profile,
                    "memory_applied": True,
                    "evidence": RetrievalResult(),
                    "calc_results": [],
                    "skipped_calculations": [],
                    "specialist_outputs": {},
                    "draft_response": response,
                    "final_response": response,
                    "pipeline_runs": pipeline_runs,
                }

            # -------------------------------------------------------
            # Confirmation required
            # -------------------------------------------------------

            if memory_action.requires_confirmation:

                logger.info(
                    "Memory update to field %r "
                    "requires confirmation.",
                    memory_action.field,
                )

                response = (
                    "I understand that you want to update "
                    "your financial profile, but I need "
                    "confirmation before changing the "
                    "stored value."
                )

                return {
                    **state,
                    "profile": profile,
                    "memory_applied": True,
                    "evidence": RetrievalResult(),
                    "calc_results": [],
                    "skipped_calculations": [],
                    "specialist_outputs": {},
                    "draft_response": response,
                    "final_response": response,
                    "pipeline_runs": pipeline_runs,
                }

        # -----------------------------------------------------------
        # Could not determine a valid update
        # -----------------------------------------------------------

        response = (
            "I couldn't determine exactly which profile "
            "value you want to update. Please state the "
            "field and new value explicitly."
        )

        return {
            **state,
            "profile": profile,
            "memory_applied": True,
            "evidence": RetrievalResult(),
            "calc_results": [],
            "skipped_calculations": [],
            "specialist_outputs": {},
            "draft_response": response,
            "final_response": response,
            "pipeline_runs": pipeline_runs,
        }

    # ---------------------------------------------------------------
    # Memory
    # ---------------------------------------------------------------

    # Memory is deliberately executed only on the first pass.
    # Critic retries must not duplicate memory writes.

    if (
        "memory" in stages
        and not state.get("memory_applied")
    ):

        memory_action = run_memory_agent(
            llm_client,
            state["user_message"],
            profile,
        )

        if (
            memory_action.memory_action != "none"
            and not memory_action.requires_confirmation
            and memory_action.field
            and memory_action.new_value is not None
        ):

            profile = apply_memory_action(
                profile,
                memory_action,
            )

            db.upsert_profile(
                profile,
                db_path=db_path,
            )

        elif memory_action.requires_confirmation:

            logger.info(
                "Memory update to field %r "
                "requires confirmation.",
                memory_action.field,
            )

    # ---------------------------------------------------------------
    # Evidence / RAG
    # ---------------------------------------------------------------

    # Always use a RetrievalResult, even when RAG is not required.
    #
    # This prevents:
    #
    #     'NoneType' object has no attribute 'evidence'
    #
    # during calculation or specialist execution.

    evidence = (
        state.get("evidence")
        or RetrievalResult()
    )

    if "rag" in stages:

        retrieved = run_retrieval(
            user_id,
            router_output,
            state["user_message"],
            db_path=db_path,
            llm_client=llm_client,
            cache=state.get("retrieval_cache"),
        )

        if retrieved is not None:
            evidence = retrieved

    # Final defensive fallback.

    if evidence is None:
        evidence = RetrievalResult()

    # ---------------------------------------------------------------
    # Deterministic calculations
    # ---------------------------------------------------------------

    calc_results: list[CalculationResult] = []

    skipped: list[str] = []

    if (
        "calculation" in stages
        and router_output.calculations_needed
    ):

        calc_results, skipped = run_calculations(
            router_output.calculations_needed,
            profile,
            evidence,
        )

    # ---------------------------------------------------------------
    # Specialist agents
    # ---------------------------------------------------------------

    specialist_outputs: dict[str, Any] = {}

    for stage in stages:

        runner = _SPECIALIST_RUNNERS.get(stage)

        if runner is None:
            continue

        try:

            output = runner(
                llm_client,
                state["user_message"],
                profile,
                evidence,
                calc_results,
                skipped,
            )

            if hasattr(output, "model_dump"):
                specialist_outputs[stage] = (
                    output.model_dump()
                )
            else:
                specialist_outputs[stage] = output

        except Exception as exc:

            logger.exception(
                "Specialist '%s' failed.",
                stage,
            )

            specialist_outputs[stage] = {
                "error": str(exc),
            }

    return {
        **state,
        "profile": profile,
        "memory_applied": True,
        "evidence": evidence,
        "calc_results": calc_results,
        "skipped_calculations": skipped,
        "specialist_outputs": specialist_outputs,
        "pipeline_runs": pipeline_runs,
    }


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------

def _node_synthesis(
    state: GraphState,
    llm_client: LLMClient,
) -> GraphState:

    # Profile updates should already have returned directly
    # from _node_pipeline(). This is a defensive fallback.
    if state["router_output"].intent == "profile_update":

        response = state.get(
            "final_response",
            state.get(
                "draft_response",
                "Your profile has been updated.",
            ),
        )

        return {
            **state,
            "draft_response": response,
            "final_response": response,
        }

    draft = run_synthesis_agent(
        llm_client,
        state["user_message"],
        state["profile"],
        state.get("evidence") or RetrievalResult(),
        state.get("calc_results", []),
        state.get("skipped_calculations", []),
        state.get(
            "specialist_outputs",
            {},
        ),
        conversation_history=_render_history_block(state.get("conversation_history", [])),
        verification_feedback=_render_verification_feedback(state.get("critic_result")),
    )

    return {
        **state,
        "draft_response": draft,
    }


# ---------------------------------------------------------------------------
# Critic
# ---------------------------------------------------------------------------

def _node_critic(
    state: GraphState,
    llm_client: LLMClient,
) -> GraphState:

    # Profile updates never need critic verification.
    if state["router_output"].intent == "profile_update":

        critic_result = CriticResult(
            passed=True,
            confidence=1.0,
            errors=[],
            unsupported_claims=[],
            calculation_errors=[],
            privacy_issues=[],
            safety_issues=[],
            required_research=[],
        )

        return {
            **state,
            "critic_result": critic_result,
        }

    # Priority 2: skip verification entirely when this turn produced
    # nothing concrete to verify -- see _turn_needs_verification's
    # docstring and module docstring "Critic: conditional, not
    # always-on". Saves the call *and* the retry loop it could
    # otherwise trigger, since there would be nothing a retry could fix.
    if not _turn_needs_verification(state):

        critic_result = CriticResult(
            passed=True,
            confidence=1.0,
            errors=[],
            unsupported_claims=[],
            calculation_errors=[],
            privacy_issues=[],
            safety_issues=[],
            required_research=[],
        )

        return {
            **state,
            "critic_result": critic_result,
        }

    try:

        critic_result = run_critic_agent(
            llm_client,
            state["draft_response"],
            state["profile"],
            state.get("evidence") or RetrievalResult(),
            state.get("calc_results", []),
        )

    except Exception as exc:

        logger.exception(
            "Critic execution failed."
        )

        critic_result = CriticResult(
            passed=False,
            confidence=0.0,
            errors=[
                f"Critic execution error: {exc}"
            ],
            unsupported_claims=[],
            calculation_errors=[],
            privacy_issues=[],
            safety_issues=[],
            required_research=[],
        )

    # Defensive protection if the critic somehow returns None.

    if critic_result is None:

        critic_result = CriticResult(
            passed=False,
            confidence=0.0,
            errors=[
                "Critic returned no result."
            ],
            unsupported_claims=[],
            calculation_errors=[],
            privacy_issues=[],
            safety_issues=[],
            required_research=[],
        )

    logger.info(
        "Critic result: passed=%s confidence=%s",
        critic_result.passed,
        critic_result.confidence,
    )

    if critic_result.errors:
        logger.warning(
            "Critic errors: %s",
            critic_result.errors,
        )

    if critic_result.unsupported_claims:
        logger.warning(
            "Critic unsupported claims: %s",
            critic_result.unsupported_claims,
        )

    return {
        **state,
        "critic_result": critic_result,
    }


# ---------------------------------------------------------------------------
# Critic routing
# ---------------------------------------------------------------------------

def _critic_route(
    state: GraphState,
) -> str:

    critic_result = state.get(
        "critic_result"
    )

    pipeline_runs = state.get(
        "pipeline_runs",
        1,
    )

    # ---------------------------------------------------------------
    # Critic passed
    # ---------------------------------------------------------------

    if (
        critic_result is not None
        and critic_result.passed
    ):

        return "finalize"

    # ---------------------------------------------------------------
    # Retry available
    # ---------------------------------------------------------------

    # pipeline_runs:
    #
    # 1 = first pipeline execution
    # 2 = first retry
    # 3 = second retry
    #
    # We retry while pipeline_runs <= MAX_CRITIC_RETRIES.

    if pipeline_runs <= MAX_CRITIC_RETRIES:

        logger.warning(
            "Critic failed. Starting retry %d/%d.",
            pipeline_runs,
            MAX_CRITIC_RETRIES,
        )

        return "pipeline"

    # ---------------------------------------------------------------
    # Retry budget exhausted
    # ---------------------------------------------------------------

    logger.warning(
        "Critic failed after %d retries. "
        "Finalizing response with verification warning.",
        MAX_CRITIC_RETRIES,
    )

    return "finalize"


# ---------------------------------------------------------------------------
# Finalize
#
# Replaces the old "Formatter" node/LLM call (see prompts.py's
# FORMATTER_AGENT header and agents/formatter.py's module docstring for
# where that step went -- it's merged into synthesis now).
# ---------------------------------------------------------------------------

def _node_finalize(
    state: GraphState,
) -> GraphState:
    """No LLM call, unlike the `_node_formatter` this replaced: see
    module docstring "Fewer sequential calls: synthesis + formatter
    merged". `draft_response` from `_node_synthesis` already IS the
    final user-facing reply -- SYNTHESIS_AGENT produces it directly now
    -- so the only thing left to do here is append a fixed disclaimer,
    and only when verification actually ran and rejected the answer on
    every retry (a *skipped* critic -- see `_turn_needs_verification` --
    means there was nothing to verify in the first place, so no
    disclaimer is added; `critic_result.passed` is True in both the
    "passed" and "skipped" cases, so this one check correctly covers
    both without needing to distinguish them).

    A fixed template, not another LLM call asking a model to "rewrite
    this to mention it might be wrong": that instruction is exactly the
    kind a model can quietly soften or drop, which is the one place this
    response must not fail quietly. Takes no `llm_client` at all --
    intentionally: see `build_graph`, which registers this node directly
    instead of wrapping it in the `lambda state: ...(state, llm_client)`
    every other node needs.
    """

    # Profile updates already have a final response; do not touch it.
    if state["router_output"].intent == "profile_update":

        return {
            **state,
            "final_response": state.get(
                "final_response",
                state.get(
                    "draft_response",
                    "Your profile has been updated.",
                ),
            ),
        }

    draft = state.get(
        "draft_response",
        "I was unable to generate a response.",
    )

    critic_result = state.get("critic_result")

    if critic_result is not None and not critic_result.passed:

        draft = (
            f"{draft}\n\n*I wasn't able to fully verify every figure in "
            "this answer against your stored data -- please treat it "
            "with extra caution and double-check anything important.*"
        )

    return {
        **state,
        "final_response": draft,
    }


# ---------------------------------------------------------------------------
# Build LangGraph
# ---------------------------------------------------------------------------

def build_graph(
    llm_client: LLMClient,
):

    graph = StateGraph(
        GraphState
    )

    graph.add_node(
        "router",
        lambda state: _node_router(
            state,
            llm_client,
        ),
    )

    graph.add_node(
        "pipeline",
        lambda state: _node_pipeline(
            state,
            llm_client,
        ),
    )

    graph.add_node(
        "synthesis",
        lambda state: _node_synthesis(
            state,
            llm_client,
        ),
    )

    graph.add_node(
        "critic",
        lambda state: _node_critic(
            state,
            llm_client,
        ),
    )

    # No llm_client -- see _node_finalize's docstring for why.
    graph.add_node(
        "finalize",
        _node_finalize,
    )

    graph.set_entry_point(
        "router"
    )

    graph.add_edge(
        "router",
        "pipeline",
    )

    graph.add_edge(
        "pipeline",
        "synthesis",
    )

    graph.add_edge(
        "synthesis",
        "critic",
    )

    graph.add_conditional_edges(
        "critic",
        _critic_route,
        {
            "pipeline": "pipeline",
            "finalize": "finalize",
        },
    )

    graph.add_edge(
        "finalize",
        END,
    )

    return graph.compile()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_finmate(
    user_id: str,
    user_message: str,
    llm_client: LLMClient,
    db_path: str = db.DEFAULT_DB_PATH,
    conversation_history: Optional[list[dict]] = None,
) -> FinMateResult:
    """Run one user turn through the full pipeline.

    `conversation_history`: optional recent prior turns for short-term
    conversational memory -- e.g. so "how can I reduce it" after "what's
    my biggest expense category" actually resolves "it". Each entry is
    `{"role": "user"|"assistant", "content": str}`, oldest first, and
    should NOT include the current `user_message` itself (that's passed
    separately, as always). This is a *caller-supplied* transcript, not
    something this function stores or looks up on its own -- see module
    docstring "Conversation history" for why (no server-side session
    store, by design). The caller (backend/app/routers/chat.py from the
    frontend's own message list, or app.py from Streamlit's
    st.session_state) owns keeping and trimming its own copy of the
    conversation for display; this function only trims what it's handed
    down to what actually goes in a prompt (see `_trim_history`).

    None or [] (the default) is exactly today's behavior: no history
    reaches the router or synthesis agent, zero extra tokens spent. This
    is distinct from `finmate/agents/memory.py`'s profile-fact memory
    (income updates, goals, etc., persisted to the database) -- that
    system is unchanged and unaffected by this parameter; see this
    module's "Conversation history" section and prompts.py's
    ROUTER/SYNTHESIS_AGENT docstrings for how the two differ and where
    each one's context actually flows.
    """

    compiled = build_graph(
        llm_client
    )

    initial_state: GraphState = {
        "user_id": user_id,
        "user_message": user_message,
        "db_path": db_path,
        "conversation_history": _trim_history(conversation_history or []),
        # Fresh dict every call -- never reused across turns or users.
        # Lets a same-turn Critic retry skip redundant RAG recomputation
        # (see finmate/rag.py "Performance") without any risk of a later
        # turn reading a stale retrieval from an earlier one.
        "retrieval_cache": {},
    }

    final_state = compiled.invoke(
        initial_state
    )

    return _finmate_result_from_state(final_state)


def _finmate_result_from_state(final_state: GraphState) -> FinMateResult:
    """Shared tail of `run_finmate` and `run_finmate_stream` -- builds
    the public `FinMateResult` the same way regardless of which one ran
    the pipeline, so a caller can't tell (other than by how they called
    in) which path produced a given result."""

    critic_result = final_state.get(
        "critic_result"
    )

    pipeline_runs = final_state.get(
        "pipeline_runs",
        1,
    )

    return FinMateResult(
        final_response=final_state[
            "final_response"
        ],
        critic_passed=bool(
            critic_result
            and critic_result.passed
        ),
        critic_result=critic_result,
        router_output=final_state[
            "router_output"
        ],
        retry_count=max(
            pipeline_runs - 1,
            0,
        ),
        evidence=final_state.get("evidence"),
        calc_results=final_state.get("calc_results"),
        specialist_outputs=final_state.get("specialist_outputs"),
        skipped_calculations=final_state.get("skipped_calculations"),
        # profile_update's fast path never reaches _node_critic at all
        # (see _node_pipeline), so there's nothing to call "skipped" in
        # the Priority-2 conditional-critic sense here -- True is the
        # correct/inert value for that path, same as a real pass.
        verification_ran=(
            final_state["router_output"].intent == "profile_update"
            or _turn_needs_verification(final_state)
        ),
    )


# ---------------------------------------------------------------------------
# Streaming entry point (Priority 2) -- see module docstring "Streaming"
# ---------------------------------------------------------------------------

@dataclass
class StreamEvent:
    """One event yielded by `run_finmate_stream`.

    `type`:
      - `"token"`: another piece of the final response's text (`.text`).
        Concatenate every `"token"` event's text, in order, since the
        last `"restart"` (or the start of the stream, if none) to get
        the text so far.
      - `"restart"`: a critic-triggered retry is starting. Every
        `"token"` received before this belongs to a now-discarded
        attempt -- a consumer (see `backend/app/routers/chat.py`'s SSE
        endpoint and `frontend/components/MessageBubble.tsx`) should
        visibly clear what it displayed and start fresh on the next
        `"token"`. Rare in practice: most turns skip the critic entirely
        (see `_turn_needs_verification`), and this only fires when the
        critic actually ran AND rejected the draft. See module docstring
        "Streaming" for why this is the one deliberate, flagged
        tradeoff in the streaming design, rather than something hidden.
      - `"done"`: the pipeline finished. `.result` is the same
        `FinMateResult` that `run_finmate` would have returned for an
        identical call -- a caller that only cares about the final
        answer, not the streaming itself, can ignore every event before
        this one and just read `.result`.
      - `"error"`: the pipeline raised before finishing. `.error` is
        `str(exception)`. No `"done"` event follows.
    """

    type: Literal["token", "restart", "done", "error"]
    text: str = ""
    result: Optional[FinMateResult] = None
    error: str = ""


def run_finmate_stream(
    user_id: str,
    user_message: str,
    llm_client: LLMClient,
    db_path: str = db.DEFAULT_DB_PATH,
    conversation_history: Optional[list[dict]] = None,
) -> Iterator[StreamEvent]:
    """Streaming counterpart of `run_finmate` -- identical pipeline
    behavior and identical `conversation_history` contract (see that
    function's docstring), but yields `StreamEvent`s as they happen
    instead of returning one `FinMateResult` at the end. See module
    docstring "Streaming" for the design and its one real tradeoff.

    Calls the same node functions `run_finmate` uses
    (`_node_router`, `_node_pipeline`, `_node_critic`, `_node_finalize`)
    directly, rather than through `build_graph`'s compiled LangGraph: a
    generator can't `yield` from inside a compiled graph's own node
    execution, so the critic-retry loop that graph expresses
    declaratively via conditional edges is an explicit `while` loop
    here instead. Only the synthesis step actually differs in what it
    calls (`stream_synthesis_agent` instead of `run_synthesis_agent`) --
    everything else is the exact same function call `run_finmate` would
    have made, in the same order, so the two can't silently drift into
    answering the same turn differently.

    Never raises: a pipeline failure is caught and yielded as a
    `StreamEvent(type="error", ...)` instead, since a raised exception
    has nowhere clean to go once a caller may already be mid-stream to
    an open SSE connection (see `backend/app/routers/chat.py`).
    """
    try:
        state: GraphState = {
            "user_id": user_id,
            "user_message": user_message,
            "db_path": db_path,
            "conversation_history": _trim_history(conversation_history or []),
            "retrieval_cache": {},
        }

        state = _node_router(state, llm_client)
        state = _node_pipeline(state, llm_client)

        # Profile updates: _node_pipeline's fast path already set
        # final_response; there is nothing to stream (see that
        # function's "FAST PATH" comment -- no synthesis, no critic).
        if state["router_output"].intent != "profile_update":

            while True:
                accumulated: list[str] = []
                for delta in stream_synthesis_agent(
                    llm_client,
                    state["user_message"],
                    state["profile"],
                    state.get("evidence") or RetrievalResult(),
                    state.get("calc_results", []),
                    state.get("skipped_calculations", []),
                    state.get("specialist_outputs", {}),
                    conversation_history=_render_history_block(state.get("conversation_history", [])),
                    verification_feedback=_render_verification_feedback(state.get("critic_result")),
                ):
                    accumulated.append(delta)
                    yield StreamEvent(type="token", text=delta)

                state["draft_response"] = "".join(accumulated)
                state = _node_critic(state, llm_client)

                if _critic_route(state) == "finalize":
                    break

                # Retry: recompute rag/calculation/specialists (RAG
                # hits retrieval_cache -- see rag.py "Performance" -- so
                # this doesn't repeat every stage from scratch), then
                # tell the caller the stream so far is being discarded
                # before starting a fresh one -- see StreamEvent's
                # docstring.
                state = _node_pipeline(state, llm_client)
                yield StreamEvent(type="restart")

        state = _node_finalize(state)
        yield StreamEvent(type="done", result=_finmate_result_from_state(state))

    except Exception as exc:  # noqa: BLE001 -- see docstring: never raise, always yield an "error" event
        logger.exception("run_finmate_stream failed.")
        yield StreamEvent(type="error", error=str(exc))
