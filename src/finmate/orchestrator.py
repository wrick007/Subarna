
"""
LangGraph orchestrator for FinMate AI.

Pipeline:

    router
        ↓
    pipeline
        ↓
    synthesis
        ↓
    critic
        ↓
    ┌───────────────┐
    │ pass          │ → formatter → END
    │ fail + retry  │ → pipeline → synthesis → critic
    └───────────────┘

Simple profile updates use a fast path and do not invoke
RAG, calculations, specialist agents, synthesis, or critic.

The pipeline supports up to two critic retries.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, TypedDict

from langgraph.graph import END, StateGraph

from . import db
from .agents.anomaly import run_anomaly_agent
from .agents.budget import run_budget_agent
from .agents.calculation import run_calculations
from .agents.cashflow import run_cashflow_agent
from .agents.critic import run_critic_agent
from .agents.debt import run_debt_agent
from .agents.formatter import run_formatter_agent
from .agents.goal import run_goal_agent
from .agents.investment import run_investment_agent
from .agents.memory import (
    apply_memory_action,
    run_memory_agent,
)
from .agents.rag_agent import run_retrieval
from .agents.router import run_router
from .agents.synthesis import run_synthesis_agent
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

        return "formatter"

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
        "Sending response to formatter with verification warning.",
        MAX_CRITIC_RETRIES,
    )

    return "formatter"


# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------

def _node_formatter(
    state: GraphState,
    llm_client: LLMClient,
) -> GraphState:

    # Profile updates already have a final response.
    # Do not send them through the formatter LLM.
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

    critic_result = state.get(
        "critic_result"
    )

    critic_notes = ""

    if (
        critic_result is not None
        and not critic_result.passed
    ):

        issues: list[str] = []

        issues.extend(
            getattr(
                critic_result,
                "errors",
                [],
            )
            or []
        )

        issues.extend(
            getattr(
                critic_result,
                "unsupported_claims",
                [],
            )
            or []
        )

        issues.extend(
            getattr(
                critic_result,
                "calculation_errors",
                [],
            )
            or []
        )

        if issues:

            issue_text = "; ".join(
                str(issue)
                for issue in issues
            )

        else:

            issue_text = (
                "The verification critic did not approve "
                "the response."
            )

        critic_notes = (
            "VERIFICATION DID NOT PASS.\n\n"
            f"Known verification issues: {issue_text}\n\n"
            "Do not present unsupported financial information "
            "as verified fact. Clearly tell the user when "
            "information could not be fully verified."
        )

    try:

        final_response = run_formatter_agent(
            llm_client,
            state["user_message"],
            state["draft_response"],
            critic_notes,
        )

    except Exception as exc:

        logger.exception(
            "Formatter failed."
        )

        # If formatter itself fails, return the draft instead
        # of crashing the entire Streamlit application.

        final_response = (
            state.get(
                "draft_response",
                "I was unable to generate a response.",
            )
        )

        if critic_notes:

            final_response += (
                "\n\n"
                "Verification warning: "
                "the response could not be fully verified."
            )

    return {
        **state,
        "final_response": final_response,
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

    graph.add_node(
        "formatter",
        lambda state: _node_formatter(
            state,
            llm_client,
        ),
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
            "formatter": "formatter",
        },
    )

    graph.add_edge(
        "formatter",
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
) -> FinMateResult:

    compiled = build_graph(
        llm_client
    )

    initial_state: GraphState = {
        "user_id": user_id,
        "user_message": user_message,
        "db_path": db_path,
        # Fresh dict every call -- never reused across turns or users.
        # Lets a same-turn Critic retry skip redundant RAG recomputation
        # (see finmate/rag.py "Performance") without any risk of a later
        # turn reading a stale retrieval from an earlier one.
        "retrieval_cache": {},
    }

    final_state = compiled.invoke(
        initial_state
    )

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
    )
