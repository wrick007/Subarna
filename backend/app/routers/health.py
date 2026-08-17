"""GET /api/health -- for Render's health check, uptime monitors, and the
frontend's own "is the backend reachable" indicator.

Deliberately does no blocking work itself (no DB query, no model call):
it reports whatever `app.main`'s startup hook already determined, so it
stays fast under load and during a cold LLM provider outage alike --
exactly the property a platform health check needs to be trustworthy.
"""

from __future__ import annotations

from fastapi import APIRouter, Request

from ..api_schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    llm_client = getattr(request.app.state, "llm_client", None)
    return HealthResponse(
        status="ok",
        llm_configured=llm_client is not None,
        provider=getattr(llm_client, "provider", None),
        model=getattr(llm_client, "model", None),
        llm_error=getattr(request.app.state, "llm_client_error", None),
        warm_up=getattr(request.app.state, "warm_up_result", {}),
    )
