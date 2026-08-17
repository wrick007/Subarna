"""
FastAPI application entry point.

    Local dev:   uvicorn app.main:app --reload --app-dir backend
    Production:  see backend/Dockerfile / DEPLOYMENT.md (Render)

Everything under `/api` reuses `src/finmate` exactly as app.py's
Streamlit UI does -- same orchestrator, same database, same optional
vector index -- through the routers in `app.routers`.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import config
from .routers import chat, health, users

from finmate import db, rag  # noqa: E402
from finmate.llm import LLMClient, LLMConfigError  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("finmate.backend")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---------------------------------------------------
    db.init_db(config.DB_PATH)

    try:
        app.state.llm_client = LLMClient()
        app.state.llm_client_error = None
        logger.info("LLM client ready: %s / %s", app.state.llm_client.provider, app.state.llm_client.model)
    except LLMConfigError as exc:
        # Matches app.py's Streamlit behavior: the app still starts and
        # serves everything that doesn't need an LLM (health, profile,
        # transactions, seed-demo, delete) -- only /api/chat 503s, with
        # this exact message, until a key is configured. A misconfigured
        # or missing API key should never prevent the *process* from
        # starting and passing a platform health check (see
        # app.routers.health's own docstring for why that split matters
        # on Render specifically).
        app.state.llm_client = None
        app.state.llm_client_error = str(exc)
        logger.warning("Starting without a configured LLM provider: %s", exc)

    if config.WARM_UP_ON_STARTUP:
        # See finmate/rag.py:warm_up's docstring -- pays the "load the
        # embedder/cross-encoder/Qdrant client" cost once, now, instead
        # of on whichever user's chat message happens to arrive first.
        app.state.warm_up_result = rag.warm_up(qdrant_path=config.QDRANT_PATH)
        logger.info("RAG warm-up: %s", app.state.warm_up_result)
    else:
        app.state.warm_up_result = {}

    yield
    # --- Shutdown ----------------------------------------------------
    rag.clear_cache()  # closes cached Qdrant client(s) cleanly; see its docstring


app = FastAPI(
    title="FinMate AI API",
    description="Multi-agent personal finance assistant -- REST API over finmate.orchestrator.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.FRONTEND_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    # A catch-all so a bug deep in the pipeline (any specialist agent, a
    # provider hiccup that slips past LLMClient's own handling, etc.)
    # reaches the frontend as a clean {"detail": "..."} JSON 500 -- what
    # every other error path in this API already returns -- rather than
    # an unhandled-exception stack trace or a connection reset. Logged
    # server-side either way, for whoever's operating the deployment.
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error. Check server logs."})


app.include_router(health.router, prefix="/api")
app.include_router(chat.router, prefix="/api")
app.include_router(users.router, prefix="/api")


@app.get("/")
async def root() -> dict:
    return {"name": "FinMate AI API", "docs": "/docs", "health": "/api/health"}
