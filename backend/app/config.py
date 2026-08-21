"""
Backend configuration, resolved once from environment variables at import
time. Every setting here has a sane local-dev default, so `uvicorn
app.main:app` works out of the box against the same `data/finmate.db`
the Streamlit app and test suite use -- only a real deployment needs to
override anything, and does so purely through environment variables (no
code change), matching how `finmate/llm.py:resolve_provider_config`
already treats the LLM provider.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# backend/app/config.py -> repo root is two directories up.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = REPO_ROOT / "src"

# Load the root .env first (so a key already set up for app.py's
# Streamlit UI just works here too), then backend/.env on top with
# override=True so backend-specific values (e.g. a different
# FINMATE_FRONTEND_ORIGINS) win where both files set the same key.
# Neither file needs to exist -- load_dotenv() is a silent no-op for a
# missing path -- and neither is used at all in a real deployment
# (Render sets real environment variables, not a .env file); this is
# purely a local-development convenience, resolved by explicit path
# rather than python-dotenv's directory-walking auto-search so it works
# the same regardless of which directory the process was started from
# (`cd backend && uvicorn app.main:app` vs `uvicorn app.main:app
# --app-dir backend` from the repo root -- see backend/.env.example).
load_dotenv(REPO_ROOT / ".env")
load_dotenv(REPO_ROOT / "backend" / ".env", override=True)

#: SQLite database path. Override with FINMATE_DB_PATH -- same variable
#: app.py already reads, so a single .env works for either frontend.
DB_PATH = os.environ.get("FINMATE_DB_PATH", str(REPO_ROOT / "data" / "finmate.db"))

#: Embedded Chroma on-disk path. Override with FINMATE_CHROMA_PATH.
CHROMA_PATH = os.environ.get("FINMATE_CHROMA_PATH", str(REPO_ROOT / "data" / "chroma_store"))

#: Comma-separated list of allowed CORS origins for the deployed frontend,
#: e.g. "https://finmate.vercel.app,https://finmate-ai.example.com".
#: Defaults to the two ports Vite/Next.js/CRA dev servers commonly use,
#: so local frontend development works with zero configuration. A
#: deployed backend MUST set this explicitly (see DEPLOYMENT.md) --
#: no wildcard fallback in production, since the API is credentialed
#: (it reads/writes a specific user_id's private financial data).
_default_origins = "http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000,http://127.0.0.1:5173"
FRONTEND_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("FINMATE_FRONTEND_ORIGINS", _default_origins).split(",")
    if origin.strip()
]

#: Pre-load the embedder/cross-encoder/Chroma client at startup (see
#: finmate/rag.py:warm_up) rather than on the first request. On by
#: default; the only reason to turn it off is a platform with a strict
#: startup-time budget where you'd rather eat the cost on first request
#: than fail a deploy health check.
WARM_UP_ON_STARTUP = os.environ.get("FINMATE_WARM_UP_ON_STARTUP", "true").strip().lower() not in ("false", "0", "no")

#: Cross-encoder reranking (finmate/rag.py stage 6) is **off by
#: default** -- read this the same way finmate.rag.retrieve reads it
#: itself (same env var, same "true"/"1"/"yes" allow-list, same opt-IN
#: polarity), so this one flag controls both the per-request behavior
#: and main.py's startup warm-up consistently. Off by default because
#: it's the one dependency in this whole pipeline heavy enough to OOM a
#: memory-constrained deployment (a 512MB free-tier container, say) --
#: loading it at startup on top of the embedder that just finished
#: loading is exactly the kind of thing that gets a process killed
#: before it ever serves a request. Retrieval still works fully without
#: it (metadata filter + keyword + vector search, fused by reciprocal
#: rank fusion -- see README.md's "RAG retrieval" section); reranking is
#: an accuracy improvement on top, not a requirement. Set to "true" on
#: any deployment with memory to spare for the accuracy gain.
ENABLE_RERANKER = os.environ.get("ENABLE_RERANKER", "").strip().lower() in ("true", "1", "yes")

#: Demo/default user_id, used only as a UI convenience default -- the
#: same convention app.py's sidebar and scripts/seed_demo_data.py use.
DEFAULT_USER_ID = os.environ.get("FINMATE_DEFAULT_USER_ID", "demo_user")
