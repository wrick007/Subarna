"""
FastAPI backend for FinMate AI.

Reuses `src/finmate` as-is -- the exact same multi-agent pipeline app.py's
Streamlit UI calls -- rather than reimplementing or duplicating any of
it. This is purely a new frontend-facing surface (a JSON API instead of
server-rendered Streamlit widgets) over the same engine, the same
database, and the same optional vector index.

Adds `src/` to `sys.path` here, once, at package-import time -- the same
convention `app.py` and `scripts/*.py` already use (see their own
`sys.path.insert` lines) -- so every submodule below (`app.routers.*`,
etc.) can `from finmate import ...` without repeating this.
"""

from __future__ import annotations

import sys

from .config import SRC_DIR

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
