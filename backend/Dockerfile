# FastAPI backend image. See ../DEPLOYMENT.md for the full Render walkthrough.
#
# IMPORTANT -- build context is the REPO ROOT, not this backend/ folder:
# this Dockerfile needs both src/ (the finmate engine) and backend/ (this
# API) copied in, so it can't build from backend/ alone. On Render:
#   Dockerfile Path                  backend/Dockerfile
#   Docker Build Context Directory   .   (repo root)
# Locally, that's:  docker build -f backend/Dockerfile .   (run from repo root)

FROM python:3.12-slim
WORKDIR /app

# curl is needed for the HEALTHCHECK below -- not present in the slim base image.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch wheel first, from PyTorch's own index -- the default
# PyPI wheel bundles CUDA libraries that are pure dead weight on a CPU-only
# host like Render's, and roughly triple the image size for nothing this
# app uses (sentence-transformers' MiniLM/cross-encoder models run fine,
# fast enough, on CPU). Installed *before* requirements.txt (which also
# pins torch/torchvision) so pip sees them already satisfied and doesn't
# pull the larger default build back in on the next step.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt ./requirements.txt
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt -r backend/requirements.txt

COPY src/ ./src/
COPY backend/ ./backend/
COPY data/ ./data/

# Render sets $PORT at runtime and routes traffic there -- the app must
# bind 0.0.0.0:$PORT, not a hardcoded port. Shell-form CMD/HEALTHCHECK
# (not exec/JSON-array form) so ${PORT:-8000} expands at container
# start, not at build time; the 8000 fallback only matters for a plain
# `docker run` with no -e PORT=... set, e.g. local testing.
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD curl --fail "http://localhost:${PORT:-8000}/api/health" || exit 1
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --app-dir backend
