# FinMate AI — backend

FastAPI JSON API over `../src/finmate` — the same multi-agent pipeline
`../app.py`'s Streamlit UI calls, exposed as a REST API instead. For the
full architecture and both deploy paths (Render for this app, Vercel for
`../frontend`), see **[`../DEPLOYMENT.md`](../DEPLOYMENT.md)**.

## Local development

```bash
cd backend
cp .env.example .env              # then add GROQ_API_KEY or GEMINI_API_KEY
pip install -r ../requirements.txt -r requirements.txt
uvicorn app.main:app --reload --app-dir .
```

Interactive API docs at [http://localhost:8000/docs](http://localhost:8000/docs)
(FastAPI generates these automatically from `app/api_schemas.py`).

## What's here

```
app/
  main.py           FastAPI app: CORS, startup warm-up, error handling, router registration
  config.py         Environment-driven settings (DB/Qdrant paths, CORS origins, ...)
  deps.py           Shared dependency: pulls the app-lifetime LLM client, or a clean 503
  api_schemas.py     Request/response models -- the API's actual wire contract (see its own docstring for why this is separate from finmate/schemas.py)
  routers/
    health.py         GET /api/health
    chat.py            POST /api/chat
    users.py           GET/DELETE profile+transactions, POST seed-demo-data
```

## Endpoints

| Method | Path | Needs an LLM key? |
|---|---|---|
| `GET` | `/api/health` | No |
| `POST` | `/api/chat` | Only for non-casual messages — see `finmate/casual.py` |
| `GET` | `/api/users/{user_id}/profile` | No |
| `GET` | `/api/users/{user_id}/transactions` | No |
| `DELETE` | `/api/users/{user_id}` | No |
| `POST` | `/api/users/seed-demo-data` | No |

## Tests

The engine's own test suite (`../tests/`, run from the repo root — see
`../README.md`'s "Run the tests") already covers everything this API
wraps. This directory has no separate test suite of its own: the routers
are thin translation layers (HTTP ↔ the already-tested `finmate.*`
functions), verified by hand against a running server rather than with
additional automated tests that would mostly just re-assert what the
engine's tests already guarantee.
