# FinMate

A private, AI-powered personal finance assistant. FinMate helps people understand spending, manage financial details, and plan next steps using their own stored data.

## What it does

- Secure email/password accounts with Supabase Auth
- Persistent profiles, transactions, and chat history
- Natural-language updates such as “update EMI to 30k a month”
- Spending, budget, cash-flow, goals, debt, investment, and anomaly analysis
- Evidence-aware answers backed by stored records and deterministic calculations
- Hybrid retrieval across transaction data

## Stack

- **Frontend:** Next.js, React, Tailwind CSS, Supabase JS
- **Backend:** FastAPI, Python, LangGraph
- **Data & auth:** Supabase Auth and Postgres with Row-Level Security
- **AI:** Groq or Gemini through an OpenAI-compatible client
- **Retrieval:** keyword search, vector search, and reranking when available

## Project structure

```text
frontend/   Next.js chat application and authentication UI
backend/    FastAPI API and Supabase token verification
src/        Finance agents, retrieval, calculations, and persistence layer
supabase/   Database migrations and access-control policies
tests/      Automated backend tests
```

## Local development

1. Copy `frontend/.env.example` to `frontend/.env.local`.
2. Copy `backend/.env.example` to `backend/.env`.
3. Add your Supabase and LLM provider keys.
4. Run the Supabase migration in `supabase/migrations/001_finmate_schema.sql`.
5. Start the backend and frontend:

```bash
uvicorn app.main:app --reload --app-dir backend
cd frontend && npm install && npm run dev
```

For production environment variables and Supabase setup, see [SUPABASE_SETUP.md](SUPABASE_SETUP.md).

## Privacy

Every account is isolated by Supabase Row-Level Security. The API verifies the logged-in user’s token and does not trust a user ID supplied by the browser.
