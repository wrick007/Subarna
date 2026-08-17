# FinMate AI

A multi-agent personal-finance assistant. Locally runnable, no cloud
infrastructure required beyond one **free** LLM API key — Groq or Gemini,
no credit card required.

FinMate AI never guesses a number: every figure you see traces back to a
pure, unit-tested Python function, your stored profile, or a retrieved
transaction — never an LLM doing arithmetic in its head. It also can't move
money, execute a trade, or send anything externally: the only three
"external action" functions in the codebase (`prepare_transfer`,
`prepare_payment`, `prepare_trade`) are stubs that return a
`ProposedAction(status="proposed_pending_confirmation")` object and are not
wired into any automatic code path.

## Two ways to run this

The engine (`src/finmate/`) is the same either way — these differ only in
how you talk to it:

1. **`app.py`** — a single-file Streamlit app. Fastest way to run this
   locally or try the pipeline (`streamlit run app.py`, see "Run the app"
   below). No separate frontend/backend to stand up.
2. **`backend/` + `frontend/`** — a FastAPI JSON API (`backend/`) and a
   Next.js chat UI (`frontend/`) as two independently deployable services —
   the shape you'd want for a real deployed web app with its own domain,
   rather than a Streamlit-hosted demo. See **[`DEPLOYMENT.md`](DEPLOYMENT.md)**
   for local dev and both deploy paths (Vercel + Render).

Both read/write the same `data/finmate.db` by default, so seeding demo data
from one is visible in the other.

## Architecture

```
User → Streamlit chat UI (app.py)
  → Orchestrator (LangGraph StateGraph, finmate/orchestrator.py)

      router ──► pipeline ──► synthesis ──► critic ──► formatter ──► END
                    ▲                          │
                    └────── retry (≤2×) ───────┘

  1. Intent Router          – structured task plan (JSON)
  2. Memory/Profile Agent    – reads/updates the stored profile (SQLite)
  3. Transaction/RAG Agent    – 6-stage hybrid retrieval (metadata filter →
                                 query rewrite → keyword → vector → RRF
                                 fusion → cross-encoder rerank; see below)
  4. Calculation Engine        – deterministic Python, dispatched by metric name
  5. Specialist agents (per intent, via the Stage 14 routing table):
       Budget · Cash-Flow Forecast · Goal Planning · Debt Analysis ·
       Investment Information · Anomaly Detection
  6. Synthesis Agent            – evidence-graded analysis (FACT/CALCULATION/
                                   FORECAST/INTERPRETATION/RECOMMENDATION)
  7. Critic/Verification Agent   – audits the draft; on failure, loops back
                                    to the pipeline node, up to 2 retries
  8. Response Formatter          – final natural-language answer
```

Steps 2–5 are combined into a single LangGraph node (`pipeline`) that
internally dispatches to whichever of memory/rag/calculation/specialists
the Stage 14 routing table says this intent needs — see "Deviations" below
for why.

Persistence:
- **SQLite** (`finmate/db.py`, stdlib `sqlite3`, no ORM) for the structured
  profile (JSON blob, one row per user) and a normalized transactions table.
- **Qdrant in embedded/on-disk mode** (`QdrantClient(path=...)`) for the
  vector half of retrieval, using `sentence-transformers` locally. Both are
  imported lazily (see Deviations) so the core engine works without them.

LLM: the `openai` Python SDK used as an **OpenAI-compatible client**, one
shared wrapper (`finmate/llm.py:LLMClient`), pointed at whichever free-tier
provider you choose — see "LLM provider (free tier)" below. Provider and
model are read from env so either can be swapped without a code change.

## RAG retrieval (Stage 3 hybrid pipeline)

The Transaction/RAG Agent's retrieval (`finmate/rag.py:retrieve`) runs up
to six stages, each independently toggleable and individually
fallback-safe — every stage from 3 onward degrades to the next tier
rather than raising if its dependency is missing, unreachable, or fails:

```
query, filters
   │
   ▼
1. Metadata filter (finmate/db.py:search_transactions) — always runs,
   no ML dependency, no network, no API key. The floor: every later
   stage narrows or reorders this stage's output, never replaces it.
   │
   ▼
2. Query rewrite (OPTIONAL, ≤1 Groq/Gemini call, finmate/query_rewrite.py)
   — expands the query into 2-3 short search-oriented phrasings (e.g.
   "food expenses" -> "dining", "groceries") to help stage 3 match
   transactions with no literal keyword overlap. Cached per (user_id,
   normalized query); skips cleanly on any failure, a missing key, or
   FINMATE_RAG_MODE=no_llm.
   │
   ▼
3. Keyword search (finmate/keyword_search.py) — SQLite FTS5, probed at
   runtime, falling back to rank_bm25, falling back to substring
   scoring. Searches description + category text for every phrasing
   from stage 2 (plus the original query), best score per transaction.
   │
   ▼
4. Dense vector search (Qdrant + sentence-transformers, reused from the
   pre-upgrade implementation) — scoped to the metadata-filtered
   candidate set via a Qdrant payload filter (see "Deviations" below).
   │
   ▼
5. Fusion (finmate/reranker.py:reciprocal_rank_fusion) — Reciprocal Rank
   Fusion of the keyword and vector rankings. Never a raw score blend:
   BM25 magnitude and cosine similarity live on incompatible scales, so
   blending them directly would let whichever happens to have the
   larger numeric range dominate for reasons that have nothing to do
   with relevance. RRF uses only rank position, which is scale-free.
   │
   ▼
6. Local cross-encoder rerank (finmate/reranker.py:cross_encoder_rerank,
   cross-encoder/ms-marco-MiniLM-L-6-v2) of the fused top-30 against the
   (possibly-rewritten) query.
   │
   ▼
top_k EvidenceItems, each carrying keyword_score / vector_score /
rerank_score / retrieval_stage so every ranking decision is auditable.
```

**Fallback ladder** — `RetrievalResult.stage` reports exactly which tier a
given answer's evidence actually came from, and `.note` explains why any
stage didn't run:

```
full hybrid (rewrite+keyword+vector+rerank)
keyword+vector+rerank (no query rewrite)
keyword+vector, no rerank
vector+rerank (no keyword)   ─┐ reachable, but not obviously implied by a
keyword+rerank (no vector)   ─┘ simple "keyword→vector→rerank" reading —
                                rerank runs on whatever fusion produced,
                                independent of which single signal fed it
vector only
keyword only
metadata-only, recency-ordered   (the floor — never removed)
```

`RetrievalResult` keeps the existing `vector_search_used: bool` / `note:
str` fields other agents already read (via `finmate/agents/_shared.py`)
and adds `keyword_search_used`, `rerank_used`, `query_rewrite_used`, and
`stage` — additive, nothing removed.

**`FINMATE_RAG_MODE=no_llm`** (env var) disables stage 2 outright, with no
client ever constructed. Every other stage is controlled per-call via
`retrieve()`'s `enable_keyword_search` / `enable_vector_search` /
`enable_rerank` / `enable_query_rewrite` parameters (all default to "on,
auto-degrades if unavailable") — see `finmate/rag.py:retrieve`'s
docstring.

### Performance: speed & token cost

Same providers (Groq/Gemini), same ranking output — `scripts/eval_rag.py`'s
Recall/MRR numbers below are unchanged before/after this work, since none
of it touches *what* gets ranked or *how*, only how expensively. Full
detail lives in each module's docstring; summary:

| Change | Where | Effect |
|---|---|---|
| Model/client singletons | `finmate/rag.py`, `finmate/reranker.py` | The embedder, the Qdrant client, and the cross-encoder each loaded fresh on *every single retrieval call* before this change — by far the slowest thing in the pipeline. Now cached per process (thread-safe, including a failed load, so an unreachable model fails fast instead of re-attempting a network fetch each time). |
| Concurrency | `finmate/rag.py:retrieve` | Vector search depends only on the query text, never on query-rewrite's output, so it now runs on a background thread starting *before* query rewrite (an LLM network call) even begins, instead of waiting behind it. Wall-clock cost drops from roughly `rewrite + keyword + vector` to roughly `max(vector, rewrite + keyword)`. |
| Batched keyword search | `finmate/keyword_search.py:keyword_rank_multi` | Ranking the candidate set against the original query *and* every query-rewrite phrasing used to rebuild the FTS5/BM25 index from scratch per phrasing (up to 4x). Now built once, queried once per phrasing. |
| Request-scoped retrieval cache | `finmate/rag.py:retrieve(cache=...)` | Off by default. The orchestrator passes a fresh dict per user turn, so a Critic-triggered retry — which re-runs this exact pipeline stage for the exact same query — hits an in-memory cache instead of redoing keyword+vector+rerank (and a second query-rewrite call) from scratch. |
| Shared LLM client for query rewrite | `finmate/orchestrator.py` → `finmate/agents/rag_agent.py` | Query rewrite used to construct its own `LLMClient` (a second OpenAI-SDK client, no connection reuse) instead of reusing the one every other agent in the turn already has. |
| Compact, trimmed prompt JSON | `finmate/agents/_shared.py`, `critic.py`, `synthesis.py`, `cashflow.py` | Evidence sent to every specialist agent, synthesis, and the critic is serialized compactly (not pretty-printed) and trimmed to the fields an LLM actually needs — audit-only fields (`keyword_score`/`vector_score`/`rerank_score`/`retrieval_stage`, and `page`, which this app's transaction evidence never populates) stay on `RetrievalResult`/`EvidenceItem` for the API and UI, just not in the tokens billed to Groq/Gemini. |
| Skip the constitution for query rewrite | `finmate/llm.py:LLMClient.call(include_constitution=...)` | The ~400-token CONSTITUTION governs how FinMate talks to *the user* and handles *their* data — query rewrite does neither (it only turns their own question into a few search phrasings), so it's the one call site that opts out. |

None of this changes the fallback ladder, the ranking algorithm, or the
LLM provider — it changes how many times a model gets loaded, how many
stages wait on each other unnecessarily, and how many tokens a fixed set
of facts costs to describe. See `tests/test_rag_performance.py` and
`tests/test_shared_context.py` for the tests that pin this down (e.g.
"index built once, not per phrasing" and "no behavior change" are both
asserted directly, not just implied by the eval numbers staying flat).

### Evaluate retrieval quality

```bash
python scripts/eval_rag.py                     # keyword+vector+fusion+rerank, no LLM calls
python scripts/eval_rag.py --with-query-rewrite # also exercise stage 2 (needs a configured key)
```

Seeds an isolated copy of the demo data at `data/eval_finmate.db` /
`data/eval_qdrant_store` (never touches your real `data/finmate.db` /
`data/qdrant_store`), runs it against 19 hand-labeled queries in
`data/rag_eval_set.json`, and prints Recall@5, Recall@10, and MRR —
macro-averaged across queries and broken down by query type — for both
the new pipeline and a frozen snapshot of the pre-upgrade one, so the
improvement is a measured number, not an assertion.

**Numbers from this build** (this sandbox could reach PyPI but not
Hugging Face Hub or the Groq/Gemini APIs — see "Deviations" for exactly
what that does and doesn't affect):

| | Recall@5 | Recall@10 | MRR |
|---|---|---|---|
| **Before** (metadata filter + single-pass vector rerank) | 0.216 | 0.359 | 0.318 |
| **After** (keyword + RRF fusion + rerank; vector/rewrite unavailable here) | **0.663** | **0.767** | **0.767** |

By query type (after / before):

| query type | n | Recall@5 | Recall@10 | MRR |
|---|---|---|---|---|
| exact_keyword | 7 | 1.000 / 0.071 | 1.000 / 0.071 | 1.000 / 0.143 |
| category_date_filter | 6 | 0.671 / 0.505 | 0.782 / 0.741 | 0.917 / 0.660 |
| anomaly | 3 | 0.333 / 0.000 | 0.667 / 0.333 | 0.375 / 0.042 |
| semantic | 3 | 0.189 / 0.189 | 0.295 / 0.295 | 0.317 / 0.317 |

The `semantic` row is flat before vs. after **on purpose**: those 3
queries ("food expenses", "transportation costs", "online retail
purchases") have zero literal keyword overlap with their target
transactions by design — finding them needs dense vector search or query
rewrite, and this build's sandbox could reach neither (no route to
Hugging Face Hub to download `all-MiniLM-L6-v2`, no route to
Groq/Gemini). `exact_keyword` reaching a clean 1.000 and
`category_date_filter`/`anomaly` improving substantially is the real,
measured signal that the fusion/rerank wiring itself is correct. Re-run
with real internet access and a configured key to see the `semantic` row
move too.

### RAG upgrade — deviations, and why

*(Scoped to this hybrid-retrieval upgrade specifically. Two other "see
Deviations below" references earlier in this README — the pipeline-node
combination and the lazy-import rationale — point to a broader
deviations write-up from an earlier stage of this project that isn't
part of this README snapshot.)*

- **Two real bugs, found by actually running the pre-upgrade code, not
  just theoretical robustness improvements:**
  1. `_get_embedder`/`_get_qdrant_client` previously only guarded the
     *import* of `sentence-transformers`/`qdrant-client`
     (`except ImportError`), not *construction*. If the package is
     installed but the model can't be downloaded — verified as a real
     failure mode in this build's sandbox, which can reach PyPI but not
     Hugging Face Hub — `SentenceTransformer(model_name)` raises `OSError`
     *outside* the old try/except and crashes `retrieve()` entirely,
     contradicting the documented "falls back if unavailable" contract.
     Construction is now guarded too.
  2. The pre-upgrade `retrieve()` already called `client.query_points(...)`
     (correctly avoiding `.search()`, which `qdrant-client` has since
     removed) but with **no payload filter** — it fetched a fixed number
     of *globally*-ranked hits and filtered down to the metadata-filtered
     candidate set afterward, in Python. For a small candidate set inside
     a larger collection, a relevant item can be missed entirely if it
     doesn't fall within the top-`limit` global hits, even when it would
     be the best match *within* the candidate set. Fixed by moving the
     filter into the query itself (a Qdrant payload filter, `source_id`
     `MatchAny` the candidate IDs) — proven with a deterministic
     regression test
     (`tests/test_rag_hybrid.py::test_vector_search_finds_target_even_crowded_by_a_tiny_global_limit`,
     `limit=1` against 5 identically-scored out-of-candidate-set "noise"
     points), not just asserted.
- **This build's sandbox could reach PyPI but not Hugging Face Hub or the
  Groq/Gemini APIs** (verified empirically, not assumed). `rank-bm25`,
  `qdrant-client`, and `sentence-transformers` are genuinely installed and
  exercised in this build's test suite (keyword search, RRF fusion, and
  the Qdrant wiring above are verified against real code, real embedded
  Qdrant); the embedding model and cross-encoder weights could not be
  downloaded, so those two stages were verified with fake/mocked scores
  instead (unit tests, no model download needed) plus their real
  graceful-fallback path, which is exactly the condition this sandbox is
  actually in.
- **`rank_bm25`'s classic IDF formula can legitimately score every term
  as exactly 0.0** for a genuine match in a small enough corpus (verified
  directly against the installed package: a term appearing in exactly
  half of a 2-document corpus gets `idf = log(1.5/1.5) = 0`). A candidate
  set that small is the normal case here, not an edge case, so
  `finmate/keyword_search.py` filters on literal token overlap instead of
  `score > 0`.
- **Query-rewrite phrasings feed the keyword stage only, not vector
  search** — dense embeddings already generalize semantically; keyword
  search can only match text actually present in a transaction, so it
  benefits far more.
- **The query-rewrite cache is an in-process Python dict**, not persisted
  to SQLite — a restart clears it, which is an acceptable trade-off
  against adding a general-purpose KV table to a schema that's otherwise
  purposely narrow (profile + transactions).
- **Eval metrics are macro-averaged** across queries (each counts
  equally) rather than micro-averaged, since `semantic_food_expenses` has
  17 gold items by design versus 2 for most `exact_keyword` queries.

## LLM provider (free tier)

FinMate AI doesn't use the Anthropic/OpenAI APIs directly — it uses the
`openai` SDK purely as an *OpenAI-compatible client*, pointed at a
provider's OpenAI-compatible endpoint. Both of the built-in options are
genuinely free with no credit card:

| Provider | Get a key | Env vars | Default model |
|---|---|---|---|
| **Groq** (default) | [console.groq.com](https://console.groq.com) | `GROQ_API_KEY` | `llama-3.3-70b-versatile` |
| **Gemini** | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) | `GEMINI_API_KEY` | `gemini-2.5-flash` |

Pick one in `.env` (see `.env.example`):

```bash
FINMATE_LLM_PROVIDER=groq        # or "gemini"
GROQ_API_KEY=gsk_...             # or GEMINI_API_KEY=AIza...
```

You can also point at any other OpenAI-compatible endpoint with
`FINMATE_LLM_PROVIDER=custom`, `FINMATE_BASE_URL`, `FINMATE_API_KEY`, and
`FINMATE_MODEL`. Override just the model on either built-in provider with
`FINMATE_MODEL` (free-tier model lineups change over time — check the
provider's current docs if a default here has been retired).

**Rate limits.** Free tiers are gated by requests-per-minute, not a credit
balance (Groq: ~30 RPM; Gemini 2.5 Flash: ~15 RPM). A single FinMate
question can trigger 4–11 LLM calls in a row (router → memory/specialists →
synthesis → critic → formatter, plus up to 2 critic-triggered retries), so
`finmate/llm.py` retries on HTTP 429 with exponential backoff before
raising. If you're seeing frequent 429s, wait a few seconds between
questions or switch provider.

## Prompt → file/function map

| Stage | Prompt constant | Owned by |
|---|---|---|
| 0 | `CONSTITUTION` | `finmate/prompts.py`; prepended to every call in `finmate/llm.py:LLMClient.call` |
| 1 | `ROUTER` | `finmate/agents/router.py:run_router` |
| 2 | `MEMORY_AGENT` | `finmate/agents/memory.py:run_memory_agent` / `apply_memory_action` |
| 3 | `RAG_AGENT` | `finmate/agents/rag_agent.py:run_retrieval` (documents the contract; implemented deterministically) → `finmate/rag.py:retrieve` (6-stage hybrid pipeline) → `keyword_search.py` / `query_rewrite.py` / `reranker.py` |
| 4 | `CALCULATION_AGENT` | `finmate/tools.py:calculate_metric` + `finmate/agents/calculation.py:run_calculations` (documents the contract; pure Python, never an LLM call, per spec) |
| 5 | `BUDGET_AGENT` | `finmate/agents/budget.py:run_budget_agent` |
| 6 | `CASHFLOW_AGENT` | `finmate/agents/cashflow.py:run_cashflow_agent` (narrates `tools.forecast_cash_flow`'s BASE/CONSERVATIVE/STRESS output) |
| 7 | `GOAL_AGENT` | `finmate/agents/goal.py:run_goal_agent` |
| 8 | `DEBT_AGENT` | `finmate/agents/debt.py:run_debt_agent` |
| 9 | `INVESTMENT_AGENT` | `finmate/agents/investment.py:run_investment_agent` |
| 10 | `ANOMALY_AGENT` | `finmate/agents/anomaly.py:run_anomaly_agent` |
| 11 | `SYNTHESIS_AGENT` | `finmate/agents/synthesis.py:run_synthesis_agent` |
| 12 | `CRITIC_AGENT` | `finmate/agents/critic.py:run_critic_agent` |
| 13 | `FORMATTER_AGENT` | `finmate/agents/formatter.py:run_formatter_agent` |
| 14 | `ROUTING_TABLE` (plain dict) | `finmate/prompts.py`; consumed by `finmate/orchestrator.py:_node_pipeline` |
| 15 | tool contracts | `finmate/tools.py` (calculations, `create_budget`, `prepare_*` stubs), `finmate/db.py` (profile/transaction lookups), `finmate/rag.py` (`retrieve`, `index_transactions_for_user`) |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # then edit .env: pick a provider and add its free key
```

Get a free key from [console.groq.com](https://console.groq.com) (Groq,
the default) or [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
(Gemini) — see "LLM provider (free tier)" above for details.

`sentence-transformers` pulls in PyTorch and can take a while / a few GB of
disk on first install. If you only want to run the deterministic engine and
tests, you can skip it. `rank-bm25` (added in the RAG upgrade) is pure
Python and tiny by comparison — it's only used as a fallback if your
`sqlite3` build lacks the FTS5 extension, which most modern Python
installs have.

## Seed demo data

```bash
python scripts/seed_demo_data.py
```

Loads `data/synthetic_profile.json` and `data/synthetic_transactions.csv`
into SQLite for `user_id = "demo_user"`, and best-effort builds the vector
index. Two months of realistic transactions, with three planted anomalies
for the Anomaly Agent to find:

1. A subscription price jump (Netflix ₹499 → ₹649 between June and July).
2. A duplicate-looking charge (two identical "Zomato - dinner order" ₹650
   charges on the same day, 2026-07-06).
3. One unusually large one-off transaction (a ₹18,500 MacBook repair).

## Run the app

```bash
streamlit run app.py
```

Open the sidebar, set User ID to `demo_user` (or seed your own), and ask
things like:
- "What's my savings rate this month?"
- "Am I overspending on dining out?"
- "Do I have any duplicate or unusual charges?"
- "How long until I hit my Japan trip goal?"
- "What would happen if I paid off the credit card balance first vs the car loan?"

## Run the tests

```bash
pytest
```

`tests/test_tools.py` and `tests/test_db.py` cover the deterministic
calculation engine and the SQLite layer — **zero network access, no API
key required**. `tests/test_rag.py` covers the pre-existing metadata-filter
path (still passing, unchanged). `tests/test_keyword_search.py`,
`tests/test_reranker.py`, `tests/test_query_rewrite.py`, and
`tests/test_rag_hybrid.py` cover the hybrid pipeline added in the RAG
upgrade — the fallback chain at every level, RRF fusion math,
cross-encoder rerank fallback, query-rewrite caching/skip-on-failure
(mocked `LLMClient`, no real call), and the Qdrant candidate-set-filter
fix (real embedded on-disk Qdrant, fake dependency-free embeddings — no
model download). **All 128 tests pass with only `pydantic`, `pytest`, and
the stdlib installed** — verified in a clean virtualenv with
`sentence-transformers`, `qdrant-client`, `rank-bm25`, and `openai` all
absent; the handful of tests that specifically test one of those
packages' integration skip themselves rather than failing when it's not
installed.

See "RAG retrieval" above for how to run `scripts/eval_rag.py` and the
measured before/after retrieval-quality numbers.

## Deployment

**Deploying the FastAPI + Next.js version (`backend/` + `frontend/`) as a
real web app with its own domain?** See **[`DEPLOYMENT.md`](DEPLOYMENT.md)**
— step-by-step Render (backend) + Vercel (frontend) instructions, plus
alternatives, cost/persistence tradeoffs, and troubleshooting.

Everything below is specifically about deploying `app.py`, the Streamlit
app, on its own.

FinMate AI is a stateful Streamlit app: both its databases
(`data/finmate.db` — SQLite, and `data/qdrant_store` — embedded Qdrant)
are **plain files on local disk**, not a managed service. That one fact
drives every choice below — pick based on whether you need that data to
survive a restart/redeploy.

### Option A — Streamlit Community Cloud (fastest, free, demo-friendly)

Good for sharing a working demo. **Not** good for real persistent user
data — see the warning below.

1. Push this repo to GitHub (public or private).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** →
   pick the repo/branch → set **Main file path** to `app.py` → **Deploy**.
3. Before or after deploying, open **App settings → Secrets** and paste
   your provider key(s) in TOML form, matching `.env.example`:
   ```toml
   FINMATE_LLM_PROVIDER = "groq"
   GROQ_API_KEY = "gsk_..."
   ```
   Streamlit exposes root-level secrets as environment variables
   automatically, so `finmate/llm.py:resolve_provider_config` (which
   reads `os.environ`) picks these up with no code changes.
4. `requirements.txt` is auto-installed; no extra config needed. First
   deploy is slower because of `sentence-transformers`/PyTorch — that's
   expected.
5. Seed demo data once, from a terminal on your machine, against the
   *same* repo you deployed (Community Cloud has no shell access):
   this only matters if you're relying on step 6 below instead of an
   in-app seed button.
6. **⚠️ Ephemeral filesystem.** Community Cloud containers are recreated
   on redeploy, and may also restart on their own after inactivity —
   `data/finmate.db` and `data/qdrant_store` do **not** persist across
   that. Either treat this as a demo that reseeds itself on boot (e.g.
   call `seed_demo_data.py`'s logic at the top of `app.py` if the DB file
   doesn't exist yet — not wired in by default, to avoid silently
   overwriting real data on platforms where the disk *does* persist), or
   use Option B/C with a real mounted volume for anything meant to keep
   a user's actual data.

### Option B — Docker (portable, works anywhere that runs a container)

A ready-to-use `Dockerfile` and `.dockerignore` are included in the repo
root (same pattern as [Streamlit's own Docker deployment
guide](https://docs.streamlit.io/deploy/tutorials/docker)):

```bash
docker build -t finmate-ai .
docker run -d --name finmate-ai -p 8501:8501 \
  --env-file .env \
  -v "$(pwd)/data:/app/data" \
  finmate-ai
docker exec -it finmate-ai python scripts/seed_demo_data.py   # run once
```

The `-v "$(pwd)/data:/app/data"` volume mount is what makes data survive
a `docker restart`/redeploy — omit it and you're back to Option A's
ephemeral behavior, just in container form. This same image deploys as-is
to Render, Railway, Fly.io, or any VPS that runs Docker; attach a
persistent volume at `/app/data` on whichever platform you use.

### Option C — Plain VPS with systemd (full control, own domain/HTTPS)

1. `git clone` the repo on the server, then the same `Setup` steps above
   (venv, `pip install -r requirements.txt`, `.env`, seed demo data).
2. Create `/etc/systemd/system/finmate-ai.service`:
   ```ini
   [Unit]
   Description=FinMate AI
   After=network.target

   [Service]
   Type=simple
   User=finmate
   WorkingDirectory=/opt/finmate-ai
   EnvironmentFile=/opt/finmate-ai/.env
   ExecStart=/opt/finmate-ai/.venv/bin/streamlit run app.py --server.port=8501 --server.address=127.0.0.1
   Restart=on-failure

   [Install]
   WantedBy=multi-user.target
   ```
3. `sudo systemctl enable --now finmate-ai`
4. Put nginx or Caddy in front for HTTPS + your domain, proxying to
   `127.0.0.1:8501` (Caddy gets you a free auto-renewing cert with a
   3-line Caddyfile: `your-domain.com { reverse_proxy 127.0.0.1:8501 }`).

Since everything already lives on local disk under the app's working
directory, this option needs no extra persistence configuration — normal
filesystem backups of `data/` are enough.

