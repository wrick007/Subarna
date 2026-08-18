# Deploying FinMate AI

This app is two deployable pieces talking to each other over HTTPS:

```
┌─────────────────────────┐         HTTPS, JSON          ┌──────────────────────────────┐
│   frontend/  (Next.js)  │ ────────────────────────────▶ │   backend/  (FastAPI)         │
│   → Vercel               │ ◀──────────────────────────── │   → Render                    │
│                           │                                │                                │
│  Chat UI, evidence panel │        NEXT_PUBLIC_API_URL     │  src/finmate/ (the actual      │
│  spending snapshot        │        FINMATE_FRONTEND_ORIGINS│  multi-agent engine) + SQLite  │
└─────────────────────────┘                                │  + optional embedded Qdrant    │
                                                              └──────────────────────────────┘
                                                                          │
                                                                          ▼
                                                              Groq or Gemini (your API key —
                                                              same provider as before, see
                                                              README.md "LLM provider")
```

Vercel and Render are the two named in this guide because they're a genuine
good fit, not just familiar names: Vercel is where Next.js deploys with
zero configuration, and Render runs an arbitrary Docker container with a
persistent disk — which this backend needs, since its dependencies
(`sentence-transformers`, `torch`, an embedded Qdrant index) are too heavy
for Vercel's serverless functions (function size and execution-time limits
that a torch-based ML backend routinely exceeds). "Anything else" that also
fits the backend's actual requirements — a Dockerfile, a process listening
on `$PORT`, optionally a persistent disk — is covered in
["Alternatives to Render"](#alternatives-to-render) below: Railway and
Fly.io both work with the same `backend/Dockerfile` essentially unchanged.

**Nothing here changes your LLM provider.** You still bring your own free
Groq or Gemini API key exactly as README.md's "LLM provider" section
describes — deployment only decides *where the code runs*, not which model
answers.

---

## Contents

1. [Before you start](#before-you-start)
2. [Part 1 — Deploy the backend to Render](#part-1--deploy-the-backend-to-render)
3. [Part 2 — Deploy the frontend to Vercel](#part-2--deploy-the-frontend-to-vercel)
4. [Part 3 — Connect them (CORS)](#part-3--connect-them-cors)
5. [Part 4 — Smoke test](#part-4--smoke-test)
6. [Run locally](#run-locally)
7. [Custom domains](#custom-domains)
8. [Alternatives to Render](#alternatives-to-render)
9. [Persistence, cost, and what's realistic](#persistence-cost-and-whats-realistic)
10. [Troubleshooting](#troubleshooting)

---

## Before you start

- A GitHub (or GitLab/Bitbucket) repo containing this project — Render and
  Vercel both deploy by connecting to a repo, not by file upload.
- A free API key from **[Groq](https://console.groq.com)** or
  **[Google AI Studio](https://aistudio.google.com/apikey)** (Gemini) — no
  credit card required for either, per README.md.
- Free accounts on **[render.com](https://render.com)** and
  **[vercel.com](https://vercel.com)** (both offer GitHub sign-in).

You'll deploy the backend first — the frontend needs its URL.

---

## Part 1 — Deploy the backend to Render

### Option A: Blueprint (fastest)

This repo includes `render.yaml` at the root, which Render reads
automatically.

1. Push this repo to GitHub (with `render.yaml` at the root).
2. In the Render Dashboard: **New +** → **Blueprint** → select your repo.
3. Render reads `render.yaml` and shows you one service, `finmate-backend`,
   with one prompt: your `GROQ_API_KEY` (that field is `sync: false` in the
   blueprint, meaning Render always asks rather than ever storing it in a
   file you might commit). Paste your key.
4. Click **Apply**. First build takes several minutes — it's installing
   `torch`/`sentence-transformers`, not a small dependency set.
5. Once live, note the service URL Render assigns, e.g.
   `https://finmate-backend.onrender.com`. You'll need it in Part 2.

`render.yaml` provisions the **Starter** plan (~$7/mo) with a 1GB
persistent disk, so `data/finmate.db` and the vector index survive
restarts. If you'd rather start free and accept that demo data resets on
every redeploy, open `render.yaml`, delete the `disk:` block, and change
`plan: starter` to `plan: free` before applying the blueprint — see
["Persistence, cost, and what's realistic"](#persistence-cost-and-whats-realistic)
for the actual tradeoff, not just the one-line summary.

### Option B: Manual dashboard setup

If you'd rather configure it by hand (or don't want a `render.yaml` in your
repo):

1. **New +** → **Web Service** → connect your repo.
2. **Runtime**: Docker.
3. **Dockerfile Path**: `backend/Dockerfile`
4. **Docker Build Context Directory**: `.` (repo root — the Dockerfile
   needs both `src/` and `backend/`, so it can't build from `backend/`
   alone; see the comment at the top of `backend/Dockerfile`).
5. **Plan**: Starter (or Free — see the tradeoff note above).
6. **Health Check Path**: `/api/health`
7. **Environment Variables** (add each of these):

   | Key | Value |
   |---|---|
   | `FINMATE_LLM_PROVIDER` | `groq` (or `gemini`) |
   | `GROQ_API_KEY` | your key (or `GEMINI_API_KEY` if using Gemini) |
   | `FINMATE_FRONTEND_ORIGINS` | `https://your-frontend.vercel.app` (placeholder for now — you'll update this in Part 3) |
   | `FINMATE_DB_PATH` | `/app/data/finmate.db` |
   | `FINMATE_QDRANT_PATH` | `/app/data/qdrant_store` |
   | `FINMATE_WARM_UP_ON_STARTUP` | `true` |

8. If you want persistence (Starter plan or higher): **Add Disk**, mount
   path `/app/data`, size 1GB.
9. **Create Web Service**.

### Verify the backend deployed correctly

```bash
curl https://YOUR-BACKEND-URL.onrender.com/api/health
```

Expect something like:

```json
{"status": "ok", "llm_configured": true, "provider": "groq", "model": "...", "warm_up": {"embedder": true, "qdrant_client": true, "cross_encoder": true}}
```

`"status": "ok"` even with `"llm_configured": false` is normal and by
design (see `backend/app/routers/health.py`'s docstring) — it means the
process is healthy but no API key was found; fix the env var and it'll
pick it up on the next restart. If `warm_up` shows `false` for
`embedder`/`cross_encoder`, that's `sentence-transformers` not finding its
model weights (rare on Render, which has normal internet access during
build/runtime) — chat still works via the metadata-filter + keyword-search
fallback described in README.md's "RAG retrieval" section, just without
semantic vector search.

---

## Part 2 — Deploy the frontend to Vercel

1. In the Vercel Dashboard: **Add New** → **Project** → import the same
   repo.
2. **Root Directory**: click **Edit** and set it to `frontend`. This is the
   one setting that matters for a monorepo like this — Vercel then treats
   `frontend/` as if it were the whole repo, auto-detects Next.js, and gets
   the build command/output directory right with no further config.
3. **Environment Variables**: add
   `NEXT_PUBLIC_API_URL` = `https://YOUR-BACKEND-URL.onrender.com` (the
   Render URL from Part 1, **no trailing slash**).
4. **Deploy**.

Vercel gives you a URL like `https://finmate-ai.vercel.app` (or a
project-specific `*.vercel.app` subdomain — exact naming depends on what's
available/your project name).

---

## Part 3 — Connect them (CORS)

The backend only accepts browser requests from origins listed in
`FINMATE_FRONTEND_ORIGINS` (see `backend/app/config.py` and
`backend/app/main.py`'s CORS middleware) — deliberately no wildcard `"*"`
in production, since this API reads and writes a specific `user_id`'s
financial data.

1. Copy your real Vercel URL from Part 2.
2. Back in the Render Dashboard → your backend service → **Environment**:
   update `FINMATE_FRONTEND_ORIGINS` to that exact URL (comma-separate
   multiple origins if you have a custom domain too — see
   ["Custom domains"](#custom-domains)).
3. Render redeploys automatically when you save an env var change.

---

## Part 4 — Smoke test

Open your Vercel URL. You should see the chat UI with a sidebar. If the
backend is reachable, the sidebar shows "Backend connected"; if not, a
banner explains what's wrong (usually a CORS mismatch from skipping Part 3,
or a typo in `NEXT_PUBLIC_API_URL`).

1. Click **Load demo data** — seeds the same synthetic profile + 48
   transactions `scripts/seed_demo_data.py` always seeds (`user_id
   "demo_user"`), so there's something to ask about immediately.
2. Try one of the suggested questions, or ask your own — e.g. *"What's my
   savings rate this month?"*
3. You should get a real answer with a **Verified** strip underneath it —
   click it to see exactly which transactions and calculations it's
   grounded in (this is the pipeline-transparency feature described in
   `frontend/components/VerifiedStrip.tsx`, not decoration).

If step 3 instead shows a "chat is unavailable" error, your API key isn't
configured correctly on Render — recheck `GROQ_API_KEY`/`GEMINI_API_KEY` in
Part 1.

---

## Run locally

Two terminals, from the repo root:

```bash
# Terminal 1 — backend
cd backend
cp .env.example .env              # then add your GROQ_API_KEY or GEMINI_API_KEY
pip install -r ../requirements.txt -r requirements.txt
uvicorn app.main:app --reload --app-dir .

# Terminal 2 — frontend
cd frontend
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) — the frontend's
`.env.example` already defaults `NEXT_PUBLIC_API_URL` to
`http://localhost:8000`, and the backend's default
`FINMATE_FRONTEND_ORIGINS` already includes `http://localhost:3000`
(`backend/app/config.py`), so local dev needs no CORS configuration at all.

This is a **third way to run the project**, alongside the Streamlit app
(`streamlit run app.py`, see README.md's own "Run the app" section) and the
test suite — all three read/write the same `data/finmate.db` by default, so
seeding demo data from one is visible in the others.

---

## Custom domains

**Vercel**: Project → Settings → Domains → add your domain, follow the DNS
instructions shown (a CNAME to `cname.vercel-dns.com`, or Vercel's
nameservers if you want them to manage DNS). HTTPS is automatic.

**Render**: Service → Settings → Custom Domains → add your domain, add the
CNAME Render shows you. HTTPS is automatic (Let's Encrypt).

Either way, after adding a custom domain: update `NEXT_PUBLIC_API_URL` on
Vercel and `FINMATE_FRONTEND_ORIGINS` on Render to the new domains, and
redeploy both (Part 3's CORS step, just with real domains instead of the
default `*.vercel.app`/`*.onrender.com` ones).

---

## Alternatives to Render

The backend is a standard Docker image reading `$PORT` at runtime (see
`backend/Dockerfile`) — nothing Render-specific. Two platforms that work
with the same Dockerfile, in case Render isn't the right fit for you:

- **[Railway](https://railway.app)**: connect the repo, set the same
  environment variables from Part 1's table, set **Root Directory** to `.`
  (repo root) and point it at `backend/Dockerfile` in the service's build
  settings. Railway attaches persistent volumes similarly to Render's disks.
- **[Fly.io](https://fly.io)**: `fly launch` from the repo root, pointing
  `fly.toml`'s `dockerfile` at `backend/Dockerfile`; `fly volumes create`
  for persistence, mounted at `/app/data`. More manual than Render/Railway
  (no dashboard-driven setup), but gives you the most control, including
  region-pinning close to your users.

For the frontend, any static-hosting-plus-Node platform that supports
Next.js works the same way Vercel does (Netlify and Cloudflare Pages both
have first-class Next.js support) — Vercel is simplest because it's built
by the same team as Next.js itself, not because the app needs anything
Vercel-specific.

**Streamlit Community Cloud** (README.md's existing "Option A") is still
the fastest path if you just want to demo the underlying engine without
standing up a separate frontend/backend at all — `app.py` is unaffected by
anything in this guide and keeps working exactly as before.

---

## Persistence, cost, and what's realistic

Being direct about tradeoffs rather than just listing steps:

| | Render Free | Render Starter (~$7/mo) |
|---|---|---|
| Persistent disk | **Not available at all** — Render requires Starter or higher | Yes (uncomment the `disk:` block in `render.yaml`, and set `plan: starter`) |
| What that means here | `data/finmate.db` and the vector index (if built) live only on the container's local, ephemeral disk — they reset on every redeploy, restart, or free-tier spin-down/spin-up. There is currently **no durable hosting for the database** on the free plan; it isn't "somewhere else," it's just a file inside the container that gets wiped. Fine for a portfolio/demo deployment — re-run "Load demo data" (or `POST /api/users/seed-demo-data`) afterward | Your data (and anyone else's, if you add real users) survives restarts |
| Spin-down | After ~15 min idle; next request pays a ~30-60s cold start | Always on |
| Memory | 512MB | 512MB on the base Starter instance — see note below |

**Memory: `render.yaml` defaults to `ENABLE_RERANKER=false` specifically
because of this.** Cross-encoder reranking (`finmate/rag.py` stage 6) is
the one piece of this pipeline heavy enough to matter on a 512MB
container: the embedder alone loads fine, but loading the cross-encoder
*on top of it* at startup is what actually gets the process OOM-killed on
Render free — the exact failure mode this default now avoids. Retrieval
still works fully with reranking off: metadata filter + keyword search +
vector search + RRF fusion all still run, per the fallback ladder in
README.md's "RAG retrieval" section — you lose the cross-encoder's extra
accuracy pass specifically, not retrieval itself. If you move to a plan
with memory to spare and want that accuracy back, set `ENABLE_RERANKER=true`
(README.md's "RAG retrieval" section documents exactly what turning it on
guarantees).

**Vector search is separately optional too** — if `sentence-transformers`
still can't load even with reranking off (unlikely at 512MB with just the
embedder, but possible under other memory pressure), retrieval
automatically falls back to metadata-filtering + keyword search. Chat
still works; you lose semantic ("similar meaning, different words")
matching specifically.

None of this is a reason to avoid Render's free tier for trying things
out — it's a reason to know what you're trading for $0, so a cold start,
a reset dataset, or reranking being off doesn't look like a bug.

---

## Troubleshooting

**Sidebar shows "Backend unreachable"**
Open your browser's dev tools → Network tab, reload. A CORS error in the
console means Part 3 wasn't completed (or `FINMATE_FRONTEND_ORIGINS`
doesn't exactly match your Vercel URL — check for a trailing slash
mismatch, and that you redeployed after changing it). A failed/timed-out
request instead usually means `NEXT_PUBLIC_API_URL` is wrong, or the
backend is still spinning up from a cold start (free tier) — wait ~30-60s
and reload.

**Chat says "Chat is unavailable: No API key found..."**
`GROQ_API_KEY` (or `GEMINI_API_KEY` + `FINMATE_LLM_PROVIDER=gemini`) isn't
set correctly on Render. Recheck Part 1 step 7, save, wait for the
automatic redeploy.

**A greeting like "hi" works but real questions don't**
That's actually meaningful signal, not a coincidence — see
`finmate/casual.py`'s docstring: greetings take a zero-LLM-call fast path
by design, so they work even without a configured provider. If greetings
work but nothing else does, it confirms the issue is specifically the LLM
key, not general backend connectivity.

**First message after a while is slow**
Free-tier cold start (~30-60s) or, on any plan, the very first request
after a fresh deploy while `warm_up` finishes loading models — both
expected, see `finmate/rag.py:warm_up`'s docstring and this file's
persistence table above.

**Backend gets killed / restarts during startup on Render Free ("Out of
memory")**
This was the cross-encoder reranker loading on top of the embedder at
startup and exceeding the free tier's 512MB — `render.yaml` now sets
`ENABLE_RERANKER=false` by default specifically to avoid it (see
"Persistence, cost, and what's realistic" above). If you're seeing this
on a repo checked out before that default existed, add
`ENABLE_RERANKER=false` (or just remove any `ENABLE_RERANKER=true` you
set) to the service's environment variables on Render and redeploy. If
it's *still* OOMing with reranking off, that means even the embedder
alone plus FastAPI plus Qdrant's embedded index doesn't fit in 512MB in
your case — at that point the fix genuinely is a bigger instance type,
not a further code change.

**Render build fails or times out**
`torch`/`sentence-transformers` make this a heavier build than a typical
FastAPI service — a first build taking 5-10 minutes is normal, not stuck.
If it genuinely fails, check that **Docker Build Context Directory** is
set to `.` (repo root), not `backend/` — the Dockerfile's first comment
block explains exactly why that distinction matters.

**Vercel build fails on fonts**
`next/font/google` (used for the three type roles described in
`frontend/README.md`) needs to reach `fonts.googleapis.com` at build time.
Vercel's build servers have normal internet access, so this should never
happen there — if it does, it's almost certainly a Vercel-side or DNS
outage, not a config issue in this repo.

**"Load demo data" seems to do nothing, or duplicates something**
It's idempotent — see `scripts/seed_demo_data.py:seed`'s docstring — safe
to click more than once; it always resets `"demo_user"` to the canonical
48-transaction dataset rather than appending to whatever was there before.
