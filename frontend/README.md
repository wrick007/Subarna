# FinMate AI — frontend

Next.js (App Router, TypeScript, Tailwind v4) chat interface for the FastAPI
backend in `../backend`. For the full architecture and both deploy paths
(Vercel for this app, Render for the backend), see
**[`../DEPLOYMENT.md`](../DEPLOYMENT.md)**.

## Local development

```bash
cd frontend
npm install
cp .env.example .env.local        # then edit NEXT_PUBLIC_API_URL if your backend isn't on localhost:8000
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). You'll need the backend
running too — see `../backend/README.md` or `../DEPLOYMENT.md`'s "Run
locally" section to run both together.

## What's here

```
app/            Next.js App Router: layout (fonts, metadata), the one page, globals.css (design tokens)
components/     ChatShell (state/orchestration), AccountMenu, MessageList/MessageBubble,
                VerifiedStrip + EvidenceDrawer (the pipeline-transparency UI), Composer, SpendingSnapshot
lib/            api.ts (typed fetch client, incl. SSE streaming), types.ts (mirrors backend/app/api_schemas.py), format.ts
```

`VerifiedStrip`/`EvidenceDrawer` are the one part of this UI worth reading
before changing anything else: they're a deliberate design choice, not
boilerplate — see the comment at the top of `components/VerifiedStrip.tsx`
for why.

Chat replies stream in token-by-token via `POST /api/chat/stream` (Server-Sent
Events) rather than arriving as one blob — see `lib/api.ts`'s `chatStream`
and `finmate/orchestrator.py`'s module docstring "Streaming" on the backend
for the full design, including the one thing to know as a consumer: a
`restart` event means a verification-triggered retry is discarding every
token sent so far, and `MessageBubble.tsx`/`ChatShell.tsx` handle that by
clearing back to the thinking indicator rather than mixing old and new text.

## Design system

Palette, type roles, and the rest of the visual decisions are documented as
comments in `app/globals.css` (token definitions) and
`components/VerifiedStrip.tsx` (the signature element). Three type roles:
Space Grotesk (wordmark only — one glance per screen, not a running display
face), Inter (everything else), and IBM Plex Mono specifically for monetary
figures, so amounts line up like a printed statement.

## Build

```bash
npm run build
```

Requires internet access to fetch Space Grotesk/Inter/IBM Plex Mono from
Google Fonts at build time (`next/font/google`) — normal on Vercel, just
worth knowing if you're building somewhere network-restricted.
