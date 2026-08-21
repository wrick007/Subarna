# Supabase production setup

FinMate now supports authenticated, durable user data. Complete these steps
before deploying this version.

1. Create a Supabase project.
2. In **Authentication → Providers**, enable Email. Configure the Site URL
   and redirect URL with your Vercel domain (for example,
   `https://your-app.vercel.app`).
3. Open **SQL Editor** and run
   [`supabase/migrations/001_finmate_schema.sql`](supabase/migrations/001_finmate_schema.sql).
   This creates `profiles`, `transactions`, and `chat_messages`, with Row
   Level Security enabled for each user.
4. In **Project Settings → API**, copy the project URL, anon key, and
   service-role key.

## Environment variables

Set these in **Render** (backend):

| Variable | Value |
| --- | --- |
| `SUPABASE_URL` | Your Supabase project URL |
| `SUPABASE_ANON_KEY` | Your Supabase anon/publishable key |
| `SUPABASE_SERVICE_ROLE_KEY` | Your service-role key — backend only |
| `FINMATE_FRONTEND_ORIGINS` | Your deployed Vercel URL |

Set these in **Vercel** (frontend):

| Variable | Value |
| --- | --- |
| `NEXT_PUBLIC_API_URL` | Your Render backend URL |
| `NEXT_PUBLIC_SUPABASE_URL` | Your Supabase project URL |
| `NEXT_PUBLIC_SUPABASE_ANON_KEY` | Your Supabase anon/publishable key |

Never put `SUPABASE_SERVICE_ROLE_KEY` in Vercel, frontend files, or Git.

## Result

Users sign up with email/password. Their account is owned by Supabase Auth;
their profile, transactions, and saved conversations live in Supabase
Postgres. The backend verifies each bearer token and ignores a user ID sent
by the browser, so users cannot request someone else's records.
