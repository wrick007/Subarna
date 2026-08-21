-- Run this in Supabase SQL Editor (or `supabase db push`) before deploying.
-- Every row belongs to exactly one authenticated Supabase user.

create table if not exists public.profiles (
  user_id uuid primary key references auth.users(id) on delete cascade,
  profile jsonb not null default '{}'::jsonb,
  updated_at timestamptz not null default now()
);

create table if not exists public.transactions (
  id bigint generated always as identity primary key,
  user_id uuid not null references auth.users(id) on delete cascade,
  date date not null,
  description text not null,
  amount numeric not null,
  currency text not null default 'INR',
  category text not null default 'uncategorized',
  account text not null default '',
  type text not null default 'expense',
  source_id text not null default '',
  created_at timestamptz not null default now()
);

create index if not exists transactions_user_date_idx on public.transactions (user_id, date);
create index if not exists transactions_user_category_idx on public.transactions (user_id, category);

create table if not exists public.chat_messages (
  id bigint generated always as identity primary key,
  user_id uuid not null references auth.users(id) on delete cascade,
  role text not null check (role in ('user', 'assistant')),
  content text not null check (char_length(content) <= 12000),
  created_at timestamptz not null default now()
);

create index if not exists chat_messages_user_created_idx on public.chat_messages (user_id, created_at);

alter table public.profiles enable row level security;
alter table public.transactions enable row level security;
alter table public.chat_messages enable row level security;

create policy "Users manage their own profile" on public.profiles
  for all using (auth.uid() = user_id) with check (auth.uid() = user_id);
create policy "Users manage their own transactions" on public.transactions
  for all using (auth.uid() = user_id) with check (auth.uid() = user_id);
create policy "Users manage their own chat messages" on public.chat_messages
  for all using (auth.uid() = user_id) with check (auth.uid() = user_id);
