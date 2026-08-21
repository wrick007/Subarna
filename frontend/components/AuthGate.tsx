"use client";

import { useEffect, useState } from "react";
import type { Session } from "@supabase/supabase-js";

import { getSupabaseClient } from "@/lib/supabase";

import ChatShell from "./ChatShell";

export default function AuthGate() {
  const [session, setSession] = useState<Session | null>(null);
  const [ready, setReady] = useState(false);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [creatingAccount, setCreatingAccount] = useState(false);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");

  useEffect(() => {
    let active = true;
    let unsubscribe: (() => void) | undefined;
    try {
      const supabase = getSupabaseClient();
      supabase.auth.getSession().then(({ data }) => {
        if (active) {
          setSession(data.session);
          setReady(true);
        }
      });
      const listener = supabase.auth.onAuthStateChange((_event, nextSession) => {
        if (active) setSession(nextSession);
      });
      unsubscribe = () => listener.data.subscription.unsubscribe();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to start authentication.");
      setReady(true);
    }
    return () => {
      active = false;
      unsubscribe?.();
    };
  }, []);

  async function submit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setBusy(true);
    setMessage("");
    try {
      const supabase = getSupabaseClient();
      const result = creatingAccount
        ? await supabase.auth.signUp({ email, password })
        : await supabase.auth.signInWithPassword({ email, password });
      if (result.error) throw result.error;
      if (creatingAccount && !result.data.session) setMessage("Check your email to confirm your account, then sign in.");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to continue. Please try again.");
    } finally {
      setBusy(false);
    }
  }

  if (!ready) return <main className="grid h-dvh place-items-center bg-paper text-ink">Loading FinMate…</main>;
  if (session) return <ChatShell userId={session.user.id} email={session.user.email ?? ""} />;

  return (
    <main className="grid min-h-dvh place-items-center bg-paper p-6 text-ink">
      <form onSubmit={submit} className="w-full max-w-sm space-y-4 rounded-2xl border border-border bg-surface p-6 shadow-sm">
        <div><h1 className="font-display text-2xl font-semibold">FinMate</h1><p className="mt-1 text-sm text-mist">Your private financial workspace.</p></div>
        <label className="block text-sm">Email<input required type="email" value={email} onChange={(e) => setEmail(e.target.value)} className="mt-1 w-full rounded-lg border border-border bg-paper px-3 py-2" /></label>
        <label className="block text-sm">Password<input required minLength={6} type="password" value={password} onChange={(e) => setPassword(e.target.value)} className="mt-1 w-full rounded-lg border border-border bg-paper px-3 py-2" /></label>
        {message && <p className="text-sm text-brick">{message}</p>}
        <button disabled={busy} className="w-full rounded-lg bg-gold px-3 py-2 font-medium text-ink disabled:opacity-60">{busy ? "Please wait…" : creatingAccount ? "Create account" : "Sign in"}</button>
        <button type="button" onClick={() => { setCreatingAccount((v) => !v); setMessage(""); }} className="w-full text-sm text-gold-deep">{creatingAccount ? "Already have an account? Sign in" : "New here? Create an account"}</button>
      </form>
    </main>
  );
}
