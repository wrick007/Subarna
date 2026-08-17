"use client";

import { useCallback, useEffect, useState } from "react";
import { Menu, X } from "lucide-react";

import { api, ApiError } from "@/lib/api";
import { generateId } from "@/lib/format";
import type { ChatMessage, HealthResponse, Transaction } from "@/lib/types";

import Composer from "./Composer";
import MessageList from "./MessageList";
import Sidebar from "./Sidebar";
import StatusBanner from "./StatusBanner";

const USER_ID_STORAGE_KEY = "finmate:user_id";
const DEFAULT_USER_ID = "demo_user";

export default function ChatShell() {
  // Not persisted via localStorage inside artifacts (that API isn't
  // available there), but this is a real deployed Next.js app, not an
  // artifact preview -- browser storage works normally here, and doing
  // this keeps "which user am I" across a page refresh, which people
  // reasonably expect from a chat app.
  const [userId, setUserId] = useState(DEFAULT_USER_ID);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [transactions, setTransactions] = useState<Transaction[]>([]);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthChecked, setHealthChecked] = useState(false);
  const [isSending, setIsSending] = useState(false);
  const [isSeedingDemo, setIsSeedingDemo] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  useEffect(() => {
    // A lazy useState(() => ...) initializer would avoid this effect
    // (and the resulting extra render) entirely, but can't be used here:
    // it would run during server rendering too, where `window` doesn't
    // exist yet -- this has to be an effect specifically because effects
    // never run on the server, only after the client has hydrated.
    const stored = window.localStorage.getItem(USER_ID_STORAGE_KEY);
    // eslint-disable-next-line react-hooks/set-state-in-effect -- see comment above
    if (stored) setUserId(stored);
  }, []);

  // Both effects below follow React's documented pattern for effects
  // that fetch data (react.dev/learn/synchronizing-with-effects#fetching-data):
  // an `ignore` flag set on cleanup, checked before every setState call.
  // Not just lint-rule compliance -- it fixes a real bug this component
  // would otherwise have: switching `userId` twice in quick succession
  // (typing a new one, or Load Demo Data right after) could let an
  // in-flight fetch for the *previous* user resolve after the newer
  // one and silently overwrite its data with stale results.
  useEffect(() => {
    let ignore = false;
    api
      .health()
      .then((h) => {
        if (!ignore) setHealth(h);
      })
      .catch(() => {
        if (!ignore) setHealth(null);
      })
      .finally(() => {
        if (!ignore) setHealthChecked(true);
      });
    return () => {
      ignore = true;
    };
  }, []);

  const refreshTransactions = useCallback(async (forUserId: string) => {
    try {
      const res = await api.getTransactions(forUserId, 500);
      setTransactions(res.transactions);
    } catch {
      setTransactions([]);
    }
  }, []);

  useEffect(() => {
    let ignore = false;
    window.localStorage.setItem(USER_ID_STORAGE_KEY, userId);
    // Deliberately synchronous, not inside the .then() below: clearing
    // the previous user's messages must happen immediately when `userId`
    // changes, not after their transactions finish loading -- otherwise
    // their old chat history would stay on screen, misattributed to the
    // new user, for however long that fetch takes.
    // eslint-disable-next-line react-hooks/set-state-in-effect -- see above
    setMessages([]);
    api
      .getTransactions(userId, 500)
      .then((res) => {
        if (!ignore) setTransactions(res.transactions);
      })
      .catch(() => {
        if (!ignore) setTransactions([]);
      });
    return () => {
      ignore = true;
    };
  }, [userId]);

  const handleSend = useCallback(
    async (text: string) => {
      const userMessage: ChatMessage = { id: generateId(), role: "user", content: text, createdAt: Date.now() };
      const pendingId = generateId();
      const pendingMessage: ChatMessage = {
        id: pendingId,
        role: "assistant",
        content: "",
        createdAt: Date.now(),
        pending: true,
      };
      setMessages((prev) => [...prev, userMessage, pendingMessage]);
      setIsSending(true);

      try {
        const res = await api.chat(userId, text);
        setMessages((prev) =>
          prev.map((m) => (m.id === pendingId ? { ...m, pending: false, content: res.response, meta: res } : m)),
        );
      } catch (err) {
        const message = err instanceof ApiError ? err.message : "Something went wrong. Please try again.";
        setMessages((prev) => prev.map((m) => (m.id === pendingId ? { ...m, pending: false, error: message } : m)));
      } finally {
        setIsSending(false);
      }
    },
    [userId],
  );

  const handleLoadDemoData = useCallback(async () => {
    setIsSeedingDemo(true);
    try {
      const res = await api.seedDemoData();
      if (res.user_id !== userId) setUserId(res.user_id);
      else await refreshTransactions(userId);
    } catch {
      // Surfaced implicitly: the sidebar's transaction snapshot simply
      // won't update, and the person can just try the button again --
      // a toast/modal for one button's failure would be more machinery
      // than this moment needs.
    } finally {
      setIsSeedingDemo(false);
    }
  }, [userId, refreshTransactions]);

  const handleForgetData = useCallback(async () => {
    setIsDeleting(true);
    try {
      await api.deleteUser(userId);
      setTransactions([]);
      setMessages([]);
    } finally {
      setIsDeleting(false);
    }
  }, [userId]);

  return (
    <div className="flex h-dvh w-full overflow-hidden bg-paper">
      {sidebarOpen && (
        <button
          type="button"
          aria-label="Close menu"
          onClick={() => setSidebarOpen(false)}
          className="fixed inset-0 z-30 bg-ink/30 sm:hidden"
        />
      )}
      <div
        className={`fixed inset-y-0 left-0 z-40 w-72 transform transition-transform duration-200 ease-out sm:static sm:z-auto sm:w-auto sm:translate-x-0 ${
          sidebarOpen ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        <Sidebar
          userId={userId}
          onUserIdChange={setUserId}
          transactions={transactions}
          health={health}
          onLoadDemoData={handleLoadDemoData}
          onForgetData={handleForgetData}
          isSeedingDemo={isSeedingDemo}
          isDeleting={isDeleting}
        />
      </div>

      <div className="flex min-w-0 flex-1 flex-col">
        <div className="flex items-center gap-3 border-b border-border px-4 py-3 sm:hidden">
          <button
            type="button"
            onClick={() => setSidebarOpen((v) => !v)}
            aria-label={sidebarOpen ? "Close menu" : "Open menu"}
            className="text-ink-soft"
          >
            {sidebarOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
          </button>
          <span className="font-display text-lg italic text-ledger-dark">FinMate</span>
        </div>

        <StatusBanner health={health} checked={healthChecked} />
        <MessageList messages={messages} onSuggestionClick={handleSend} />
        <Composer onSend={handleSend} disabled={isSending} />
      </div>
    </div>
  );
}
