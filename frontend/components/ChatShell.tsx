"use client";

import { useCallback, useEffect, useState } from "react";
import { Settings, Wifi, WifiOff } from "lucide-react";

import { api } from "@/lib/api";
import { generateId } from "@/lib/format";
import type { ChatHistoryTurn, ChatMessage, HealthResponse, Transaction } from "@/lib/types";

import AccountMenu from "./AccountMenu";
import Composer from "./Composer";
import MessageList from "./MessageList";
import StatusBanner from "./StatusBanner";

const USER_ID_STORAGE_KEY = "finmate:user_id";
const DEFAULT_USER_ID = "demo_user";

// Short-term conversational memory (see finmate/orchestrator.py's module
// docstring "Conversation history"): how many of the most recent
// *settled* messages this client sends up with each turn. The backend
// re-trims to a smaller number anyway (finmate.orchestrator.
// MAX_HISTORY_MESSAGES) regardless of what's sent -- this cap exists to
// keep the request body small, not because 20 is the "real" limit.
const MAX_HISTORY_MESSAGES_SENT = 20;

/** `messages` state -> the `history` the API wants: settled turns only
 * (a still-`pending` reply has no content yet, and a failed one has
 * none worth resending as context), oldest-first, most recent
 * MAX_HISTORY_MESSAGES_SENT. Excludes whatever message is about to be
 * sent as the *current* turn -- callers pass the state as it was before
 * appending that one. */
function historyForApi(messages: ChatMessage[]): ChatHistoryTurn[] {
  return messages
    .filter((m) => !m.pending && !m.error && m.content.trim().length > 0)
    .slice(-MAX_HISTORY_MESSAGES_SENT)
    .map((m) => ({ role: m.role, content: m.content }));
}

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
  const [accountMenuOpen, setAccountMenuOpen] = useState(false);

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
      const history = historyForApi(messages);
      setMessages((prev) => [...prev, userMessage, pendingMessage]);
      setIsSending(true);

      // Priority 2 streaming: token-by-token via SSE (see lib/api.ts's
      // chatStream and finmate/orchestrator.py's module docstring
      // "Streaming"). `accumulated` tracks the current attempt's text
      // outside React state (state updates are async/batched; the next
      // token's callback needs the *actual* current value, not a stale
      // closure over `content` from the last render).
      let accumulated = "";
      await api.chatStream(userId, text, history, {
        onToken: (delta) => {
          accumulated += delta;
          const streamedSoFar = accumulated;
          setMessages((prev) =>
            prev.map((m) => (m.id === pendingId ? { ...m, pending: false, streaming: true, content: streamedSoFar } : m)),
          );
        },
        onRestart: () => {
          // See StreamEvent's docstring in orchestrator.py: a
          // critic-triggered retry discards every token streamed so
          // far. Go back to the thinking indicator rather than leave
          // stale, superseded text on screen while the retry streams in.
          accumulated = "";
          setMessages((prev) =>
            prev.map((m) => (m.id === pendingId ? { ...m, pending: true, streaming: false, content: "" } : m)),
          );
        },
        onDone: (res) => {
          // Use res.response, not `accumulated`, as the final text: on
          // an exhausted-retry verification failure, the backend appends
          // a disclaimer *after* streaming already finished (see
          // orchestrator.py's _node_finalize) -- accumulated tokens
          // alone would be missing it.
          setMessages((prev) =>
            prev.map((m) =>
              m.id === pendingId ? { ...m, pending: false, streaming: false, content: res.response, meta: res } : m,
            ),
          );
        },
        onError: (message) => {
          setMessages((prev) =>
            prev.map((m) => (m.id === pendingId ? { ...m, pending: false, streaming: false, error: message } : m)),
          );
        },
      });

      setIsSending(false);
    },
    [userId, messages],
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
    <div className="flex h-dvh w-full flex-col overflow-hidden bg-paper">
      {/* Top bar -- replaces the pre-redesign always-visible Sidebar
          entirely (see AccountMenu.tsx's docstring): one thin row,
          present on every screen size, no separate mobile/desktop
          layouts to keep in sync. This is the whole point of "closer to
          ChatGPT's actual interaction pattern" from the redesign brief
          -- minimal chrome, one thing on screen at a time, nothing
          competing with the chat itself. */}
      <header className="relative flex shrink-0 items-center justify-between border-b border-border px-4 py-3 sm:px-6">
        <span className="font-display text-lg font-semibold tracking-tight text-ink">FinMate</span>

        <div className="flex items-center gap-3">
          <span title={health?.status === "ok" ? "Backend connected" : "Backend unreachable"}>
            {health?.status === "ok" ? (
              <Wifi className="h-4 w-4 text-mist" aria-hidden="true" />
            ) : (
              <WifiOff className="h-4 w-4 text-brick" aria-hidden="true" />
            )}
          </span>
          <button
            type="button"
            onClick={() => setAccountMenuOpen((v) => !v)}
            aria-label="Account and data settings"
            aria-expanded={accountMenuOpen}
            className="rounded-lg p-1.5 text-ink-soft transition-colors hover:bg-gold-soft hover:text-gold-deep"
          >
            <Settings className="h-4.5 w-4.5" aria-hidden="true" />
          </button>
        </div>

        <AccountMenu
          open={accountMenuOpen}
          onClose={() => setAccountMenuOpen(false)}
          userId={userId}
          onUserIdChange={setUserId}
          transactions={transactions}
          onLoadDemoData={handleLoadDemoData}
          onForgetData={handleForgetData}
          isSeedingDemo={isSeedingDemo}
          isDeleting={isDeleting}
        />
      </header>

      <StatusBanner health={health} checked={healthChecked} />
      <MessageList messages={messages} onSuggestionClick={handleSend} />
      <Composer onSend={handleSend} disabled={isSending} />
    </div>
  );
}
