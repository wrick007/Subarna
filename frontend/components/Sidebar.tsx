"use client";

import { useEffect, useRef, useState } from "react";
import { Database, Loader2, Trash2, Wifi, WifiOff } from "lucide-react";

import type { HealthResponse, Transaction } from "@/lib/types";

import SpendingSnapshot from "./SpendingSnapshot";

export default function Sidebar({
  userId,
  onUserIdChange,
  transactions,
  health,
  onLoadDemoData,
  onForgetData,
  isSeedingDemo,
  isDeleting,
}: {
  userId: string;
  onUserIdChange: (id: string) => void;
  transactions: Transaction[];
  health: HealthResponse | null;
  onLoadDemoData: () => void;
  onForgetData: () => void;
  isSeedingDemo: boolean;
  isDeleting: boolean;
}) {
  const [confirmingDelete, setConfirmingDelete] = useState(false);
  const confirmTimeout = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Cleanup-only (no setup logic, nothing set in the body -- just a
  // teardown function) so this doesn't trip the same "setState in
  // effect" concern the userId-sync effect above was rewritten to avoid:
  // it exists solely to cancel the pending timer if this component
  // unmounts (e.g. switching away) while "click again to confirm" is
  // still showing, so that timer's callback can't fire setState on an
  // already-unmounted component.
  useEffect(() => {
    return () => {
      if (confirmTimeout.current) clearTimeout(confirmTimeout.current);
    };
  }, []);

  // The user-id field is deliberately uncontrolled, keyed on `userId`:
  // typing a new id and blurring commits it via onUserIdChange, while a
  // `userId` change from *outside* (Load Demo Data switching to
  // "demo_user") remounts the input fresh with the new value as its
  // defaultValue -- the standard React pattern for "reset local state
  // when a prop changes" (react.dev/learn/preserving-and-resetting-state),
  // with no separate state-mirroring effect needed at all.
  function handleUserIdBlur(e: React.FocusEvent<HTMLInputElement>) {
    const trimmed = e.target.value.trim();
    if (trimmed && trimmed !== userId) onUserIdChange(trimmed);
    else e.target.value = userId;
  }

  function handleDeleteClick() {
    if (!confirmingDelete) {
      setConfirmingDelete(true);
      confirmTimeout.current = setTimeout(() => setConfirmingDelete(false), 4000);
      return;
    }
    if (confirmTimeout.current) clearTimeout(confirmTimeout.current);
    setConfirmingDelete(false);
    onForgetData();
  }

  return (
    <aside className="flex h-full w-full flex-col gap-6 overflow-y-auto themed-scroll border-border bg-paper p-5 sm:w-72 sm:border-r">
      <div>
        <span className="font-display text-xl italic text-ledger-dark">FinMate</span>
        <p className="mt-0.5 text-xs text-mist">Evidence-grounded finance, checked before it answers.</p>
      </div>

      <div className="flex items-center gap-1.5 text-xs">
        {health?.status === "ok" ? (
          <Wifi className="h-3.5 w-3.5 text-ledger" aria-hidden="true" />
        ) : (
          <WifiOff className="h-3.5 w-3.5 text-brick" aria-hidden="true" />
        )}
        <span className={health?.status === "ok" ? "text-ink-soft" : "text-brick"}>
          {health?.status === "ok" ? "Backend connected" : "Backend unreachable"}
        </span>
      </div>

      <div>
        <label htmlFor="user-id" className="text-[11px] font-medium uppercase tracking-wide text-mist">
          Signed in as
        </label>
        <input
          id="user-id"
          key={userId}
          defaultValue={userId}
          onBlur={handleUserIdBlur}
          onKeyDown={(e) => {
            if (e.key === "Enter") (e.target as HTMLInputElement).blur();
          }}
          className="mt-1 w-full rounded-lg border border-border bg-surface px-3 py-1.5 text-sm text-ink focus:border-ledger/50 focus:outline-none"
          placeholder="user_id"
        />
      </div>

      <div>
        <p className="mb-2 text-[11px] font-medium uppercase tracking-wide text-mist">Financial pulse</p>
        <SpendingSnapshot transactions={transactions} />
      </div>

      <div className="mt-auto space-y-2 border-t border-border pt-4">
        <button
          type="button"
          onClick={onLoadDemoData}
          disabled={isSeedingDemo}
          className="flex w-full items-center justify-center gap-2 rounded-lg border border-border bg-surface px-3 py-2 text-sm font-medium text-ink-soft transition-colors hover:border-ledger/30 hover:text-ledger-dark disabled:opacity-60"
        >
          {isSeedingDemo ? (
            <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
          ) : (
            <Database className="h-4 w-4" aria-hidden="true" />
          )}
          Load demo data
        </button>
        <button
          type="button"
          onClick={handleDeleteClick}
          disabled={isDeleting}
          className={`flex w-full items-center justify-center gap-2 rounded-lg border px-3 py-2 text-sm font-medium transition-colors disabled:opacity-60 ${
            confirmingDelete
              ? "border-brick bg-brick-soft text-brick"
              : "border-border bg-surface text-ink-soft hover:border-brick/30 hover:text-brick"
          }`}
        >
          {isDeleting ? <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" /> : <Trash2 className="h-4 w-4" aria-hidden="true" />}
          {confirmingDelete ? "Click again to confirm" : "Forget my data"}
        </button>
      </div>
    </aside>
  );
}
