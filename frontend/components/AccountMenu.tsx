"use client";

import { useEffect, useRef, useState } from "react";
import { Database, Loader2, Trash2 } from "lucide-react";

import type { Transaction } from "@/lib/types";

import SpendingSnapshot from "./SpendingSnapshot";

/**
 * Replaces the pre-redesign Sidebar.tsx: same responsibilities (account
 * identity, data snapshot, demo data, "forget my data"), but rendered as
 * a small popover from a top-bar trigger instead of a permanently-
 * visible panel -- see this redesign's notes on "no dense sidebar of
 * financial widgets competing for attention" and moving "Load demo
 * data" specifically out of primary view. Renamed from Sidebar.tsx to
 * AccountMenu.tsx since it no longer renders as a sidebar at all; this
 * is a component-file rename, not a path any deploy config references
 * (contrast the redesign brief's hard constraint on backend/render.yaml/
 * frontend/app/ paths, which this doesn't touch) -- called out here and
 * in the redesign summary rather than done silently.
 */
export default function AccountMenu({
  open,
  onClose,
  userId,
  onUserIdChange,
  transactions,
  onLoadDemoData,
  onForgetData,
  isSeedingDemo,
  isDeleting,
}: {
  open: boolean;
  onClose: () => void;
  userId: string;
  onUserIdChange: (id: string) => void;
  transactions: Transaction[];
  onLoadDemoData: () => void;
  onForgetData: () => void;
  isSeedingDemo: boolean;
  isDeleting: boolean;
}) {
  const [confirmingDelete, setConfirmingDelete] = useState(false);
  const confirmTimeout = useRef<ReturnType<typeof setTimeout> | null>(null);
  const panelRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    return () => {
      if (confirmTimeout.current) clearTimeout(confirmTimeout.current);
    };
  }, []);

  // Escape-to-close, matching the click-outside backdrop below --
  // standard popover accessibility, not just a nicety here since the
  // panel can contain a destructive "Forget my data" action a keyboard
  // user should be able to back out of quickly.
  useEffect(() => {
    if (!open) return;
    function handleKey(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    document.addEventListener("keydown", handleKey);
    return () => document.removeEventListener("keydown", handleKey);
  }, [open, onClose]);

  if (!open) return null;

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
    <>
      {/* Invisible full-screen backdrop: click anywhere outside the
          panel to close, same technique this file's predecessor used
          for the mobile sidebar overlay. */}
      <button type="button" aria-label="Close menu" onClick={onClose} className="fixed inset-0 z-30" />
      <div
        ref={panelRef}
        role="menu"
        className="absolute right-0 top-full z-40 mt-2 w-80 max-w-[calc(100vw-2rem)] space-y-4 rounded-xl border border-border bg-surface p-4 text-left shadow-lg"
      >
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
            className="mt-1 w-full rounded-lg border border-border bg-paper px-3 py-1.5 text-sm text-ink focus:border-gold/50 focus:outline-none"
            placeholder="user_id"
          />
        </div>

        <div>
          <p className="mb-2 text-[11px] font-medium uppercase tracking-wide text-mist">Financial pulse</p>
          <SpendingSnapshot transactions={transactions} />
        </div>

        <div className="space-y-2 border-t border-border pt-3">
          <button
            type="button"
            onClick={onLoadDemoData}
            disabled={isSeedingDemo}
            className="flex w-full items-center justify-center gap-2 rounded-lg border border-border bg-paper px-3 py-2 text-sm font-medium text-ink-soft transition-colors hover:border-gold/40 hover:text-gold-deep disabled:opacity-60"
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
                : "border-border bg-paper text-ink-soft hover:border-brick/30 hover:text-brick"
            }`}
          >
            {isDeleting ? <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" /> : <Trash2 className="h-4 w-4" aria-hidden="true" />}
            {confirmingDelete ? "Click again to confirm" : "Forget my data"}
          </button>
        </div>
      </div>
    </>
  );
}
