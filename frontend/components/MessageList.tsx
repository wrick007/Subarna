"use client";

import { useEffect, useRef } from "react";
import { Compass } from "lucide-react";

import type { ChatMessage } from "@/lib/types";

import MessageBubble from "./MessageBubble";

const SUGGESTIONS = [
  "What's my savings rate this month?",
  "Am I overspending on dining out?",
  "Do I have any duplicate or unusual charges?",
  "How long until I hit my Japan trip goal?",
];

function EmptyState({ onPick }: { onPick: (text: string) => void }) {
  return (
    <div className="flex h-full flex-col items-center justify-center px-6 py-12 text-center">
      <span className="font-display text-2xl italic text-ledger-dark">Ask FinMate anything.</span>
      <p className="mt-3 max-w-sm text-sm leading-relaxed text-mist">
        Every answer here traces back to a real transaction or a calculation you can check yourself — nothing is
        estimated from memory.
      </p>
      <div className="mt-6 flex w-full max-w-md flex-col gap-2">
        {SUGGESTIONS.map((s) => (
          <button
            key={s}
            type="button"
            onClick={() => onPick(s)}
            className="flex items-center gap-2.5 rounded-xl border border-border bg-surface px-4 py-2.5 text-left text-sm text-ink-soft transition-colors hover:border-ledger/30 hover:bg-ledger-soft hover:text-ledger-dark"
          >
            <Compass className="h-4 w-4 shrink-0 text-mist" aria-hidden="true" />
            {s}
          </button>
        ))}
      </div>
    </div>
  );
}

export default function MessageList({
  messages,
  onSuggestionClick,
}: {
  messages: ChatMessage[];
  onSuggestionClick: (text: string) => void;
}) {
  const bottomRef = useRef<HTMLDivElement>(null);
  const lastMessage = messages[messages.length - 1];
  const lastMessageContent = lastMessage?.content;
  const lastMessagePending = lastMessage?.pending;

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages.length, lastMessageContent, lastMessagePending]);

  if (messages.length === 0) {
    return <EmptyState onPick={onSuggestionClick} />;
  }

  return (
    <div className="themed-scroll flex-1 space-y-5 overflow-y-auto px-4 py-6 sm:px-8">
      {messages.map((m) => (
        <MessageBubble key={m.id} message={m} />
      ))}
      <div ref={bottomRef} />
    </div>
  );
}
