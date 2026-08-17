"use client";

import { AlertCircle } from "lucide-react";
import ReactMarkdown from "react-markdown";

import { formatTime } from "@/lib/format";
import type { ChatMessage } from "@/lib/types";

import VerifiedStrip from "./VerifiedStrip";

function ThinkingIndicator() {
  return (
    <div className="flex items-center gap-1 py-1" role="status" aria-label="FinMate is thinking">
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          className="h-1.5 w-1.5 animate-bounce rounded-full bg-mist"
          style={{ animationDelay: `${i * 120}ms` }}
        />
      ))}
    </div>
  );
}

export default function MessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user";

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div className={`flex max-w-[85%] flex-col sm:max-w-[75%] ${isUser ? "items-end" : "items-start"}`}>
        <div
          className={
            isUser
              ? "rounded-2xl rounded-tr-md bg-ledger px-4 py-2.5 text-[0.95rem] leading-relaxed text-white"
              : "rounded-2xl rounded-tl-md border border-border bg-surface px-4 py-2.5 text-[0.95rem] leading-relaxed text-ink"
          }
        >
          {message.pending ? (
            <ThinkingIndicator />
          ) : message.error ? (
            <div className="flex items-start gap-2 text-brick">
              <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
              <span>{message.error}</span>
            </div>
          ) : isUser ? (
            <p className="whitespace-pre-wrap">{message.content}</p>
          ) : (
            <div className="prose-chat">
              <ReactMarkdown>{message.content}</ReactMarkdown>
            </div>
          )}
        </div>
        <span className="mt-1 px-1 text-[11px] text-mist">{formatTime(message.createdAt)}</span>
        {!isUser && message.meta && <VerifiedStrip meta={message.meta} />}
      </div>
    </div>
  );
}
