"use client";

import { AlertCircle } from "lucide-react";
import ReactMarkdown from "react-markdown";

import { formatTime } from "@/lib/format";
import type { ChatMessage } from "@/lib/types";


function ThinkingIndicator() {
  return (
    <div className="flex items-center gap-1 py-1" role="status" aria-label="FinMate is thinking">
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          className="h-1.5 w-1.5 animate-bounce rounded-full bg-gold"
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
            // User bubbles get a warm gold *tint*, not a saturated gold
            // fill -- see globals.css's token comment: gold is spent
            // where the product does distinctive work (the thinking
            // indicator and streaming cursor below, both literally the
            // moment of live generation), not as base chrome for every
            // message a person sends, which would both dilute what the
            // color means and add more visual weight than a "minimal,
            // low-noise" redesign should.
            isUser
              ? "rounded-2xl rounded-tr-md bg-gold-soft px-4 py-2.5 text-[0.95rem] leading-relaxed text-ink"
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
              {message.streaming && (
                <span
                  className="ml-0.5 inline-block h-3.5 w-1.5 animate-pulse bg-gold align-text-bottom"
                  aria-hidden="true"
                />
              )}
            </div>
          )}
        </div>
        <span className="mt-1 px-1 text-[11px] text-mist">{formatTime(message.createdAt)}</span>
      </div>
    </div>
  );
}
