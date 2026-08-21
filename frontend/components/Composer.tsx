"use client";

import { useEffect, useRef, useState } from "react";
import { ArrowUp } from "lucide-react";

export default function Composer({
  onSend,
  disabled,
  disabledReason,
}: {
  onSend: (text: string) => void;
  disabled?: boolean;
  disabledReason?: string;
}) {
  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "0px";
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  }, [value]);

  function submit() {
    const trimmed = value.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setValue("");
  }

  return (
    <div className="border-t border-border bg-paper px-4 py-3 sm:px-8 sm:py-4">
      <div className="mx-auto w-full max-w-3xl">
        {disabled && disabledReason && (
          <p className="mb-2 text-center text-xs text-brick">{disabledReason}</p>
        )}
        <div className="flex items-end gap-2 rounded-2xl border border-border bg-surface p-2 shadow-sm focus-within:border-gold/50">
          <textarea
            ref={textareaRef}
            rows={1}
            value={value}
            disabled={disabled}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                submit();
              }
            }}
            placeholder="Ask about your spending, budget, goals…"
            aria-label="Message FinMate"
            className="max-h-40 flex-1 resize-none bg-transparent px-2 py-1.5 text-[0.95rem] leading-relaxed text-ink placeholder:text-mist focus:outline-none disabled:cursor-not-allowed disabled:opacity-60"
          />
          <button
            type="button"
            onClick={submit}
            disabled={disabled || !value.trim()}
            aria-label="Send message"
            className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-gold text-white transition-opacity enabled:hover:bg-gold-deep disabled:cursor-not-allowed disabled:opacity-30"
          >
            <ArrowUp className="h-4.5 w-4.5" aria-hidden="true" />
          </button>
        </div>
      </div>
    </div>
  );
}
