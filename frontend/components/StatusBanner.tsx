import { AlertTriangle, WifiOff } from "lucide-react";

import { API_BASE_URL } from "@/lib/api";
import type { HealthResponse } from "@/lib/types";

export default function StatusBanner({ health, checked }: { health: HealthResponse | null; checked: boolean }) {
  if (!checked) return null;

  if (!health) {
    return (
      <div className="flex items-start gap-2 border-b border-brick/20 bg-brick-soft px-4 py-2.5 text-sm text-brick sm:px-8">
        <WifiOff className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
        <span>
          Can&apos;t reach the backend at <code className="font-mono text-xs">{API_BASE_URL}</code>. Confirm it&apos;s
          running and that <code className="font-mono text-xs">NEXT_PUBLIC_API_URL</code> is set correctly.
        </span>
      </div>
    );
  }

  if (!health.llm_configured) {
    // Deliberately a neutral/muted treatment, not gold: see globals.css's
    // token comment -- gold means "verified"/"live" specifically, and a
    // second, unrelated meaning for the same color would blur that.
    return (
      <div className="flex items-start gap-2 border-b border-border bg-paper px-4 py-2.5 text-sm text-ink-soft sm:px-8">
        <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-mist" aria-hidden="true" />
        <span>
          No LLM provider is configured on the backend, so real questions won&apos;t work yet (greetings still will).
          Set <code className="font-mono text-xs">GROQ_API_KEY</code> or{" "}
          <code className="font-mono text-xs">GEMINI_API_KEY</code> and restart it — see DEPLOYMENT.md.
        </span>
      </div>
    );
  }

  return null;
}
