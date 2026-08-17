"use client";

import { useState } from "react";
import { BadgeCheck, ChevronDown, ShieldAlert } from "lucide-react";

import { formatLatency } from "@/lib/format";
import type { ChatApiResponse } from "@/lib/types";

import EvidenceDrawer from "./EvidenceDrawer";

/**
 * The one element this UI is meant to be remembered by (see
 * frontend-design skill's "signature" guidance): FinMate's actual
 * differentiator, per the project README, is that nothing it says is
 * un-traceable -- every claim comes from a retrieved transaction or a
 * deterministic calculation, checked by a critic agent before the
 * person ever sees it. Most chat UIs bury that kind of provenance in a
 * debug panel, if they expose it at all. This makes it the *first*
 * thing under every answer, not the last -- a thin brass-accented strip
 * that expands into exactly what the pipeline actually did.
 */
export default function VerifiedStrip({ meta }: { meta: ChatApiResponse }) {
  const [open, setOpen] = useState(false);

  if (meta.is_casual) return null;

  const sourceCount = meta.retrieval?.evidence.length ?? 0;
  const hasIssues = meta.critic_errors.length > 0 || meta.critic_unsupported_claims.length > 0;
  const passed = meta.critic_passed && !hasIssues;

  const summaryParts: string[] = [];
  if (sourceCount > 0) summaryParts.push(`${sourceCount} source${sourceCount === 1 ? "" : "s"}`);
  if (meta.calculations.length > 0) {
    summaryParts.push(`${meta.calculations.length} calculation${meta.calculations.length === 1 ? "" : "s"}`);
  }
  summaryParts.push(formatLatency(meta.latency_ms));

  return (
    <div className="mt-2">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className={`group flex w-full items-center gap-2 rounded-lg border px-3 py-1.5 text-left text-xs font-medium transition-colors ${
          passed
            ? "border-brass/25 bg-brass-soft text-brass hover:border-brass/40"
            : "border-brick/25 bg-brick-soft text-brick hover:border-brick/40"
        }`}
      >
        {passed ? (
          <BadgeCheck className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
        ) : (
          <ShieldAlert className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
        )}
        <span>{passed ? "Verified" : "Needs review"}</span>
        <span aria-hidden="true" className="opacity-50">
          ·
        </span>
        <span className="truncate font-normal opacity-80">{summaryParts.join(" · ")}</span>
        <ChevronDown
          className={`ml-auto h-3.5 w-3.5 shrink-0 transition-transform ${open ? "rotate-180" : ""}`}
          aria-hidden="true"
        />
      </button>
      {open && <EvidenceDrawer meta={meta} />}
    </div>
  );
}
