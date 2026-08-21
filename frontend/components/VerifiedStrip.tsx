"use client";

import { useState } from "react";
import { BadgeCheck, ChevronDown, Info, ShieldAlert } from "lucide-react";

import { formatLatency } from "@/lib/format";
import type { ChatApiResponse } from "@/lib/types";

import EvidenceDrawer from "./EvidenceDrawer";

/**
 * The one element this UI is meant to be remembered by (see
 * frontend-design skill's "signature" guidance, and this redesign's
 * deliberate call on the evidence panel: keep it, not cut it -- see
 * README/redesign notes for the full reasoning). FinMate's actual
 * differentiator, per the project README, is that nothing it says is
 * un-traceable -- every claim comes from a retrieved transaction or a
 * deterministic calculation. Most chat UIs bury that kind of provenance
 * in a debug panel, if they expose it at all. This keeps it visible
 * without competing for attention: a single line of small text under
 * the answer, no border/background box, that expands into exactly what
 * the pipeline did -- collapsed by default, plain text rather than a
 * button-shaped chip, deliberately quieter than the pre-redesign
 * bordered pill it replaces (see globals.css's token comment on
 * "closer to ChatGPT... low visual noise").
 *
 * Three honest states, not two: as of the Priority-2 pipeline redesign,
 * the critic doesn't run on every turn anymore (see
 * finmate/orchestrator.py's "Critic: conditional, not always-on") --
 * `meta.verification_ran` tells this component whether a check actually
 * happened, so a general-information answer reads as exactly that,
 * never as "Verified" when nothing was actually checked against
 * anything. Gold is spent only on the one state that earned it.
 */
export default function VerifiedStrip({ meta }: { meta: ChatApiResponse }) {
  const [open, setOpen] = useState(false);

  if (meta.is_casual) return null;

  const sourceCount = meta.retrieval?.evidence.length ?? 0;
  const hasIssues = meta.critic_errors.length > 0 || meta.critic_unsupported_claims.length > 0;
  const passed = meta.verification_ran && meta.critic_passed && !hasIssues;
  const flagged = meta.verification_ran && (!meta.critic_passed || hasIssues);
  // The remaining case -- !meta.verification_ran -- is "general information."

  const summaryParts: string[] = [];
  if (sourceCount > 0) summaryParts.push(`${sourceCount} source${sourceCount === 1 ? "" : "s"}`);
  if (meta.calculations.length > 0) {
    summaryParts.push(`${meta.calculations.length} calculation${meta.calculations.length === 1 ? "" : "s"}`);
  }
  summaryParts.push(formatLatency(meta.latency_ms));

  const label = passed ? "Verified" : flagged ? "Needs review" : "General information";
  const Icon = passed ? BadgeCheck : flagged ? ShieldAlert : Info;
  const textClass = passed ? "text-gold-deep" : flagged ? "text-brick" : "text-mist";

  return (
    <div className="mt-1.5">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className={`group flex items-center gap-1.5 text-left text-xs transition-colors hover:opacity-80 ${textClass}`}
      >
        <Icon className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
        <span className="font-medium">{label}</span>
        <span aria-hidden="true" className="opacity-50">
          ·
        </span>
        <span className="truncate font-normal opacity-70">{summaryParts.join(" · ")}</span>
        <ChevronDown
          className={`h-3.5 w-3.5 shrink-0 transition-transform ${open ? "rotate-180" : ""}`}
          aria-hidden="true"
        />
      </button>
      {open && <EvidenceDrawer meta={meta} />}
    </div>
  );
}
