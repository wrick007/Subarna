import { Calculator, FileWarning, Sparkles } from "lucide-react";

import { formatDate, formatSignedCurrency } from "@/lib/format";
import type { ChatApiResponse, EvidenceItem } from "@/lib/types";

/** Stage badge color is drawn from the *existing* palette (ledger-soft /
 * brass-soft / plain border) rather than inventing a fourth accent color
 * just to color-code three retrieval tiers -- see globals.css's token
 * comment on keeping brass meaningful by not spreading it everywhere. */
function stageBadgeClasses(stage: string): string {
  if (stage === "rerank") return "bg-brass-soft text-brass border-brass/30";
  if (stage === "vector") return "bg-ledger-soft text-ledger-dark border-ledger/20";
  if (stage === "keyword") return "bg-paper text-ink-soft border-border";
  return "bg-paper text-mist border-border";
}

function EvidenceRow({ item }: { item: EvidenceItem }) {
  return (
    <li className="flex items-start justify-between gap-3 border-b border-border/70 py-2.5 last:border-b-0">
      <div className="min-w-0 flex-1">
        <p className="truncate text-sm text-ink">{item.description}</p>
        <div className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-mist">
          <span>{formatDate(item.date)}</span>
          <span aria-hidden="true">·</span>
          <span>{item.category}</span>
          {item.document && (
            <>
              <span aria-hidden="true">·</span>
              <span className="truncate">{item.document}</span>
            </>
          )}
          {item.retrieval_stage && (
            <span
              className={`rounded-full border px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide ${stageBadgeClasses(item.retrieval_stage)}`}
            >
              {item.retrieval_stage}
            </span>
          )}
        </div>
      </div>
      <span
        className={`tabular-nums shrink-0 text-sm font-medium ${item.amount < 0 ? "text-ink" : "text-ledger-dark"}`}
      >
        {formatSignedCurrency(item.amount, item.currency)}
      </span>
    </li>
  );
}

function CalculationRow({ calc }: { calc: ChatApiResponse["calculations"][number] }) {
  return (
    <li className="border-b border-border/70 py-2.5 last:border-b-0">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="text-sm text-ink">{calc.metric.replace(/_/g, " ")}</p>
          <p className="mt-0.5 truncate font-mono text-xs text-mist">{calc.formula}</p>
        </div>
        <span className="tabular-nums shrink-0 text-sm font-medium text-ink">
          {Number.isFinite(calc.value)
            ? calc.metric.toLowerCase().includes("rate") || calc.metric.toLowerCase().includes("percent")
              ? `${(calc.value * 100).toFixed(1)}%`
              : formatSignedCurrency(calc.value, calc.currency || "INR")
            : String(calc.value)}
        </span>
      </div>
    </li>
  );
}

export default function EvidenceDrawer({ meta }: { meta: ChatApiResponse }) {
  const evidence = meta.retrieval?.evidence ?? [];
  const hasEvidence = evidence.length > 0;
  const hasCalcs = meta.calculations.length > 0;
  const hasIssues = meta.critic_errors.length > 0 || meta.critic_unsupported_claims.length > 0;

  return (
    <div className="mt-2 space-y-4 rounded-xl border border-border bg-surface p-4 text-left shadow-sm">
      {meta.specialists_used.length > 0 && (
        <div className="flex flex-wrap items-center gap-1.5">
          <Sparkles className="h-3.5 w-3.5 text-mist" aria-hidden="true" />
          {meta.specialists_used.map((name) => (
            <span
              key={name}
              className="rounded-full border border-border bg-paper px-2 py-0.5 text-[11px] font-medium capitalize text-ink-soft"
            >
              {name.replace(/_/g, " ")}
            </span>
          ))}
        </div>
      )}

      {hasCalcs && (
        <div>
          <h4 className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-mist">
            <Calculator className="h-3.5 w-3.5" aria-hidden="true" />
            Calculations
          </h4>
          <ul className="mt-1">
            {meta.calculations.map((c, i) => (
              <CalculationRow key={`${c.metric}-${i}`} calc={c} />
            ))}
          </ul>
        </div>
      )}

      {hasEvidence && (
        <div>
          <h4 className="text-xs font-semibold uppercase tracking-wide text-mist">
            Transactions ({evidence.length})
          </h4>
          <ul className="mt-1 max-h-64 overflow-y-auto themed-scroll">
            {evidence.map((item) => (
              <EvidenceRow key={item.source_id} item={item} />
            ))}
          </ul>
        </div>
      )}

      {!hasEvidence && !hasCalcs && (
        <p className="text-sm text-mist">
          {meta.retrieval?.note || "No transactions or calculations were needed for this answer."}
        </p>
      )}

      {meta.skipped_calculations.length > 0 && (
        <div>
          <h4 className="text-xs font-semibold uppercase tracking-wide text-mist">Not calculated</h4>
          <ul className="mt-1 space-y-1">
            {meta.skipped_calculations.map((s, i) => (
              <li key={i} className="text-xs text-mist">
                {s}
              </li>
            ))}
          </ul>
        </div>
      )}

      {hasIssues && (
        <div className="rounded-lg border border-brick/25 bg-brick-soft p-3">
          <h4 className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-brick">
            <FileWarning className="h-3.5 w-3.5" aria-hidden="true" />
            The critic flagged this answer
          </h4>
          <ul className="mt-1.5 space-y-1">
            {[...meta.critic_errors, ...meta.critic_unsupported_claims].map((issue, i) => (
              <li key={i} className="text-xs text-brick/90">
                {issue}
              </li>
            ))}
          </ul>
          {meta.critic_retries_used > 0 && (
            <p className="mt-1.5 text-[11px] text-brick/70">
              Retried {meta.critic_retries_used} {meta.critic_retries_used === 1 ? "time" : "times"} before this
              reply was returned.
            </p>
          )}
        </div>
      )}

      <p className="border-t border-border/70 pt-2.5 text-[11px] leading-relaxed text-mist">
        {meta.retrieval?.note || "No retrieval was needed for this answer."}
      </p>
    </div>
  );
}
