/** Formatting helpers. Small and dependency-free on purpose -- pulling
 * in a currency-formatting library for what Intl already does natively
 * would be needless weight for a handful of call sites. */

export function formatCurrency(amount: number, currency: string): string {
  try {
    return new Intl.NumberFormat("en-IN", {
      style: "currency",
      currency: currency || "INR",
      maximumFractionDigits: 0,
    }).format(amount);
  } catch {
    // An unrecognized currency code (Intl throws on those) -- fall back
    // to a plain, still-readable rendering rather than crashing the
    // message that contains it.
    return `${currency} ${amount.toLocaleString("en-IN", { maximumFractionDigits: 0 })}`;
  }
}

export function formatSignedCurrency(amount: number, currency: string): string {
  const formatted = formatCurrency(Math.abs(amount), currency);
  return amount < 0 ? `-${formatted}` : `+${formatted}`;
}

export function formatDate(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleDateString("en-IN", { day: "numeric", month: "short", year: "numeric" });
}

export function formatDateShort(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleDateString("en-IN", { day: "numeric", month: "short" });
}

export function formatLatency(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

export function formatTime(ts: number): string {
  return new Date(ts).toLocaleTimeString("en-IN", { hour: "numeric", minute: "2-digit" });
}

/** Stable-enough id for a locally-created chat message -- not sent to
 * the backend, only used as a React key and for locating "the message
 * currently pending a reply" in local state. */
export function generateId(): string {
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 9)}`;
}
