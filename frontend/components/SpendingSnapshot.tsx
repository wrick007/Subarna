"use client";

import { useMemo } from "react";
import { Bar, BarChart, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { formatCurrency } from "@/lib/format";
import type { Transaction } from "@/lib/types";

const BAR_COLOR = "#1e5f52"; // --color-ledger, hardcoded: recharts renders to SVG fill attrs, which don't resolve CSS custom properties reliably across browsers
const BAR_COLOR_MUTED = "#a9c4bd";

export default function SpendingSnapshot({ transactions }: { transactions: Transaction[] }) {
  const { income, spend, currency, topCategories } = useMemo(() => {
    let income = 0;
    let spend = 0;
    const byCategory = new Map<string, number>();
    let currency = "INR";

    for (const t of transactions) {
      currency = t.currency || currency;
      if (t.amount >= 0) {
        income += t.amount;
      } else {
        spend += Math.abs(t.amount);
        byCategory.set(t.category, (byCategory.get(t.category) ?? 0) + Math.abs(t.amount));
      }
    }

    const topCategories = [...byCategory.entries()]
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5)
      .map(([category, amount]) => ({ category, amount }));

    return { income, spend, currency, topCategories };
  }, [transactions]);

  if (transactions.length === 0) {
    return (
      <p className="text-xs leading-relaxed text-mist">
        No transactions loaded yet. Load the demo dataset below, or ask a question once your own data is connected.
      </p>
    );
  }

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 gap-2">
        <div className="rounded-lg border border-border bg-surface px-3 py-2">
          <p className="text-[11px] uppercase tracking-wide text-mist">Income</p>
          <p className="tabular-nums mt-0.5 text-sm font-medium text-ledger-dark">
            {formatCurrency(income, currency)}
          </p>
        </div>
        <div className="rounded-lg border border-border bg-surface px-3 py-2">
          <p className="text-[11px] uppercase tracking-wide text-mist">Spend</p>
          <p className="tabular-nums mt-0.5 text-sm font-medium text-ink">{formatCurrency(spend, currency)}</p>
        </div>
      </div>

      {topCategories.length > 0 && (
        <div>
          <p className="mb-1 text-[11px] uppercase tracking-wide text-mist">Top categories</p>
          <div style={{ width: "100%", height: 130 }}>
            <ResponsiveContainer>
              <BarChart data={topCategories} layout="vertical" margin={{ top: 0, right: 8, bottom: 0, left: 0 }}>
                <XAxis type="number" hide />
                <YAxis
                  type="category"
                  dataKey="category"
                  width={92}
                  tick={{ fontSize: 11, fill: "#4b5563" }}
                  axisLine={false}
                  tickLine={false}
                />
                <Tooltip
                  cursor={{ fill: "rgba(30,95,82,0.06)" }}
                  formatter={(value) => formatCurrency(Number(value ?? 0), currency)}
                  labelFormatter={() => ""}
                  contentStyle={{
                    fontSize: 12,
                    borderRadius: 8,
                    border: "1px solid #e1e3da",
                    boxShadow: "none",
                  }}
                />
                <Bar dataKey="amount" radius={[0, 4, 4, 0]} maxBarSize={14}>
                  {topCategories.map((_, i) => (
                    <Cell key={i} fill={i === 0 ? BAR_COLOR : BAR_COLOR_MUTED} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  );
}
