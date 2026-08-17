#!/usr/bin/env python3
"""
Offline evaluation harness for finmate.rag's hybrid retrieval pipeline.

Computes Recall@5, Recall@10, and MRR (each within the same top-10 ranked
list -- effectively "MRR@10") against the hand-labeled golden set in
data/rag_eval_set.json, run against demo_user's real seeded transactions
(data/synthetic_transactions.csv). All three metrics are macro-averaged
across queries: each query counts equally regardless of how many gold
transactions it has, since a couple of broad-category queries in the
eval set ("food expenses": 17 gold items) have far more gold items than
the keyword-exact ones (2 each) -- micro-averaging would let those few
broad queries dominate the aggregate.

Also runs a frozen snapshot of the pre-upgrade retrieve() (metadata
filter -> optional single-pass vector rerank, no keyword stage, no
fusion, no cross-encoder -- see `legacy_retrieve` below) for a genuine
before/after comparison, not just an assertion that the new pipeline is
better.

Fully offline by default: builds an isolated demo DB + vector index (at
--db-path/--qdrant-path, which default to eval-only files so this never
touches a real seeded data/finmate.db), and makes zero LLM calls unless
--with-query-rewrite is passed. Every optional stage degrades exactly the
way finmate/rag.py documents if its dependency isn't available -- this
script prints a capability report up front so the numbers below are
self-explanatory without cross-referencing the README.

Usage:
    python scripts/eval_rag.py
    python scripts/eval_rag.py --with-query-rewrite
    python scripts/eval_rag.py --db-path data/finmate.db --qdrant-path data/qdrant_store
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import statistics
import sys
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from finmate import db, keyword_search, rag, reranker  # noqa: E402
from finmate.schemas import Transaction  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
DEFAULT_EVAL_SET_PATH = DATA_DIR / "rag_eval_set.json"
TOP_KS = (5, 10)
RETRIEVE_DEPTH = 10  # one ranked list per query; Recall@5/@10/MRR are all sliced from it


# ---------------------------------------------------------------------------
# Isolated seeding -- deliberately NOT importing scripts/seed_demo_data.py:
# that script always builds its vector index at finmate.rag's *default*
# qdrant_path, but this harness needs its own isolated qdrant_path (see
# module docstring) so a `python scripts/eval_rag.py` run never touches,
# or is affected by, data/qdrant_store from a real seeded app instance.
# This is a small, deliberate duplication of that script's CSV-loading
# logic, not a divergence in what gets seeded -- same CSV, same fields.
# ---------------------------------------------------------------------------


def seed_eval_db(user_id: str, db_path: str, qdrant_path: str) -> tuple[list[Transaction], int]:
    db.init_db(db_path)
    tx_path = DATA_DIR / "synthetic_transactions.csv"
    with tx_path.open(newline="") as f:
        transactions = [
            Transaction(
                user_id=row["user_id"], date=row["date"], description=row["description"],
                amount=float(row["amount"]), currency=row["currency"], category=row["category"],
                account=row["account"], type=row["type"], source_id=row["source_id"],
            )
            for row in csv.DictReader(f)
            if row["user_id"] == user_id
        ]
    db.insert_transactions(transactions, db_path=db_path)
    indexed = rag.index_transactions_for_user(user_id, transactions, db_path=db_path, qdrant_path=qdrant_path)
    return transactions, indexed


# ---------------------------------------------------------------------------
# Capability probe
# ---------------------------------------------------------------------------


def probe_capabilities(embedding_model: str) -> dict[str, str]:
    caps = {
        "fts5": "available" if keyword_search.fts5_available() else "unavailable -> falls back to rank_bm25/substring",
        "rank_bm25 package": "installed" if keyword_search.rank_bm25_available() else "not installed",
        "embedding model (vector search)": "loaded" if rag._get_embedder(embedding_model) is not None
            else "unavailable (sentence-transformers not installed, or the model couldn't be loaded/downloaded)",
        "cross-encoder (rerank)": "loaded" if reranker._get_cross_encoder() is not None
            else "unavailable (sentence-transformers not installed, or the model couldn't be loaded/downloaded)",
        "Groq/Gemini key (query rewrite)": "configured" if (os.environ.get("GROQ_API_KEY") or os.environ.get("GEMINI_API_KEY"))
            else "not configured -- stage 2 will report used=False if exercised",
    }
    return caps


def print_capability_report(caps: dict[str, str]) -> None:
    print("=== Capability probe (what THIS run could actually exercise) ===")
    for name, status in caps.items():
        print(f"  {name:34s}: {status}")
    print()


# ---------------------------------------------------------------------------
# Frozen snapshot of the pre-upgrade retrieve() -- kept ONLY so this
# harness can report a genuine before/after; not used anywhere else in
# the app. Copied verbatim (logic-for-logic) from this repo's actual
# finmate/rag.py as it existed before this build -- it already called
# `.query_points()` rather than the now-removed `.search()`, so unlike a
# separate copy of this codebase this build also touched, there's no
# AttributeError guard needed here to keep it runnable. What it did NOT
# have is a payload filter: it fetched a fixed number of *globally*-
# ranked hits and filtered down to the metadata-filtered candidate set
# afterward, in Python -- see README "Deviations" for why that's a real
# gap, not just a style difference, and
# tests/test_rag_hybrid.py::test_vector_search_finds_target_even_crowded_by_a_tiny_global_limit
# for a deterministic reproduction of it.
# ---------------------------------------------------------------------------


def legacy_retrieve(
    user_id: str,
    query: str,
    top_k: int,
    db_path: str,
    qdrant_path: str,
    embedding_model: str,
    start_date: str | None = None,
    end_date: str | None = None,
    category: str | None = None,
    account: str | None = None,
) -> list[str]:
    candidates = db.search_transactions(
        user_id, start_date=start_date, end_date=end_date, category=category, account=account, db_path=db_path,
    )

    def _key(t: Transaction) -> str:
        return t.source_id or str(t.id)

    def _recency(items: list[Transaction]) -> list[str]:
        return [_key(t) for t in sorted(items, key=lambda t: t.date, reverse=True)[:top_k]]

    if not query or not query.strip() or not candidates:
        return _recency(candidates)

    embedder = rag._get_embedder(embedding_model)
    client = rag._get_qdrant_client(qdrant_path)
    collection = f"{rag.COLLECTION_PREFIX}{user_id}"
    if embedder is None or client is None or not client.collection_exists(collection):
        return _recency(candidates)

    candidate_ids = {_key(t) for t in candidates}
    query_vector = embedder.encode([query])[0].tolist()
    # No query_filter here -- this IS the pre-upgrade behavior being
    # measured, not a simplification for this script's sake.
    search_result = client.query_points(collection_name=collection, query=query_vector, limit=top_k * 3)
    hits = search_result.points

    ranked: list[str] = []
    for hit in hits:
        source_id = (hit.payload or {}).get("source_id")
        if source_id in candidate_ids and source_id not in ranked:
            ranked.append(source_id)
        if len(ranked) >= top_k:
            break
    return ranked if ranked else _recency(candidates)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def recall_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    return len(set(ranked_ids[:k]) & relevant_ids) / len(relevant_ids)


def reciprocal_rank(ranked_ids: list[str], relevant_ids: set[str]) -> float:
    for i, sid in enumerate(ranked_ids, start=1):
        if sid in relevant_ids:
            return 1.0 / i
    return 0.0


def _macro_avg(rows: list[dict], key: str) -> float:
    return statistics.mean(r[key] for r in rows) if rows else 0.0


def evaluate(eval_set: dict, retrieve_fn: Callable[..., list[str]]) -> dict:
    per_query = []
    for q in eval_set["queries"]:
        relevant = set(q["relevant_source_ids"])
        filters = q.get("filters") or {}
        ranked_ids = retrieve_fn(query=q["query"], **filters)
        row = {"id": q["id"], "query_type": q.get("query_type", "unlabeled"), "n_relevant": len(relevant)}
        for k in TOP_KS:
            row[f"recall@{k}"] = recall_at_k(ranked_ids, relevant, k)
        row["mrr"] = reciprocal_rank(ranked_ids, relevant)
        per_query.append(row)

    summary = {"n_queries": len(per_query)}
    for k in TOP_KS:
        summary[f"recall@{k}"] = _macro_avg(per_query, f"recall@{k}")
    summary["mrr"] = _macro_avg(per_query, "mrr")

    by_type: dict[str, list[dict]] = {}
    for row in per_query:
        by_type.setdefault(row["query_type"], []).append(row)
    type_summary = {}
    for qtype, rows in sorted(by_type.items()):
        ts = {"n_queries": len(rows)}
        for k in TOP_KS:
            ts[f"recall@{k}"] = _macro_avg(rows, f"recall@{k}")
        ts["mrr"] = _macro_avg(rows, "mrr")
        type_summary[qtype] = ts

    return {"per_query": per_query, "summary": summary, "by_type": type_summary}


def print_report(title: str, report: dict) -> None:
    s = report["summary"]
    print(f"--- {title} ---")
    print(f"  ALL (n={s['n_queries']:2d})            Recall@5={s['recall@5']:.3f}  Recall@10={s['recall@10']:.3f}  MRR={s['mrr']:.3f}")
    for qtype, ts in report["by_type"].items():
        print(f"    {qtype:26s} (n={ts['n_queries']:2d})  Recall@5={ts['recall@5']:.3f}  Recall@10={ts['recall@10']:.3f}  MRR={ts['mrr']:.3f}")
    print()


def print_comparison(before: dict, after: dict) -> None:
    b, a = before["summary"], after["summary"]
    print("=== Before / after, all queries (macro-averaged) ===")
    print(f"  {'metric':10s} {'before':>8s} {'after':>8s} {'delta':>8s}")
    for metric in ("recall@5", "recall@10", "mrr"):
        print(f"  {metric:10s} {b[metric]:8.3f} {a[metric]:8.3f} {a[metric] - b[metric]:+8.3f}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db-path", default=str(DATA_DIR / "eval_finmate.db"),
                         help="Isolated by default -- never touches your real data/finmate.db.")
    parser.add_argument("--qdrant-path", default=str(DATA_DIR / "eval_qdrant_store"),
                         help="Isolated by default -- never touches your real data/qdrant_store.")
    parser.add_argument("--eval-set", default=str(DEFAULT_EVAL_SET_PATH))
    parser.add_argument("--with-query-rewrite", action="store_true",
                         help="Also exercise stage 2 (<=1 Groq/Gemini call per unique query, via "
                              "FINMATE_LLM_PROVIDER + the matching API key). Off by default so this "
                              "script makes zero LLM calls and needs no key to run.")
    parser.add_argument("--top-k", type=int, default=RETRIEVE_DEPTH)
    args = parser.parse_args()

    eval_set = json.loads(Path(args.eval_set).read_text())
    user_id = eval_set["user_id"]

    Path(args.db_path).unlink(missing_ok=True)
    shutil.rmtree(args.qdrant_path, ignore_errors=True)
    _transactions, indexed = seed_eval_db(user_id, args.db_path, args.qdrant_path)
    print(f"Seeded {len(_transactions)} transactions for user_id={user_id!r} "
          f"({indexed} indexed into the vector store).\n")

    caps = probe_capabilities(rag.DEFAULT_EMBEDDING_MODEL)
    print_capability_report(caps)

    def before_fn(query: str, **filters) -> list[str]:
        return legacy_retrieve(
            user_id=user_id, query=query, top_k=args.top_k, db_path=args.db_path,
            qdrant_path=args.qdrant_path, embedding_model=rag.DEFAULT_EMBEDDING_MODEL, **filters,
        )

    def after_fn(query: str, **filters) -> list[str]:
        result = rag.retrieve(
            user_id=user_id, query=query, top_k=args.top_k, db_path=args.db_path,
            qdrant_path=args.qdrant_path, enable_query_rewrite=args.with_query_rewrite, **filters,
        )
        return [e.source_id for e in result.evidence]

    before_report = evaluate(eval_set, before_fn)
    after_report = evaluate(eval_set, after_fn)

    print_report("BEFORE  (pre-upgrade: metadata filter + single-pass vector rerank only)", before_report)
    after_title = "AFTER   (hybrid: keyword + vector + RRF fusion + cross-encoder rerank"
    after_title += " + query rewrite)" if args.with_query_rewrite else ", query rewrite off)"
    print_report(after_title, after_report)
    print_comparison(before_report, after_report)

    out_path = DATA_DIR / "rag_eval_results.json"
    out_path.write_text(json.dumps(
        {"capabilities": caps, "with_query_rewrite": args.with_query_rewrite, "before": before_report, "after": after_report},
        indent=2,
    ))
    print(f"Full per-query results written to {out_path}")


if __name__ == "__main__":
    main()
