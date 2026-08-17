"""
Stage 3 of hybrid retrieval: exact/keyword ranking over an already
metadata-filtered candidate set of transactions.

Three-tier fallback, probed at runtime (never assumed -- a sqlite3 build
without the FTS5 extension compiled in is common enough, e.g. some system
Pythons on older distros, that this must be checked, not asserted):

  1. SQLite FTS5 (stdlib `sqlite3`, in-memory virtual table) -- real
     BM25-style ranking via FTS5's built-in `bm25()` auxiliary function.
     No new dependency.
  2. `rank_bm25` (pure-Python, pip package) -- used when the local
     sqlite3 build lacks FTS5.
  3. Case-insensitive substring/token-count scoring -- used when
     `rank_bm25` isn't installed either. Zero dependencies, so this tier
     can never itself be unavailable; it is the floor for stage 3, the
     way recency-ordering is the floor for the whole pipeline.

Every tier operates purely in-memory over the `candidates` list already
returned by `db.search_transactions` -- none of them touch the
transactions table directly or accept a raw SQL fragment from the query
string, so keyword search can never surface a transaction outside the
caller's metadata filter or run arbitrary SQL against user input.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from typing import Literal

from .schemas import Transaction

KeywordTier = Literal["fts5", "rank_bm25", "substring", "none"]

_TOKEN_RE = re.compile(r"\w+")


@dataclass
class KeywordHit:
    source_id: str
    score: float


def _tx_key(tx: Transaction) -> str:
    """Same identity convention used throughout finmate.rag: prefer the
    stable external source_id, fall back to the DB row id."""
    return tx.source_id or (str(tx.id) if tx.id is not None else "")


def searchable_text(tx: Transaction) -> str:
    """Text indexed for keyword matching. Includes `category` alongside
    `description` deliberately: the RAG_AGENT contract treats category as
    a first-class metadata field, and a query like "insurance" or
    "groceries" should be able to hit transactions by their category even
    when the description text itself doesn't repeat that word."""
    return f"{tx.description} {tx.category}"


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def fts5_available() -> bool:
    """Runtime probe -- never assume. A no-op in-memory virtual table
    create/drop; cheap enough to call on every retrieval."""
    try:
        conn = sqlite3.connect(":memory:")
        try:
            conn.execute("CREATE VIRTUAL TABLE probe_fts USING fts5(x)")
        finally:
            conn.close()
        return True
    except sqlite3.OperationalError:
        return False


def rank_bm25_available() -> bool:
    try:
        import rank_bm25  # noqa: F401,PLC0415
    except ImportError:
        return False
    return True


def _build_fts5_index(conn: sqlite3.Connection, candidates: list[Transaction]) -> None:
    """Create and populate the in-memory FTS5 table once. Split out of
    `_keyword_rank_fts5` so `_keyword_rank_multi_fts5` (see "Batched
    multi-query ranking" below) can build the index exactly once and run
    several MATCH queries against it, instead of rebuilding it per query."""
    conn.execute("CREATE VIRTUAL TABLE tx_fts USING fts5(source_id UNINDEXED, text)")
    conn.executemany(
        "INSERT INTO tx_fts (source_id, text) VALUES (?, ?)",
        [(_tx_key(t), searchable_text(t)) for t in candidates],
    )


def _query_fts5_index(conn: sqlite3.Connection, query: str) -> list[KeywordHit]:
    """Run one MATCH query against an already-built FTS5 table."""
    tokens = _tokenize(query)
    if not tokens:
        return []
    # Tokens come from user (or LLM-rewritten) free text, not a trusted
    # FTS5 query string -- quote each token as a literal phrase and OR
    # them together rather than passing raw text to MATCH, so stray
    # FTS5 query-syntax characters in the input (", *, NEAR, -, etc.)
    # can't change the query's meaning or raise a syntax error.
    match_expr = " OR ".join('"' + tok.replace('"', '""') + '"' for tok in tokens)
    rows = conn.execute(
        "SELECT source_id, bm25(tx_fts) AS rank FROM tx_fts WHERE tx_fts MATCH ? ORDER BY rank",
        (match_expr,),
    ).fetchall()
    # FTS5's bm25() returns a *cost* (lower/more negative = better match).
    # Negate it so every tier in this module reports "higher is better",
    # matching how vector cosine-similarity and cross-encoder scores are
    # oriented downstream in finmate/reranker.py.
    return [KeywordHit(source_id=source_id, score=-rank) for source_id, rank in rows]


def _keyword_rank_fts5(candidates: list[Transaction], query: str) -> list[KeywordHit]:
    tokens = _tokenize(query)
    if not tokens:
        return []
    conn = sqlite3.connect(":memory:")
    try:
        _build_fts5_index(conn, candidates)
        return _query_fts5_index(conn, query)
    finally:
        conn.close()


def _keyword_rank_bm25_pkg(candidates: list[Transaction], query: str) -> list[KeywordHit]:
    from rank_bm25 import BM25Okapi  # noqa: PLC0415

    doc_tokens = [_tokenize(searchable_text(t)) for t in candidates]
    if not doc_tokens or not any(doc_tokens):
        return []
    query_tokens = _tokenize(query)
    if not query_tokens:
        return []
    bm25 = BM25Okapi(doc_tokens)
    scores = bm25.get_scores(query_tokens)
    query_token_set = set(query_tokens)
    # Filter on genuine lexical overlap, NOT "score > 0": rank_bm25's
    # classic IDF formula (log((N-df+0.5)/(df+0.5)), no smoothing "+1"
    # inside the log) legitimately produces an IDF of exactly 0 -- for
    # every term -- whenever a term appears in exactly half of a small
    # corpus; verified directly against this package on a 2-document
    # fixture during this build (see README "Deviations"). A candidate
    # set this small is the normal case here (a date/category filter
    # commonly narrows to well under a hundred transactions), so this
    # isn't a rare edge case to shrug off -- "score > 0" would have
    # silently dropped genuine keyword matches for exactly the small
    # candidate sets this tier most often runs against.
    hits = [
        KeywordHit(source_id=_tx_key(t), score=float(s))
        for t, s, tokens in zip(candidates, scores, doc_tokens)
        if query_token_set & set(tokens)
    ]
    hits.sort(key=lambda h: h.score, reverse=True)
    return hits


def _keyword_rank_substring(candidates: list[Transaction], query: str) -> list[KeywordHit]:
    tokens = _tokenize(query)
    if not tokens:
        return []
    hits: list[KeywordHit] = []
    for t in candidates:
        text = searchable_text(t).lower()
        score = sum(text.count(tok) for tok in tokens)
        if score > 0:
            hits.append(KeywordHit(source_id=_tx_key(t), score=float(score)))
    hits.sort(key=lambda h: h.score, reverse=True)
    return hits


def keyword_rank(candidates: list[Transaction], query: str) -> tuple[list[KeywordHit], KeywordTier]:
    """Rank `candidates` by lexical match to `query`, best tier available.

    Returns (hits, tier) where `hits` is sorted best-first (empty if
    nothing matched, or if `query`/`candidates` is empty) and `tier` is
    one of "fts5" | "rank_bm25" | "substring" | "none" -- "none" only
    when there was nothing to search (empty query or empty candidate
    list), so callers can distinguish "ran and found nothing" from
    "never ran". This function never raises: a missing optional
    dependency is a documented fallback, not an error.

    A thin single-query wrapper over `keyword_rank_multi` (see below) --
    kept as its own function because most callers (and every existing
    test in this file) only ever have one query in hand, and because
    `finmate/rag.py` needs the batched form for its own reasons that
    don't belong in this module's public single-query contract.
    """
    if not query or not query.strip() or not candidates:
        return [], "none"
    results, tier = keyword_rank_multi(candidates, [query])
    return results.get(query, []), tier


# ---------------------------------------------------------------------------
# Batched multi-query ranking
# ---------------------------------------------------------------------------
#
# finmate/rag.py's hybrid pipeline ranks the SAME candidate set against
# several query phrasings at once (the user's original wording plus up to
# `query_rewrite.MAX_PHRASINGS` LLM-generated variants -- see
# finmate/query_rewrite.py). Calling `keyword_rank` once per phrasing
# would rebuild the FTS5 table (or the rank_bm25 BM25Okapi index) from
# scratch for every one of them, even though they all search the exact
# same documents -- pure waste for what is, by construction, tens to a
# few hundred transactions, not a fresh corpus each time. `keyword_rank_multi`
# builds the index once and queries it once per phrasing instead; the
# per-query scores are identical to calling `keyword_rank` separately for
# each one (same tokens, same documents, same ranking formula), just
# cheaper to compute.


def keyword_rank_multi(
    candidates: list[Transaction], queries: list[str]
) -> tuple[dict[str, list[KeywordHit]], KeywordTier]:
    """Rank `candidates` against every string in `queries` in one pass,
    sharing a single index build across all of them.

    Returns ({query: hits}, tier). `tier` is the one value `keyword_rank`
    would return for any of these queries in this environment -- it
    reflects which fallback tier is available (FTS5 / rank_bm25 /
    substring), not anything query-specific, so a single environment
    probe covers the whole batch. An empty `queries` or `candidates`
    returns ({}, "none") without building any index. Duplicate entries in
    `queries` are each given their own (identical) result rather than
    deduplicated -- callers with few enough phrasings that this could
    matter (this module's only caller passes at most 1 + MAX_PHRASINGS)
    don't need the added bookkeeping.
    """
    if not candidates or not queries:
        return {}, "none"
    if fts5_available():
        return _keyword_rank_multi_fts5(candidates, queries), "fts5"
    if rank_bm25_available():
        return _keyword_rank_multi_bm25_pkg(candidates, queries), "rank_bm25"
    return {q: _keyword_rank_substring(candidates, q) for q in queries}, "substring"


def _keyword_rank_multi_fts5(candidates: list[Transaction], queries: list[str]) -> dict[str, list[KeywordHit]]:
    conn = sqlite3.connect(":memory:")
    try:
        _build_fts5_index(conn, candidates)
        return {q: _query_fts5_index(conn, q) for q in queries}
    finally:
        conn.close()


def _keyword_rank_multi_bm25_pkg(candidates: list[Transaction], queries: list[str]) -> dict[str, list[KeywordHit]]:
    from rank_bm25 import BM25Okapi  # noqa: PLC0415

    doc_tokens = [_tokenize(searchable_text(t)) for t in candidates]
    if not doc_tokens or not any(doc_tokens):
        return {q: [] for q in queries}

    # The relatively expensive part -- computing IDF weights over the
    # whole candidate set -- happens once here rather than once per query.
    bm25 = BM25Okapi(doc_tokens)

    results: dict[str, list[KeywordHit]] = {}
    for q in queries:
        query_tokens = _tokenize(q)
        if not query_tokens:
            results[q] = []
            continue
        scores = bm25.get_scores(query_tokens)
        # See _keyword_rank_bm25_pkg's docstring for why this filters on
        # genuine lexical overlap rather than "score > 0".
        query_token_set = set(query_tokens)
        hits = [
            KeywordHit(source_id=_tx_key(t), score=float(s))
            for t, s, tokens in zip(candidates, scores, doc_tokens)
            if query_token_set & set(tokens)
        ]
        hits.sort(key=lambda h: h.score, reverse=True)
        results[q] = hits
    return results
