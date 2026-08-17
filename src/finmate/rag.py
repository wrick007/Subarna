"""
Hybrid retrieval for the Transaction/RAG Agent (Stage 3).

Six-stage pipeline, per the RAG_AGENT prompt contract ("Use hybrid
retrieval when available: 1. exact/keyword search; 2. vector retrieval;
3. metadata filtering; 4. reranking" -- metadata filtering runs first in
code, since every later stage searches *within* its output):

  1. Metadata filter (SQLite, via db.search_transactions) -- always runs,
     needs no ML dependency, no network, no API key. This is the floor:
     every other stage narrows or reorders this stage's output, never
     replaces it, and every EvidenceItem returned traces back to a row
     this stage actually fetched.
  2. Query understanding (OPTIONAL, ≤1 LLM call, finmate/query_rewrite.py,
     Groq/Gemini via the existing LLMClient) -- expands the query into
     2-3 short search-oriented phrasings to help stage 3 match
     transactions with no literal keyword overlap (e.g. "food expenses"
     -> "dining", "groceries"). Cached per (user_id, normalized query);
     skips cleanly on any failure, a missing key, or FINMATE_RAG_MODE=no_llm.
  3. Keyword/exact search (finmate/keyword_search.py) over the
     metadata-filtered candidates -- SQLite FTS5, falling back to
     rank_bm25, falling back to substring scoring, whichever is
     available (probed at runtime).
  4. Dense vector search (Qdrant + sentence-transformers, reused from the
     pre-upgrade implementation) -- scoped to the metadata-filtered
     candidate set via a Qdrant payload filter rather than fetching a
     fixed number of globally-ranked hits and filtering to the candidate
     set in Python afterward (see "Deviations" in the README for why
     that could silently miss a relevant item).
  5. Fusion (finmate/reranker.py:reciprocal_rank_fusion) -- Reciprocal
     Rank Fusion of the keyword and vector rankings; never a raw score
     blend, since the two scores live on incompatible scales.
  6. Local cross-encoder rerank (finmate/reranker.py:cross_encoder_rerank)
     of the fused top-N against the (possibly-rewritten) query.

Fallback ladder -- `RetrievalResult.stage` reports exactly which ran, and
`.note` explains why any didn't:

  "full hybrid (rewrite+keyword+vector+rerank)"
  "keyword+vector+rerank (no query rewrite)"
  "keyword+vector, no rerank"
  "vector+rerank (no keyword)"        [reachable; not named in the spec's
  "keyword+rerank (no vector)"         6-tier ladder, since rerank runs on
                                        whatever fusion produced, independent
                                        of *which* signal fed it -- see README]
  "vector only"
  "keyword only"
  "metadata-only, recency-ordered"    (today's floor -- never removed)

Every stage from 3 onward degrades on its own: a missing dependency, an
empty index, a network-unreachable model download, or an unexpected
error from the ML/vector stack all fall through to the next tier rather
than raising. `RetrievalResult.vector_search_used` / `.note` are kept
(existing contract other agents and the Critic already read via
`finmate/agents/_shared.py`); `.keyword_search_used`, `.rerank_used`,
`.query_rewrite_used`, and `.stage` are additive.

Performance (speed + token cost, same providers, same ranking output):

  - Model/client singletons. `_get_embedder`, `_get_qdrant_client` (here)
    and `reranker._get_cross_encoder` are cached per model_name/path for
    the process lifetime, including a *failed* load. Loading model
    weights (or reconnecting an embedded Qdrant client) is by far the
    slowest thing in this pipeline; the pre-upgrade code paid that cost
    on every single retrieval call. A long-lived server process now pays
    it once. See each function's docstring and `clear_cache()`.
  - Concurrency. Stage 4 (vector search) depends only on the query text
    and the candidate set -- never on stage 2's output -- so it's kicked
    off on a background thread before stage 2 (query rewrite, an LLM
    network round-trip) even starts, and only awaited once stage 3
    (keyword search) is done. Wall-clock cost drops from roughly
    rewrite + keyword + vector to roughly max(vector, rewrite + keyword).
    The ranking output is unaffected -- this only changes *when* the
    already-independent stage 4 computation happens, not what it computes.
  - Batched keyword search. Stage 3 ranks the candidate set against the
    original query *and* every query-rewrite phrasing. The pre-upgrade
    code called `keyword_search.keyword_rank` once per phrasing, rebuilding
    the in-memory FTS5 table (or rank_bm25 index) from scratch each time.
    `keyword_search.keyword_rank_multi` builds it once and queries it once
    per phrasing -- identical scores, a fraction of the cost.
  - Request-scoped memoization (`cache=` on `retrieve`). Optional, off by
    default. The orchestrator passes a fresh dict per user turn, so a
    Critic-triggered retry -- which re-runs this exact pipeline stage for
    the exact same query -- hits an in-memory cache instead of redoing
    keyword+vector+rerank (and a second query-rewrite LLM call) from
    scratch. Safe because retrieval is a pure function of on-disk state
    that a single turn's retries never mutate in between, and the cache
    never outlives the turn that created it.
  - Token cost downstream. This module's own output is unaffected, but
    see `finmate/agents/_shared.py` and `finmate/agents/critic.py`: the
    evidence this module returns is serialized compactly (not
    pretty-printed) and trimmed to the fields an LLM actually needs
    before being sent to any prompt -- audit-only fields like
    `keyword_score`/`vector_score`/`rerank_score`/`retrieval_stage` stay
    on `RetrievalResult`/`EvidenceItem` for the API/UI, just not in the
    tokens billed to Groq/Gemini.
"""

from __future__ import annotations

import atexit
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

from . import db, keyword_search, query_rewrite, reranker
from .llm import LLMClient
from .schemas import EvidenceItem, Transaction

logger = logging.getLogger("finmate.rag")

DEFAULT_QDRANT_PATH = "data/qdrant_store"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_PREFIX = "finmate_tx_"
DEFAULT_CROSS_ENCODER_MODEL = reranker.DEFAULT_CROSS_ENCODER_MODEL

# How many fused candidates the cross-encoder reranks. Larger than the
# typical top_k so the (cheap) fusion stage has room to be wrong and the
# (expensive, more accurate) cross-encoder can still promote a good match
# fusion under-ranked -- but bounded, since cross-encoder scoring cost is
# roughly linear in this number.
DEFAULT_RERANK_POOL = 30

# How many hits to request from Qdrant per query before candidate-set
# filtering narrows them. Kept well above top_k so the payload filter
# (scoped to the metadata-filtered candidate set) still has enough raw
# hits to work with.
DEFAULT_VECTOR_FETCH_LIMIT = 50

# Upper bound on how long retrieve() will wait on the background vector
# search before giving up on it for this call (see "Performance" above).
# Generous for a local embedded-Qdrant + small local embedding model --
# this exists to bound worst-case latency if something is badly wrong,
# not because that path is normally anywhere close to this slow.
VECTOR_SEARCH_TIMEOUT_SECONDS = 15.0

# --- process-lifetime singleton caches (see module docstring "Performance") ---
# Each maps its key (model_name, or on-disk path) to the loaded object,
# or to None for "already tried, unavailable" -- see _get_embedder /
# _get_qdrant_client docstrings.
_embedder_cache: dict[str, object] = {}
_embedder_lock = threading.Lock()
_qdrant_client_cache: dict[str, object] = {}
_qdrant_client_lock = threading.Lock()

# Shared across every retrieve() call in this process. Submitting a task
# to an existing pool is cheap; creating/tearing down a new
# ThreadPoolExecutor per call would not be. Bounded rather than
# unbounded: today, the only thing ever submitted here is one vector
# search per retrieve() call, so a handful of workers is generous
# headroom for several concurrent chat turns in a server process, not a
# resource hog.
_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="finmate-rag")


def delete_user_vector_index(user_id: str, qdrant_path: str = DEFAULT_QDRANT_PATH) -> bool:
    """Drop this user's entire Qdrant collection (`{COLLECTION_PREFIX}{user_id}`),
    if one exists. The data-deletion counterpart to `index_transactions_for_user`:
    `finmate.db.delete_user_data` only ever touched SQLite, so calling
    just that (as this module's very first version did -- see
    backend/app/routers/users.py and app.py's "forget this user's data")
    left an orphaned vector collection behind under that same user_id.
    Never raises: a missing client/collection is treated as "nothing to
    delete", same as every other `_get_*`-backed function in this
    module.
    """
    client = _get_qdrant_client(qdrant_path)
    if client is None:
        return False
    collection = f"{COLLECTION_PREFIX}{user_id}"
    try:
        if not client.collection_exists(collection):
            return False
        client.delete_collection(collection_name=collection)
        return True
    except Exception as exc:  # noqa: BLE001 -- deletion failing must degrade, never crash the caller
        logger.info("Could not delete vector collection for %s (%s).", user_id, exc)
        return False


def warm_up(
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    cross_encoder_model: str = DEFAULT_CROSS_ENCODER_MODEL,
    qdrant_path: str = DEFAULT_QDRANT_PATH,
) -> dict[str, bool]:
    """Pre-load every cacheable model/client this module and `reranker`
    use, so the *first real retrieval* doesn't pay the (by far largest --
    see module docstring "Performance") cost of loading them. A one-off
    background script paid that cost once regardless; a server process
    should pay it at startup, once, deliberately, not on whichever user's
    turn happens to be first.

    Safe to call even when the ML dependencies aren't installed --
    reuses the same never-raises `_get_*` accessors `retrieve` itself
    calls, so a missing/unreachable dependency here is reported in the
    returned dict, not raised. Intended for `backend/app/main.py`'s
    startup hook; also handy to call once at the top of a long batch job
    for the same reason.

    Returns which pieces actually became available, e.g.
    `{"embedder": True, "qdrant_client": True, "cross_encoder": False}`.
    """
    return {
        "embedder": _get_embedder(embedding_model) is not None,
        "qdrant_client": _get_qdrant_client(qdrant_path) is not None,
        "cross_encoder": reranker._get_cross_encoder(cross_encoder_model) is not None,
    }


def clear_cache() -> None:
    """Test hook / operational escape hatch: forget cached embedder and
    Qdrant client instances (including cached load failures), so the
    next call attempts a fresh load. See module docstring "Performance".
    Does not affect `reranker`'s cross-encoder cache or
    `query_rewrite`'s rewrite cache -- see `reranker.clear_cache()` and
    `query_rewrite.clear_cache()` for those."""
    with _embedder_lock:
        _embedder_cache.clear()
    with _qdrant_client_lock:
        for client in _qdrant_client_cache.values():
            _close_qdrant_client(client)
        _qdrant_client_cache.clear()


def _close_qdrant_client(client: object) -> None:
    if client is None:
        return
    try:
        client.close()
    except Exception:  # noqa: BLE001 -- best-effort cleanup only
        pass


@atexit.register
def _close_all_cached_qdrant_clients_at_exit() -> None:
    """Embedded-mode QdrantClient holds an on-disk lock and its own
    __del__ tries to release it -- harmless, but by the time the
    interpreter is tearing down module globals that can fire after
    `sys.meta_path` is already cleared, which prints a scary-looking (but
    functionally inert) "Exception ignored" traceback. Closing every
    cached client explicitly here, while the interpreter is still fully
    intact, avoids that -- worth doing now that clients are long-lived
    singletons instead of the pre-upgrade one-per-call instances, which
    were typically already gone (refcounted away) long before shutdown."""
    for client in _qdrant_client_cache.values():
        _close_qdrant_client(client)


@dataclass
class RetrievalResult:
    evidence: list[EvidenceItem] = field(default_factory=list)
    vector_search_used: bool = False
    note: str = ""
    # --- RAG hybrid-retrieval upgrade: additive fields ---
    keyword_search_used: bool = False
    rerank_used: bool = False
    query_rewrite_used: bool = False
    stage: str = "metadata-only, recency-ordered"


def _tx_key(tx: Transaction) -> str:
    return tx.source_id or (str(tx.id) if tx.id is not None else "")


def _transaction_to_evidence(
    tx: Transaction,
    relevance: float = 0.0,
    keyword_score: Optional[float] = None,
    vector_score: Optional[float] = None,
    rerank_score: Optional[float] = None,
    retrieval_stage: str = "",
) -> EvidenceItem:
    return EvidenceItem(
        source_id=_tx_key(tx),
        date=tx.date,
        description=tx.description,
        amount=tx.amount,
        currency=tx.currency,
        category=tx.category,
        document=tx.account,
        page=None,
        relevance=relevance,
        keyword_score=keyword_score,
        vector_score=vector_score,
        rerank_score=rerank_score,
        retrieval_stage=retrieval_stage,
    )


def _recency_ordered_evidence(candidates: list[Transaction], top_k: int) -> list[EvidenceItem]:
    ordered = sorted(candidates, key=lambda t: t.date, reverse=True)[:top_k]
    return [_transaction_to_evidence(t, retrieval_stage="metadata") for t in ordered]


def _get_embedder(model_name: str = DEFAULT_EMBEDDING_MODEL):
    """Lazily import + instantiate sentence-transformers' SentenceTransformer.

    Returns None -- never raises -- if the package isn't installed, OR if
    instantiation fails for any reason (most commonly: model weights
    can't be downloaded because Hugging Face Hub is unreachable --
    verified as a real failure mode in this build's sandboxed
    environment; see README "Deviations").

    Cached per `model_name` for the process lifetime -- see module
    docstring "Performance". Double-checked locking: the common case
    (already cached) never touches the lock; only a genuine first load
    for a given model_name contends for it. This matters more than it
    might for a typical cache, since `_vector_search` calls this from a
    background thread (see "Performance") that can genuinely race a
    concurrent request's main-thread call.
    """
    if model_name in _embedder_cache:
        return _embedder_cache[model_name]
    with _embedder_lock:
        if model_name in _embedder_cache:  # re-check: another thread may have just finished loading
            return _embedder_cache[model_name]
        try:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415
        except ImportError:
            _embedder_cache[model_name] = None
            return None
        try:
            embedder = SentenceTransformer(model_name)
        except Exception as exc:  # noqa: BLE001 -- model load failure must degrade, not crash
            logger.info("Embedding model unavailable (%s); falling back.", exc)
            embedder = None
        _embedder_cache[model_name] = embedder
        return embedder


def _get_qdrant_client(path: str = DEFAULT_QDRANT_PATH):
    """Lazily import + instantiate QdrantClient in embedded/on-disk mode.
    Returns None -- never raises -- if the package isn't installed or
    instantiation fails.

    Cached per `path` for the process lifetime -- see module docstring
    "Performance". This is not just a speed optimization: embedded-mode
    Qdrant only allows one open client per on-disk path, so repeatedly
    constructing a fresh, never-explicitly-closed client against the
    same path on every call (the pre-upgrade behavior) relied on
    CPython's reference-counting GC collecting the previous instance
    before the next one opened -- reusing a single cached client removes
    that fragility along with the repeated open/lock overhead.
    """
    if path in _qdrant_client_cache:
        return _qdrant_client_cache[path]
    with _qdrant_client_lock:
        if path in _qdrant_client_cache:
            return _qdrant_client_cache[path]
        try:
            from qdrant_client import QdrantClient  # noqa: PLC0415
        except ImportError:
            _qdrant_client_cache[path] = None
            return None
        try:
            client = QdrantClient(path=path)
        except Exception as exc:  # noqa: BLE001
            logger.info("Qdrant client unavailable (%s); falling back.", exc)
            client = None
        _qdrant_client_cache[path] = client
        return client


def index_transactions_for_user(
    user_id: str,
    transactions: Optional[list[Transaction]] = None,
    db_path: str = db.DEFAULT_DB_PATH,
    qdrant_path: str = DEFAULT_QDRANT_PATH,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> int:
    """(Re)build the vector index for one user's transactions.

    Returns the number of points indexed, or 0 if the ML dependencies are
    not installed or not loadable (a warning-level no-op, not an error --
    metadata and keyword search still work fully without this).
    """
    embedder = _get_embedder(embedding_model)
    client = _get_qdrant_client(qdrant_path)
    if embedder is None or client is None:
        return 0

    from qdrant_client.http import models as qmodels  # noqa: PLC0415

    txs = transactions if transactions is not None else db.search_transactions(user_id, db_path=db_path)
    if not txs:
        return 0

    collection = f"{COLLECTION_PREFIX}{user_id}"
    vector_size = embedder.get_sentence_embedding_dimension()
    if not client.collection_exists(collection):
        client.create_collection(
            collection_name=collection,
            vectors_config=qmodels.VectorParams(size=vector_size, distance=qmodels.Distance.COSINE),
        )

    texts = [f"{t.date} {t.description} {t.category} {t.amount}{t.currency}" for t in txs]
    vectors = embedder.encode(texts).tolist()
    points = [
        qmodels.PointStruct(
            id=t.id if t.id is not None else i,
            vector=vectors[i],
            payload={"user_id": user_id, "date": t.date, "category": t.category,
                     "account": t.account, "source_id": t.source_id or str(t.id)},
        )
        for i, t in enumerate(txs)
    ]
    client.upsert(collection_name=collection, points=points)
    return len(points)


def _vector_search(
    user_id: str,
    query: str,
    candidate_ids: set[str],
    qdrant_path: str,
    embedding_model: str,
    limit: int,
) -> tuple[list[str], dict[str, float]]:
    """Best-effort dense vector search, scoped server-side to
    `candidate_ids` via a Qdrant payload filter.

    The pre-upgrade version of this code already called
    `client.query_points(...)` (correctly avoiding `.search()`, which
    `qdrant-client` has since removed) but with no `query_filter` -- it
    fetched a fixed number of globally-ranked hits and filtered to the
    candidate set *after the fact*, in Python. That's a real gap: for a
    small candidate set inside a larger collection, a relevant item can
    be missed entirely if it doesn't happen to fall within the top
    `limit` *global* hits, even though it would easily be the best match
    *within* the candidate set. Filtering inside the query itself doesn't
    have that blind spot -- proven, not just asserted, by
    `tests/test_rag_hybrid.py::test_vector_search_finds_target_even_crowded_by_a_tiny_global_limit`.

    Returns ([], {}) -- never raises -- if the embedder or Qdrant client
    aren't available, the user's collection doesn't exist, or the search
    call fails for any reason (a broken/incompatible vector stack must
    degrade this stage, not crash retrieval). Called from a background
    thread by `retrieve` (see module docstring "Performance") as well as
    directly in tests, so it must not assume it's on the main thread.
    """
    if not candidate_ids:
        return [], {}
    embedder = _get_embedder(embedding_model)
    client = _get_qdrant_client(qdrant_path)
    if embedder is None or client is None:
        return [], {}

    collection = f"{COLLECTION_PREFIX}{user_id}"
    try:
        from qdrant_client.http import models as qmodels  # noqa: PLC0415

        if not client.collection_exists(collection):
            return [], {}
        query_vector = embedder.encode([query])[0].tolist()
        response = client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=limit,
            query_filter=qmodels.Filter(
                must=[qmodels.FieldCondition(key="source_id", match=qmodels.MatchAny(any=list(candidate_ids)))]
            ),
        )
        points = response.points
    except Exception as exc:  # noqa: BLE001 -- a broken/incompatible/unavailable vector stack must degrade, not crash
        logger.info("Vector search unavailable or failed, falling back: %s", exc)
        return [], {}

    ranked_ids: list[str] = []
    scores: dict[str, float] = {}
    for point in points:
        source_id = (point.payload or {}).get("source_id")
        if source_id and source_id in candidate_ids and source_id not in scores:
            ranked_ids.append(source_id)
            scores[source_id] = float(point.score)
    return ranked_ids, scores


def _describe_stage(keyword_used: bool, vector_used: bool, rerank_used: bool, query_rewrite_used: bool) -> str:
    """Map which stages actually ran to one of the documented fallback-
    ladder labels (see module docstring). Only called once we already
    know at least one of keyword_used/vector_used is True -- retrieve()
    returns the "metadata-only" floor directly otherwise."""
    if keyword_used and vector_used:
        if rerank_used:
            return (
                "full hybrid (rewrite+keyword+vector+rerank)"
                if query_rewrite_used
                else "keyword+vector+rerank (no query rewrite)"
            )
        return "keyword+vector, no rerank"
    if vector_used:
        return "vector+rerank (no keyword)" if rerank_used else "vector only"
    return "keyword+rerank (no vector)" if rerank_used else "keyword only"


def retrieve(
    user_id: str,
    query: str = "",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    category: Optional[str] = None,
    account: Optional[str] = None,
    top_k: int = 20,
    db_path: str = db.DEFAULT_DB_PATH,
    qdrant_path: str = DEFAULT_QDRANT_PATH,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    cross_encoder_model: str = DEFAULT_CROSS_ENCODER_MODEL,
    llm_client: Optional[LLMClient] = None,
    enable_query_rewrite: Optional[bool] = None,
    enable_keyword_search: bool = True,
    enable_vector_search: bool = True,
    enable_rerank: bool = True,
    rerank_pool_size: int = DEFAULT_RERANK_POOL,
    cache: Optional[dict] = None,
) -> RetrievalResult:
    """Hybrid retrieval entry point used by the Transaction/RAG Agent.

    Backward compatible with the pre-upgrade signature: every new
    parameter is optional and defaults to "on, but auto-degrades if
    unavailable" -- existing callers (`finmate/agents/rag_agent.py`, and
    the tests in `tests/test_rag.py`) work unchanged.

    Step 1 (always): metadata filter via db.search_transactions -- exact,
    deterministic, no fabrication risk. If there's no query text, or no
    candidates survive the filter, this is also the last step: recency-
    ordered metadata results, unranked by relevance (today's floor).

    Steps 2-6 (best-effort, each independently toggleable):
      - `enable_query_rewrite` (None = auto from `FINMATE_RAG_MODE` env
        var / key availability): at most one Groq/Gemini call to expand
        the query for stage 3's benefit.
      - `enable_keyword_search` / `enable_vector_search` / `enable_rerank`:
        force a stage off regardless of what's available (useful for
        tests and for ops -- e.g. disabling rerank to save CPU). Each
        still auto-degrades to "unavailable" on its own when left on but
        its dependency is missing.

    `llm_client`: inject an existing (or mocked) LLMClient for stage 2.
    If None, `finmate.query_rewrite` constructs one lazily from
    environment config on first use -- this is what lets query rewrite
    activate in the real app without `finmate/agents/rag_agent.py` or
    `finmate/orchestrator.py` having to change how they call this
    function.

    `cache`: optional dict for request-scoped memoization (see module
    docstring "Performance"). None (the default) disables caching
    entirely -- every existing caller/test that doesn't pass this
    argument behaves exactly as before. Pass a dict you created fresh for
    this user turn (the orchestrator does exactly this) to make a
    same-turn retry -- e.g. the Critic sending the pipeline back around --
    reuse the first call's result instead of recomputing it. Do not share
    one cache dict across different users' turns or across a long-lived
    process: nothing here ever expires an entry, by design, since it's
    meant to live exactly as long as the caller's `dict` does.
    """

    def _compute() -> RetrievalResult:
        return _retrieve_impl(
            user_id=user_id, query=query, start_date=start_date, end_date=end_date, category=category,
            account=account, top_k=top_k, db_path=db_path, qdrant_path=qdrant_path,
            embedding_model=embedding_model, cross_encoder_model=cross_encoder_model, llm_client=llm_client,
            enable_query_rewrite=enable_query_rewrite, enable_keyword_search=enable_keyword_search,
            enable_vector_search=enable_vector_search, enable_rerank=enable_rerank,
            rerank_pool_size=rerank_pool_size,
        )

    if cache is None:
        return _compute()

    cache_key = (
        user_id, query, start_date, end_date, category, account, top_k, db_path, qdrant_path,
        embedding_model, cross_encoder_model, enable_query_rewrite, enable_keyword_search,
        enable_vector_search, enable_rerank, rerank_pool_size,
    )
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    result = _compute()
    cache[cache_key] = result
    return result


def _retrieve_impl(
    user_id: str,
    query: str,
    start_date: Optional[str],
    end_date: Optional[str],
    category: Optional[str],
    account: Optional[str],
    top_k: int,
    db_path: str,
    qdrant_path: str,
    embedding_model: str,
    cross_encoder_model: str,
    llm_client: Optional[LLMClient],
    enable_query_rewrite: Optional[bool],
    enable_keyword_search: bool,
    enable_vector_search: bool,
    enable_rerank: bool,
    rerank_pool_size: int,
) -> RetrievalResult:
    """The pipeline itself -- see `retrieve`'s docstring for the full
    contract. Split out from `retrieve` only so `retrieve` can wrap it
    with the optional memoization described there without duplicating
    this body for the cached and uncached paths."""
    candidates = db.search_transactions(
        user_id, start_date=start_date, end_date=end_date, category=category, account=account, db_path=db_path,
    )
    if not candidates:
        return RetrievalResult(
            evidence=[],
            vector_search_used=False, keyword_search_used=False, rerank_used=False, query_rewrite_used=False,
            stage="metadata-only, recency-ordered",
            note="No metadata-filtered candidates for this user/filter combination.",
        )

    if not query or not query.strip():
        return RetrievalResult(
            evidence=_recency_ordered_evidence(candidates, top_k),
            vector_search_used=False, keyword_search_used=False, rerank_used=False, query_rewrite_used=False,
            stage="metadata-only, recency-ordered",
            note="No query text supplied; returned recency-ordered metadata results only.",
        )

    by_id = {_tx_key(t): t for t in candidates}

    # --- Stage 4, kicked off early, in the background -----------------
    # Vector search depends only on `query` and this candidate set --
    # never on stage 2's output (unlike keyword search below, which
    # searches the rewritten phrasings too) -- so it starts here, before
    # stage 2 even runs, and is only collected once stage 3 is done. See
    # module docstring "Performance".
    vector_future = None
    if enable_vector_search:
        vector_future = _EXECUTOR.submit(
            _vector_search, user_id=user_id, query=query, candidate_ids=set(by_id),
            qdrant_path=qdrant_path, embedding_model=embedding_model,
            limit=max(top_k * 3, DEFAULT_VECTOR_FETCH_LIMIT),
        )

    # --- Stage 2: optional query rewrite (<=1 LLM call) ---
    rag_mode = os.environ.get("FINMATE_RAG_MODE", "")
    want_rewrite = enable_query_rewrite if enable_query_rewrite is not None else (rag_mode.strip().lower() != "no_llm")
    rewrite_result = query_rewrite.QueryRewriteResult()
    if want_rewrite:
        rewrite_result = query_rewrite.rewrite_query(user_id, query, llm_client=llm_client, mode=rag_mode or None)
    keyword_queries = [query, *rewrite_result.phrasings]

    # --- Stage 3: keyword search. One index build shared across every
    # phrasing (original + rewritten) -- see keyword_search.keyword_rank_multi. ---
    keyword_score_map: dict[str, float] = {}
    keyword_tier = "none"
    if enable_keyword_search:
        hits_by_query, keyword_tier = keyword_search.keyword_rank_multi(candidates, keyword_queries)
        for hits in hits_by_query.values():
            for hit in hits:
                # A source_id matched by more than one phrasing keeps its
                # single best score rather than summing -- summing would
                # let a query that happens to expand into more phrasings
                # win purely by having more terms, which isn't relevance.
                if hit.source_id not in keyword_score_map or hit.score > keyword_score_map[hit.source_id]:
                    keyword_score_map[hit.source_id] = hit.score
    keyword_ranked_ids = sorted(keyword_score_map, key=lambda sid: keyword_score_map[sid], reverse=True)
    keyword_used = bool(keyword_ranked_ids)

    # --- Stage 4, continued: collect the vector search kicked off above ---
    vector_ranked_ids: list[str] = []
    vector_score_map: dict[str, float] = {}
    if vector_future is not None:
        try:
            vector_ranked_ids, vector_score_map = vector_future.result(timeout=VECTOR_SEARCH_TIMEOUT_SECONDS)
        except Exception as exc:  # noqa: BLE001 -- includes concurrent.futures.TimeoutError; must degrade, not raise
            logger.info("Vector search did not complete in time, falling back: %s", exc)
            vector_ranked_ids, vector_score_map = [], {}
    vector_used = bool(vector_ranked_ids)

    if not keyword_used and not vector_used:
        note = (
            "Keyword and vector search both found no matches (or were unavailable/disabled) "
            "within the metadata-filtered set; fell back to recency-ordered metadata results."
        )
        if rewrite_result.note:
            note = f"{note} Query rewrite: {rewrite_result.note}"
        return RetrievalResult(
            evidence=_recency_ordered_evidence(candidates, top_k),
            vector_search_used=False, keyword_search_used=False, rerank_used=False,
            query_rewrite_used=rewrite_result.used, stage="metadata-only, recency-ordered", note=note,
        )

    # --- Stage 5: fusion (RRF) ---
    fused = reranker.reciprocal_rank_fusion(
        keyword_ranked_ids=keyword_ranked_ids, vector_ranked_ids=vector_ranked_ids,
        keyword_scores=keyword_score_map, vector_scores=vector_score_map,
    )

    # --- Stage 6: local cross-encoder rerank of the fused top-N ---
    rerank_pool = fused[:rerank_pool_size]
    rerank_score_map: dict[str, float] = {}
    rerank_used = False
    if enable_rerank:
        docs = [
            (hit.source_id, keyword_search.searchable_text(by_id[hit.source_id]))
            for hit in rerank_pool if hit.source_id in by_id
        ]
        rerank_result = reranker.cross_encoder_rerank(query, docs, model_name=cross_encoder_model)
        if rerank_result is not None:
            rerank_used = True
            rerank_score_map = dict(rerank_result)

    if rerank_used:
        pool_ids_final = sorted(
            (hit.source_id for hit in rerank_pool if hit.source_id in rerank_score_map),
            key=lambda sid: rerank_score_map[sid], reverse=True,
        )
    else:
        pool_ids_final = [hit.source_id for hit in rerank_pool]
    final_order_ids = pool_ids_final + [hit.source_id for hit in fused[rerank_pool_size:]]

    fused_by_id = {hit.source_id: hit for hit in fused}
    evidence: list[EvidenceItem] = []
    for source_id in final_order_ids:
        tx = by_id.get(source_id)
        if tx is None:
            continue
        fused_hit = fused_by_id.get(source_id)
        in_rerank = source_id in rerank_score_map
        relevance = rerank_score_map[source_id] if in_rerank else (fused_hit.rrf_score if fused_hit else 0.0)
        evidence.append(_transaction_to_evidence(
            tx,
            relevance=relevance,
            keyword_score=keyword_score_map.get(source_id),
            vector_score=vector_score_map.get(source_id),
            rerank_score=rerank_score_map.get(source_id),
            retrieval_stage="rerank" if in_rerank else "fusion",
        ))
        if len(evidence) >= top_k:
            break

    stage_label = _describe_stage(
        keyword_used=keyword_used, vector_used=vector_used, rerank_used=rerank_used,
        query_rewrite_used=rewrite_result.used,
    )
    note_bits = [
        f"stages: keyword={keyword_tier}, vector={'used' if vector_used else 'unavailable/disabled'}, "
        f"rerank={'used' if rerank_used else 'unavailable/disabled'}, "
        f"query_rewrite={'used' if rewrite_result.used else 'skipped'}."
    ]
    if rewrite_result.note:
        note_bits.append(f"Query rewrite: {rewrite_result.note}")
    return RetrievalResult(
        evidence=evidence,
        vector_search_used=vector_used,
        keyword_search_used=keyword_used,
        rerank_used=rerank_used,
        query_rewrite_used=rewrite_result.used,
        stage=stage_label,
        note=" ".join(note_bits),
    )
