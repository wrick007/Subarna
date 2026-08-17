"""
Stages 5-6 of hybrid retrieval: fusion, then local cross-encoder rerank.

Fusion uses Reciprocal Rank Fusion (RRF) rather than blending raw scores.
Keyword scores (BM25 magnitude or substring counts) and vector scores
(cosine similarity, roughly 0..1) live on different, incompatible scales
that also shift with corpus size and query -- averaging or weighting them
directly would let whichever signal happens to have the larger numeric
range dominate, for reasons that have nothing to do with actual
relevance. RRF (Cormack, Clarke & Buettcher, 2009) sidesteps this by
using only *rank position* within each list, which is scale-free by
construction, and naturally rewards items that multiple signals agree on.

Cross-encoder reranking is a second, more expensive, more accurate pass
over just the fused top-N. A cross-encoder scores a (query, document)
pair jointly -- unlike the bi-encoder used for dense retrieval, which
embeds query and document independently and compares vectors -- so it
catches relevance signal a bi-encoder's cosine similarity misses, at the
cost of being too slow to run over an entire candidate set. That's why it
only ever sees the fused top-N, never the full metadata-filtered pool.

`reciprocal_rank_fusion` has zero dependencies and is tested with plain
fake (id, score) inputs. `cross_encoder_rerank` lazily imports
`sentence_transformers.CrossEncoder` and returns None -- never raises --
if the model can't be loaded, so callers fall back to the fused ranking
unchanged.

Performance: `_get_cross_encoder` is the single slowest thing in the
whole retrieval pipeline the first time it runs (loading model weights
from disk, or worse, downloading them). The pre-upgrade version of this
module re-ran that load on *every single retrieval call* -- fine for a
one-shot script, a real problem for a long-lived server process handling
many chat turns. It's now a process-lifetime singleton, keyed by
`model_name` and guarded by a lock (a real concern once `finmate/rag.py`
calls this from a background thread -- see that module's docstring): the
model loads once, including a *failed* load, so a permanently-unreachable
model fails fast on every later call instead of re-attempting a network
fetch every time. See `clear_cache()` to force a fresh attempt.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("finmate.reranker")

# --- process-lifetime cross-encoder cache -----------------------------
# Keyed by model_name; a cached `None` means "already tried, unavailable"
# (see _get_cross_encoder's docstring) rather than "not yet looked up".
_cross_encoder_cache: dict[str, object] = {}
_cross_encoder_lock = threading.Lock()


def clear_cache() -> None:
    """Test hook / operational escape hatch: forget any cached
    CrossEncoder instance (or cached load-failure) so the next call to
    `_get_cross_encoder` attempts a fresh load. Useful in tests that
    exercise the load path itself, and for an operator who wants to retry
    after fixing a transient failure (e.g. restored network access)
    without restarting the whole process."""
    with _cross_encoder_lock:
        _cross_encoder_cache.clear()

RRF_K = 60  # standard damping constant from the RRF paper; not sensitive
# to small changes for candidate pools this small (tens, not millions, of
# transactions), so it's left as a named constant rather than exposed as
# a tuning knob nothing in this codebase yet needs to vary.

DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass
class FusedHit:
    source_id: str
    rrf_score: float
    keyword_rank: Optional[int] = None
    vector_rank: Optional[int] = None
    keyword_score: Optional[float] = None
    vector_score: Optional[float] = None


def reciprocal_rank_fusion(
    keyword_ranked_ids: list[str],
    vector_ranked_ids: list[str],
    keyword_scores: Optional[dict[str, float]] = None,
    vector_scores: Optional[dict[str, float]] = None,
    k: int = RRF_K,
) -> list[FusedHit]:
    """Combine two rank-ordered id lists into one fused ranking via RRF.

    score(id) = sum, over each input list containing id, of
    1 / (k + rank_in_that_list) -- rank is 1-indexed. An id missing from
    one list simply doesn't get that list's term (not penalized beyond
    not receiving it), so keyword-only and vector-only matches both
    surface in the fused output; items both signals agree on rank
    highest, since they accumulate a term from each list.

    Either input list may be empty (e.g. the vector stage was
    unavailable) -- fusion over a single non-empty list degrades
    gracefully to that list's own rank order, which is exactly the
    "keyword only" / "vector only" tiers in the pipeline's documented
    fallback ladder.

    `keyword_scores`/`vector_scores` are optional raw-score lookups
    (source_id -> score) carried through onto the returned `FusedHit`s
    purely for audit/display (`EvidenceItem.keyword_score` /
    `.vector_score`) -- they play no part in the RRF math itself.
    """
    keyword_scores = keyword_scores or {}
    vector_scores = vector_scores or {}
    rrf_scores: dict[str, float] = {}
    kw_rank_map: dict[str, int] = {}
    vec_rank_map: dict[str, int] = {}

    for rank, sid in enumerate(keyword_ranked_ids, start=1):
        rrf_scores[sid] = rrf_scores.get(sid, 0.0) + 1.0 / (k + rank)
        kw_rank_map.setdefault(sid, rank)
    for rank, sid in enumerate(vector_ranked_ids, start=1):
        rrf_scores[sid] = rrf_scores.get(sid, 0.0) + 1.0 / (k + rank)
        vec_rank_map.setdefault(sid, rank)

    fused = [
        FusedHit(
            source_id=sid,
            rrf_score=score,
            keyword_rank=kw_rank_map.get(sid),
            vector_rank=vec_rank_map.get(sid),
            keyword_score=keyword_scores.get(sid),
            vector_score=vector_scores.get(sid),
        )
        for sid, score in rrf_scores.items()
    ]
    # Stable sort + id as a secondary key gives deterministic ordering for
    # exact ties (matters for reproducible tests/eval numbers).
    fused.sort(key=lambda h: (-h.rrf_score, h.source_id))
    return fused


def _get_cross_encoder(model_name: str = DEFAULT_CROSS_ENCODER_MODEL):
    """Lazily import, instantiate, and cache sentence_transformers.CrossEncoder.
    Returns None -- never raises -- if the package isn't installed, or if
    construction fails for any reason (most commonly: the model weights
    can't be downloaded because Hugging Face Hub is unreachable -- a
    real, verified failure mode in a network-restricted environment, see
    README "Deviations"). A bare `except ImportError` here would not be
    enough: instantiation failure is a distinct, later failure than the
    import succeeding, and both must degrade the same way.

    Cached per `model_name` for the process lifetime -- see module
    docstring "Performance" section. Double-checked locking: the common
    case (already cached) never touches the lock at all; only a genuine
    first load for a given model_name contends for it.
    """
    if model_name in _cross_encoder_cache:
        return _cross_encoder_cache[model_name]
    with _cross_encoder_lock:
        if model_name in _cross_encoder_cache:  # re-check: another thread may have just finished loading
            return _cross_encoder_cache[model_name]
        try:
            from sentence_transformers import CrossEncoder  # noqa: PLC0415
        except ImportError:
            _cross_encoder_cache[model_name] = None
            return None
        try:
            encoder = CrossEncoder(model_name)
        except Exception as exc:  # noqa: BLE001 -- any load failure must degrade, never crash retrieval
            logger.info("Cross-encoder model unavailable (%s); skipping rerank.", exc)
            encoder = None
        _cross_encoder_cache[model_name] = encoder
        return encoder


def cross_encoder_rerank(
    query: str,
    documents: list[tuple[str, str]],
    model_name: str = DEFAULT_CROSS_ENCODER_MODEL,
) -> Optional[list[tuple[str, float]]]:
    """Score each (source_id, text) pair in `documents` against `query`
    with a local cross-encoder and return them sorted best-first.

    Returns `None` -- never raises -- if the cross-encoder can't be
    loaded, or if scoring itself fails for any reason; callers must treat
    `None` as "rerank unavailable, keep the fused order", exactly like
    every other optional stage in this pipeline. Returns `[]` (not None)
    for an empty `documents` list, since that's a successful no-op, not a
    failure.
    """
    if not documents:
        return []
    encoder = _get_cross_encoder(model_name)
    if encoder is None:
        return None
    try:
        pairs = [(query, text) for _, text in documents]
        scores = encoder.predict(pairs)
    except Exception as exc:  # noqa: BLE001 -- a broken local model must not crash retrieval
        logger.info("Cross-encoder scoring failed (%s); skipping rerank.", exc)
        return None
    ids = [source_id for source_id, _ in documents]
    ranked = sorted(zip(ids, (float(s) for s in scores)), key=lambda pair: pair[1], reverse=True)
    return ranked
