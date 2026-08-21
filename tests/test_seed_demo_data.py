"""
Tests for scripts/seed_demo_data.py.

The one behavior worth pinning down directly: `db.insert_transactions`
is a plain INSERT with no dedup (correctly so -- see that function's
docstring and `seed()`'s own docstring for why this project doesn't
enforce uniqueness on `source_id` at the database layer). That means
`seed()` itself is responsible for being safe to call more than once
against the same db_path -- both `scripts/seed_demo_data.py`'s own CLI
usage and `backend/app/routers/users.py`'s `/api/users/seed-demo-data`
endpoint (which a person could plausibly call twice, e.g. a double
click) depend on that being true.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from finmate import db  # noqa: E402

from scripts.seed_demo_data import seed  # noqa: E402


def test_seed_is_idempotent_when_called_twice(tmp_path):
    """Regression test: before `seed()` cleared the target user's
    existing rows first, calling it twice back to back would silently
    double every transaction (no unique constraint exists, or should
    exist, on `source_id` -- see module docstring) -- corrupting every
    downstream calculation (spend totals, budgets, anomaly detection)
    for that user without raising any error."""
    db_path = str(tmp_path / "test.db")

    first = seed(db_path, build_vector_index=False, verbose=False)
    second = seed(db_path, build_vector_index=False, verbose=False)

    assert first.user_id == second.user_id == "demo_user"
    assert first.transactions_seeded == second.transactions_seeded
    assert first.transactions_seeded > 0

    stored = db.search_transactions("demo_user", db_path=db_path)
    assert len(stored) == first.transactions_seeded  # not 2x

    # No duplicate source_ids either -- a weaker but complementary check
    # to the raw count above (a bug that dropped as many rows as it
    # duplicated could otherwise slip through a count-only assertion).
    source_ids = [t.source_id for t in stored]
    assert len(source_ids) == len(set(source_ids))


def test_seed_returns_the_seeded_profile_unmodified(tmp_path):
    db_path = str(tmp_path / "test.db")
    seed(db_path, build_vector_index=False, verbose=False)

    profile = db.get_user_profile("demo_user", db_path=db_path)
    assert profile is not None
    assert profile.user_id == "demo_user"


def test_seed_skips_vector_indexing_when_requested(tmp_path):
    db_path = str(tmp_path / "test.db")
    result = seed(db_path, build_vector_index=False, verbose=False)
    assert result.vector_indexed == 0


def test_seed_reports_vector_indexed_count_when_chroma_available(tmp_path, monkeypatch):
    pytest.importorskip("chromadb")
    from finmate import rag

    class _FakeArray(list):
        def tolist(self):
            return [x.tolist() if isinstance(x, _FakeArray) else x for x in self]

    class _ConstantFakeEmbedder:
        def get_sentence_embedding_dimension(self) -> int:
            return 3

        def encode(self, texts):
            return _FakeArray([_FakeArray([1.0, 0.0, 0.0]) for _ in texts])

    monkeypatch.setattr(rag, "_get_embedder", lambda model_name=rag.DEFAULT_EMBEDDING_MODEL: _ConstantFakeEmbedder())

    db_path = str(tmp_path / "test.db")
    chroma_path = str(tmp_path / "chroma")

    result = seed(db_path, build_vector_index=True, verbose=False, chroma_path=chroma_path)
    assert result.vector_indexed == result.transactions_seeded

    # And -- the actual point of threading chroma_path through seed() at
    # all -- the index really did land at the path we asked for, not at
    # rag.DEFAULT_CHROMA_PATH regardless of what was requested.
    client = rag._get_chroma_client(chroma_path)
    collection_names = {c.name for c in client.list_collections()}
    assert f"{rag.COLLECTION_PREFIX}demo_user" in collection_names
