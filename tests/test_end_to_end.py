"""
A genuine end-to-end pass, not just unit tests in isolation -- per the
redesign brief's testing requirements: "run the backend locally, seed
demo data, have an actual multi-turn conversation through the API...
and confirm memory, speed, and the Chroma-backed retrieval all work
together, not just in isolation."

This is as close to that as this sandbox's constraints allow: a real
FastAPI TestClient (not a mocked router function), a real seed-demo-data
call through the real endpoint (not a hand-built fixture), a real
embedded on-disk Chroma index actually queried (not skipped), and a real
two-turn conversation through POST /api/chat/stream where turn 2 only
makes sense with turn 1's context. The one thing genuinely swapped for a
fake is the LLM provider itself (RecordingLLMClient, see tests/_support.py)
-- this sandbox has no network to Groq/Gemini regardless (see README
"Deviations"), so this is the same substitution every other test in this
suite makes, not a shortcut specific to this file.
"""

from __future__ import annotations

import json

import pytest

# See tests/test_chat_endpoint.py's identical guard for why this is
# module-level: `from backend.app.main import app` below transitively
# needs fastapi too.
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from backend.app import config
from backend.app.main import app
from finmate import db, rag
from finmate.schemas import RouterOutput
from tests._support import RecordingLLMClient


class _FakeArray(list):
    def tolist(self):
        return [x.tolist() if isinstance(x, _FakeArray) else x for x in self]


class _ConstantFakeEmbedder:
    """Same fake used by tests/test_rag_hybrid.py -- real Chroma, fake
    vectors, so this exercises the actual storage/query wiring without
    needing Hugging Face Hub access (unreachable in this sandbox)."""

    def get_sentence_embedding_dimension(self) -> int:
        return 3

    def encode(self, texts):
        return _FakeArray([_FakeArray([1.0, 0.0, 0.0]) for _ in texts])


def _parse_sse(text: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    event_name = None
    for line in text.split("\n"):
        if line.startswith("event: "):
            event_name = line[len("event: "):]
        elif line.startswith("data: "):
            events.append((event_name, json.loads(line[len("data: "):])))
    return events


def test_full_stack_seed_then_two_turn_conversation_with_memory_and_chroma(tmp_path, monkeypatch):
    pytest.importorskip("chromadb")

    # Real Chroma wiring, fake vectors (see module docstring).
    monkeypatch.setattr(rag, "_get_embedder", lambda model_name=rag.DEFAULT_EMBEDDING_MODEL: _ConstantFakeEmbedder())

    config.DB_PATH = str(tmp_path / "test.db")
    config.CHROMA_PATH = str(tmp_path / "chroma")
    db.init_db(config.DB_PATH)

    fake_llm = RecordingLLMClient(
        router_intent="transaction_question",
        router_output_overrides={"search_phrasings": ["dining", "restaurants"]},
        synthesis_text="Your biggest expense category this month is Dining at 8200 INR.",
    )
    app.state.llm_client = fake_llm
    app.state.llm_client_error = None
    client = TestClient(app)

    # --- Step 1: seed demo data through the REAL endpoint ---
    health = client.get("/api/health")
    assert health.status_code == 200

    seed_resp = client.post("/api/users/seed-demo-data")
    assert seed_resp.status_code == 200
    seed_body = seed_resp.json()
    demo_user_id = seed_body["user_id"]
    assert seed_body["transactions_seeded"] > 0

    # Confirm the vector index genuinely exists on disk for this user --
    # not just that the endpoint returned 200 -- via the real Chroma client.
    chroma_client = rag._get_chroma_client(config.CHROMA_PATH)
    collection_names = {c.name for c in chroma_client.list_collections()}
    assert f"{rag.COLLECTION_PREFIX}{demo_user_id}" in collection_names

    # --- Step 2: turn 1, streamed, real transaction_question through the real pipeline ---
    resp1 = client.post(
        "/api/chat/stream",
        json={"user_id": demo_user_id, "message": "What's my biggest expense category?"},
    )
    assert resp1.status_code == 200
    events1 = _parse_sse(resp1.text)
    token_events1 = [d for name, d in events1 if name == "token"]
    done_events1 = [d for name, d in events1 if name == "done"]
    assert len(token_events1) > 1, "streaming should deliver more than one chunk"
    assert len(done_events1) == 1
    turn1_response = done_events1[0]["response"]
    assert turn1_response == fake_llm.synthesis_text
    # The Chroma-backed retrieval actually ran and found something (fake
    # embeddings are identical for every text, so every candidate matches).
    assert done_events1[0]["retrieval"] is not None
    assert len(done_events1[0]["retrieval"]["evidence"]) > 0

    fake_llm.calls.clear()  # only inspect what turn 2 sends

    # --- Step 3: turn 2, with history, only makes sense given turn 1 ---
    resp2 = client.post(
        "/api/chat/stream",
        json={
            "user_id": demo_user_id,
            "message": "How can I reduce it?",
            "history": [
                {"role": "user", "content": "What's my biggest expense category?"},
                {"role": "assistant", "content": turn1_response},
            ],
        },
    )
    assert resp2.status_code == 200
    events2 = _parse_sse(resp2.text)
    token_events2 = [d for name, d in events2 if name == "token"]
    done_events2 = [d for name, d in events2 if name == "done"]
    assert len(token_events2) > 1
    assert len(done_events2) == 1

    # Memory: the router and synthesis calls for turn 2 actually received
    # turn 1's content -- the whole point of the conversation-memory fix.
    router_calls = fake_llm.calls_with(response_model=RouterOutput)
    assert router_calls, "router should have been called for turn 2"
    assert "biggest expense category" in router_calls[0]["user_message"]

    synthesis_calls = [c for c in fake_llm.calls if c["response_model"] is None]
    assert synthesis_calls, "synthesis should have been called for turn 2"
    assert "biggest expense category" in synthesis_calls[0]["user_message"]

    # Speed: this representative turn's evidence retrieval genuinely
    # found matching transactions (see the retrieval assertion on turn
    # 1 above) -- so, correctly, Priority 2's conditional critic
    # (finmate/orchestrator.py's `_turn_needs_verification`) does NOT
    # skip verification here: there's a concrete, transaction-backed
    # claim to check, which is exactly the case the critic must still
    # run for unconditionally (see that module's "Critic: conditional,
    # not always-on" docstring -- what gets skipped is turns with
    # nothing concrete, not turns like this one). Confirms the
    # conditional-critic wiring end-to-end on both sides: it engages
    # when there's something to check, and (per
    # test_general_finance_turn_never_calls_critic in
    # tests/test_call_reduction.py) it doesn't when there isn't.
    from finmate.schemas import CriticResult
    assert len(fake_llm.calls_with(response_model=CriticResult)) == 1
    assert done_events2[0]["verification_ran"] is True
