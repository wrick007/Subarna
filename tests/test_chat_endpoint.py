"""
Backend-level tests for `backend/app/routers/chat.py` -- the first tests
in this project to exercise the FastAPI app directly (via TestClient)
rather than calling `finmate.orchestrator` functions in-process. See
`backend/app/__init__.py` for how `backend.app` becomes importable
without needing `finmate` installed separately (it inserts `src/` onto
`sys.path` at package-import time); confirmed empirically importable
from a test file the same way `tests/test_conversation_memory.py` etc.
import `finmate` directly.

`app.state.llm_client` is set directly (mirroring what
`backend/app/main.py`'s lifespan handler does at real startup) rather
than going through a real provider key -- this sandbox has no network
to Groq/Gemini regardless (see README "Deviations"), and every other
test in this suite mocks the LLM client the same way.
"""

from __future__ import annotations

import json

import pytest

# Module-level, not per-function: `from backend.app.main import app` below
# transitively imports fastapi too, so without this guard a minimal
# environment (no backend/requirements.txt installed) would hit a
# collection ERROR for this whole file, not the clean skip every other
# optional-dependency test in this suite gets via importorskip -- see
# README "Run the tests" for the "pydantic/pytest/stdlib only" claim
# this preserves. Expected/reasonable for these two files specifically
# (test_chat_endpoint.py, test_end_to_end.py) to need fastapi+httpx --
# they test the *backend*, whose own requirements.txt is additive to
# the engine's (see backend/requirements.txt's own comment) -- unlike
# every other file in this directory, which tests the engine and
# shouldn't need backend-only dependencies at all.
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from backend.app import config
from backend.app.main import app
from tests._support import RecordingLLMClient


def _client(tmp_path, llm_client=None) -> TestClient:
    config.DB_PATH = str(tmp_path / "test.db")
    from finmate import db
    db.init_db(config.DB_PATH)
    app.state.llm_client = llm_client
    app.state.llm_client_error = None if llm_client else "no provider key configured for this test"
    return TestClient(app)


def _parse_sse(text: str) -> list[tuple[str, dict]]:
    """Parse a `text/event-stream` body into a list of (event, data)
    pairs, in order -- mirrors the frontend's own parsing contract (see
    `frontend/lib/api.ts`): each event is an `event:` line, a `data:`
    line holding one JSON object, then a blank line."""
    events: list[tuple[str, dict]] = []
    event_name = None
    for line in text.split("\n"):
        if line.startswith("event: "):
            event_name = line[len("event: "):]
        elif line.startswith("data: "):
            events.append((event_name, json.loads(line[len("data: "):])))
    return events


# ---------------------------------------------------------------------------
# /api/chat (non-streaming) -- history wiring, unchanged casual fast path
# ---------------------------------------------------------------------------


def test_chat_casual_message_needs_no_llm_client(tmp_path):
    client = _client(tmp_path, llm_client=None)
    resp = client.post("/api/chat", json={"user_id": "u1", "message": "hello"})
    assert resp.status_code == 200
    assert resp.json()["is_casual"] is True


def test_chat_non_casual_without_llm_client_returns_503(tmp_path):
    client = _client(tmp_path, llm_client=None)
    resp = client.post("/api/chat", json={"user_id": "u1", "message": "what's my savings rate?"})
    assert resp.status_code == 503


def test_chat_history_reaches_the_router(tmp_path):
    fake = RecordingLLMClient(router_intent="general_finance")
    client = _client(tmp_path, llm_client=fake)
    resp = client.post("/api/chat", json={
        "user_id": "u1",
        "message": "how can I reduce it?",
        "history": [
            {"role": "user", "content": "what's my biggest expense category?"},
            {"role": "assistant", "content": "Your biggest category is dining."},
        ],
    })
    assert resp.status_code == 200
    from finmate.schemas import RouterOutput
    router_calls = fake.calls_with(response_model=RouterOutput)
    assert "biggest expense category" in router_calls[0]["user_message"]


def test_chat_rejects_oversized_history(tmp_path):
    fake = RecordingLLMClient()
    client = _client(tmp_path, llm_client=fake)
    huge_history = [{"role": "user", "content": "hi"} for _ in range(61)]  # over the 60-entry cap
    resp = client.post("/api/chat", json={"user_id": "u1", "message": "hello there", "history": huge_history})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# /api/chat/stream -- SSE wire format, actual incremental delivery
# ---------------------------------------------------------------------------


def test_chat_stream_casual_message(tmp_path):
    client = _client(tmp_path, llm_client=None)
    resp = client.post("/api/chat/stream", json={"user_id": "u1", "message": "hi there"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse(resp.text)
    assert events[0][0] == "token"
    assert events[-1][0] == "done"
    assert events[-1][1]["is_casual"] is True


def test_chat_stream_emits_multiple_token_events_and_a_final_done(tmp_path):
    fake = RecordingLLMClient(router_intent="general_finance", synthesis_text="a fairly long streamed answer here")
    client = _client(tmp_path, llm_client=fake)
    resp = client.post("/api/chat/stream", json={"user_id": "u1", "message": "what's a good savings rate?"})
    assert resp.status_code == 200
    events = _parse_sse(resp.text)

    token_events = [d for name, d in events if name == "token"]
    done_events = [(name, d) for name, d in events if name == "done"]
    assert len(token_events) > 1
    assert "".join(t["text"] for t in token_events) == "a fairly long streamed answer here"
    assert len(done_events) == 1
    assert done_events[0][1]["response"] == "a fairly long streamed answer here"
    assert done_events[0][1]["is_casual"] is False


def test_chat_stream_actually_streams_not_buffers(tmp_path):
    """The whole point of this endpoint: confirm chunks genuinely arrive
    incrementally through `iter_lines()` rather than the server building
    the full body first -- i.e. more than one read from the connection
    is needed to see all the SSE events, not just that the final string
    happens to contain multiple "event:" markers."""
    fake = RecordingLLMClient(router_intent="general_finance", synthesis_text="one two three four five six seven")
    client = _client(tmp_path, llm_client=fake)
    with client.stream("POST", "/api/chat/stream", json={"user_id": "u1", "message": "give me a tip"}) as resp:
        assert resp.status_code == 200
        chunks_received = 0
        for _ in resp.iter_lines():
            chunks_received += 1
        assert chunks_received > 2  # more than "one line total" -- genuinely multiple reads


def test_chat_stream_error_event_on_pipeline_failure(tmp_path):
    class _BrokenClient(RecordingLLMClient):
        def call(self, *a, **kw):
            raise RuntimeError("simulated provider outage")

    client = _client(tmp_path, llm_client=_BrokenClient())
    resp = client.post("/api/chat/stream", json={"user_id": "u1", "message": "what's my savings rate?"})
    assert resp.status_code == 200  # the HTTP response itself succeeded; the *pipeline* failed mid-stream
    events = _parse_sse(resp.text)
    assert events[-1][0] == "error"
    assert "simulated provider outage" in events[-1][1]["message"]
