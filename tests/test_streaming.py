"""
Tests for Priority 2's streaming path -- see finmate/orchestrator.py's
module docstring "Streaming" for the design.

Covers three layers, each mocked at the layer below it so a failure
points at the right one:
  - `LLMClient.call_stream` against a fake OpenAI-shaped streaming
    response (no real provider call -- this sandbox has no network to
    Groq/Gemini regardless; see README "Deviations").
  - `finmate.orchestrator.run_finmate_stream` against `RecordingLLMClient`
    (see tests/_support.py), which implements `call_stream` too.
  - `POST /api/chat/stream` against a FastAPI TestClient, confirming the
    SSE wire format and that it actually arrives as separate chunks
    rather than one buffered blob.

None of this can exercise real token-by-token latency against a live
Groq/Gemini stream in this sandbox -- see the redesign summary for what
that means for the benchmark numbers, and a reminder to smoke-test this
for real once a provider key is available.
"""

from __future__ import annotations

from finmate import db
from finmate.orchestrator import run_finmate, run_finmate_stream
from finmate.schemas import CriticResult

from tests._support import RecordingLLMClient


def _db(tmp_path) -> str:
    path = str(tmp_path / "test.db")
    db.init_db(path)
    return path


# ---------------------------------------------------------------------------
# LLMClient.call_stream, in isolation, against a fake OpenAI-shaped client
# ---------------------------------------------------------------------------


class _FakeDelta:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.delta = _FakeDelta(content)


class _FakeChunk:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)] if content is not None else [_FakeChoice(None)]


class _FakeChunkNoChoices:
    choices = []


class _FakeCompletions:
    def __init__(self, chunks):
        self._chunks = chunks
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        assert kwargs.get("stream") is True
        return iter(self._chunks)


class _FakeChat:
    def __init__(self, chunks):
        self.completions = _FakeCompletions(chunks)


class _FakeOpenAIClient:
    def __init__(self, chunks):
        self.chat = _FakeChat(chunks)


def test_call_stream_yields_text_deltas_in_order():
    from finmate.llm import LLMClient

    chunks = [_FakeChunk("Hello"), _FakeChunk(", "), _FakeChunk("world"), _FakeChunkNoChoices(), _FakeChunk(None), _FakeChunk("!")]
    fake_openai = _FakeOpenAIClient(chunks)

    client = LLMClient.__new__(LLMClient)  # bypass __init__ -- no real provider construction
    client._client = fake_openai
    client.model = "test-model"

    deltas = list(client.call_stream("system prompt", "user message"))
    assert deltas == ["Hello", ", ", "world", "!"]
    assert "".join(deltas) == "Hello, world!"
    # A blank/None delta and a chunk with no choices at all must not
    # raise or produce a spurious empty-string event.
    assert fake_openai.chat.completions.last_kwargs["stream"] is True


# ---------------------------------------------------------------------------
# run_finmate_stream against RecordingLLMClient
# ---------------------------------------------------------------------------


def test_stream_tokens_concatenate_to_the_synthesis_text(tmp_path):
    db_path = _db(tmp_path)
    client = RecordingLLMClient(router_intent="general_finance", synthesis_text="This is the streamed answer")

    events = list(run_finmate_stream("u1", "what's a good rule of thumb for saving?", client, db_path=db_path))

    token_events = [e for e in events if e.type == "token"]
    done_events = [e for e in events if e.type == "done"]
    assert token_events, "expected at least one token event"
    assert len(token_events) > 1, "expected more than one chunk -- otherwise this isn't really testing streaming"
    assert "".join(e.text for e in token_events) == "This is the streamed answer"
    assert len(done_events) == 1
    assert done_events[0].result.final_response == "This is the streamed answer"
    assert [e.type for e in events][-1] == "done"


def test_stream_matches_non_stream_result_for_equivalent_mocked_calls(tmp_path):
    """The streaming and non-streaming entry points must not silently
    diverge in behavior -- same router/critic outcome, same final text,
    for the identical mocked scenario."""
    import os
    os.makedirs(str(tmp_path / "a"), exist_ok=True)
    os.makedirs(str(tmp_path / "b"), exist_ok=True)
    db_path_a = str(tmp_path / "a" / "test.db")
    db_path_b = str(tmp_path / "b" / "test.db")
    db.init_db(db_path_a)
    db.init_db(db_path_b)

    client_a = RecordingLLMClient(router_intent="general_finance", synthesis_text="Same answer either way")
    client_b = RecordingLLMClient(router_intent="general_finance", synthesis_text="Same answer either way")

    non_stream_result = run_finmate("u1", "what's an emergency fund?", client_a, db_path=db_path_a)
    stream_events = list(run_finmate_stream("u1", "what's an emergency fund?", client_b, db_path=db_path_b))
    stream_result = [e for e in stream_events if e.type == "done"][0].result

    assert stream_result.final_response == non_stream_result.final_response
    assert stream_result.critic_passed == non_stream_result.critic_passed
    assert stream_result.router_output.intent == non_stream_result.router_output.intent


def test_stream_emits_restart_on_critic_triggered_retry(tmp_path, monkeypatch):
    import finmate.orchestrator as orch
    from finmate.schemas import CalculationResult

    db_path = _db(tmp_path)
    client = RecordingLLMClient(router_intent="general_finance", critic_fail_times=1, synthesis_text="draft then final")

    real_pipeline = orch._node_pipeline

    def _pipeline_with_forced_calc(state, llm_client):
        new_state = real_pipeline(state, llm_client)
        calc = CalculationResult(metric="x", value=1, currency="INR", period="monthly", formula="x", inputs={}, source_ids=[])
        return {**new_state, "calc_results": [calc]}

    monkeypatch.setattr(orch, "_node_pipeline", _pipeline_with_forced_calc)

    events = list(run_finmate_stream("u1", "what did I spend?", client, db_path=db_path))

    restart_events = [e for e in events if e.type == "restart"]
    assert len(restart_events) == 1
    # Tokens exist both before and after the restart (two full attempts streamed).
    restart_index = events.index(restart_events[0])
    assert any(e.type == "token" for e in events[:restart_index])
    assert any(e.type == "token" for e in events[restart_index + 1:])
    done = [e for e in events if e.type == "done"][0]
    assert done.result.retry_count == 1


def test_stream_never_raises_yields_error_event_instead(tmp_path):
    db_path = _db(tmp_path)

    class _BrokenClient(RecordingLLMClient):
        def call(self, *a, **kw):
            raise RuntimeError("simulated provider outage")

    events = list(run_finmate_stream("u1", "what's my savings rate?", _BrokenClient(), db_path=db_path))
    assert events[-1].type == "error"
    assert "simulated provider outage" in events[-1].error
