"""
Tests for short-term conversational memory: finmate/orchestrator.py's
`conversation_history` parameter, threaded into the router (to resolve
"it"/"that") and the synthesis agent (to answer as a continuation of the
conversation), per that module's docstring "Conversation history".

Deliberately distinct from finmate/agents/memory.py's *profile-fact*
memory (income updates, goals, etc.) -- that system has its own test
coverage elsewhere and is untouched by this file.

No real LLM call anywhere here (no network in this sandbox regardless --
see README "Deviations"): `RecordingLLMClient` (see tests/_support.py)
stands in for a real provider, returns the minimum valid structured
response for whatever `response_model` is asked for, and -- the actual
point of these tests -- records exactly what `user_message` text every
call received, so a test can assert on it directly rather than trusting
behavior it can't observe.
"""

from __future__ import annotations

from finmate import db
from finmate.orchestrator import (
    MAX_HISTORY_CHARS,
    MAX_HISTORY_MESSAGES,
    _render_history_block,
    _trim_history,
    run_finmate,
)
from finmate.schemas import MemoryAction, RouterOutput

from tests._support import RecordingLLMClient as _RecordingLLMClient


def _db(tmp_path) -> str:
    path = str(tmp_path / "test.db")
    db.init_db(path)
    return path


# ---------------------------------------------------------------------------
# The headline scenario: a two-turn exchange where turn 2 only makes sense
# with turn 1's context.
# ---------------------------------------------------------------------------


def test_turn_two_router_and_synthesis_receive_turn_one_context(tmp_path):
    db_path = _db(tmp_path)
    client = _RecordingLLMClient()

    turn1 = run_finmate("u1", "What's my biggest expense category?", client, db_path=db_path)

    history = [
        {"role": "user", "content": "What's my biggest expense category?"},
        {"role": "assistant", "content": turn1.final_response},
    ]
    client.calls.clear()  # only inspect what turn 2 actually sends

    run_finmate("u1", "How can I reduce it?", client, db_path=db_path, conversation_history=history)

    router_calls = client.calls_with(response_model=RouterOutput)
    assert len(router_calls) == 1
    assert "What's my biggest expense category?" in router_calls[0]["user_message"]
    assert "How can I reduce it?" in router_calls[0]["user_message"]

    synthesis_calls = client.calls_with(response_model=None, system_prompt_contains="Senior Personal Financial Analyst")
    assert len(synthesis_calls) == 1
    assert "What's my biggest expense category?" in synthesis_calls[0]["user_message"]


def test_first_turn_has_no_history_and_costs_no_extra_tokens(tmp_path):
    """A conversation's first message has nothing to remember yet --
    this must be indistinguishable from calling run_finmate before this
    feature existed: no history header text anywhere in any prompt."""
    db_path = _db(tmp_path)
    client = _RecordingLLMClient()

    run_finmate("u1", "What's my savings rate?", client, db_path=db_path)

    assert client.calls  # sanity: the pipeline actually ran
    for call in client.calls:
        assert "Recent conversation" not in call["user_message"]


def test_conversation_history_none_behaves_like_no_history(tmp_path):
    """The default (no conversation_history argument at all) must match
    passing an explicit empty list -- existing callers that don't know
    about this parameter yet keep working exactly as before."""
    db_path = _db(tmp_path)
    client_default = _RecordingLLMClient()
    client_explicit_empty = _RecordingLLMClient()

    run_finmate("u1", "What's my savings rate?", client_default, db_path=db_path)
    run_finmate("u1", "What's my savings rate?", client_explicit_empty, db_path=db_path, conversation_history=[])

    router_default = client_default.calls_with(response_model=RouterOutput)[0]["user_message"]
    router_explicit = client_explicit_empty.calls_with(response_model=RouterOutput)[0]["user_message"]
    assert router_default == router_explicit


def test_profile_update_fast_path_ignores_history_and_stays_untouched(tmp_path):
    """Priority 1's own constraint: the existing profile-fact memory fast
    path (finmate/agents/memory.py, via the profile_update intent) must
    keep working exactly as before -- this is a *different* kind of
    memory, not something conversation_history should change."""
    db_path = _db(tmp_path)

    class _ProfileUpdateClient(_RecordingLLMClient):
        def call(self, agent_system_prompt, user_message, response_model=None, **kw):
            super().call(agent_system_prompt, user_message, response_model=response_model, **kw)
            if response_model is RouterOutput:
                return RouterOutput(intent="profile_update")
            if response_model is MemoryAction:
                return MemoryAction(
                    memory_action="update", field="monthly_income", new_value=50000, requires_confirmation=False,
                )
            return "ok"

    client = _ProfileUpdateClient()
    history = [{"role": "user", "content": "some earlier turn"}, {"role": "assistant", "content": "some earlier reply"}]
    result = run_finmate(
        "u1", "my income is 50000 INR monthly", client, db_path=db_path, conversation_history=history,
    )
    assert "₹50,000" in result.final_response
    # Fast path never reaches synthesis/critic/formatter at all.
    assert client.calls_with(response_model=None, system_prompt_contains="Senior Personal Financial Analyst") == []
    profile = db.get_user_profile("u1", db_path=db_path)
    assert profile.monthly_income == 50000


# ---------------------------------------------------------------------------
# Trimming: message-count cap and character-budget cap (_trim_history),
# and the shared renderer (_render_history_block).
# ---------------------------------------------------------------------------


def test_render_history_block_empty_is_empty_string():
    assert _render_history_block([]) == ""


def test_render_history_block_labels_speakers_and_preserves_order():
    history = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second question"},
    ]
    block = _render_history_block(history)
    assert block.index("User: first question") < block.index("Assistant: first answer") < block.index("User: second question")


def test_trim_history_keeps_only_the_most_recent_messages():
    long_history = [{"role": "user", "content": f"message {i}"} for i in range(MAX_HISTORY_MESSAGES + 10)]
    trimmed = _trim_history(long_history)
    assert len(trimmed) <= MAX_HISTORY_MESSAGES
    # The tail (most recent) survives, not the head.
    assert trimmed[-1]["content"] == f"message {MAX_HISTORY_MESSAGES + 9}"
    assert trimmed[0]["content"] != "message 0"


def test_trim_history_enforces_character_budget_even_under_the_message_cap():
    # Two messages, well under MAX_HISTORY_MESSAGES, but one is huge.
    huge = "x" * (MAX_HISTORY_CHARS * 2)
    history = [{"role": "user", "content": huge}, {"role": "assistant", "content": "short reply"}]
    trimmed = _trim_history(history)
    assert len(_render_history_block(trimmed)) <= MAX_HISTORY_CHARS
    # The most recent message is what survives the character trim.
    assert trimmed[-1]["content"] == "short reply"


def test_trim_history_drops_malformed_entries_without_raising():
    history = [
        {"role": "user", "content": "fine"},
        {"role": "system", "content": "not a supported role"},
        {"content": "missing role entirely"},
        {"role": "assistant", "content": ""},  # blank content
        {"role": "assistant"},  # missing content entirely
        "not even a dict",
        {"role": "assistant", "content": "also fine"},
    ]
    trimmed = _trim_history(history)  # must not raise
    assert trimmed == [{"role": "user", "content": "fine"}, {"role": "assistant", "content": "also fine"}]


def test_trim_history_on_empty_input():
    assert _trim_history([]) == []
