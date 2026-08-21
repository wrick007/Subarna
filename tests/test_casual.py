"""
Tests for finmate/casual.py -- the zero-LLM-call greeting fast path.

No prior dedicated test file existed for this module (confirmed by
searching the suite before writing this one) -- these tests cover both
its pre-existing `is_casual_message` behavior (which nothing else in
this build's added test coverage exercised) and Priority 4's actual
change: a pool of response variations instead of one fixed string.
"""

from __future__ import annotations

from finmate.casual import CASUAL_RESPONSES, is_casual_message, pick_casual_response


# ---------------------------------------------------------------------------
# is_casual_message -- pre-existing behavior, now with coverage
# ---------------------------------------------------------------------------


def test_recognizes_simple_greetings():
    for message in ["hi", "hello", "hey", "Hi There", "GOOD MORNING", "thanks", "bye"]:
        assert is_casual_message(message) is True, message


def test_is_case_and_whitespace_insensitive():
    assert is_casual_message("  Hello   ") is True
    assert is_casual_message("HeLLo") is True
    assert is_casual_message("hello\n") is True


def test_does_not_treat_a_real_question_as_casual_even_with_a_greeting_prefix():
    """The narrow, exact-match design is deliberate (see module
    docstring): a false positive here would silently skip a real
    financial question."""
    assert is_casual_message("hi, what's my savings rate?") is False
    assert is_casual_message("hello can you tell me my balance") is False
    assert is_casual_message("") is False
    assert is_casual_message("what's a Roth IRA?") is False


# ---------------------------------------------------------------------------
# Priority 4: a pool, not one fixed string
# ---------------------------------------------------------------------------


def test_response_pool_has_more_than_one_entry():
    assert len(CASUAL_RESPONSES) > 1


def test_responses_are_not_always_identical_across_many_calls():
    seen = {pick_casual_response() for _ in range(200)}
    assert len(seen) > 1, "200 draws from the pool produced only one distinct response"


def test_every_response_is_a_real_pool_member():
    for _ in range(50):
        assert pick_casual_response() in CASUAL_RESPONSES


def test_every_response_names_finmate_and_invites_a_financial_question():
    """The properties the original single CASUAL_RESPONSE guaranteed --
    who's answering, and what to do next -- must survive across every
    variation, not just some of them."""
    invite_words = ("finance", "financ", "budget", "spending", "goal")
    for response in CASUAL_RESPONSES:
        assert "FinMate" in response
        assert any(word in response.lower() for word in invite_words), response


def test_pool_entries_are_short_single_sentiment_replies():
    """A casual greeting reply should read like a quick hello, not a
    paragraph -- guards against a future edit accidentally turning one
    pool entry into something structurally different from the rest."""
    for response in CASUAL_RESPONSES:
        assert len(response) < 150
