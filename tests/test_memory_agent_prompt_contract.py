"""
Regression tests for a real bug reported by an actual user of the
redesigned app: every attempt to update `monthly_income` (via
`finmate/agents/memory.py`'s profile_update fast path in
`finmate/orchestrator.py`) failed with "I couldn't determine exactly
which profile value you want to update" -- regardless of phrasing, even
when the phrasing exactly matched what the assistant itself suggested.

Root cause, found by reading `prompts.MEMORY_AGENT` and
`schemas.MemoryAction` together rather than in isolation: the prompt's
own few-shot JSON examples used `"value"` as the key for the number
being set, but `MemoryAction`'s actual pydantic field is `new_value`.
Pydantic v2's default `model_validate` behavior silently drops
unrecognized keys and falls back to a field's default for anything
missing -- so `{"...", "value": 200000, ...}` parses "successfully" with
`new_value` left at its `None` default, and
`finmate/orchestrator.py`'s fast path (`memory_action.new_value is not
None`) always fails the check, every time, for every phrasing. No
`ValidationError` is ever raised, so the automatic single-retry-on-
validation-failure in `LLMClient.call` never even triggers -- this
fails silently and permanently, not loudly-then-recovers.

A second, related bug found alongside it: the prompt's "no update
needed" example showed `"field": null`, which -- unlike a silently
dropped extra key -- IS a hard `ValidationError` against
`MemoryAction.field` (a non-optional `str`, not `Optional[str]`):
pydantic only substitutes a field's default when the key is *absent*
from the input, not when it's explicitly `null`.

Neither bug could have been caught by this suite's existing tests: every
mocked-LLM test elsewhere (`tests/_support.py`'s `RecordingLLMClient`)
constructs `MemoryAction` objects directly in Python, bypassing the
JSON-parsing step entirely -- which is exactly the step this bug lived
in. These tests instead go through the real parsing path
(`LLMClient._parse_and_validate`) with a raw JSON string shaped exactly
like what a real LLM would emit if it followed the prompt's own
examples literally, since that's the only way to actually exercise the
prompt-text/schema-field-name contract.
"""

from __future__ import annotations

from finmate.llm import LLMClient
from finmate.prompts import MEMORY_AGENT
from finmate.schemas import MemoryAction


def test_memory_agent_prompt_uses_schema_field_name_new_value_not_value():
    """The prompt's own JSON examples must use the exact key
    `MemoryAction` reads (`new_value`), not a plausible-looking but
    wrong one (`value`) -- a mismatch here doesn't raise, it just makes
    every update silently fail (see module docstring)."""
    assert '"new_value"' in MEMORY_AGENT
    # A bare `"value":` key (the original bug) must not reappear --
    # deliberately checking for the colon so this doesn't false-positive
    # on the word "value" appearing in prose elsewhere in the prompt.
    assert '"value":' not in MEMORY_AGENT


def test_memory_agent_prompt_none_example_uses_empty_string_field_not_null():
    """The "no update needed" example must show `"field": ""`, not
    `"field": null` -- the latter is a hard ValidationError against the
    schema's non-optional str field (see module docstring)."""
    assert '"field": ""' in MEMORY_AGENT
    assert '"field": null' not in MEMORY_AGENT


def test_realistic_update_json_now_parses_with_new_value_populated():
    """Simulates exactly what a real LLM emits if it follows
    MEMORY_AGENT's current (fixed) few-shot examples literally -- through
    the real JSON-parsing path, not a MemoryAction built directly in
    Python. This is the actual bug report's scenario: a user saying
    "Set monthly_income to 200,000 INR" (i.e., a clean, unambiguous
    update)."""
    raw = '{"memory_action": "update", "field": "monthly_income", "new_value": 200000, "requires_confirmation": false}'
    action = LLMClient._parse_and_validate(raw, MemoryAction)
    assert action.field == "monthly_income"
    assert action.new_value == 200000
    # Exactly the condition orchestrator.py's profile_update fast path
    # checks before applying the update -- this must be True now.
    assert (
        action.memory_action != "none"
        and not action.requires_confirmation
        and action.field
        and action.new_value is not None
    )


def test_realistic_update_json_with_the_old_buggy_key_still_fails_the_check():
    """Documents the actual failure mode (not just that it's fixed): if
    an LLM ever again emits the legacy "value" key -- e.g. a different
    provider/model that doesn't follow the prompt's examples as
    literally -- this still parses "successfully" (no exception) with
    new_value left at None, and still correctly fails orchestrator.py's
    update check. This is why fixing the prompt (see the two tests
    above) is the real fix: nothing downstream can distinguish this
    case from "the LLM genuinely couldn't find an update" without an
    exception to catch."""
    raw = '{"memory_action": "update", "field": "monthly_income", "value": 200000, "requires_confirmation": false}'
    action = LLMClient._parse_and_validate(raw, MemoryAction)  # does not raise
    assert action.new_value is None
    assert not (
        action.memory_action != "none"
        and not action.requires_confirmation
        and action.field
        and action.new_value is not None
    )


def test_realistic_none_example_json_parses_without_raising():
    """Regression test for the second bug: field: null used to raise a
    ValidationError for the "no update needed" case."""
    raw = '{"memory_action": "none", "field": "", "new_value": null, "requires_confirmation": false}'
    action = LLMClient._parse_and_validate(raw, MemoryAction)  # must not raise
    assert action.memory_action == "none"
    assert action.field == ""


def test_realistic_none_example_json_with_the_old_buggy_null_field_does_raise():
    """Documents the failure mode this fix prevents: unlike the
    value/new_value bug above, this one WAS loud (a real
    ValidationError) -- which is exactly why LLMClient.call's built-in
    single retry could sometimes paper over it (the retry message tells
    the model what failed and asks again), making it an intermittent
    rather than a total failure. Still worth having fixed at the source
    rather than relying on a retry to mask it every time."""
    import pydantic

    raw = '{"memory_action": "none", "field": null, "new_value": null, "requires_confirmation": false}'
    try:
        LLMClient._parse_and_validate(raw, MemoryAction)
        assert False, "expected a ValidationError"
    except pydantic.ValidationError:
        pass
