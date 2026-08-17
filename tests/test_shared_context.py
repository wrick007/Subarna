"""
Tests for the token-efficiency work described in
finmate/agents/_shared.py's module docstring: trimming EvidenceItem down
to the fields an agent actually needs before it hits a prompt, and
compact (not pretty-printed) JSON serialization. Covers `_shared.py`
itself (used by every specialist agent plus synthesis) and
`critic.py` (which builds a structurally similar payload of its own).

Neither module had prior direct test coverage -- correctness here was
implicitly exercised only by whichever agent happened to be running end
to end. These tests isolate the context-building logic itself, with no
LLM call involved (critic.py's is covered via a minimal fake client that
just records what it was asked to send).
"""

from __future__ import annotations

import json

from finmate.agents import critic as critic_module
from finmate.agents._shared import build_context_block, evidence_for_prompt
from finmate.rag import RetrievalResult
from finmate.schemas import CalculationResult, CriticResult, EvidenceItem, UserProfile


def _evidence_item(**overrides) -> EvidenceItem:
    defaults = dict(
        source_id="tx_1",
        date="2026-06-05",
        description="Zomato - dinner order",
        amount=-650.0,
        currency="INR",
        category="Dining out",
        document="HDFC Credit Card",
        page=None,
        relevance=0.87,
        keyword_score=12.4,
        vector_score=0.62,
        rerank_score=3.9,
        retrieval_stage="rerank",
    )
    defaults.update(overrides)
    return EvidenceItem(**defaults)


def _retrieval_result(items: list[EvidenceItem]) -> RetrievalResult:
    return RetrievalResult(evidence=items, vector_search_used=True, note="ok", stage="full hybrid")


# ---------------------------------------------------------------------------
# evidence_for_prompt
# ---------------------------------------------------------------------------


def test_evidence_for_prompt_keeps_only_the_fields_an_agent_needs():
    result = _retrieval_result([_evidence_item()])
    trimmed = evidence_for_prompt(result)

    assert len(trimmed) == 1
    assert trimmed[0] == {
        "source_id": "tx_1",
        "date": "2026-06-05",
        "description": "Zomato - dinner order",
        "amount": -650.0,
        "currency": "INR",
        "category": "Dining out",
        "document": "HDFC Credit Card",
        "relevance": 0.87,
    }


def test_evidence_for_prompt_drops_retrieval_audit_only_fields():
    result = _retrieval_result([_evidence_item()])
    trimmed = evidence_for_prompt(result)[0]

    for audit_field in ("keyword_score", "vector_score", "rerank_score", "retrieval_stage", "page"):
        assert audit_field not in trimmed


def test_evidence_for_prompt_does_not_mutate_the_original_evidence_items():
    item = _evidence_item()
    result = _retrieval_result([item])
    evidence_for_prompt(result)

    # The full object -- audit fields included -- is unchanged, since the
    # orchestrator/API/UI still need it (see finmate/rag.py "Performance").
    assert item.keyword_score == 12.4
    assert item.retrieval_stage == "rerank"


def test_evidence_for_prompt_preserves_order_and_handles_empty_evidence():
    items = [_evidence_item(source_id="a"), _evidence_item(source_id="b"), _evidence_item(source_id="c")]
    trimmed = evidence_for_prompt(_retrieval_result(items))
    assert [t["source_id"] for t in trimmed] == ["a", "b", "c"]

    assert evidence_for_prompt(_retrieval_result([])) == []


# ---------------------------------------------------------------------------
# build_context_block
# ---------------------------------------------------------------------------


def test_build_context_block_uses_compact_not_pretty_printed_json():
    profile = UserProfile(user_id="u1")
    result = _retrieval_result([_evidence_item()])
    calc = CalculationResult(metric="savings_rate", value=0.2, currency="INR", period="2026-06", formula="x/y")

    block = build_context_block("How much did I spend on food?", profile, result, [calc], [])
    # The natural-language instruction preamble legitimately contains ", "
    # and ": " as ordinary English punctuation -- only the JSON payload
    # itself (everything after the first blank line) is under test here.
    json_part = block.split("\n\n", 1)[1]

    # A pretty-printed (indent=2) dump always contains "\n  " between a
    # brace/bracket and its first key; compact json.dumps(separators=(",", ":"))
    # never does. This is a direct, mechanical check that indent=2 isn't
    # being used anymore -- not just an indirect inference from length.
    assert "\n" not in json_part
    assert ": " not in json_part  # compact separators use ":" with no trailing space
    assert ", " not in json_part  # ...and "," with no trailing space
    assert json.loads(json_part)  # still well-formed, parseable JSON


def test_build_context_block_embeds_valid_trimmed_json():
    profile = UserProfile(user_id="u1")
    result = _retrieval_result([_evidence_item(source_id="tx_9")])

    block = build_context_block("test message", profile, result, [], [])

    json_part = block.split("\n\n", 1)[1]
    payload = json.loads(json_part)
    assert payload["user_message"] == "test message"
    assert payload["evidence"] == [
        {
            "source_id": "tx_9",
            "date": "2026-06-05",
            "description": "Zomato - dinner order",
            "amount": -650.0,
            "currency": "INR",
            "category": "Dining out",
            "document": "HDFC Credit Card",
            "relevance": 0.87,
        }
    ]
    assert "keyword_score" not in json_part


def test_build_context_block_still_includes_retrieval_metadata_fields():
    profile = UserProfile(user_id="u1")
    result = RetrievalResult(evidence=[], vector_search_used=True, note="fell back to keyword only")
    block = build_context_block("msg", profile, result, [], ["net_worth: no account balances on file"])
    payload = json.loads(block.split("\n\n", 1)[1])

    assert payload["vector_search_used"] is True
    assert payload["retrieval_note"] == "fell back to keyword only"
    assert payload["skipped_calculations"] == ["net_worth: no account balances on file"]


# ---------------------------------------------------------------------------
# critic.py: same trim + compact-JSON treatment, own payload shape
# ---------------------------------------------------------------------------


class _RecordingLLMClient:
    def __init__(self, result: CriticResult):
        self._result = result
        self.last_user_message: str = ""

    def call(self, agent_system_prompt, user_message, response_model=None, **kwargs):
        self.last_user_message = user_message
        return self._result


def test_critic_payload_is_compact_and_evidence_is_trimmed():
    fake_result = CriticResult(passed=True)
    client = _RecordingLLMClient(fake_result)
    profile = UserProfile(user_id="u1")
    evidence = _retrieval_result([_evidence_item(source_id="tx_5")])

    returned = critic_module.run_critic_agent(client, "You spent 650 on dining.", profile, evidence, [])

    assert returned is fake_result
    assert "\n  " not in client.last_user_message  # not pretty-printed

    json_part = client.last_user_message.split("\n\n", 1)[1]
    assert "keyword_score" not in json_part
    assert "retrieval_stage" not in json_part
    payload = json.loads(json_part)
    assert payload["draft_response"] == "You spent 650 on dining."
    assert payload["evidence"][0]["source_id"] == "tx_5"
    assert "keyword_score" not in payload["evidence"][0]
