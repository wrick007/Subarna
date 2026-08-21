"""
Shared test support -- NOT a test file itself (no `test_` prefix, so
pytest never collects it directly).

`RecordingLLMClient`: a fake `finmate.llm.LLMClient` standing in for a
real provider. No real LLM call happens anywhere in this test suite (no
network in this sandbox regardless -- see README "Deviations", and this
project's actual tests were always designed to run without one). Records
every `.call()`/`.call_stream()` -- system prompt, user message, and
which `response_model` was asked for -- so a test can assert on exactly
what an agent sent, and returns a minimally valid response for whatever
was requested, configurable enough to drive the scenarios Priority 1 and
2's tests need (a specific router intent, a critic that fails N times
then passes, streamed text) without every test needing its own subclass.
"""

from __future__ import annotations

from finmate.schemas import CriticResult, MemoryAction, RouterOutput


class RecordingLLMClient:
    """
    `router_intent`: what RouterOutput.intent every router call returns.
      `general_finance` (the default) maps to ROUTING_TABLE's `[]`
      stages -- no rag/calculation/specialist agent ever runs -- which
      keeps a test's call graph small and predictable unless the test
      specifically wants more (pass a different intent, or override
      `router_output_overrides`).

    `critic_fail_times`: the first N critic calls return passed=False
      (with a fixed, recognizable error message); every call after that
      passes. 0 (the default) means the critic always passes on its
      first call, when it's called at all.

    `synthesis_text`: what every raw-text (response_model=None) call --
      i.e. synthesis -- returns. `.call_stream` yields this same text
      split into words (a crude but sufficient stand-in for token
      deltas: tests only need "more than one chunk arrives, and they
      concatenate back to the whole thing", not real tokenization).

    `router_output_overrides`: extra RouterOutput field values (e.g.
    `{"search_phrasings": ["dining", "restaurants"]}`) merged into every
    router response.
    """

    def __init__(
        self,
        router_intent: str = "general_finance",
        critic_fail_times: int = 0,
        synthesis_text: str = "ok",
        router_output_overrides: dict | None = None,
    ):
        self.calls: list[dict] = []
        self.router_intent = router_intent
        self.critic_fail_times = critic_fail_times
        self.synthesis_text = synthesis_text
        self.router_output_overrides = router_output_overrides or {}
        self._critic_calls_made = 0

    def call(
        self,
        agent_system_prompt,
        user_message,
        response_model=None,
        max_tokens: int = 2000,
        temperature: float = 0.0,
        include_constitution: bool = True,
    ):
        self.calls.append({
            "system_prompt": agent_system_prompt,
            "user_message": user_message,
            "response_model": response_model,
        })

        if response_model is RouterOutput:
            return RouterOutput(intent=self.router_intent, **self.router_output_overrides)

        if response_model is CriticResult:
            self._critic_calls_made += 1
            if self._critic_calls_made <= self.critic_fail_times:
                return CriticResult(
                    passed=False, confidence=0.2,
                    errors=[f"mocked verification failure #{self._critic_calls_made}"],
                )
            return CriticResult(passed=True, confidence=1.0)

        if response_model is MemoryAction:
            return MemoryAction(memory_action="none")

        # Raw text: synthesis (and, pre-Priority-2, the formatter).
        return self.synthesis_text

    def call_stream(
        self,
        agent_system_prompt,
        user_message,
        max_tokens: int = 2000,
        temperature: float = 0.0,
        include_constitution: bool = True,
    ):
        self.calls.append({
            "system_prompt": agent_system_prompt,
            "user_message": user_message,
            "response_model": None,
            "streamed": True,
        })
        words = self.synthesis_text.split(" ")
        for i, word in enumerate(words):
            yield word if i == len(words) - 1 else word + " "

    def calls_with(self, response_model=None, system_prompt_contains: str = "") -> list[dict]:
        return [
            c for c in self.calls
            if c["response_model"] is response_model and system_prompt_contains in c["system_prompt"]
        ]
