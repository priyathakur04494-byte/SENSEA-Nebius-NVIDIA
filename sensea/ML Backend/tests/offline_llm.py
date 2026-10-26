"""Test-only offline LLM harness.

This module never performs network I/O. It imports ``llm`` (which builds
an OpenAI client object but does not contact the provider), keeps a
reference to that client so timeout/retry bounds can be asserted, then
replaces the network-facing functions and client with fakes/guards.

Usage in a standalone test script:

    from offline_llm import install_offline_llm
    fake = install_offline_llm()

After installation every ``llm.generate_answer`` call goes to ``fake`` and
any accidental direct use of ``llm._client`` raises ``NetworkBlocked``
instead of contacting a real API.
"""

import llm as _llm


# Kept so tests can inspect the real transport configuration without
# making a request, and can exercise the production error handlers with a
# raising stub client.
ORIGINAL_CLIENT = _llm._client
ORIGINAL_GENERATE_ANSWER = _llm.generate_answer
ORIGINAL_GENERATE_FALLBACK = _llm.generate_fallback


class NetworkBlocked(RuntimeError):
    """Raised if production code tries to reach the real LLM client."""


class _BlockedCompletions:
    def create(self, *args, **kwargs):
        raise NetworkBlocked(
            "real LLM call attempted during an offline test"
        )


class _BlockedChat:
    def __init__(self):
        self.completions = _BlockedCompletions()


class _BlockedClient:
    """Stand-in client whose only behaviour is to fail loudly."""

    def __init__(self):
        self.chat = _BlockedChat()


class FakeLLM:
    """Deterministic in-process replacement for the LLM functions."""

    def __init__(self, response: str | None = None):
        # None means "derive a grounded answer from the first memory Text
        # line", so the production grounding guard can be exercised with
        # realistic input. Set response explicitly to test other output.
        self.response = response
        self.last_response = None
        self.calls = []
        self.fallback_calls = []
        self.total_calls = 0
        self.total_fallback_calls = 0
        self.answer_exception = None
        self.fallback_exception = None

    @staticmethod
    def _grounded_default(context: str) -> str:
        for line in (context or "").splitlines():
            stripped = line.strip()

            if stripped.lower().startswith("text:"):
                return stripped[len("text:"):].strip()

        return "Offline fake LLM answer."

    def generate_answer(self, question: str, context: str) -> str:
        self.total_calls += 1
        self.calls.append({
            "question": question,
            "context": context,
        })

        if self.answer_exception is not None:
            raise self.answer_exception

        if self.response is None:
            answer = self._grounded_default(context)
        else:
            answer = self.response

        self.last_response = answer
        return answer

    def generate_fallback(self, question: str, mode: str) -> str:
        self.total_fallback_calls += 1
        self.fallback_calls.append({
            "question": question,
            "mode": mode,
        })

        if self.fallback_exception is not None:
            raise self.fallback_exception

        return self.response

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def reset(self) -> None:
        self.calls.clear()
        self.fallback_calls.clear()
        self.answer_exception = None
        self.fallback_exception = None
        self.response = None
        self.last_response = None

    def install(self) -> "FakeLLM":
        _llm.generate_answer = self.generate_answer
        _llm.generate_fallback = self.generate_fallback
        _llm._client = _BlockedClient()
        return self


_FAKE = None


def install_offline_llm() -> FakeLLM:
    """Install (once) and return the process-wide fake LLM."""
    global _FAKE

    if _FAKE is None:
        _FAKE = FakeLLM().install()

    return _FAKE
