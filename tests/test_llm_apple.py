"""Contract tests for AppleFoundationClient against a fake apple_fm_sdk.

The real `apple-fm-sdk` package only installs on macOS 26+ with Apple
Silicon (see https://apple.github.io/python-apple-fm-sdk/), so it can't be
installed or exercised in this (Linux) sandbox. What CAN be verified here,
without a Mac, is that AppleFoundationClient correctly drives the SDK's
documented shape: SystemLanguageModel().is_available(), constructing a
LanguageModelSession with `instructions`/`model`, passing
GenerationOptions(maximum_response_tokens=...), awaiting session.respond(),
and translating apple_fm_sdk.FoundationModelsError into a RuntimeError.

The fake below mirrors the real SDK's public surface (class names, method
signatures, return shapes) as documented at
https://apple.github.io/python-apple-fm-sdk/api/ -- it's injected via the
`sdk` constructor parameter rather than mocking internals, so these tests
exercise the same call sequence AppleFoundationClient would make against
the real package.
"""
from __future__ import annotations

import pytest

from jitskilled.llm_apple import AppleFoundationClient, _import_sdk


class FakeFoundationModelsError(Exception):
    """Stand-in for apple_fm_sdk.FoundationModelsError."""


class FakeGenerationOptions:
    def __init__(self, maximum_response_tokens=None, **kwargs):
        self.maximum_response_tokens = maximum_response_tokens


class FakeSystemLanguageModel:
    def __init__(self, available: bool, reason: str):
        self._available = available
        self._reason = reason

    def is_available(self):
        return self._available, self._reason


class FakeLanguageModelSession:
    def __init__(self, instructions, model, behavior: str):
        self.instructions = instructions
        self.model = model
        self._behavior = behavior
        self.last_prompt = None
        self.last_options = None

    async def respond(self, prompt, options=None):
        self.last_prompt = prompt
        self.last_options = options
        if self._behavior == "error":
            raise FakeFoundationModelsError("simulated generation failure")
        if self._behavior == "empty":
            return ""
        max_tok = options.maximum_response_tokens if options else None
        return f"echo: {prompt} (max_tokens={max_tok})"


class FakeSDK:
    """Fake apple_fm_sdk module, injected via AppleFoundationClient(sdk=...)."""

    FoundationModelsError = FakeFoundationModelsError

    def __init__(self, model_available=True, model_reason="ok", session_behavior="echo"):
        self._model_available = model_available
        self._model_reason = model_reason
        self._session_behavior = session_behavior
        self.last_session: FakeLanguageModelSession | None = None

    def SystemLanguageModel(self):
        return FakeSystemLanguageModel(self._model_available, self._model_reason)

    def GenerationOptions(self, **kwargs):
        return FakeGenerationOptions(**kwargs)

    def LanguageModelSession(self, instructions=None, model=None):
        session = FakeLanguageModelSession(instructions, model, self._session_behavior)
        self.last_session = session
        return session


def test_success_round_trip():
    client = AppleFoundationClient(sdk=FakeSDK())
    result = client._complete("system prompt", "hello world", max_tokens=50)
    assert result == "echo: hello world (max_tokens=50)"


def test_solve_uses_complete_end_to_end():
    # Exercises the inherited PromptedLLM.solve() path, not just
    # _complete() directly, to prove the base-class wiring works against
    # this backend too.
    client = AppleFoundationClient(sdk=FakeSDK())
    answer = client.solve("What is the capital?", "doc text")
    assert answer.startswith("echo:")


def test_instructions_passed_as_session_instructions():
    fake_sdk = FakeSDK()
    client = AppleFoundationClient(sdk=fake_sdk)
    client._complete("SYSTEM TEXT", "user text", max_tokens=100)
    assert fake_sdk.last_session.instructions == "SYSTEM TEXT"


def test_max_tokens_passed_through_as_generation_option():
    fake_sdk = FakeSDK()
    client = AppleFoundationClient(sdk=fake_sdk)
    client._complete("system", "user", max_tokens=321)
    assert fake_sdk.last_session.last_options.maximum_response_tokens == 321


def test_model_unavailable_raises_at_construction():
    fake_sdk = FakeSDK(model_available=False, model_reason="Apple Intelligence is disabled")
    with pytest.raises(RuntimeError, match="not available.*Apple Intelligence is disabled"):
        AppleFoundationClient(sdk=fake_sdk)


def test_generation_error_wrapped_as_runtime_error():
    client = AppleFoundationClient(sdk=FakeSDK(session_behavior="error"))
    with pytest.raises(RuntimeError, match="Apple Foundation Models request failed"):
        client._complete("system", "user")


def test_empty_response_does_not_raise():
    client = AppleFoundationClient(sdk=FakeSDK(session_behavior="empty"))
    assert client._complete("system", "user") == ""


def test_missing_package_raises_clear_import_error():
    # apple_fm_sdk is an optional extra; skip if it IS installed.
    try:
        import apple_fm_sdk  # noqa: F401
        pytest.skip("apple-fm-sdk is installed; "
                    "missing-dependency path not exercisable")
    except ImportError:
        pass
    with pytest.raises(ImportError, match="pip install jit-skilled\\[apple\\]"):
        AppleFoundationClient()


def test_import_sdk_reraises_as_import_error_with_install_hint():
    try:
        import apple_fm_sdk  # noqa: F401
        pytest.skip("apple-fm-sdk is installed; "
                    "missing-dependency path not exercisable")
    except ImportError:
        pass
    with pytest.raises(ImportError, match="apple-fm-sdk"):
        _import_sdk()
