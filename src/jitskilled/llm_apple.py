"""Apple Foundation Models backend: talks to Apple's on-device model
directly via `apple-fm-sdk`, Apple's own Python bindings for the
`FoundationModels` framework (see
https://apple.github.io/python-apple-fm-sdk/). No subprocess, no Swift to
compile -- earlier versions of this backend shelled out to a hand-written
Swift CLI helper because there was no Python API; Apple has since shipped
one directly, which this backend uses instead.

Requirements (on your machine, not this pipeline):
  - macOS 26+ with Apple Intelligence turned on and the on-device model
    downloaded (Settings > Apple Intelligence & Siri).
  - Apple Silicon Mac, Xcode 26+, Python 3.10+.
  - `pip install jit-skilled[apple]` (installs apple-fm-sdk).

This backend cannot be imported or exercised in a Linux sandbox -- the SDK
itself only installs on macOS. tests/test_llm_apple.py contract-tests the
Python-side logic (session creation, instructions/options mapping, error
translation, availability checks) against a fake module matching
apple_fm_sdk's documented shape, injected via the `sdk` constructor
parameter -- not the real (macOS-only) package.
"""
from __future__ import annotations

import asyncio
from typing import Any

from .llm import PromptedLLM


def _import_sdk() -> Any:
    try:
        import apple_fm_sdk
    except ImportError as exc:
        raise ImportError(
            "The 'apple' backend needs apple-fm-sdk, which only installs "
            "on macOS 26+ with an Apple Silicon Mac (see "
            "https://apple.github.io/python-apple-fm-sdk/ for full "
            "requirements). Install it with: pip install jit-skilled[apple]"
        ) from exc
    return apple_fm_sdk


class AppleFoundationClient(PromptedLLM):
    def __init__(self, sdk: Any = None):
        # `sdk` is overridable so tests can inject a fake apple_fm_sdk
        # module without the real (macOS-only) package being installed.
        self._fm = sdk if sdk is not None else _import_sdk()

        model = self._fm.SystemLanguageModel()
        is_available, reason = model.is_available()
        if not is_available:
            raise RuntimeError(
                f"Apple on-device Foundation Model is not available: "
                f"{reason}. Check System Settings > Apple Intelligence & "
                "Siri, and that the on-device model has finished "
                "downloading."
            )
        self._model = model

    def _complete(self, system: str, user: str, max_tokens: int = 800) -> str:
        try:
            return asyncio.run(self._respond(system, user, max_tokens))
        except self._fm.FoundationModelsError as exc:
            raise RuntimeError(
                f"Apple Foundation Models request failed: {exc}"
            ) from exc

    async def _respond(self, system: str, user: str, max_tokens: int) -> str:
        # A fresh session per call, not a persisted multi-turn conversation:
        # each SkillTTA step (synthesize/solve/critic/editor/judge) is an
        # independent one-shot request, matching how every other backend
        # in this codebase is used (see PromptedLLM in llm.py).
        session = self._fm.LanguageModelSession(instructions=system, model=self._model)
        options = self._fm.GenerationOptions(maximum_response_tokens=max_tokens)
        response = await session.respond(user, options=options)
        return response.strip() if isinstance(response, str) else str(response)
