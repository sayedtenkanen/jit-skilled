"""Apple Foundation Models backend: shells out to a small Swift CLI helper
that wraps Apple's on-device `FoundationModels` framework.

There is no public Python (or any non-Swift) API for Apple's on-device
model, so this backend works by subprocess: Python writes one JSON object
to the helper's stdin and reads one JSON object back from its stdout. The
helper's source is in apple_foundation_cli/ at the repo root -- you build
it once with Swift on your Mac; this Python code never talks to Apple's
framework directly.

Requirements (on your machine, not this pipeline):
  - macOS 26 (Tahoe) or newer, Apple Intelligence enabled, model downloaded
    (Settings > Apple Intelligence & Siri).
  - Xcode / Swift toolchain to build apple_foundation_cli/.
  - `swift build -c release` inside apple_foundation_cli/, then either put
    the resulting binary on PATH as `jitskilled-apple-fm`, or point
    APPLE_FM_CLI_PATH at it directly.

This backend cannot be exercised in a Linux sandbox -- the stdin/stdout
JSON contract below is what's tested from the Python side; the Swift half
needs to be built and run on an actual Mac.

Configuration (env var, optional):
  APPLE_FM_CLI_PATH  default: "jitskilled-apple-fm" (must be on PATH)
"""
from __future__ import annotations

import json
import os
import subprocess

from .llm import PromptedLLM

_DEFAULT_CLI = "jitskilled-apple-fm"
_TIMEOUT_SECONDS = 120


class AppleFoundationClient(PromptedLLM):
    def __init__(self, cli_path: str | None = None):
        self._cli_path = cli_path or os.environ.get("APPLE_FM_CLI_PATH") or _DEFAULT_CLI

    def _complete(self, system: str, user: str, max_tokens: int = 800) -> str:
        request = json.dumps({"system": system, "user": user, "max_tokens": max_tokens})
        try:
            proc = subprocess.run(
                [self._cli_path],
                input=request,
                capture_output=True,
                text=True,
                timeout=_TIMEOUT_SECONDS,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Could not find Apple Foundation Models CLI helper "
                f"{self._cli_path!r}. Build it from apple_foundation_cli/ "
                "with `swift build -c release` on macOS 26+, then set "
                "APPLE_FM_CLI_PATH to the built binary (or put it on PATH "
                f"as {_DEFAULT_CLI!r})."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"Apple Foundation Models CLI helper timed out after "
                f"{_TIMEOUT_SECONDS}s."
            ) from exc

        if proc.returncode != 0:
            raise RuntimeError(
                f"Apple Foundation Models CLI helper exited with code "
                f"{proc.returncode}: {proc.stderr.strip()}"
            )

        try:
            return json.loads(proc.stdout)["text"].strip()
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise RuntimeError(
                f"Unexpected output from Apple Foundation Models CLI helper: "
                f"{proc.stdout!r}"
            ) from exc
