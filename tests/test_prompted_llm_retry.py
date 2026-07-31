"""Regression tests for PromptedLLM.critic/editor/judge retrying across
the generate+parse boundary, not just the raw completion call.

This was a real bug found via a live run against the Apple Foundation
Models backend: the on-device model returned JSON with an unescaped quote
inside a string value, parse_json_response raised, and -- because
retry_with_backoff previously wrapped only self._complete(), not the
parse step -- the whole optimize run crashed on a single malformed
generation instead of retrying. critic()/editor()/judge() now wrap
generation AND parsing together so a fresh (likely well-formed) retry can
recover.
"""
from __future__ import annotations

from typing import Any

import pytest

from jitskilled.llm import PromptedLLM


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    # critic()/editor()/judge() call retry_with_backoff with its default
    # base_delay=1.0 (they don't expose a base_delay override), so without
    # this the exponential backoff (1s, 2s, 4s...) actually sleeps during
    # these tests. Patch it out so retry behavior is still exercised for
    # real, just not in real time.
    monkeypatch.setattr("jitskilled._util.time.sleep", lambda _seconds: None)


class _FlakyJSONClient(PromptedLLM):
    """Returns malformed JSON on the first call, valid JSON after that."""

    def __init__(self, valid_payload: str):
        self._valid_payload = valid_payload
        self.call_count = 0

    def _complete(self, system: str, user: str, max_tokens: int = 800) -> str:
        self.call_count += 1
        if self.call_count == 1:
            # Mirrors the real failure: an unescaped quote inside a string
            # value breaks json.loads with "Expecting ',' delimiter".
            return '{"reason": "quoting "this" breaks json"}'
        return self._valid_payload


def test_critic_retries_past_malformed_json_response():
    client = _FlakyJSONClient(
        '{"task_id": "t0", "failure_attribution": "skill", '
        '"success_attribution": null, "evidence": ["e"], '
        '"lesson": "l", "implicated_slot_id": null}'
    )
    result = client.critic({"task_id": "t0", "outcome": "failed"})
    assert result["task_id"] == "t0"
    assert client.call_count == 2


def test_editor_retries_past_malformed_json_response():
    client = _FlakyJSONClient('{"operations": []}')
    result = client.editor({"case_critic_results": []})
    assert result == {"operations": []}
    assert client.call_count == 2


def test_judge_retries_past_malformed_json_response():
    client = _FlakyJSONClient('{"correct": true, "reason": "matches"}')
    result = client.judge("q", "answer", "ground truth")
    assert result == {"correct": True, "reason": "matches"}
    assert client.call_count == 2


def test_critic_still_raises_if_every_attempt_is_malformed():
    class _AlwaysBrokenClient(PromptedLLM):
        def __init__(self):
            self.call_count = 0

        def _complete(self, system: str, user: str, max_tokens: int = 800) -> str:
            self.call_count += 1
            return '{"a": "broke "here"}'

    client = _AlwaysBrokenClient()

    with pytest.raises(RuntimeError, match="Could not parse JSON"):
        client.critic({"task_id": "t0", "outcome": "failed"})
    # 1 initial attempt + 3 retries = 4 total calls (retry_with_backoff's
    # default max_retries=3).
    assert client.call_count == 4


def test_critic_editor_judge_return_type_annotation_is_consistent() -> None:
    # Sanity check that the retry-wrapped closures still return plain
    # dicts (not e.g. accidentally returning the closure itself).
    client = _FlakyJSONClient('{"operations": []}')
    result: dict[str, Any] = client.editor({"case_critic_results": []})
    assert isinstance(result, dict)
