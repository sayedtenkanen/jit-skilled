"""Contract tests for AppleFoundationClient against a stub CLI.

The real `jitskilled-apple-fm` binary is Swift, needs macOS 26+ with Apple
Intelligence, and can't be built or run in this (Linux) sandbox -- see
apple_foundation_cli/README.md. What CAN be verified here, without a Mac,
is that AppleFoundationClient correctly speaks the stdin/stdout JSON
contract documented in llm_apple.py and apple_foundation_cli/Sources/
jitskilled-apple-fm/main.swift: request encoding, response decoding, and
every documented failure mode (missing binary, non-zero exit, malformed
JSON, missing "text" key, timeout).

These tests spawn a real subprocess (tests/fixtures/fake_apple_cli.py)
rather than mocking subprocess.run, so they exercise the actual
serialization/deserialization boundary -- not just that the right function
was called.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from jitskilled.llm_apple import AppleFoundationClient

FAKE_CLI = str(Path(__file__).parent / "fixtures" / "fake_apple_cli.py")


def _client(**kwargs) -> AppleFoundationClient:
    kwargs.setdefault("cli_path", FAKE_CLI)
    kwargs.setdefault("timeout", 5)
    return AppleFoundationClient(**kwargs)


def test_success_round_trip():
    client = _client()
    result = client._complete("system prompt", "hello world", max_tokens=50)
    assert result == "echo: hello world"


def test_solve_uses_complete_and_strips_result():
    # Exercises the inherited PromptedLLM.solve() path end to end, not just
    # _complete() directly, to prove the base-class wiring works against a
    # real backend implementation too.
    client = _client()
    answer = client.solve("What is the capital?", "doc text")
    assert answer.startswith("echo:")


def test_missing_binary_raises_clear_runtime_error():
    client = _client(cli_path="/nonexistent/path/to/jitskilled-apple-fm")
    with pytest.raises(RuntimeError, match="Could not find Apple Foundation Models CLI helper"):
        client._complete("system", "hello")


def test_cli_nonzero_exit_raises_with_stderr_content():
    client = _client()
    with pytest.raises(RuntimeError, match="exited with code"):
        client._complete("system", "TRIGGER_CLI_ERROR please answer")


def test_cli_nonzero_exit_includes_apple_error_message():
    client = _client()
    with pytest.raises(RuntimeError, match="not available"):
        client._complete("system", "TRIGGER_CLI_ERROR please answer")


def test_malformed_json_output_raises_clear_error():
    client = _client()
    with pytest.raises(RuntimeError, match="Unexpected output"):
        client._complete("system", "TRIGGER_MALFORMED_OUTPUT please answer")


def test_missing_text_key_raises_clear_error():
    client = _client()
    with pytest.raises(RuntimeError, match="Unexpected output"):
        client._complete("system", "TRIGGER_MISSING_TEXT_KEY please answer")


def test_timeout_raises_clear_error():
    # timeout=1 in the client, fake CLI sleeps 30s on this marker -- proves
    # the TimeoutExpired path is wired up, without a slow test.
    client = _client(timeout=1)
    with pytest.raises(RuntimeError, match="timed out after 1s"):
        client._complete("system", "TRIGGER_TIMEOUT please answer")


def test_default_timeout_is_120_seconds_when_unset():
    client = AppleFoundationClient(cli_path=FAKE_CLI)
    assert client._timeout == 120


def test_cli_path_env_var_used_when_no_explicit_path(monkeypatch):
    monkeypatch.setenv("APPLE_FM_CLI_PATH", FAKE_CLI)
    client = AppleFoundationClient()
    assert client._cli_path == FAKE_CLI


def test_explicit_cli_path_overrides_env_var(monkeypatch):
    monkeypatch.setenv("APPLE_FM_CLI_PATH", "/some/other/path")
    client = AppleFoundationClient(cli_path=FAKE_CLI)
    assert client._cli_path == FAKE_CLI


def test_defaults_to_jitskilled_apple_fm_on_path(monkeypatch):
    monkeypatch.delenv("APPLE_FM_CLI_PATH", raising=False)
    client = AppleFoundationClient()
    assert client._cli_path == "jitskilled-apple-fm"


@pytest.mark.skipif(
    sys.platform != "linux" and sys.platform != "darwin",
    reason="shebang execution assumed for the fixture script",
)
def test_fake_cli_is_directly_executable():
    """Sanity check that the fixture itself is runnable as `[path]` (no
    `python3` prefix), matching how AppleFoundationClient invokes
    self._cli_path -- i.e. it's a faithful stand-in for a compiled binary.
    """
    import subprocess

    proc = subprocess.run(
        [FAKE_CLI],
        input='{"system": "s", "user": "hi", "max_tokens": 10}',
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert proc.returncode == 0
    assert '"text"' in proc.stdout
