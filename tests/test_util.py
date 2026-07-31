"""Tests for _util.py: retry_with_backoff + parse_json_response."""
import pytest

from jitskilled._util import parse_json_response, retry_with_backoff


def test_retry_succeeds_on_first_try():
    call_count = 0

    def fn():
        nonlocal call_count
        call_count += 1
        return "ok"

    result = retry_with_backoff(fn, max_retries=3, base_delay=0)
    assert result == "ok"
    assert call_count == 1


def test_retry_retries_on_transient_error():
    call_count = 0

    def fn():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ConnectionError("connection reset")
        return "recovered"

    result = retry_with_backoff(fn, max_retries=3, base_delay=0)
    assert result == "recovered"
    assert call_count == 3


def test_retry_retries_on_malformed_json_message():
    # Regression test: real on-device model output has been observed to
    # emit near-valid JSON with an unescaped quote inside a string value
    # (e.g. reason text quoting a phrase like "many units"), which
    # parse_json_response reports as "Could not parse JSON from LLM
    # response: ...". That failure mode should be retried like any other
    # transient generation hiccup, since a fresh generation often doesn't
    # repeat the same formatting mistake.
    call_count = 0

    def fn():
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise RuntimeError(
                "Could not parse JSON from LLM response: '{\"a\": \"broke \"here\"}'"
            )
        return {"a": "fixed"}

    result = retry_with_backoff(fn, max_retries=3, base_delay=0)
    assert result == {"a": "fixed"}
    assert call_count == 2


def test_retry_raises_on_permanent_error():
    def fn():
        raise PermissionError("unauthorized")

    with pytest.raises(PermissionError, match="unauthorized"):
        retry_with_backoff(fn, max_retries=3, base_delay=0)


def test_retry_raises_after_exhausted_retries():
    call_count = 0

    def fn():
        nonlocal call_count
        call_count += 1
        raise ConnectionError("rate limit exceeded")

    with pytest.raises(ConnectionError):
        retry_with_backoff(fn, max_retries=2, base_delay=0)
    assert call_count == 3  # 1 initial + 2 retries


def test_retry_passes_args_and_kwargs():
    def fn(a, b, c=None):
        return a + b + (c or 0)

    result = retry_with_backoff(fn, 1, 2, c=3, max_retries=0, base_delay=0)
    assert result == 6


def test_parse_json_response_extracts_object():
    assert parse_json_response('{"a": 1}') == {"a": 1}


def test_parse_json_response_extracts_from_prose():
    text = 'Here is the result:\n{"key": "value"}\nDone.'
    assert parse_json_response(text) == {"key": "value"}
