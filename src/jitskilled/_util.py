"""Small helpers shared across LLM client implementations."""
from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable
from typing import Any, TypeVar

_T = TypeVar("_T")
_log = logging.getLogger(__name__)

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def retry_with_backoff(fn: Callable[..., _T], *args: Any,
                        max_retries: int = 3, base_delay: float = 1.0,
                        **kwargs: Any) -> _T:
    """Call *fn* with retries on transient errors (API errors, timeouts).

    Uses exponential backoff: 1s, 2s, 4s. Only retries on exceptions that
    look transient (network errors, rate limits, 5xx). Permanent errors
    (auth, bad request) are raised immediately.
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if attempt == max_retries:
                raise
            msg = str(exc).lower()
            _retryable = (
                "rate" in msg or "timeout" in msg or "500" in msg
                or "502" in msg or "503" in msg or "529" in msg
                or "overloaded" in msg or "try again" in msg
                or "connection" in msg or "temporarily" in msg
                or "decoding" in msg or "decode" in msg
            )
            if not _retryable:
                raise
            delay = base_delay * (2 ** attempt)
            _log.warning("LLM call failed (attempt %d/%d): %s — retrying in %.1fs",
                         attempt + 1, max_retries + 1, exc, delay)
            time.sleep(delay)
    raise last_exc  # unreachable but satisfies type checker


def parse_json_response(text: str) -> dict[str, Any]:
    """Extract and parse a JSON object from an LLM text response.

    Models frequently wrap JSON in prose or code fences despite being asked
    for JSON only. This grabs the first {...} block before parsing, and
    raises a RuntimeError with the offending text (truncated) on failure
    rather than a bare, hard-to-debug JSONDecodeError.
    """
    stripped = text.strip()
    match = _JSON_OBJECT_RE.search(stripped)
    candidate = match.group(0) if match else stripped
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        preview = text[:300] + ("..." if len(text) > 300 else "")
        raise RuntimeError(
            f"Could not parse JSON from LLM response: {preview!r}"
        ) from exc
