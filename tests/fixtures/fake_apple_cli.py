#!/usr/bin/env python3
"""Stub stand-in for the compiled Swift `jitskilled-apple-fm` binary.

This is what tests/test_llm_apple.py uses to contract-test
AppleFoundationClient against: the real Swift CLI can't be built or run in
this (Linux) sandbox, but the stdin/stdout JSON contract it implements
(documented at the top of apple_foundation_cli/Sources/jitskilled-apple-fm/
main.swift) is plain text I/O and can be faithfully emulated here in
Python. Exercising AppleFoundationClient against this script -- via a real
subprocess, not a mock of subprocess.run -- proves the Python side holds up
its end of that contract: request encoding, response decoding, and error
handling for every failure mode the real binary can produce.

Behavior is selected by a marker substring in the request's "user" field
(see MARKER_* below), so each test in test_llm_apple.py can select a
specific CLI behavior just by choosing what question it asks.
"""
from __future__ import annotations

import json
import sys
import time

MARKER_ERROR = "TRIGGER_CLI_ERROR"
MARKER_MALFORMED = "TRIGGER_MALFORMED_OUTPUT"
MARKER_MISSING_TEXT = "TRIGGER_MISSING_TEXT_KEY"
MARKER_TIMEOUT = "TRIGGER_TIMEOUT"
MARKER_EMPTY_STDIN = "TRIGGER_EMPTY_STDIN"  # handled by the test, not here


def main() -> None:
    raw = sys.stdin.read()
    if not raw.strip():
        sys.stderr.write("No input on stdin.\n")
        sys.exit(1)

    try:
        request = json.loads(raw)
    except json.JSONDecodeError:
        sys.stderr.write("Could not parse stdin as JSON.\n")
        sys.exit(1)

    user = request.get("user", "")

    if MARKER_TIMEOUT in user:
        time.sleep(30)  # tests set a short client-side timeout to beat this
        return

    if MARKER_ERROR in user:
        sys.stderr.write("Apple on-device model is not available (unavailable).\n")
        sys.exit(1)

    if MARKER_MALFORMED in user:
        sys.stdout.write("this is not json at all {{{\n")
        return

    if MARKER_MISSING_TEXT in user:
        sys.stdout.write(json.dumps({"unexpected_key": "no text field here"}) + "\n")
        return

    # Default: well-formed success response, contract-shaped like the real
    # Swift CLI's {"text": "..."} output.
    sys.stdout.write(json.dumps({"text": f"echo: {user}"}) + "\n")


if __name__ == "__main__":
    main()
