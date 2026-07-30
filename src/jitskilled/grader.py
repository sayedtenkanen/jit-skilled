"""Automatic answer checking against ground truth."""
from __future__ import annotations

import re

_VALUE_TOKEN_RE = re.compile(r"\$?\d[\d.]*%?")


def _normalize(s: str) -> str:
    s = s.lower().strip()
    s = s.replace(",", "")
    s = re.sub(r"[^a-z0-9%$.]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _value_tokens(s: str) -> set[str]:
    return set(_VALUE_TOKEN_RE.findall(s))


def grade(answer: str, ground_truth: str) -> bool:
    """Automatic correctness check for short factual answers.

    Rules, in order:
      1. Exact match after normalization -> pass.
      2. If the ground truth contains a number/currency/percentage, require
         an exact shared numeric token between answer and ground truth
         (NOT substring containment -- "3" must not match "30%").
      3. Otherwise (non-numeric ground truth), fall back to loose substring
         containment.

    This is a stand-in for domain-specific grading. Replace it with your
    own validator (exact-match rules, a test suite, a business-rule check)
    for anything beyond this demo.
    """
    norm_answer = _normalize(answer)
    norm_gt = _normalize(ground_truth)
    if not norm_gt:
        return False
    if norm_answer == norm_gt:
        return True

    gt_tokens = _value_tokens(norm_gt)
    if gt_tokens:
        return bool(gt_tokens & _value_tokens(norm_answer))

    # For non-numeric answers, require word-boundary containment to avoid
    # false positives like "yes" matching inside "years".
    def _has_boundary(haystack: str, needle: str) -> bool:
        idx = haystack.find(needle)
        if idx == -1:
            return False
        left_ok = idx == 0 or not haystack[idx - 1].isalnum()
        right = idx + len(needle)
        right_ok = right >= len(haystack) or not haystack[right].isalnum()
        return left_ok and right_ok

    return _has_boundary(norm_answer, norm_gt) or _has_boundary(norm_gt, norm_answer)
