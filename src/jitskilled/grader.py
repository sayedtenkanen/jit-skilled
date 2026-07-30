"""Automatic answer checking against ground truth."""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .llm import SkillTTALLM

_VALUE_TOKEN_RE = re.compile(r"\$?\d[\d.]*%?")


def _normalize(s: str) -> str:
    s = s.lower().strip()
    s = s.replace(",", "")
    s = re.sub(r"[^a-z0-9%$.]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _value_tokens(s: str) -> set[str]:
    return set(_VALUE_TOKEN_RE.findall(s))


def _has_boundary(haystack: str, needle: str) -> bool:
    """True if needle appears in haystack at word boundaries on both
    sides, avoiding false positives like "yes" matching inside "years".
    """
    idx = haystack.find(needle)
    if idx == -1:
        return False
    left_ok = idx == 0 or not haystack[idx - 1].isalnum()
    right = idx + len(needle)
    right_ok = right >= len(haystack) or not haystack[right].isalnum()
    return left_ok and right_ok


def grade(answer: str, ground_truth: str) -> bool:
    """Automatic correctness check for short factual answers.

    Rules, in order:
      1. Exact match after normalization -> pass.
      2. If the ground truth contains a number/currency/percentage, require
         an exact shared numeric token between answer and ground truth
         (NOT substring containment -- "3" must not match "30%").
      3. Otherwise (non-numeric ground truth), fall back to word-boundary
         substring containment (avoids "yes" matching inside "years").

    This is a fast, free, fully deterministic check -- reliable for the
    numeric/currency/percentage answers most of this demo dataset uses.
    For free-text ground truth it's necessarily shallower (surface-level
    text matching, no real semantic understanding); see grade_with_judge
    below for an LLM-backed escalation path on exactly those cases.
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

    return _has_boundary(norm_answer, norm_gt) or _has_boundary(norm_gt, norm_answer)


def grade_with_judge(llm: SkillTTALLM | None, question: str, answer: str,
                      ground_truth: str) -> tuple[bool, str]:
    """Two-tier grading: run the fast deterministic checks first, and only
    escalate to an LLM judge for the case grade() is least trustworthy on
    -- free-text ground truth with no numeric/currency/percentage token,
    where the deterministic boundary-match came back False (it could be a
    real miss, or just different phrasing of the same answer).

    Numeric/currency/percentage ground truth is NEVER escalated: the
    deterministic token check is authoritative there, cheaper, and doesn't
    depend on an LLM being available or correct.

    Returns (passed, reason). `llm` may be None (e.g. mock/offline runs);
    in that case an escalation-eligible case just falls back to the
    deterministic False rather than raising.
    """
    norm_answer = _normalize(answer)
    norm_gt = _normalize(ground_truth)
    if not norm_gt:
        return False, "empty ground truth"
    if norm_answer == norm_gt:
        return True, "exact match after normalization"

    gt_tokens = _value_tokens(norm_gt)
    if gt_tokens:
        matched = bool(gt_tokens & _value_tokens(norm_answer))
        reason = ("matched numeric/currency/percentage token" if matched
                  else "no shared numeric/currency/percentage token")
        return matched, reason

    if _has_boundary(norm_answer, norm_gt) or _has_boundary(norm_gt, norm_answer):
        return True, "word-boundary substring match"

    if llm is None:
        return False, "no boundary match; no LLM judge available to escalate to"

    verdict = llm.judge(question, answer, ground_truth)
    reason = verdict.get("reason") or "no reason given"
    return bool(verdict.get("correct")), f"LLM judge: {reason}"
