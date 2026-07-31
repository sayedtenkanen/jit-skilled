"""LLM client abstraction.

Four implementations, all behind the same SkillTTALLM interface so the rest
of the codebase never needs to know which backend is in use:

  - AnthropicClient: real Claude calls via the Anthropic API (needs
    ANTHROPIC_API_KEY + `pip install jit-skilled[anthropic]`).
  - OllamaClient (llm_ollama.py): a locally-running Ollama model over its
    REST API. No API key, no extra Python dependency -- needs `ollama
    serve` running with a model pulled.
  - AppleFoundationClient (llm_apple.py): Apple's on-device Foundation
    Models framework, via Apple's own `apple-fm-sdk` Python bindings
    (`pip install jit-skilled[apple]`). macOS 26+/Apple Silicon with
    Apple Intelligence only.
  - MockClient: a deterministic, keyword-overlap heuristic. It exists so
    the whole pipeline is runnable and testable with zero setup. It does
    NOT demonstrate that skills improve accuracy -- it proves the plumbing
    (retrieval -> synthesis -> solve -> grade -> critic -> editor) is wired
    correctly end to end. Use a real backend for actual skill quality.

Pick a backend with get_client(backend), the JITSKILLED_LLM_BACKEND env
var, or --llm on the CLI (highest to lowest precedence: explicit argument,
env var, auto-detect from ANTHROPIC_API_KEY, mock).
"""
from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from typing import Any

from . import prompts
from ._util import parse_json_response, retry_with_backoff

_BACKENDS = ("anthropic", "ollama", "apple", "mock")


class SkillTTALLM(ABC):
    @abstractmethod
    def synthesize_skill(self, framework: str, slot_library_text: str,
                          target: dict[str, Any],
                          retrieved: list[dict[str, Any]]) -> str:
        """Return a SKILL.md markdown string for this target task."""

    @abstractmethod
    def solve(self, question: str, document_text: str,
              skill_text: str | None = None) -> str:
        """Return an answer string. skill_text is None for zero-shot."""

    @abstractmethod
    def critic(self, case_payload: dict[str, Any]) -> dict[str, Any]:
        """Return a dict matching the critic schema (see optimize.py)."""

    @abstractmethod
    def editor(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Return a dict: {"operations": [...]}."""

    @abstractmethod
    def judge(self, question: str, answer: str, ground_truth: str) -> dict[str, Any]:
        """Return {"correct": bool, "reason": str}. Used by
        grader.grade_with_judge as a fallback for answers the fast
        deterministic check can't confidently resolve."""


class PromptedLLM(SkillTTALLM):
    """Base class for backends driven by a single text-completion call.

    Subclasses implement only `_complete(system, user, max_tokens) -> str`;
    the four SkillTTA-step methods are implemented once here using the
    shared prompts in prompts.py, so every real backend gets identical
    prompts and identical JSON-parsing behavior.
    """

    @abstractmethod
    def _complete(self, system: str, user: str, max_tokens: int = 800) -> str:
        ...

    def synthesize_skill(self, framework, slot_library_text, target, retrieved):
        system, user = prompts.synthesize_skill_prompt(
            framework, slot_library_text, target, retrieved
        )
        return retry_with_backoff(self._complete, system, user, max_tokens=800).strip()

    def solve(self, question, document_text, skill_text=None):
        system, user = prompts.solve_prompt(question, document_text, skill_text)
        return retry_with_backoff(self._complete, system, user, max_tokens=100).strip()

    def critic(self, case_payload):
        system, user = prompts.critic_prompt(case_payload)

        def _generate_and_parse() -> dict[str, Any]:
            return parse_json_response(self._complete(system, user, max_tokens=500))

        # The retry wraps generation AND parsing together, not just the raw
        # API call: a malformed-JSON response (e.g. an on-device model
        # emitting an unescaped quote inside a string value) is a one-off
        # generation quirk worth retrying, not just a network hiccup. See
        # _util.retry_with_backoff's "could not parse json" retry trigger.
        return retry_with_backoff(_generate_and_parse)

    def editor(self, payload):
        system, user = prompts.editor_prompt(payload)

        def _generate_and_parse() -> dict[str, Any]:
            return parse_json_response(self._complete(system, user, max_tokens=600))

        return retry_with_backoff(_generate_and_parse)

    def judge(self, question, answer, ground_truth):
        system, user = prompts.judge_prompt(question, answer, ground_truth)

        def _generate_and_parse() -> dict[str, Any]:
            return parse_json_response(self._complete(system, user, max_tokens=150))

        return retry_with_backoff(_generate_and_parse)


def get_client(backend: str | None = None) -> SkillTTALLM:
    """Resolve which SkillTTALLM implementation to use.

    Precedence: explicit `backend` argument > JITSKILLED_LLM_BACKEND env var >
    auto-detect (ANTHROPIC_API_KEY present -> anthropic) > mock.
    """
    backend = backend or os.environ.get("JITSKILLED_LLM_BACKEND")
    if backend and backend not in _BACKENDS:
        raise ValueError(f"Unknown LLM backend {backend!r}; must be one of {_BACKENDS}")

    if backend == "mock":
        return MockClient()
    if backend == "anthropic":
        return AnthropicClient()
    if backend == "ollama":
        from .llm_ollama import OllamaClient
        return OllamaClient()
    if backend == "apple":
        from .llm_apple import AppleFoundationClient
        return AppleFoundationClient()

    # No explicit backend requested: auto-detect, falling back to mock.
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return AnthropicClient()
        except ImportError:
            print("[llm] ANTHROPIC_API_KEY is set but `anthropic` package "
                  "is not installed (pip install jit-skilled[anthropic]). "
                  "Falling back to MockClient.")
    return MockClient()


class AnthropicClient(PromptedLLM):
    def __init__(self, model: str = "claude-sonnet-4-5"):
        import anthropic  # local import: only required for this backend
        self._client = anthropic.Anthropic()
        self._model = model

    def _complete(self, system: str, user: str, max_tokens: int = 800) -> str:
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(
            block.text for block in resp.content if block.type == "text"
        ).strip()


_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "what", "how", "when",
    "did", "do", "does", "of", "in", "to", "for", "and", "or", "on", "at",
    "many", "much", "than", "with", "per",
}
_VALUE_RE = re.compile(
    r"\$[\d,]+(?:\.\d+)?(?:\s?million|\s?billion)?|\d+(?:\.\d+)?%|"
    r"\d+(?:\.\d+)?\s?(?:kilograms|kg|meters?|days?|employees?)|"
    r"[A-Z][a-z]+ \d{1,2}, \d{4}|\b\d+\b"
)


def _render_mock_example(r: dict[str, Any]) -> str:
    line = f"- similar question answered as: `{r['ground_truth']}`"
    attempt, label = r.get("prior_attempt"), r.get("prior_label")
    if attempt is not None and label is not None:
        verdict = "correct" if label == "pass" else "wrong"
        line += f" (a prior zero-shot attempt guessed `{attempt}`, which was {verdict})"
    return line


class MockClient(SkillTTALLM):
    """Deterministic keyword-overlap heuristic. See module docstring."""

    def synthesize_skill(self, framework, slot_library_text, target, retrieved):
        docs = sorted({r["source_doc"] for r in retrieved})
        examples = "\n".join(_render_mock_example(r) for r in retrieved[:3])
        return (
            "# SKILL.md\n\n"
            "## Task Type\n"
            f"Fact lookup over a short document ({target['source_doc']}).\n\n"
            "## Retrieval Notes\n"
            f"- {len(retrieved)} similar questions retrieved, most from: "
            f"{', '.join(docs) or 'n/a'}.\n"
            f"{examples}\n\n"
            "## Answering Strategy\n"
            "- Find the sentence whose subject matches the question.\n"
            "- Extract the number, currency, percentage, or date it contains.\n"
            "- [MOCK] Slot library guidance was not applied by reasoning "
            "(deterministic heuristic only) -- use a real backend for "
            "actual skill quality.\n\n"
            "## Output Format\n"
            "Return the value exactly as written in the document, unit included.\n"
        )

    def solve(self, question, document_text, skill_text=None):
        q_tokens = {t for t in re.findall(r"[a-z']+", question.lower())
                    if t not in _STOPWORDS}
        sentences = re.split(r"(?<=[.\n])\s+", document_text)
        best_sentence, best_score = "", -1
        for s in sentences:
            s_tokens = set(re.findall(r"[a-z']+", s.lower()))
            score = len(q_tokens & s_tokens)
            if score > best_score:
                best_sentence, best_score = s, score
        values = _VALUE_RE.findall(best_sentence)
        if not values:
            return "not found"
        # Prefer a currency/percentage/unit/date match over a bare number:
        # a bare \d+ fallback can match an incidental year (e.g. "Q3 2026")
        # that appears earlier in the sentence than the actual answer.
        rich_values = [v for v in values if not v.isdigit()]
        return rich_values[0] if rich_values else values[0]

    def critic(self, case_payload):
        failed = case_payload.get("outcome") == "failed"
        return {
            "task_id": case_payload.get("task_id"),
            "failure_attribution": "skill" if failed else None,
            "success_attribution": None if failed else "skill",
            "evidence": ["[MOCK] deterministic placeholder critic"],
            "lesson": (
                "When multiple numbers appear near the question's subject, "
                "prefer the one in the same sentence as the exact keyword "
                "match." if failed else None
            ),
            "implicated_slot_id": "prefer_nearest_sentence" if failed else None,
        }

    def editor(self, payload):
        # candidate_index varies which deterministic patch comes back, so
        # multi-candidate optimize runs produce genuinely different
        # candidates even against the mock backend (see optimize.py).
        candidate_index = payload.get("candidate_index", 1)
        if candidate_index % 3 == 1:
            return {
                "operations": [
                    {
                        "operation": "modify_slot",
                        "category": "input",
                        "slot_id": "prefer_nearest_sentence",
                        "reason": f"[MOCK candidate {candidate_index}] reinforcing "
                                  "based on placeholder critic lessons",
                        "text": (
                            "Answers are usually contained in a single sentence "
                            "of the source document. Locate the sentence whose "
                            "subject matches the question's subject, and if "
                            "several numbers appear nearby, prefer the one in "
                            "that exact sentence over ones in neighboring "
                            "sentences."
                        ),
                    }
                ]
            }
        if candidate_index % 3 == 2:
            return {
                "operations": [
                    {
                        "operation": "add_slot",
                        "category": "input",
                        "slot_id": "cross_check_neighboring_sentences",
                        "reason": f"[MOCK candidate {candidate_index}] alternative "
                                  "strategy: explicitly rule out adjacent sentences",
                        "text": (
                            "Before finalizing a number, check the sentence "
                            "immediately before and after the matched sentence "
                            "for a similar-looking value, and prefer the one "
                            "whose surrounding words match the question's "
                            "wording most closely."
                        ),
                    }
                ]
            }
        return {
            "operations": [
                {
                    "operation": "modify_slot",
                    "category": "output",
                    "slot_id": "answer_only",
                    "reason": f"[MOCK candidate {candidate_index}] alternative "
                              "strategy: tighten output formatting instead",
                    "text": (
                        "State only the final answer, with no surrounding "
                        "explanation, caveats, or restatement of the question. "
                        "If the retrieved examples' answers include a unit or "
                        "symbol, the output must include it too."
                    ),
                }
            ]
        }

    def judge(self, question, answer, ground_truth):
        # [MOCK] word-overlap heuristic, not real semantic judgment -- good
        # enough to prove grade_with_judge's escalation wiring works: use a
        # real backend for actual grading quality on free-text answers.
        answer_words = set(re.findall(r"[a-z0-9']+", answer.lower()))
        gt_words = set(re.findall(r"[a-z0-9']+", ground_truth.lower()))
        overlap = len(answer_words & gt_words) / max(1, len(gt_words))
        correct = overlap >= 0.5
        return {
            "correct": correct,
            "reason": f"[MOCK] {overlap:.0%} word overlap with reference answer",
        }
