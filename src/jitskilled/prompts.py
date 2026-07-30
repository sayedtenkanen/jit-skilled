"""Shared prompt construction for every real (non-mock) LLM backend.

Every backend (Anthropic, Ollama, Apple Foundation Models) needs the exact
same system/user text for each of the four SkillTTA steps. Building them
here once means the three backends can't silently drift apart.
"""
from __future__ import annotations

import json
from typing import Any


def _render_retrieved_example(r: dict[str, Any]) -> str:
    """Render one retrieved pool item for the synthesis prompt.

    When the pool item carries a recorded `prior_attempt` / `prior_label`
    (see scripts/generate_evolve_attempts.py), include what was actually
    tried and whether it worked -- not just the gold answer. This mirrors
    real SkillTTA's retrieval of past trajectories rather than bare Q&A
    pairs. Items without those fields (e.g. a hand-written pool that
    hasn't been through the generator) still render fine.
    """
    lines = [
        f"- Q: {r['question']}",
        f"  correct answer: {r['ground_truth']}",
        f"  (doc: {r['source_doc']})",
    ]
    attempt, label = r.get("prior_attempt"), r.get("prior_label")
    if attempt is not None and label is not None:
        verdict = "CORRECT" if label == "pass" else "WRONG"
        lines.append(f"  prior zero-shot attempt: {attempt!r} -> {verdict}")
    return "\n".join(lines)


def synthesize_skill_prompt(framework: str, slot_library_text: str,
                             target: dict[str, Any],
                             retrieved: list[dict[str, Any]]) -> tuple[str, str]:
    system = (
        "You are the skill-synthesis step of a test-time skill adaptation "
        "pipeline. Follow the required SKILL.md structure exactly. Ground "
        "every claim in the retrieved examples given. Never reveal or "
        "guess the target question's answer -- you are writing a strategy "
        "document, not answering the question."
    )
    retrieved_block = "\n\n".join(
        _render_retrieved_example(r) for r in retrieved
    ) or "(no similar examples retrieved)"
    user = (
        f"{framework}\n\n---\nCANDIDATE SLOT LIBRARY (guidance you may draw on):\n"
        f"{slot_library_text}\n\n---\nTARGET TASK:\nQuestion: {target['question']}\n"
        f"Source document: {target['source_doc']}\n\n---\nRETRIEVED EXAMPLES "
        f"(similar past questions, their correct answers, and -- where "
        f"recorded -- a prior zero-shot attempt and whether it was right):"
        f"\n{retrieved_block}\n\nWrite the SKILL.md now."
    )
    return system, user


def solve_prompt(question: str, document_text: str,
                  skill_text: str | None = None) -> tuple[str, str]:
    system = (
        "Answer the question using only the provided document. "
        "If a SKILL is provided, follow its instructions exactly. "
        "Do NOT use any tools, functions, or API calls. "
        "Do NOT emit tool calls or function invocations. "
        "Read the document text below and answer directly. "
        "Respond with the answer only, no explanation."
    )
    skill_block = f"SKILL:\n{skill_text}\n\n" if skill_text else ""
    user = f"{skill_block}DOCUMENT:\n{document_text}\n\nQUESTION: {question}\nAnswer:"
    return system, user


def critic_prompt(case_payload: dict[str, Any]) -> tuple[str, str]:
    system = (
        "You are the critic step of an offline skill-optimization loop. "
        "You will see one task's outcome. Return ONLY a JSON object with "
        "keys: task_id, failure_attribution (retrieval|skill|execution|"
        "other|null), success_attribution (skill|execution|other|null), "
        "evidence (1-3 grounded observations), lesson (short reusable "
        "instruction or null), implicated_slot_id (existing id or null). "
        "Exactly one of failure_attribution/success_attribution must be "
        "non-null. Return JSON only, no prose, no code fences."
    )
    user = json.dumps(case_payload, indent=2)
    return system, user


def judge_prompt(question: str, answer: str, ground_truth: str) -> tuple[str, str]:
    """For grader.grade_with_judge's escalation path: cases the fast
    deterministic check in grader.py couldn't confidently resolve (no
    shared numeric token, no exact/boundary text match) -- typically
    free-text answers phrased differently than the reference. Ask an LLM
    whether they're substantively the same answer.
    """
    system = (
        "You are a grading judge. Decide whether a candidate answer is "
        "substantively correct given the reference answer, allowing for "
        "different phrasing, synonyms, or extra context -- but not for "
        "different facts, numbers, or entities. Return ONLY a JSON object: "
        '{"correct": true|false, "reason": "one grounded sentence"}. '
        "Return JSON only, no prose, no code fences."
    )
    user = (
        f"QUESTION: {question}\nREFERENCE ANSWER: {ground_truth}\n"
        f"CANDIDATE ANSWER: {answer}\n\nIs the candidate answer correct?"
    )
    return system, user


def editor_prompt(payload: dict[str, Any]) -> tuple[str, str]:
    system = (
        "You are the editor step of an offline skill-optimization loop. "
        "You see critic reports (not raw trajectories) plus the current "
        "slot library. Propose 1-4 ordered operations to improve the "
        "library. Return ONLY a JSON object: "
        '{"operations": [{"operation": "add_slot|delete_slot|modify_slot", '
        '"category": "input|output", "slot_id": "stable_snake_case_id", '
        '"reason": "grounded explanation", '
        '"text": "complete slot text, or null for delete"}]}. '
        "If the payload includes candidate_index and num_candidates > 1, "
        "you are one of several independent proposals generated from the "
        "same critic results for manual beam search -- explore a genuinely "
        "different hypothesis than a generic single best-guess edit would, "
        "so the candidates are worth comparing rather than near-duplicates. "
        "Return JSON only, no prose, no code fences."
    )
    user = json.dumps(payload, indent=2)
    return system, user
