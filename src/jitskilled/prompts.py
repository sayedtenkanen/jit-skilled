"""Shared prompt construction for every real (non-mock) LLM backend.

Every backend (Anthropic, Ollama, Apple Foundation Models) needs the exact
same system/user text for each of the four SkillTTA steps. Building them
here once means the three backends can't silently drift apart.
"""
from __future__ import annotations

import json
from typing import Any


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
        f"- Q: {r['question']}\n  A: {r['ground_truth']}\n  (doc: {r['source_doc']})"
        for r in retrieved
    ) or "(no similar examples retrieved)"
    user = (
        f"{framework}\n\n---\nCANDIDATE SLOT LIBRARY (guidance you may draw on):\n"
        f"{slot_library_text}\n\n---\nTARGET TASK:\nQuestion: {target['question']}\n"
        f"Source document: {target['source_doc']}\n\n---\nRETRIEVED EXAMPLES "
        f"(similar past questions with known-correct answers):\n{retrieved_block}\n\n"
        "Write the SKILL.md now."
    )
    return system, user


def solve_prompt(question: str, document_text: str,
                  skill_text: str | None = None) -> tuple[str, str]:
    system = (
        "Answer the question using only the provided document. "
        "If a SKILL is provided, follow its instructions exactly. "
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
        "Return JSON only, no prose, no code fences."
    )
    user = json.dumps(payload, indent=2)
    return system, user
