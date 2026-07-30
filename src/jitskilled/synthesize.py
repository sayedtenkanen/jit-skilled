"""Skill synthesis: retrieve + call the LLM to write a one-off SKILL.md."""
from __future__ import annotations

import logging

from .llm import SkillTTALLM
from .retrieval import top_k_retrieve
from .slots import render_slots

_log = logging.getLogger(__name__)

_REQUIRED_SECTIONS = ("## Task Type", "## Retrieval Notes",
                      "## Answering Strategy", "## Output Format")


def _validate_skill(skill_text: str) -> list[str]:
    """Return a list of warnings if the skill is missing required sections."""
    warnings = []
    for section in _REQUIRED_SECTIONS:
        if section not in skill_text:
            warnings.append(f"missing required section: {section}")
    if len(skill_text.split()) > 300:
        warnings.append(f"skill is {len(skill_text.split())} words (recommended max 200)")
    return warnings


def synthesize_skill_for_task(llm: SkillTTALLM, framework_text: str,
                               slot_library: dict, target: dict,
                               pool: list[dict], k: int = 3) -> tuple[str, list[dict]]:
    """Returns (skill_markdown, retrieved_examples)."""
    retrieved = top_k_retrieve(
        target["question"], pool, k=k, exclude_task_id=target["task_id"]
    )
    slot_text = render_slots(slot_library)
    skill_text = llm.synthesize_skill(framework_text, slot_text, target, retrieved)

    warnings = _validate_skill(skill_text)
    for w in warnings:
        _log.warning("skill validation for %s: %s", target["task_id"], w)

    return skill_text, retrieved
