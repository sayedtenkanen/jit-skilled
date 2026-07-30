"""The frozen solver: answers a question, optionally guided by a skill."""
from __future__ import annotations

from .llm import SkillTTALLM


def solve_task(llm: SkillTTALLM, question: str, document_text: str,
               skill_text: str | None = None) -> str:
    return llm.solve(question, document_text, skill_text=skill_text).strip()
