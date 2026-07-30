"""Tests for prompts.py: verify prompt construction for all four steps."""
from jitskilled.prompts import (
    critic_prompt,
    editor_prompt,
    solve_prompt,
    synthesize_skill_prompt,
)

TARGET = {"task_id": "t0", "question": "What was revenue?", "source_doc": "q3"}
RETRIEVED = [
    {"question": "What was margin?", "ground_truth": "61%", "source_doc": "q3"},
]


def test_synthesize_skill_prompt_contains_framework_and_retrieved():
    system, user = synthesize_skill_prompt("FRAMEWORK", "SLOTS", TARGET, RETRIEVED)
    assert "FRAMEWORK" in user
    assert "SLOTS" in user
    assert "What was revenue?" in user
    assert "margin" in user
    assert "skill-synthesis" in system.lower() or "skill" in system.lower()


def test_synthesize_skill_prompt_handles_empty_retrieved():
    system, user = synthesize_skill_prompt("FW", "SL", TARGET, [])
    assert "no similar examples" in user.lower()


def test_solve_prompt_without_skill():
    system, user = solve_prompt("Q?", "Document text.")
    assert "Document text." in user
    assert "Q?" in user
    assert "SKILL" not in user


def test_solve_prompt_with_skill():
    system, user = solve_prompt("Q?", "Document text.", skill_text="SKILL.md content")
    assert "SKILL.md content" in user
    assert "Document text." in user


def test_critic_prompt_returns_json_friendly_payload():
    import json
    case = {"task_id": "t0", "outcome": "failed"}
    system, user = critic_prompt(case)
    assert "critic" in system.lower()
    parsed = json.loads(user)
    assert parsed["task_id"] == "t0"


def test_editor_prompt_returns_json_friendly_payload():
    import json
    payload = {"case_critic_results": [], "current_slot_library": {}}
    system, user = editor_prompt(payload)
    assert "editor" in system.lower()
    parsed = json.loads(user)
    assert "case_critic_results" in parsed
