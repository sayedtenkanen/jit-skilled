"""Tests for synthesize.py: skill synthesis + validation."""

from jitskilled.llm import MockClient
from jitskilled.synthesize import _validate_skill, synthesize_skill_for_task

FRAMEWORK = (
    "## Task Type\nFact lookup.\n\n"
    "## Retrieval Notes\nBullets.\n\n"
    "## Answering Strategy\nSteps.\n\n"
    "## Output Format\nOne line."
)
POOL = [
    {"task_id": "e0", "question": "What was Q3 revenue?",
     "source_doc": "q3", "ground_truth": "$18.4 million"},
    {"task_id": "e1", "question": "What was Q3 gross margin?",
     "source_doc": "q3", "ground_truth": "61%"},
]
TARGET = {"task_id": "t0", "question": "What was net income?", "source_doc": "q3"}
SLOTS = {"slots": {"input": [{"id": "a", "text": "Copy units exactly."}],
                    "output": [{"id": "b", "text": "Answer only."}]}}


def test_synthesize_skill_for_task_returns_skill_and_retrieved():
    llm = MockClient()
    skill, retrieved = synthesize_skill_for_task(
        llm, FRAMEWORK, SLOTS, TARGET, POOL, k=2
    )
    assert isinstance(skill, str)
    assert len(skill) > 0
    assert isinstance(retrieved, list)
    assert len(retrieved) <= 2


def test_synthesize_skill_for_task_retrieved_has_expected_fields():
    llm = MockClient()
    _, retrieved = synthesize_skill_for_task(
        llm, FRAMEWORK, SLOTS, TARGET, POOL, k=2
    )
    for r in retrieved:
        assert "task_id" in r
        assert "question" in r
        assert "score" in r


def test_validate_skill_passes_for_well_formed_skill():
    skill = (
        "## Task Type\nFinancial lookup.\n\n"
        "## Retrieval Notes\n- Similar questions.\n\n"
        "## Answering Strategy\n- Find sentence.\n\n"
        "## Output Format\nReturn value.\n"
    )
    warnings = _validate_skill(skill)
    assert warnings == []


def test_validate_skill_warns_on_missing_section():
    skill = "## Task Type\nLookup.\n\n## Answering Strategy\nSteps.\n"
    warnings = _validate_skill(skill)
    assert any("Retrieval Notes" in w for w in warnings)
    assert any("Output Format" in w for w in warnings)


def test_validate_skill_warns_on_long_skill():
    long_body = "word " * 301
    skill = (
        f"## Task Type\n{long_body}\n\n"
        "## Retrieval Notes\n- notes.\n\n"
        "## Answering Strategy\n- steps.\n\n"
        "## Output Format\n- format.\n"
    )
    warnings = _validate_skill(skill)
    assert any("words" in w for w in warnings)
