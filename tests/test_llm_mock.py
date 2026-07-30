import pytest

from jitskilled._util import parse_json_response
from jitskilled.llm import MockClient, get_client

TARGET = {"task_id": "t0", "question": "What was Q3 revenue?", "source_doc": "q3"}
RETRIEVED = [
    {"task_id": "e0", "question": "What was Q3 gross margin?",
     "source_doc": "q3", "ground_truth": "61%"},
]


def test_get_client_mock_backend_explicit():
    assert isinstance(get_client("mock"), MockClient)


def test_get_client_unknown_backend_raises():
    with pytest.raises(ValueError, match="Unknown LLM backend"):
        get_client("not_a_real_backend")


def test_get_client_defaults_to_mock_without_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("JITSKILLED_LLM_BACKEND", raising=False)
    assert isinstance(get_client(), MockClient)


def test_mock_synthesize_skill_has_required_sections():
    skill = MockClient().synthesize_skill("framework", "slots", TARGET, RETRIEVED)
    for heading in ("## Task Type", "## Retrieval Notes",
                    "## Answering Strategy", "## Output Format"):
        assert heading in skill


def test_mock_solve_finds_value_near_keywords():
    doc = "Aurora Robotics reported Q3 2026 revenue of $18.4 million, up 22%."
    answer = MockClient().solve("What was Q3 revenue?", doc)
    assert "$18.4" in answer or "18.4" in answer


def test_mock_solve_no_match_returns_not_found():
    doc = "This document is entirely unrelated filler text with no numbers."
    answer = MockClient().solve("What was the launch date?", doc)
    assert answer == "not found"


def test_mock_critic_schema_exactly_one_attribution_set():
    for outcome in ("failed", "passed"):
        result = MockClient().critic({"task_id": "t0", "outcome": outcome})
        has_failure = result["failure_attribution"] is not None
        has_success = result["success_attribution"] is not None
        assert has_failure != has_success  # exactly one, not both/neither


def test_mock_editor_returns_operations_list():
    result = MockClient().editor({"case_critic_results": []})
    assert isinstance(result["operations"], list)
    assert len(result["operations"]) >= 1


def test_mock_editor_varies_by_candidate_index():
    llm = MockClient()
    results = [llm.editor({"case_critic_results": [], "candidate_index": i})
               for i in (1, 2, 3)]
    slot_ids = [r["operations"][0]["slot_id"] for r in results]
    assert len(set(slot_ids)) == 3, f"expected 3 distinct candidates, got {slot_ids}"


def test_mock_editor_defaults_to_candidate_1_when_unspecified():
    with_default = MockClient().editor({"case_critic_results": []})
    explicit_1 = MockClient().editor({"case_critic_results": [], "candidate_index": 1})
    assert with_default == explicit_1


def test_parse_json_response_extracts_object_from_prose():
    text = 'Sure, here is the JSON:\n{"a": 1, "b": [1, 2]}\nHope that helps!'
    assert parse_json_response(text) == {"a": 1, "b": [1, 2]}


def test_parse_json_response_raises_clear_error_on_garbage():
    with pytest.raises(RuntimeError, match="Could not parse JSON"):
        parse_json_response("not json at all")
