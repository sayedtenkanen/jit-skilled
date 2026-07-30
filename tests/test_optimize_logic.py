"""Unit tests for optimize.py logic functions (bucket, sample, build_case)."""

from jitskilled.optimize import _bucket, _build_case, _sample_cases


def test_bucket_classifies_all_transitions():
    current = {
        "t0": {"label": "pass"},  # was pass -> persistent_success
        "t1": {"label": "fail"},  # was pass -> regression
        "t2": {"label": "pass"},  # was fail -> improvement
        "t3": {"label": "fail"},  # was fail -> persistent_failure
    }
    previous = {
        "t0": {"label": "pass"},
        "t1": {"label": "pass"},
        "t2": {"label": "fail"},
        "t3": {"label": "fail"},
    }
    buckets = _bucket(current, previous)
    assert buckets["t0"] == "persistent_success"
    assert buckets["t1"] == "regression"
    assert buckets["t2"] == "improvement"
    assert buckets["t3"] == "persistent_failure"


def test_bucket_skips_tasks_not_in_previous():
    current = {"t0": {"label": "pass"}}
    previous = {}
    buckets = _bucket(current, previous)
    assert buckets == {}


def test_sample_cases_returns_empty_for_zero_max():
    buckets = {"t0": "regression", "t1": "improvement"}
    assert _sample_cases(buckets, max_cases=0, seed=0) == []


def test_sample_cases_returns_empty_for_negative_max():
    buckets = {"t0": "regression"}
    assert _sample_cases(buckets, max_cases=-1, seed=0) == []


def test_sample_cases_returns_all_when_max_exceeds_bucket_count():
    buckets = {"t0": "regression", "t1": "improvement"}
    result = _sample_cases(buckets, max_cases=10, seed=0)
    assert set(result) == {"t0", "t1"}


def test_sample_cases_respects_max_cases():
    buckets = {
        "t0": "regression", "t1": "regression",
        "t2": "improvement", "t3": "persistent_success",
    }
    result = _sample_cases(buckets, max_cases=2, seed=42)
    assert len(result) == 2


def test_sample_cases_reproducible_with_same_seed():
    buckets = {"t0": "regression", "t1": "improvement", "t2": "persistent_success"}
    r1 = _sample_cases(buckets, max_cases=2, seed=7)
    r2 = _sample_cases(buckets, max_cases=2, seed=7)
    assert r1 == r2


def test_build_case_structure():
    current = {
        "t0": {
            "question": "What was revenue?",
            "answer": "$18.4 million",
            "ground_truth": "$18.4 million",
            "label": "pass",
            "retrieved_task_ids": ["e0"],
        }
    }
    previous = {
        "t0": {"answer": "$18 million", "label": "fail"},
    }
    pool_by_id = {
        "e0": {"task_id": "e0", "question": "What was Q3 revenue?",
               "source_doc": "q3", "ground_truth": "$18.4 million"},
    }
    current_slots = {"slots": {"input": [], "output": []}}

    case = _build_case("t0", "improvement", current, previous,
                       pool_by_id, current_slots)
    assert case["task_id"] == "t0"
    assert case["transition"] == "improvement"
    assert case["outcome"] == "passed"
    assert case["current_run"]["label"] == "pass"
    assert case["previous_run"]["label"] == "fail"
    assert len(case["skill_retrieval"]["retrieved_examples"]) == 1


def test_build_case_persistent_success_has_empty_previous():
    current = {
        "t0": {"question": "Q?", "answer": "A", "ground_truth": "A",
               "label": "pass", "retrieved_task_ids": []}
    }
    case = _build_case("t0", "persistent_success", current, {}, {},
                       {"slots": {"input": [], "output": []}})
    assert case["previous_run"] == {}
