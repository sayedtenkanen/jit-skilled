"""Sanity checks on the shipped dataset itself -- not code, but data that
code depends on. Catches a hand-edited evolve.jsonl row that forgot to run
scripts/generate_evolve_attempts.py, or a test.jsonl/evolve.jsonl task_id
collision that would silently break the disjoint test/evolve split.
"""
import json

REQUIRED_EVOLVE_KEYS = {"task_id", "source_doc", "question", "ground_truth",
                         "prior_attempt", "prior_label"}
REQUIRED_TEST_KEYS = {"task_id", "source_doc", "question", "ground_truth"}


def _load_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_evolve_rows_have_recorded_attempt_and_label(project_root):
    rows = _load_jsonl(project_root / "data" / "evolve.jsonl")
    assert rows, "evolve.jsonl is empty"
    for row in rows:
        missing = REQUIRED_EVOLVE_KEYS - row.keys()
        assert not missing, f"{row.get('task_id')} missing keys: {missing}"
        assert row["prior_label"] in ("pass", "fail")


def test_evolve_has_both_passing_and_failing_attempts(project_root):
    # The whole point of Fix 1 is a mix of recorded successes and failures,
    # not just gold answers -- assert the pool actually has both.
    rows = _load_jsonl(project_root / "data" / "evolve.jsonl")
    labels = {row["prior_label"] for row in rows}
    assert labels == {"pass", "fail"}, (
        "evolve.jsonl should contain both pass and fail recorded attempts; "
        f"got only {labels}"
    )


def test_test_rows_have_required_keys(project_root):
    rows = _load_jsonl(project_root / "data" / "test.jsonl")
    assert rows, "test.jsonl is empty"
    for row in rows:
        missing = REQUIRED_TEST_KEYS - row.keys()
        assert not missing, f"{row.get('task_id')} missing keys: {missing}"


def test_evolve_and_test_task_ids_are_disjoint(project_root):
    evolve_ids = {r["task_id"] for r in _load_jsonl(project_root / "data" / "evolve.jsonl")}
    test_ids = {r["task_id"] for r in _load_jsonl(project_root / "data" / "test.jsonl")}
    assert not (evolve_ids & test_ids)


def test_all_source_docs_exist(project_root):
    docs_dir = project_root / "data" / "documents"
    available = {p.stem for p in docs_dir.glob("*.txt")}
    for filename in ("evolve.jsonl", "test.jsonl"):
        for row in _load_jsonl(project_root / "data" / filename):
            assert row["source_doc"] in available, (
                f"{filename}:{row['task_id']} references missing doc "
                f"{row['source_doc']!r}"
            )
