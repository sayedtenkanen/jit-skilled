from jitskilled.retrieval import top_k_retrieve

POOL = [
    {"task_id": "e0", "question": "What was Q3 revenue?",
     "source_doc": "q3", "ground_truth": "$1"},
    {"task_id": "e1", "question": "What was Q3 gross margin?",
     "source_doc": "q3", "ground_truth": "1%"},
    {"task_id": "e2", "question": "How many PTO days per year?",
     "source_doc": "hr", "ground_truth": "21"},
]


def test_returns_at_most_k():
    result = top_k_retrieve("What was the net income?", POOL, k=2)
    assert len(result) <= 2


def test_excludes_self_task_id():
    result = top_k_retrieve("What was Q3 revenue?", POOL, k=10, exclude_task_id="e0")
    assert all(r["task_id"] != "e0" for r in result)


def test_topical_match_ranks_above_unrelated():
    result = top_k_retrieve("What was Q3 net income for the quarter?", POOL, k=3)
    ids_in_order = [r["task_id"] for r in result]
    # e0/e1 (revenue/margin, same doc + overlapping vocabulary) should rank
    # above e2 (unrelated PTO question) for a revenue-flavored query.
    assert ids_in_order.index("e2") == len(ids_in_order) - 1


def test_empty_pool_returns_empty_list():
    assert top_k_retrieve("anything", [], k=3) == []


def test_each_result_has_score():
    result = top_k_retrieve("What was Q3 revenue?", POOL, k=3)
    assert all("score" in r for r in result)
