import pytest

from jitskilled.grader import grade, grade_with_judge
from jitskilled.llm import MockClient

PASS_CASES = [
    ("$3.6 million", "$3.6 million"),
    ("214 employees", "214"),
    ("12%", "12%"),
    ("5", "5 days"),
    ("The answer is 21 days.", "21 days"),
]

FAIL_CASES = [
    ("$24,500", "12%"),
    ("not found", "18 kilograms"),
    ("", "5 days"),
]


@pytest.mark.parametrize("answer,ground_truth", PASS_CASES)
def test_grade_passes(answer, ground_truth):
    assert grade(answer, ground_truth) is True


@pytest.mark.parametrize("answer,ground_truth", FAIL_CASES)
def test_grade_fails(answer, ground_truth):
    assert grade(answer, ground_truth) is False


def test_short_numeric_substring_does_not_false_positive():
    # Regression test: a naive substring check would pass "3" against "30%"
    # because "3" is a substring of "30". Locks in the fix in grader.py.
    assert grade("3", "30%") is False


def test_empty_ground_truth_never_passes():
    assert grade("anything", "") is False


# --- additional edge-case coverage ---

def test_yes_does_not_match_years():
    """'yes' should not match inside 'years' (substring false positive)."""
    assert grade("yes", "years") is False


def test_years_does_not_match_yes():
    """Reverse direction: 'years' should not match inside 'yes'."""
    assert grade("years", "yes") is False


def test_cat_matches_in_concat():
    """'cat' as a whole word should match inside 'concatenate'."""
    # 'cat' has word boundaries around it in 'concatenate'? No -- 'cat' is
    # embedded in 'concatenate' without word boundaries, so this should NOT match.
    assert grade("cat", "concatenate") is False


def test_cat_matches_in_the_cat():
    """'cat' with word boundaries should match 'the cat sat'."""
    assert grade("cat", "the cat sat") is True


def test_exact_match_after_normalization():
    assert grade("  $18.4 million  ", "$18.4 million") is True


def test_case_insensitive():
    assert grade("Yes", "yes") is True


def test_comma_stripped_from_numbers():
    assert grade("$24,500", "$24500") is True


def test_numeric_token_match_across_context():
    assert grade("The revenue was $3.6 million total.", "$3.6 million") is True


def test_no_match_similar_but_different_number():
    assert grade("$3.7 million", "$3.6 million") is False


# --- grade_with_judge: two-tier escalation ---

class _PoisonedJudge:
    """A fake LLM whose .judge() always returns the wrong verdict. Used to
    prove numeric/boundary-match cases never reach the judge at all -- if
    they did, these tests would fail because the poisoned verdict would
    flip the result.
    """

    def judge(self, question, answer, ground_truth):
        return {"correct": "SHOULD NOT BE CALLED", "reason": "poisoned"}


def test_grade_with_judge_numeric_pass_never_calls_judge():
    passed, reason = grade_with_judge(
        _PoisonedJudge(), "q", "$3.6 million", "$3.6 million"
    )
    assert passed is True
    assert "numeric" in reason.lower() or "match" in reason.lower()


def test_grade_with_judge_numeric_fail_never_calls_judge():
    passed, reason = grade_with_judge(_PoisonedJudge(), "q", "$3.7 million", "$3.6 million")
    assert passed is False
    assert reason == "no shared numeric/currency/percentage token"


def test_grade_with_judge_empty_ground_truth():
    passed, reason = grade_with_judge(_PoisonedJudge(), "q", "anything", "")
    assert passed is False
    assert reason == "empty ground truth"


def test_grade_with_judge_exact_match_never_calls_judge():
    passed, reason = grade_with_judge(_PoisonedJudge(), "q", "the cat sat", "the cat sat")
    assert passed is True
    assert reason == "exact match after normalization"


def test_grade_with_judge_boundary_match_never_calls_judge():
    passed, reason = grade_with_judge(_PoisonedJudge(), "q", "cat", "the cat sat")
    assert passed is True
    assert reason == "word-boundary substring match"


def test_grade_with_judge_no_boundary_no_llm_falls_back_false():
    passed, reason = grade_with_judge(None, "q", "totally different phrasing", "some reference")
    assert passed is False
    assert reason == "no boundary match; no LLM judge available to escalate to"


def test_grade_with_judge_escalates_to_mock_judge_on_ambiguous_free_text():
    # High word overlap with the reference, but not an exact/boundary match,
    # so this only passes if the judge was actually invoked.
    llm = MockClient()
    passed, reason = grade_with_judge(
        llm, "q", "revenue grew steadily", "revenue growth was steady"
    )
    assert isinstance(passed, bool)
    assert reason.startswith("LLM judge: ")


def test_grade_with_judge_mock_judge_low_overlap_fails():
    llm = MockClient()
    passed, reason = grade_with_judge(
        llm, "q", "completely unrelated text here", "some reference answer"
    )
    assert passed is False
    assert reason.startswith("LLM judge: ")


# --- MockClient.judge() direct coverage ---

def test_mock_judge_high_overlap_is_correct():
    verdict = MockClient().judge("q", "the total revenue was steady", "revenue was steady")
    assert verdict["correct"] is True
    assert "word overlap" in verdict["reason"]


def test_mock_judge_low_overlap_is_incorrect():
    verdict = MockClient().judge("q", "completely different words entirely", "revenue was steady")
    assert verdict["correct"] is False
    assert "word overlap" in verdict["reason"]


def test_mock_judge_reason_reports_percentage():
    verdict = MockClient().judge("q", "revenue was steady", "revenue was steady")
    assert "100%" in verdict["reason"]
