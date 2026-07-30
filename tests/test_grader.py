import pytest

from jitskilled.grader import grade

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
