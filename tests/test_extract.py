import pytest

from medeval.extract import extract_choice


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Answer: B", "B"),
        ("The correct answer is (C).", "C"),
        ("**Answer: D**", "D"),
        ("Final answer - A", "A"),
        ("Thinking... maybe A.\nAnswer: C", "C"),  # the last statement wins
        ("B", "B"),
        ("(b)", "B"),
        ("C. Radial nerve, because of the spiral groove.", "C"),
        ("answer = B", "B"),
    ],
)
def test_extracts_letter(text, expected):
    assert extract_choice(text, "ABCD") == expected


@pytest.mark.parametrize(
    "text",
    [
        "",
        None,
        "I am not sure about this one.",
        "The answer is a nerve injury.",  # lowercase article, not option A
        "Answer: E",  # outside the valid letters
    ],
)
def test_returns_none_when_no_choice(text):
    assert extract_choice(text, "ABCD") is None


def test_valid_letters_are_respected():
    assert extract_choice("Answer: E", "ABCDE") == "E"
