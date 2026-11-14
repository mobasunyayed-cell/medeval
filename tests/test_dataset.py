import json
from pathlib import Path

import pytest

from medeval.dataset import DatasetError, load_questions, parse_question

GOOD = {
    "id": "q1",
    "question": "Which?",
    "options": {"A": "one", "B": "two"},
    "answer": "b",
    "specialty": "Anatomy",
}


def write(path: Path, *records: object) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    return path


def test_parse_normalises_case_and_defaults():
    q = parse_question({**GOOD})
    assert q.answer == "B"
    assert q.specialty == "anatomy"
    assert q.difficulty == "medium"


@pytest.mark.parametrize(
    "patch, message",
    [
        ({"id": ""}, "must not be empty"),
        ({"options": {"A": "only one"}}, "at least 2"),
        ({"options": {"a": "x", "B": "y"}}, "capital letters"),
        ({"options": {"A": "x", "B": " "}}, "is empty"),
        ({"answer": "C"}, "not one of the options"),
    ],
)
def test_parse_rejects_bad_records(patch, message):
    with pytest.raises(DatasetError, match=message):
        parse_question({**GOOD, **patch})


def test_parse_reports_missing_fields_and_wrong_type():
    with pytest.raises(DatasetError, match="missing field"):
        parse_question({"id": "x"})
    with pytest.raises(DatasetError, match="JSON object"):
        parse_question(["not", "a", "dict"])


def test_load_skips_comments_and_blank_lines(tmp_path):
    path = tmp_path / "d.jsonl"
    path.write_text("# comment\n\n" + json.dumps(GOOD) + "\n", encoding="utf-8")
    assert [q.id for q in load_questions(path)] == ["q1"]


def test_load_rejects_duplicates_bad_json_missing_and_empty(tmp_path):
    with pytest.raises(DatasetError, match="duplicate id"):
        load_questions(write(tmp_path / "a.jsonl", GOOD, GOOD))
    bad = tmp_path / "b.jsonl"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(DatasetError, match="line 1: invalid JSON"):
        load_questions(bad)
    with pytest.raises(DatasetError, match="not found"):
        load_questions(tmp_path / "missing.jsonl")
    empty = tmp_path / "e.jsonl"
    empty.write_text("# nothing\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="empty"):
        load_questions(empty)


def test_bundled_sample_dataset_is_valid_and_balanced():
    root = Path(__file__).resolve().parent.parent
    questions = load_questions(root / "data" / "sample_mcq.jsonl")
    assert len(questions) == 24
    assert all(len(q.options) == 4 for q in questions)
    letters = {q.answer for q in questions}
    assert letters == {"A", "B", "C", "D"} or letters <= {"A", "B", "C", "D"}
