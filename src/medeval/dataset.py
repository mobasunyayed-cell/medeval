"""Load and validate multiple-choice question datasets stored as JSONL."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class DatasetError(ValueError):
    """Raised when a dataset file or one of its records is invalid."""


@dataclass(frozen=True)
class Question:
    """One multiple-choice question with a single correct answer."""

    id: str
    question: str
    options: dict[str, str]
    answer: str
    specialty: str = "general"
    difficulty: str = "medium"


_REQUIRED = ("id", "question", "options", "answer")


def parse_question(record: Any, line: int | None = None) -> Question:
    """Validate one decoded JSON record and turn it into a Question."""
    where = f"line {line}: " if line else ""
    if not isinstance(record, dict):
        raise DatasetError(f"{where}record must be a JSON object")
    missing = [key for key in _REQUIRED if key not in record]
    if missing:
        raise DatasetError(f"{where}missing field(s): {', '.join(missing)}")

    qid = str(record["id"]).strip()
    text = str(record["question"]).strip()
    if not qid or not text:
        raise DatasetError(f"{where}'id' and 'question' must not be empty")

    options = record["options"]
    if not isinstance(options, dict) or len(options) < 2:
        raise DatasetError(f"{where}'options' must be an object with at least 2 entries")
    for key, value in options.items():
        if not (isinstance(key, str) and len(key) == 1 and key.isalpha() and key.isupper()):
            raise DatasetError(f"{where}option keys must be single capital letters, got {key!r}")
        if not str(value).strip():
            raise DatasetError(f"{where}option {key} is empty")

    answer = str(record["answer"]).strip().upper()
    if answer not in options:
        raise DatasetError(f"{where}answer {answer!r} is not one of the options")

    return Question(
        id=qid,
        question=text,
        options={key: str(value).strip() for key, value in options.items()},
        answer=answer,
        specialty=str(record.get("specialty", "general")).strip().lower() or "general",
        difficulty=str(record.get("difficulty", "medium")).strip().lower() or "medium",
    )


def load_questions(path: str | Path) -> list[Question]:
    """Read a JSONL file. Blank lines and lines starting with '#' are ignored."""
    path = Path(path)
    if not path.is_file():
        raise DatasetError(f"dataset not found: {path}")

    questions: list[Question] = []
    seen: set[str] = set()
    for number, raw in enumerate(path.read_text(encoding="utf-8").split("\n"), start=1):
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        try:
            record = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"line {number}: invalid JSON ({exc.msg})") from exc
        question = parse_question(record, number)
        if question.id in seen:
            raise DatasetError(f"line {number}: duplicate id {question.id!r}")
        seen.add(question.id)
        questions.append(question)

    if not questions:
        raise DatasetError("dataset is empty")
    return questions
