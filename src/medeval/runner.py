"""Run a model over a dataset and store the results."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from medeval.dataset import Question
from medeval.extract import extract_choice
from medeval.models import ModelClient

SCHEMA_VERSION = 1


class RunFormatError(ValueError):
    """Raised when a saved run file is not in the expected format."""


@dataclass
class Result:
    question_id: str
    specialty: str
    difficulty: str
    gold: str
    predicted: str | None
    correct: bool
    response: str
    error: str | None = None


@dataclass
class Run:
    model: str
    dataset: str
    created_at: str
    results: list[Result] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "model": self.model,
            "dataset": self.dataset,
            "created_at": self.created_at,
            "results": [asdict(r) for r in self.results],
        }

    @classmethod
    def from_dict(cls, data: Any) -> Run:
        try:
            if data["schema_version"] != SCHEMA_VERSION:
                raise RunFormatError(f"unsupported schema_version {data['schema_version']!r}")
            return cls(
                model=data["model"],
                dataset=data["dataset"],
                created_at=data["created_at"],
                results=[Result(**item) for item in data["results"]],
            )
        except (KeyError, TypeError) as exc:
            raise RunFormatError(f"malformed run file: {exc!r}") from exc


def build_prompt(question: Question) -> str:
    """The exact text sent to the model."""
    options = "\n".join(f"{key}. {value}" for key, value in sorted(question.options.items()))
    return (
        "Answer the following medical multiple-choice question. "
        'Finish with a final line of the form "Answer: X" where X is one option letter.\n\n'
        f"Question: {question.question}\n{options}"
    )


def run_eval(model: ModelClient, questions: list[Question]) -> list[Result]:
    """Ask every question. A failing call is recorded as an error, not raised."""
    results: list[Result] = []
    for question in questions:
        response, error = "", None
        try:
            response = model.complete(question, build_prompt(question))
        except Exception as exc:  # noqa: BLE001 - one bad call must not stop the whole run
            error = f"{type(exc).__name__}: {exc}"
        predicted = extract_choice(response, "".join(question.options))
        results.append(
            Result(
                question_id=question.id,
                specialty=question.specialty,
                difficulty=question.difficulty,
                gold=question.answer,
                predicted=predicted,
                correct=predicted == question.answer,
                response=response,
                error=error,
            )
        )
    return results


def make_run(model_name: str, dataset_name: str, results: list[Result]) -> Run:
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return Run(model=model_name, dataset=dataset_name, created_at=stamp, results=results)


def save_run(path: str | Path, run: Run) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(run.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


def load_run(path: str | Path) -> Run:
    path = Path(path)
    if not path.is_file():
        raise RunFormatError(f"run file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RunFormatError(f"{path}: invalid JSON ({exc.msg})") from exc
    return Run.from_dict(data)
