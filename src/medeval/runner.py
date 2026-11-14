"""Run a model over a dataset and store the results.

Long evaluations fail halfway: rate limits, network drops, a closed laptop. The runner
therefore retries transient errors with exponential backoff, can ask questions concurrently,
and can append every finished answer to a checkpoint file so a rerun resumes where it stopped.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from medeval.dataset import Question
from medeval.extract import extract_choice
from medeval.models import ModelClient, PermanentError

SCHEMA_VERSION = 1
MAX_BACKOFF_SECONDS = 30.0


class RunFormatError(ValueError):
    """Raised when a saved run or checkpoint file is not in the expected format."""


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
    attempts: int = 1


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


def _ask(
    model: ModelClient,
    question: Question,
    retries: int,
    backoff: float,
    sleep: Callable[[float], None],
) -> Result:
    """Ask one question, retrying transient failures. Never raises."""
    prompt = build_prompt(question)
    response, error, attempts = "", None, 0
    for attempt in range(retries + 1):
        attempts = attempt + 1
        try:
            response, error = model.complete(question, prompt), None
            break
        except PermanentError as exc:
            error = f"{type(exc).__name__}: {exc}"
            break
        except Exception as exc:  # noqa: BLE001 - any provider failure is just a failed attempt
            error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                sleep(min(backoff * 2**attempt, MAX_BACKOFF_SECONDS))
    predicted = extract_choice(response, "".join(question.options))
    return Result(
        question_id=question.id,
        specialty=question.specialty,
        difficulty=question.difficulty,
        gold=question.answer,
        predicted=predicted,
        correct=predicted == question.answer,
        response=response,
        error=error,
        attempts=attempts,
    )


def load_checkpoint(path: str | Path) -> list[Result]:
    """Read a checkpoint (one JSON result per line). A torn final line is ignored.

    If a question appears more than once, the latest entry wins, so a retried failure
    replaces the earlier error.
    """
    path = Path(path)
    if not path.is_file():
        return []
    lines = [line for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]
    latest: dict[str, Result] = {}
    for number, line in enumerate(lines, start=1):
        try:
            result = Result(**json.loads(line))
        except (json.JSONDecodeError, TypeError) as exc:
            if number == len(lines):  # the process died while writing this line
                continue
            raise RunFormatError(f"{path}, line {number}: corrupt checkpoint entry") from exc
        latest[result.question_id] = result
    return list(latest.values())


def run_eval(
    model: ModelClient,
    questions: list[Question],
    *,
    workers: int = 1,
    retries: int = 0,
    backoff: float = 1.0,
    checkpoint: str | Path | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> list[Result]:
    """Ask every question and return results in dataset order.

    A failing call is recorded on its result instead of raised. With ``checkpoint`` set,
    finished answers are appended to that file as they arrive, and answers already there
    (without an error) are reused instead of asked again. The model client must be safe to
    call from several threads when ``workers`` is above 1.
    """
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if retries < 0:
        raise ValueError("retries must not be negative")
    if backoff < 0:
        raise ValueError("backoff must not be negative")

    wanted = {q.id for q in questions}
    done: dict[str, Result] = {}
    if checkpoint is not None:
        done = {
            r.question_id: r
            for r in load_checkpoint(checkpoint)
            if r.question_id in wanted and r.error is None
        }
    pending = [q for q in questions if q.id not in done]

    sink = None
    if checkpoint is not None:
        Path(checkpoint).parent.mkdir(parents=True, exist_ok=True)
        sink = open(checkpoint, "a", encoding="utf-8")  # noqa: SIM115 - closed in finally
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        futures = [pool.submit(_ask, model, q, retries, backoff, sleep) for q in pending]
        for future in as_completed(futures):
            result = future.result()
            done[result.question_id] = result
            if sink is not None:
                sink.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
                sink.flush()
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        if sink is not None:
            sink.close()
    return [done[q.id] for q in questions]


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
