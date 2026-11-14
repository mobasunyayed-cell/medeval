import json
import threading

import pytest

from medeval.dataset import Question
from medeval.models import ModelError, PermanentError, SimulatedModel
from medeval.runner import RunFormatError, load_checkpoint, run_eval

QUESTIONS = [
    Question(id=f"q{i}", question="?", options={"A": "x", "B": "y"}, answer="A") for i in range(8)
]


class Flaky:
    """Fails the first ``failures`` calls for each question, then answers correctly."""

    name = "flaky"

    def __init__(self, failures: int, error: type[Exception] = ModelError) -> None:
        self.failures = failures
        self.error = error
        self.calls: dict[str, int] = {}

    def complete(self, question, prompt):
        count = self.calls[question.id] = self.calls.get(question.id, 0) + 1
        if count <= self.failures:
            raise self.error("temporary")
        return "Answer: A"


def test_retry_uses_exponential_backoff_then_succeeds():
    delays: list[float] = []
    model = Flaky(failures=2)
    [result] = run_eval(model, QUESTIONS[:1], retries=2, backoff=0.5, sleep=delays.append)
    assert result.correct and result.error is None
    assert result.attempts == 3
    assert delays == [0.5, 1.0]


def test_retries_exhausted_records_error_without_a_final_sleep():
    delays: list[float] = []
    [result] = run_eval(Flaky(failures=99), QUESTIONS[:1], retries=2, sleep=delays.append)
    assert result.error and "temporary" in result.error
    assert result.attempts == 3 and len(delays) == 2
    assert not result.correct and result.predicted is None


def test_permanent_errors_are_not_retried():
    delays: list[float] = []
    model = Flaky(failures=99, error=PermanentError)
    [result] = run_eval(model, QUESTIONS[:1], retries=5, sleep=delays.append)
    assert result.attempts == 1 and delays == []
    assert model.calls == {"q0": 1}


def test_backoff_is_capped():
    delays: list[float] = []
    run_eval(Flaky(failures=99), QUESTIONS[:1], retries=3, backoff=100.0, sleep=delays.append)
    assert delays == [30.0, 30.0, 30.0]


@pytest.mark.parametrize("kwargs", [{"workers": 0}, {"retries": -1}, {"backoff": -0.1}])
def test_rejects_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        run_eval(SimulatedModel(), QUESTIONS, **kwargs)


class Counting:
    name = "counting"

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.seen: list[str] = []

    def complete(self, question, prompt):
        with self.lock:
            self.seen.append(question.id)
        return "Answer: A"


def test_concurrent_run_keeps_dataset_order_and_asks_each_question_once():
    model = Counting()
    results = run_eval(model, QUESTIONS, workers=4)
    assert [r.question_id for r in results] == [q.id for q in QUESTIONS]
    assert sorted(model.seen) == sorted(q.id for q in QUESTIONS)


def test_checkpoint_resume_only_asks_failed_questions(tmp_path):
    path = tmp_path / "ck" / "run.jsonl"

    class FailsOnOdd:
        name = "odd"

        def complete(self, question, prompt):
            if int(question.id[1:]) % 2:
                raise PermanentError("nope")
            return "Answer: A"

    first = run_eval(FailsOnOdd(), QUESTIONS, checkpoint=path)
    assert sum(r.error is not None for r in first) == 4

    healthy = Counting()
    second = run_eval(healthy, QUESTIONS, checkpoint=path)
    assert sorted(healthy.seen) == ["q1", "q3", "q5", "q7"]  # successes were reused
    assert all(r.correct and r.error is None for r in second)

    # latest entry wins when the checkpoint is read back
    assert all(r.error is None for r in load_checkpoint(path))
    assert len(path.read_text(encoding="utf-8").strip().split("\n")) == 8 + 4


def test_checkpoint_ignores_unknown_ids(tmp_path):
    path = tmp_path / "ck.jsonl"
    run_eval(Counting(), QUESTIONS, checkpoint=path)
    subset = run_eval(Counting(), QUESTIONS[:2], checkpoint=path)
    assert [r.question_id for r in subset] == ["q0", "q1"]


def test_torn_last_line_is_ignored_but_corruption_elsewhere_is_not(tmp_path):
    path = tmp_path / "ck.jsonl"
    run_eval(Counting(), QUESTIONS[:3], checkpoint=path)
    good = path.read_text(encoding="utf-8")

    path.write_text(good + '{"question_id": "q9", "spec', encoding="utf-8")
    assert len(load_checkpoint(path)) == 3

    lines = good.strip().split("\n")
    path.write_text("\n".join([lines[0], "{broken", lines[2]]) + "\n", encoding="utf-8")
    with pytest.raises(RunFormatError, match="line 2"):
        load_checkpoint(path)


def test_checkpoint_survives_unicode_line_separators(tmp_path):
    class Odd:
        name = "odd"

        def complete(self, question, prompt):
            return "reasoning more\u0085text\nAnswer: A"

    path = tmp_path / "ck.jsonl"
    first = run_eval(Odd(), QUESTIONS[:2], checkpoint=path)
    again = run_eval(Counting(), QUESTIONS[:2], checkpoint=path)
    assert again == first
    assert json.loads(path.read_text(encoding="utf-8").split("\n")[0])["question_id"] == "q0"


def test_missing_checkpoint_is_empty(tmp_path):
    assert load_checkpoint(tmp_path / "none.jsonl") == []
