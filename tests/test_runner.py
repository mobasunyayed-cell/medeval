import json

import pytest

from medeval.dataset import Question
from medeval.models import ModelError, SimulatedModel
from medeval.runner import (
    RunFormatError,
    build_prompt,
    load_run,
    make_run,
    run_eval,
    save_run,
)

QUESTIONS = [
    Question(id=f"q{i}", question="?", options={"A": "x", "B": "y"}, answer="A", specialty="s")
    for i in range(10)
]


class Broken:
    name = "broken"

    def complete(self, question, prompt):
        raise ModelError("api down")


class Rambler:
    name = "rambler"

    def complete(self, question, prompt):
        return "I really cannot decide."


def test_prompt_lists_options_in_order():
    prompt = build_prompt(QUESTIONS[0])
    assert "Question: ?" in prompt
    assert prompt.index("A. x") < prompt.index("B. y")
    assert "Answer: X" in prompt


def test_perfect_and_zero_models():
    assert all(r.correct for r in run_eval(SimulatedModel(accuracy=1.0), QUESTIONS))
    assert not any(r.correct for r in run_eval(SimulatedModel(accuracy=0.0), QUESTIONS))


def test_errors_are_recorded_not_raised():
    results = run_eval(Broken(), QUESTIONS)
    assert len(results) == 10
    assert all(r.error and "api down" in r.error for r in results)
    assert all(r.predicted is None and not r.correct for r in results)


def test_unparseable_reply_counts_as_unanswered():
    results = run_eval(Rambler(), QUESTIONS)
    assert all(r.predicted is None and r.error is None for r in results)


def test_save_and_load_round_trip(tmp_path):
    run = make_run("m", "d.jsonl", run_eval(SimulatedModel(accuracy=1.0), QUESTIONS))
    path = tmp_path / "nested" / "run.json"
    save_run(path, run)
    assert load_run(path) == run


def test_load_run_rejects_bad_files(tmp_path):
    with pytest.raises(RunFormatError, match="not found"):
        load_run(tmp_path / "nope.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")
    with pytest.raises(RunFormatError, match="invalid JSON"):
        load_run(bad)
    wrong_version = tmp_path / "v.json"
    wrong_version.write_text(json.dumps({"schema_version": 99}), encoding="utf-8")
    with pytest.raises(RunFormatError, match="unsupported"):
        load_run(wrong_version)
    missing = tmp_path / "m.json"
    missing.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
    with pytest.raises(RunFormatError, match="malformed"):
        load_run(missing)
