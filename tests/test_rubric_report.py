import json

import pytest

from medeval.compare import paired_compare
from medeval.report import render_comparison, render_markdown
from medeval.rubric import RubricError, load_rubric, parse_rubric, score_response
from medeval.runner import Result, Run

RUBRIC = parse_rubric(
    {
        "id": "r",
        "criteria": [
            {"name": "epi", "any_of": ["epinephrine", "adrenaline"], "weight": 3},
            {"name": "help", "any_of": ["call for help"]},
        ],
        "penalties": [{"name": "wrong", "any_of": ["antihistamine first"], "amount": 2}],
    }
)


def test_full_and_partial_scores():
    full = score_response("Give EPINEPHRINE and call for help.", RUBRIC)
    assert full.score == 1.0 and full.missed == ()
    partial = score_response("Give adrenaline.", RUBRIC)
    assert partial.score == pytest.approx(0.75)
    assert partial.missed == ("help",)


def test_penalty_applies_and_score_is_floored():
    result = score_response("Antihistamine first, nothing else.", RUBRIC)
    assert result.violations == ("wrong",)
    assert result.score == 0.0


def test_matching_respects_word_boundaries():
    assert score_response("The epinephrines are stored.", RUBRIC).matched == ()


@pytest.mark.parametrize(
    "data",
    [
        {"id": "x"},
        {"id": "x", "criteria": []},
        {"id": "x", "criteria": [{"name": "a", "any_of": []}]},
        {"id": "x", "criteria": [{"name": "a", "any_of": ["b"], "weight": 0}]},
        {"id": "x", "criteria": [{"name": "a", "any_of": ["b"], "weight": "heavy"}]},
        "not a dict",
    ],
)
def test_parse_rubric_rejects_bad_input(data):
    with pytest.raises(RubricError):
        parse_rubric(data)


def test_load_rubric_errors(tmp_path):
    with pytest.raises(RubricError, match="not found"):
        load_rubric(tmp_path / "x.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")
    with pytest.raises(RubricError, match="invalid JSON"):
        load_rubric(bad)
    good = tmp_path / "ok.json"
    good.write_text(json.dumps({"id": "x", "criteria": [{"name": "a", "any_of": ["b"]}]}))
    assert load_rubric(good).id == "x"


def make_run(name, flags):
    results = [
        Result(f"q{i}", "anatomy", "easy", "A", "A" if ok else "B", ok, "", None)
        for i, ok in enumerate(flags)
    ]
    return Run(name, "data.jsonl", "2026-01-01T00:00:00+00:00", results)


def test_markdown_report_contents():
    text = render_markdown(make_run("m", [True, False, True]))
    assert "# Evaluation report: m" in text
    assert "| anatomy | 3 | 2 | 66.7% |" in text
    assert "## Missed questions" in text and "`q1`" in text


def test_report_without_misses_has_no_missed_section():
    assert "Missed questions" not in render_markdown(make_run("m", [True, True]))


def test_comparison_report():
    a, b = make_run("A", [True] * 12), make_run("B", [False] * 12)
    text = render_comparison(a, b, paired_compare(a, b))
    assert "Only A correct: 12" in text
    assert "significant at 0.05" in text and "not significant" not in text
