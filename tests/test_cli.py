import runpy
from pathlib import Path

import pytest

from medeval.cli import main

ROOT = Path(__file__).resolve().parent.parent
DATA = str(ROOT / "data" / "sample_mcq.jsonl")


def test_validate(capsys):
    assert main(["validate", DATA]) == 0
    assert "OK: 24 questions" in capsys.readouterr().out


def test_run_report_compare_flow(tmp_path, capsys):
    a, b = str(tmp_path / "a.json"), str(tmp_path / "b.json")
    assert main(["run", "--data", DATA, "--model", "simulated:1.0", "--out", a]) == 0
    assert "24/24 correct" in capsys.readouterr().out
    assert main(["run", "--data", DATA, "--model", "random:3", "--out", b, "--limit", "12"]) == 0
    capsys.readouterr()

    assert main(["report", a]) == 0
    assert "# Evaluation report: simulated:1" in capsys.readouterr().out

    out = tmp_path / "report.md"
    assert main(["report", a, "--out", str(out)]) == 0
    assert out.read_text(encoding="utf-8").startswith("# Evaluation report")

    assert main(["compare", a, b]) == 0
    assert "Shared questions: 12" in capsys.readouterr().out


def test_replay_model_through_cli(tmp_path, capsys):
    replay = str(ROOT / "examples" / "replay_answers.jsonl")
    # the replay file only has 3 answers, the rest become recorded errors
    out = str(tmp_path / "r.json")
    assert main(["run", "--data", DATA, "--model", f"replay:{replay}", "--out", out]) == 0
    assert "3/24 correct" in capsys.readouterr().out


def test_rubric_command(capsys):
    args = [
        "rubric",
        "--rubric",
        str(ROOT / "examples" / "rubric_anaphylaxis.json"),
        "--response",
        str(ROOT / "examples" / "rubric_response_good.txt"),
    ]
    assert main(args) == 0
    assert "score: 1.00" in capsys.readouterr().out


@pytest.mark.parametrize(
    "argv",
    [
        ["validate", "missing.jsonl"],
        ["run", "--data", DATA, "--model", "bogus", "--out", "x.json"],
        ["run", "--data", DATA, "--model", "random", "--out", "x.json", "--limit", "0"],
        ["report", "missing.json"],
        ["compare", "a.json", "b.json"],
        ["rubric", "--rubric", "nope.json", "--response", "nope.txt"],
    ],
)
def test_user_errors_return_exit_code_2(argv, capsys):
    assert main(argv) == 2
    assert capsys.readouterr().err.startswith("error:")


def test_rubric_missing_response_file(capsys):
    rubric = str(ROOT / "examples" / "rubric_anaphylaxis.json")
    assert main(["rubric", "--rubric", rubric, "--response", "nope.txt"]) == 2
    assert "response file not found" in capsys.readouterr().err


def test_module_entry_point_requires_a_command(capsys):
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("medeval", run_name="__main__", alter_sys=True)
    assert exc.value.code == 2  # argparse: a subcommand is required
