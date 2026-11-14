"""Command-line interface: ``medeval validate | run | report | compare | rubric``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from medeval import __version__
from medeval.agreement import run_agreement
from medeval.compare import CompareError, paired_compare
from medeval.dataset import DatasetError, load_questions
from medeval.metrics import summarize
from medeval.models import ModelError, make_model
from medeval.report import render_comparison, render_markdown
from medeval.rubric import RubricError, load_rubric, score_response
from medeval.runner import RunFormatError, load_run, make_run, run_eval, save_run

_USER_ERRORS = (DatasetError, ModelError, RunFormatError, CompareError, RubricError)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="medeval", description="Evaluate language-model answers on medical questions."
    )
    parser.add_argument("--version", action="version", version=f"medeval {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("validate", help="check a dataset file")
    p.add_argument("data")

    p = sub.add_parser("run", help="evaluate a model on a dataset")
    p.add_argument("--data", required=True)
    p.add_argument(
        "--model", required=True, help="random | simulated:0.8 | replay:FILE | anthropic:ID"
    )
    p.add_argument("--out", required=True, help="where to save the run (JSON)")
    p.add_argument("--limit", type=int, help="only use the first N questions")
    p.add_argument("--workers", type=int, default=1, help="questions to ask at the same time")
    p.add_argument("--retries", type=int, default=0, help="extra attempts after a failed call")
    p.add_argument("--backoff", type=float, default=1.0, help="first retry delay in seconds")
    p.add_argument("--checkpoint", help="append answers here and resume from it if it exists")

    p = sub.add_parser("report", help="print a Markdown report for a saved run")
    p.add_argument("run")
    p.add_argument("--out", help="write the report to a file instead of printing it")
    p.add_argument(
        "--ci", choices=["wilson", "bootstrap"], default="wilson", help="interval method"
    )

    p = sub.add_parser("compare", help="paired comparison of two saved runs")
    p.add_argument("run_a")
    p.add_argument("run_b")

    p = sub.add_parser("agreement", help="Cohen's kappa between the answers of two runs")
    p.add_argument("run_a")
    p.add_argument("run_b")

    p = sub.add_parser("rubric", help="score a free-text response against a rubric")
    p.add_argument("--rubric", required=True)
    p.add_argument("--response", required=True, help="text file containing the response")
    return parser


def _cmd_validate(args: argparse.Namespace) -> int:
    questions = load_questions(args.data)
    specialties = sorted({q.specialty for q in questions})
    print(f"OK: {len(questions)} questions, specialties: {', '.join(specialties)}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    questions = load_questions(args.data)
    if args.limit is not None:
        if args.limit < 1:
            raise DatasetError("--limit must be at least 1")
        questions = questions[: args.limit]
    model = make_model(args.model)
    if args.checkpoint and Path(args.checkpoint).exists():
        print(f"resuming from checkpoint {args.checkpoint}", file=sys.stderr)
    try:
        results = run_eval(
            model,
            questions,
            workers=args.workers,
            retries=args.retries,
            backoff=args.backoff,
            checkpoint=args.checkpoint,
        )
    except ValueError as exc:
        raise DatasetError(str(exc)) from exc
    run = make_run(model.name, Path(args.data).name, results)
    save_run(args.out, run)
    s = summarize(run.results)
    print(
        f"{model.name}: {s.correct}/{s.n} correct ({s.accuracy:.1%}, "
        f"95% CI {s.ci_low:.1%} to {s.ci_high:.1%}), {s.unanswered} unanswered. Saved to {args.out}"
    )
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    text = render_markdown(load_run(args.run), args.ci)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"Report written to {args.out}")
    else:
        print(text)
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    a, b = load_run(args.run_a), load_run(args.run_b)
    print(render_comparison(a, b, paired_compare(a, b)))
    return 0


def _cmd_agreement(args: argparse.Namespace) -> int:
    result = run_agreement(load_run(args.run_a), load_run(args.run_b))
    print(f"shared questions: {result.shared}")
    print(f"percent agreement: {result.percent_agreement:.1%}")
    print(f"Cohen's kappa: {result.kappa:.3f} ({result.interpretation})")
    return 0


def _cmd_rubric(args: argparse.Namespace) -> int:
    rubric = load_rubric(args.rubric)
    response = Path(args.response)
    if not response.is_file():
        raise RubricError(f"response file not found: {response}")
    result = score_response(response.read_text(encoding="utf-8"), rubric)
    print(f"score: {result.score:.2f} ({result.earned:g}/{result.possible:g})")
    print(f"matched: {', '.join(result.matched) or '-'}")
    print(f"missed: {', '.join(result.missed) or '-'}")
    print(f"violations: {', '.join(result.violations) or '-'}")
    return 0


_COMMANDS = {
    "validate": _cmd_validate,
    "run": _cmd_run,
    "report": _cmd_report,
    "compare": _cmd_compare,
    "agreement": _cmd_agreement,
    "rubric": _cmd_rubric,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _COMMANDS[args.command](args)
    except _USER_ERRORS as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
