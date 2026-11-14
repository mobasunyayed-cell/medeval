"""Markdown reports."""

from __future__ import annotations

from medeval.compare import Comparison
from medeval.metrics import IntervalMethod, Summary, breakdown, summarize
from medeval.runner import Run


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _row(label: str, s: Summary) -> str:
    interval = f"{_pct(s.ci_low)} to {_pct(s.ci_high)}"
    return f"| {label} | {s.n} | {s.correct} | {_pct(s.accuracy)} | {interval} |"


def _table(title: str, groups: dict[str, Summary], method: str) -> list[str]:
    lines = [
        f"## {title}",
        "",
        f"| Group | N | Correct | Accuracy | 95% CI ({method}) |",
        "|---|---|---|---|---|",
    ]
    lines += [_row(name, s) for name, s in groups.items()]
    return [*lines, ""]


def render_markdown(run: Run, interval: IntervalMethod = "wilson") -> str:
    overall = summarize(run.results, interval)
    errors = [r for r in run.results if r.error]
    misses = [r for r in run.results if not r.correct]

    lines = [
        f"# Evaluation report: {run.model}",
        "",
        f"- Dataset: `{run.dataset}`",
        f"- Created: {run.created_at}",
        f"- Questions: {overall.n}, unanswered: {overall.unanswered}, call errors: {len(errors)}",
        "",
        *_table("Overall", {"all": overall}, interval),
        *_table("By specialty", breakdown(run.results, "specialty", interval), interval),
        *_table("By difficulty", breakdown(run.results, "difficulty", interval), interval),
    ]
    if misses:
        lines += ["## Missed questions", ""]
        for r in misses:
            lines.append(f"- `{r.question_id}`: gold {r.gold}, predicted {r.predicted or 'none'}")
        lines.append("")
    return "\n".join(lines)


def render_comparison(a: Run, b: Run, result: Comparison) -> str:
    verdict = "significant at 0.05" if result.p_value < 0.05 else "not significant at 0.05"
    return "\n".join(
        [
            f"# Comparison: {a.model} vs {b.model}",
            "",
            f"- Shared questions: {result.shared}",
            f"- Accuracy A ({a.model}): {_pct(result.accuracy_a)}",
            f"- Accuracy B ({b.model}): {_pct(result.accuracy_b)}",
            f"- Both correct: {result.both_correct}, both wrong: {result.both_wrong}",
            f"- Only A correct: {result.a_only}, only B correct: {result.b_only}",
            f"- Exact McNemar p-value: {result.p_value:.4f} ({verdict})",
            "",
        ]
    )
