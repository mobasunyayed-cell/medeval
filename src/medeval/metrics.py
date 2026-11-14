"""Accuracy with confidence intervals, overall and per group."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

from medeval.runner import Result


@dataclass(frozen=True)
class Summary:
    n: int
    correct: int
    unanswered: int
    accuracy: float
    ci_low: float
    ci_high: float


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion (95% by default).

    Unlike the normal approximation it stays inside [0, 1] and behaves for small n,
    which matters because medical test sets are often small.
    """
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def summarize(results: list[Result]) -> Summary:
    n = len(results)
    correct = sum(r.correct for r in results)
    unanswered = sum(r.predicted is None for r in results)
    low, high = wilson_interval(correct, n)
    return Summary(n, correct, unanswered, correct / n if n else 0.0, low, high)


def breakdown(results: list[Result], by: str) -> dict[str, Summary]:
    """Summaries grouped by 'specialty' or 'difficulty', sorted by group name."""
    if by not in ("specialty", "difficulty"):
        raise ValueError("by must be 'specialty' or 'difficulty'")
    groups: dict[str, list[Result]] = defaultdict(list)
    for result in results:
        groups[getattr(result, by)].append(result)
    return {name: summarize(groups[name]) for name in sorted(groups)}
