"""Accuracy with confidence intervals, overall and per group."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from medeval.runner import Result

IntervalMethod = Literal["wilson", "bootstrap"]


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


def bootstrap_interval(
    outcomes: Sequence[bool],
    confidence: float = 0.95,
    resamples: int = 5000,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap interval for accuracy. Deterministic for a given seed.

    It makes no distributional assumption, but it collapses to a single point when every
    outcome is the same (all right or all wrong), so Wilson stays the default.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if resamples < 1:
        raise ValueError("resamples must be at least 1")
    n = len(outcomes)
    if n == 0:
        return 0.0, 1.0
    values = [int(o) for o in outcomes]
    rng = random.Random(seed)
    scores = sorted(sum(rng.choices(values, k=n)) / n for _ in range(resamples))
    tail = (1.0 - confidence) / 2.0
    low = scores[int(tail * resamples)]
    high = scores[min(resamples - 1, int((1.0 - tail) * resamples))]
    return low, high


def summarize(results: list[Result], method: IntervalMethod = "wilson") -> Summary:
    if method not in ("wilson", "bootstrap"):
        raise ValueError("method must be 'wilson' or 'bootstrap'")
    n = len(results)
    correct = sum(r.correct for r in results)
    unanswered = sum(r.predicted is None for r in results)
    if method == "wilson":
        low, high = wilson_interval(correct, n)
    else:
        low, high = bootstrap_interval([r.correct for r in results])
    return Summary(n, correct, unanswered, correct / n if n else 0.0, low, high)


def breakdown(
    results: list[Result], by: str, method: IntervalMethod = "wilson"
) -> dict[str, Summary]:
    """Summaries grouped by 'specialty' or 'difficulty', sorted by group name."""
    if by not in ("specialty", "difficulty"):
        raise ValueError("by must be 'specialty' or 'difficulty'")
    groups: dict[str, list[Result]] = defaultdict(list)
    for result in results:
        groups[getattr(result, by)].append(result)
    return {name: summarize(groups[name], method) for name in sorted(groups)}
