"""Inter-rater agreement: do two raters, or two models, label things the same way?"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from medeval.compare import CompareError
from medeval.runner import Run

Weights = Literal["none", "linear", "quadratic"]


def cohens_kappa(
    a: Sequence[str],
    b: Sequence[str],
    categories: Sequence[str] | None = None,
    weights: Weights = "none",
) -> float:
    """Cohen's kappa: agreement beyond what chance alone would produce.

    1 is perfect agreement, 0 is chance level, negative is worse than chance.
    ``weights`` linear or quadratic treats ``categories`` as ordered (for example rubric
    scores 1 to 5) so near misses cost less than far misses. Returns NaN when kappa is
    undefined, which happens when both raters use only one category.
    """
    if len(a) != len(b):
        raise ValueError("both raters must label the same number of items")
    if not a:
        raise ValueError("need at least one item")
    labels = list(categories) if categories is not None else sorted(set(a) | set(b))
    index = {label: i for i, label in enumerate(labels)}
    if len(index) != len(labels):
        raise ValueError("categories must be unique")
    k = len(labels)

    observed = [[0] * k for _ in range(k)]
    try:
        for x, y in zip(a, b, strict=True):
            observed[index[x]][index[y]] += 1
    except KeyError as exc:
        raise ValueError(f"label {exc.args[0]!r} is not in categories") from None

    def disagreement(i: int, j: int) -> float:
        if weights == "none":
            return 0.0 if i == j else 1.0
        distance = abs(i - j) / (k - 1) if k > 1 else 0.0
        return distance if weights == "linear" else distance**2

    n = len(a)
    rows = [sum(row) for row in observed]
    cols = [sum(observed[i][j] for i in range(k)) for j in range(k)]
    observed_cost = sum(disagreement(i, j) * observed[i][j] for i in range(k) for j in range(k))
    expected_cost = sum(disagreement(i, j) * rows[i] * cols[j] for i in range(k) for j in range(k))
    if expected_cost == 0:
        return math.nan
    return 1.0 - (observed_cost / n) / (expected_cost / n**2)


def interpret_kappa(kappa: float) -> str:
    """Landis and Koch (1977) verbal labels. Conventional, not a law of nature."""
    if math.isnan(kappa):
        return "undefined"
    if kappa < 0:
        return "poor"
    for limit, label in (
        (0.20, "slight"),
        (0.40, "fair"),
        (0.60, "moderate"),
        (0.80, "substantial"),
    ):
        if kappa <= limit:
            return label
    return "almost perfect"


@dataclass(frozen=True)
class Agreement:
    shared: int
    percent_agreement: float
    kappa: float
    interpretation: str


def run_agreement(a: Run, b: Run) -> Agreement:
    """How often two runs pick the same option letter (no answer counts as its own label)."""
    a_by_id = {r.question_id: r.predicted or "none" for r in a.results}
    b_by_id = {r.question_id: r.predicted or "none" for r in b.results}
    shared = sorted(set(a_by_id) & set(b_by_id))
    if not shared:
        raise CompareError("the two runs share no question ids")
    left = [a_by_id[q] for q in shared]
    right = [b_by_id[q] for q in shared]
    kappa = cohens_kappa(left, right)
    same = sum(x == y for x, y in zip(left, right, strict=True))
    return Agreement(len(shared), same / len(shared), kappa, interpret_kappa(kappa))
