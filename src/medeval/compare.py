"""Paired comparison of two runs on the same questions."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb

from medeval.runner import Run


class CompareError(ValueError):
    """Raised when two runs cannot be compared."""


@dataclass(frozen=True)
class Comparison:
    shared: int
    both_correct: int
    both_wrong: int
    a_only: int
    b_only: int
    accuracy_a: float
    accuracy_b: float
    p_value: float


def mcnemar_exact(a_only: int, b_only: int) -> float:
    """Two-sided exact McNemar test.

    Only questions where the models disagree carry information. Under the null hypothesis
    each disagreement is a fair coin flip, so the p-value is a binomial tail.
    """
    n = a_only + b_only
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(a_only, b_only) + 1)) / (1 << n)
    return min(1.0, 2 * tail)


def paired_compare(a: Run, b: Run) -> Comparison:
    a_by_id = {r.question_id: r for r in a.results}
    b_by_id = {r.question_id: r for r in b.results}
    shared = sorted(set(a_by_id) & set(b_by_id))
    if not shared:
        raise CompareError("the two runs share no question ids")

    both_correct = both_wrong = a_only = b_only = 0
    for qid in shared:
        a_ok, b_ok = a_by_id[qid].correct, b_by_id[qid].correct
        if a_ok and b_ok:
            both_correct += 1
        elif a_ok:
            a_only += 1
        elif b_ok:
            b_only += 1
        else:
            both_wrong += 1

    n = len(shared)
    return Comparison(
        shared=n,
        both_correct=both_correct,
        both_wrong=both_wrong,
        a_only=a_only,
        b_only=b_only,
        accuracy_a=(both_correct + a_only) / n,
        accuracy_b=(both_correct + b_only) / n,
        p_value=mcnemar_exact(a_only, b_only),
    )
