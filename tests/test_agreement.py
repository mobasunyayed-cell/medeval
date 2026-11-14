import math

import pytest

from medeval.agreement import cohens_kappa, interpret_kappa, run_agreement
from medeval.compare import CompareError
from medeval.runner import Result, Run


def test_perfect_chance_and_worse_than_chance():
    assert cohens_kappa(list("ABAB"), list("ABAB")) == pytest.approx(1.0)
    assert cohens_kappa(list("AABB"), list("ABAB")) == pytest.approx(0.0)
    assert cohens_kappa(list("AABB"), list("BBAA")) == pytest.approx(-1.0)


def test_textbook_example():
    # 20 yes/yes, 5 yes/no, 10 no/yes, 15 no/no  ->  kappa = 0.4
    a = ["y"] * 25 + ["n"] * 25
    b = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
    assert cohens_kappa(a, b) == pytest.approx(0.4)


def test_weighted_kappa_penalises_far_misses_more():
    cats = ["1", "2", "3", "4", "5"]
    a = ["1", "2", "3", "4", "5"]
    near = ["2", "3", "4", "5", "5"]
    far = ["5", "4", "3", "2", "1"]
    near_k = cohens_kappa(a, near, cats, weights="quadratic")
    far_k = cohens_kappa(a, far, cats, weights="quadratic")
    assert near_k > far_k
    assert cohens_kappa(a, near, cats, weights="linear") < near_k  # quadratic is more lenient here


def test_undefined_when_everyone_uses_one_category():
    assert math.isnan(cohens_kappa(["A", "A"], ["A", "A"]))


def test_input_validation():
    with pytest.raises(ValueError, match="same number"):
        cohens_kappa(["A"], ["A", "B"])
    with pytest.raises(ValueError, match="at least one"):
        cohens_kappa([], [])
    with pytest.raises(ValueError, match="not in categories"):
        cohens_kappa(["A"], ["C"], categories=["A", "B"])
    with pytest.raises(ValueError, match="unique"):
        cohens_kappa(["A"], ["A"], categories=["A", "A"])


@pytest.mark.parametrize(
    "kappa, label",
    [
        (-0.1, "poor"),
        (0.1, "slight"),
        (0.3, "fair"),
        (0.5, "moderate"),
        (0.7, "substantial"),
        (0.9, "almost perfect"),
        (math.nan, "undefined"),
    ],
)
def test_interpretation(kappa, label):
    assert interpret_kappa(kappa) == label


def make_run(predictions):
    results = [
        Result(f"q{i}", "s", "d", "A", p, p == "A", "", None) for i, p in enumerate(predictions)
    ]
    return Run("m", "d", "t", results)


def test_run_agreement():
    result = run_agreement(make_run(["A", "B", None, "C"]), make_run(["A", "B", None, "D"]))
    assert result.shared == 4
    assert result.percent_agreement == pytest.approx(0.75)
    assert 0.5 < result.kappa < 1.0


def test_run_agreement_needs_overlap():
    other = Run("m", "d", "t", [Result("zzz", "s", "d", "A", "A", True, "", None)])
    with pytest.raises(CompareError):
        run_agreement(make_run(["A"]), other)
