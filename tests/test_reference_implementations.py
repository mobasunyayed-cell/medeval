"""Check our small statistics code against well-tested libraries (dev dependencies only)."""

import random

import pytest

from medeval.agreement import cohens_kappa
from medeval.compare import mcnemar_exact
from medeval.metrics import wilson_interval

stats = pytest.importorskip("scipy.stats")
metrics = pytest.importorskip("sklearn.metrics")


@pytest.mark.parametrize("n", [1, 2, 10, 24, 100, 1000])
def test_wilson_matches_scipy(n):
    for k in sorted({0, 1, n // 3, n // 2, n - 1, n}):
        ours = wilson_interval(k, n)
        ref = stats.binomtest(k, n).proportion_ci(confidence_level=0.95, method="wilson")
        assert ours[0] == pytest.approx(ref.low, abs=2e-3), (k, n)
        assert ours[1] == pytest.approx(ref.high, abs=2e-3), (k, n)


@pytest.mark.parametrize("a, b", [(0, 0), (0, 5), (3, 3), (10, 2), (2, 10), (30, 18), (100, 60)])
def test_mcnemar_matches_binomial_test(a, b):
    n = a + b
    expected = 1.0 if n == 0 else stats.binomtest(min(a, b), n, 0.5).pvalue
    assert mcnemar_exact(a, b) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("weights", ["none", "linear", "quadratic"])
def test_kappa_matches_scikit_learn(weights):
    rng = random.Random(7)
    cats = ["1", "2", "3", "4"]
    for _ in range(20):
        a = [rng.choice(cats) for _ in range(60)]
        b = [x if rng.random() < 0.6 else rng.choice(cats) for x in a]
        reference = metrics.cohen_kappa_score(
            a, b, labels=cats, weights=None if weights == "none" else weights
        )
        assert cohens_kappa(a, b, cats, weights=weights) == pytest.approx(reference, abs=1e-9)
