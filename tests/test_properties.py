"""Property-based tests: Hypothesis generates many inputs and hunts for counterexamples."""

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from medeval.agreement import cohens_kappa
from medeval.compare import mcnemar_exact
from medeval.extract import extract_choice
from medeval.metrics import bootstrap_interval, wilson_interval
from medeval.runner import Result, Run, load_run, save_run

LETTERS = "ABCDE"


@given(st.integers(0, 5000).flatmap(lambda n: st.tuples(st.integers(0, n), st.just(n))))
def test_wilson_interval_is_ordered_and_contains_the_estimate(pair):
    k, n = pair
    low, high = wilson_interval(k, n)
    assert 0.0 <= low <= high <= 1.0
    if n:
        assert low - 1e-12 <= k / n <= high + 1e-12


@given(st.integers(1, 2000), st.data())
def test_wilson_lower_bound_grows_with_successes(n, data):
    k = data.draw(st.integers(0, n - 1))
    assert wilson_interval(k, n)[0] <= wilson_interval(k + 1, n)[0] + 1e-12


@given(st.integers(0, 150), st.integers(0, 150))
def test_mcnemar_is_symmetric_and_a_probability(a, b):
    p = mcnemar_exact(a, b)
    assert 0.0 <= p <= 1.0
    assert p == mcnemar_exact(b, a)


@given(st.integers(0, 150))
def test_mcnemar_is_one_when_models_disagree_equally(a):
    assert mcnemar_exact(a, a) == 1.0


@given(st.lists(st.booleans(), max_size=60), st.integers(0, 10))
def test_bootstrap_interval_is_valid_and_deterministic(outcomes, seed):
    low, high = bootstrap_interval(outcomes, resamples=200, seed=seed)
    assert 0.0 <= low <= high <= 1.0
    assert (low, high) == bootstrap_interval(outcomes, resamples=200, seed=seed)


def test_bootstrap_validates_arguments():
    with pytest.raises(ValueError):
        bootstrap_interval([True], confidence=1.0)
    with pytest.raises(ValueError):
        bootstrap_interval([True], resamples=0)


def test_bootstrap_collapses_when_all_outcomes_match():
    assert bootstrap_interval([True] * 20) == (1.0, 1.0)


@given(st.text(), st.sampled_from(list(LETTERS)))
def test_extractor_always_finds_an_explicit_final_answer(prefix, letter):
    assert extract_choice(f"{prefix}\nAnswer: {letter}", LETTERS) == letter


@given(st.text(), st.text(alphabet=LETTERS, min_size=1, max_size=5))
def test_extractor_only_returns_valid_letters_or_none(text, valid):
    result = extract_choice(text, valid)
    assert result is None or result in valid


labels = st.lists(st.sampled_from(["w", "x", "y", "z"]), min_size=1, max_size=40)


@given(labels, labels)
def test_kappa_is_symmetric_and_bounded(a, b):
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    forward, backward = cohens_kappa(a, b), cohens_kappa(b, a)
    if math.isnan(forward):
        assert math.isnan(backward)
    else:
        assert forward == pytest.approx(backward)
        assert -1.0 - 1e-9 <= forward <= 1.0 + 1e-9


@given(labels)
def test_kappa_of_identical_labels_is_one_unless_undefined(a):
    kappa = cohens_kappa(a, a)
    assert math.isnan(kappa) if len(set(a)) == 1 else kappa == pytest.approx(1.0)


results = st.builds(
    Result,
    question_id=st.text(min_size=1),
    specialty=st.text(),
    difficulty=st.text(),
    gold=st.sampled_from(list("ABCD")),
    predicted=st.none() | st.sampled_from(list("ABCD")),
    correct=st.booleans(),
    response=st.text(),
    error=st.none() | st.text(),
    attempts=st.integers(1, 5),
)


@given(st.lists(results, max_size=8))
def test_saved_runs_load_back_unchanged(tmp_path_factory, items):
    path = tmp_path_factory.mktemp("runs") / "run.json"
    run = Run("model", "data", "2026-01-01T00:00:00+00:00", items)
    save_run(path, run)
    assert load_run(path) == run
