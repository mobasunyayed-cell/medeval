import pytest

from medeval.compare import CompareError, mcnemar_exact, paired_compare
from medeval.metrics import bootstrap_interval, breakdown, summarize, wilson_interval
from medeval.runner import Result, Run


def res(qid, correct, specialty="a", difficulty="easy", predicted="A"):
    return Result(qid, specialty, difficulty, "A", predicted if correct else None, correct, "")


def run_of(name, flags):
    results = [res(f"q{i}", ok) for i, ok in enumerate(flags)]
    return Run(model=name, dataset="d", created_at="t", results=results)


def test_wilson_matches_reference_values():
    low, high = wilson_interval(8, 10)
    assert low == pytest.approx(0.4902, abs=1e-3)
    assert high == pytest.approx(0.9433, abs=1e-3)


def test_wilson_edges_stay_in_unit_interval():
    assert wilson_interval(0, 0) == (0.0, 1.0)
    low, high = wilson_interval(10, 10)
    assert high == pytest.approx(1.0)
    assert low == pytest.approx(0.7225, abs=1e-3)
    low, high = wilson_interval(0, 10)
    assert low == pytest.approx(0.0)
    assert high == pytest.approx(0.2775, abs=1e-3)


def test_summarize_counts():
    results = [res("1", True), res("2", False), res("3", True, predicted="A")]
    s = summarize(results)
    assert (s.n, s.correct, s.unanswered) == (3, 2, 1)
    assert s.accuracy == pytest.approx(2 / 3)
    assert s.ci_low < s.accuracy < s.ci_high


def test_summarize_empty():
    s = summarize([])
    assert s.n == 0 and s.accuracy == 0.0


def test_breakdown_groups_and_validates():
    results = [res("1", True, "x"), res("2", False, "y"), res("3", True, "x")]
    groups = breakdown(results, "specialty")
    assert list(groups) == ["x", "y"]
    assert groups["x"].correct == 2 and groups["y"].correct == 0
    assert list(breakdown(results, "difficulty")) == ["easy"]
    with pytest.raises(ValueError):
        breakdown(results, "nonsense")


def test_mcnemar_known_values():
    assert mcnemar_exact(10, 2) == pytest.approx(158 / 4096)  # 2 * (1 + 12 + 66) / 2**12
    assert mcnemar_exact(5, 5) == 1.0
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(0, 8) == pytest.approx(2 / 256)


def test_paired_compare_counts():
    a = run_of("A", [True, True, False, False, True])
    b = run_of("B", [True, False, True, False, False])
    c = paired_compare(a, b)
    assert (c.shared, c.both_correct, c.both_wrong, c.a_only, c.b_only) == (5, 1, 1, 2, 1)
    assert c.accuracy_a == pytest.approx(0.6) and c.accuracy_b == pytest.approx(0.4)


def test_paired_compare_uses_only_shared_ids_and_needs_overlap():
    a = run_of("A", [True, True])
    b = Run("B", "d", "t", [res("other", True)])
    with pytest.raises(CompareError):
        paired_compare(a, b)
    partial = Run("B", "d", "t", [res("q0", True)])
    assert paired_compare(a, partial).shared == 1


def test_bootstrap_is_close_to_wilson_for_moderate_samples():
    outcomes = [True] * 50 + [False] * 50
    low, high = bootstrap_interval(outcomes, resamples=4000)
    w_low, w_high = wilson_interval(50, 100)
    assert low == pytest.approx(w_low, abs=0.03) and high == pytest.approx(w_high, abs=0.03)


def test_bootstrap_of_nothing_is_uninformative():
    assert bootstrap_interval([]) == (0.0, 1.0)


def test_summarize_supports_both_methods_and_rejects_others():
    results = [res(str(i), i % 2 == 0) for i in range(20)]
    wilson = summarize(results, "wilson")
    boot = summarize(results, "bootstrap")
    assert wilson.accuracy == boot.accuracy == 0.5
    assert (wilson.ci_low, wilson.ci_high) != (boot.ci_low, boot.ci_high)
    with pytest.raises(ValueError, match="method"):
        summarize(results, "magic")  # type: ignore[arg-type]
