# Statistics

## Wilson interval (default)

For `k` correct out of `n`, the Wilson score interval stays inside 0 to 1 and behaves well for
small `n`, unlike the textbook normal approximation. With 22 of 24 correct it gives 74.2% to
97.7%.

## Bootstrap interval

Resamples the question outcomes with replacement and takes the 2.5th and 97.5th percentiles.
It assumes nothing about the distribution, but collapses when every outcome is the same, so
Wilson remains the default.

## Exact McNemar test

Only questions where two models disagree carry information. Let `b` be the questions only A got
right and `c` the ones only B got right. Under the null hypothesis each disagreement is a fair
coin flip, so the two-sided p-value is `2 * P(X <= min(b, c))` for `X ~ Binomial(b + c, 0.5)`,
capped at 1.

## Cohen's kappa

Agreement beyond chance: `kappa = 1 - observed disagreement / expected disagreement`. 1 is
perfect agreement, 0 is chance level, negative is worse than chance. Linear and quadratic
weights treat the categories as ordered, so near misses cost less than far misses. The verbal
labels (slight, fair, moderate, substantial, almost perfect) follow Landis and Koch (1977) and
are conventions, not laws.

## How the code is verified

Wilson intervals and McNemar p-values are compared with `scipy`, and kappa with
`scikit-learn`, in the test suite. Property-based tests (Hypothesis) check symmetry, bounds and
monotonicity on generated inputs.
