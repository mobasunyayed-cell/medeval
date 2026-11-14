# medeval

[![CI](https://github.com/mobasunyayed-cell/medeval/actions/workflows/ci.yml/badge.svg)](https://github.com/mobasunyayed-cell/medeval/actions/workflows/ci.yml)
[![CodeQL](https://github.com/mobasunyayed-cell/medeval/actions/workflows/codeql.yml/badge.svg)](https://github.com/mobasunyayed-cell/medeval/actions/workflows/codeql.yml)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

**Documentation:** https://mobasunyayed-cell.github.io/medeval/

A small, dependency-free Python toolkit for evaluating language-model answers on medical questions. It answers three questions that a bare accuracy number does not:

1. **How sure are we?** Every accuracy comes with a 95% confidence interval (Wilson, or bootstrap on request), because medical test sets are small.
2. **Is model A really better than model B?** Paired comparison with an exact McNemar test on the questions both models saw.
3. **Where does it fail?** Results are broken down by specialty and difficulty, and missed questions are listed.
4. **Do two raters really agree?** Cohen's kappa (plain or weighted) between two models, or between a model and a human.

It also includes a transparent, weighted-checklist **rubric scorer** for free-text answers.

> Research and evaluation tooling only. It is not a medical device and says nothing about clinical safety.

## Install

```bash
git clone https://github.com/mobasunyayed-cell/medeval.git
cd medeval
pip install -e .            # core, no third-party dependencies
pip install -e ".[dev]"     # plus pytest, ruff, mypy
pip install -e ".[anthropic]"   # optional: evaluate Claude models through the API
```

Or with Docker:

```bash
docker build -t medeval .
docker run --rm medeval validate data/sample_mcq.jsonl
```

Python 3.10 or newer.

## Quick start

```bash
medeval validate data/sample_mcq.jsonl
medeval run --data data/sample_mcq.jsonl --model simulated:0.8 --out runs/a.json
medeval run --data data/sample_mcq.jsonl --model random:1 --out runs/b.json
medeval report runs/a.json
medeval compare runs/a.json runs/b.json
```

Real output from the commands above (the "models" here are built-in stand-ins, see below):

```
simulated:0.8: 19/24 correct (79.2%, 95% CI 59.5% to 90.8%), 0 unanswered.
random:1: 10/24 correct (41.7%, 95% CI 24.5% to 61.2%), 0 unanswered.

# Comparison: simulated:0.8 vs random:1
- Shared questions: 24
- Both correct: 9, both wrong: 4
- Only A correct: 10, only B correct: 1
- Exact McNemar p-value: 0.0117 (significant at 0.05)
```

Note how wide the intervals are with 24 questions. A random guesser over four options should score about 25%, yet this one landed at 41.7%, and its interval still contains 25%. That is exactly why the interval is reported.

## Long runs: retries, concurrency, resume

```bash
medeval run --data data/sample_mcq.jsonl --model anthropic:claude-sonnet-5-5 \
  --out runs/claude.json --workers 4 --retries 3 --backoff 1 --checkpoint runs/claude.ckpt.jsonl
```

- `--retries` retries a failed call with exponential backoff (1 s, 2 s, 4 s, capped at 30 s). Errors that retrying cannot fix, such as a missing recorded answer, are not retried.
- `--workers` asks several questions at once. Results keep dataset order.
- `--checkpoint` appends every finished answer to a file. If the run is interrupted, run the same command again: answers already saved are reused, and only missing or failed questions are asked. A half-written last line from a crash is ignored.

## Agreement and interval methods

```bash
medeval agreement runs/a.json runs/b.json
medeval report runs/a.json --ci bootstrap
```

```
shared questions: 24
percent agreement: 83.3%
Cohen's kappa: 0.700 (substantial)
```

Wilson and bootstrap intervals differ at the edges. For 22 of 24 correct, Wilson gives 74.2% to 97.7% while the bootstrap gives 79.2% to 100%. The bootstrap upper bound reaches 100% because resampling can draw an all-correct sample, which is why Wilson is the default. In Python, `cohens_kappa(a, b, categories, weights="linear")` supports ordered scales such as 1 to 5 rubric scores.

## Models

| Spec | What it does |
|---|---|
| `random[:seed]` | Guesses a letter. A chance baseline. |
| `simulated:<p>` | Knows the answer and gets it right with probability `p`. For demos and pipeline tests only. |
| `replay:<file.jsonl>` | Replays answers you collected elsewhere, for example pasted from a chat. One `{"id", "response"}` object per line (see `examples/replay_answers.jsonl`). |
| `anthropic:<model-id>` | Calls the Anthropic Messages API. Needs the `anthropic` extra and `ANTHROPIC_API_KEY`. |

Adding a provider means writing a class with a `name` and a `complete(question, prompt)` method. Nothing else changes.

## Dataset format

One JSON object per line. Lines starting with `#` are comments.

```json
{"id": "anat-001", "question": "Which nerve ...?", "options": {"A": "Ulnar", "B": "Median", "C": "Radial", "D": "Axillary"}, "answer": "C", "specialty": "anatomy", "difficulty": "easy"}
```

`medeval validate` checks required fields, option letters, that the answer is one of the options, and that ids are unique.

`data/sample_mcq.jsonl` has 24 short questions written for this project. It exists to demo and test the tooling. **It is far too small to rank real models, and its medical content should be reviewed by a clinician before anyone relies on it.**

## How answers are scored

The prompt asks the model to finish with `Answer: X`. The extractor takes the last "answer ..." statement, or a reply that is just a letter, or one that starts with `B.` or `(B)`. Anything else counts as *unanswered* and is reported separately from wrong answers. A failed API call is recorded as an error on that question and does not stop the run.

## Rubric scoring for free text

```bash
medeval rubric --rubric examples/rubric_anaphylaxis.json --response examples/rubric_response_good.txt
```

A rubric lists weighted criteria (each satisfied by any one of several phrases) and optional penalties. This is a keyword baseline. It can tell you a required idea is missing, not whether the reasoning is good. Use it as a first filter next to human or model-based grading.

## Design notes

- **No runtime dependencies.** Statistics are implemented directly (`math.comb`, the Wilson formula, kappa) and verified in the tests against `scipy` and `scikit-learn`, plus property-based tests (Hypothesis) that look for counterexamples such as asymmetric p-values or intervals outside 0 to 1.
- **Wilson, not the normal approximation,** so intervals stay inside 0 to 1 for small samples and extreme accuracies.
- **Exact McNemar, not a chi-square approximation,** because discordant pairs are often few.
- **Strict typing** (`mypy --strict`) and linting (`ruff`) run in CI on Python 3.10, 3.11 and 3.12, with a 90% coverage gate. The suite has 144 tests.
- **Failures are data, not crashes.** One failed call becomes an error on that question. A crash mid-run loses nothing thanks to the checkpoint.
- **Saved runs are versioned** (`schema_version`) so old files fail loudly instead of being misread.

## Limitations

- Multiple choice only for accuracy; free text goes through the rubric scorer.
- The bundled dataset is tiny and author-written.
- Accuracy on multiple-choice questions is not clinical competence.
- The rubric scorer matches words, not meaning.

## Development

```bash
pip install -e ".[dev]"
ruff check . && ruff format --check .
mypy
pytest --cov=medeval
```

See [CONTRIBUTING.md](CONTRIBUTING.md).

```
src/medeval/
  dataset.py   load and validate JSONL questions
  extract.py   pull the chosen letter out of free text
  models.py    random, simulated, replay and Anthropic clients
  runner.py    run a model with retries, workers and checkpoints; save and load runs
  metrics.py   accuracy, Wilson and bootstrap intervals, breakdowns
  compare.py   paired comparison, exact McNemar test
  agreement.py Cohen's kappa (plain, linear, quadratic) between runs or raters
  report.py    Markdown reports
  rubric.py    weighted checklist scoring
  cli.py       command-line interface
tests/         pytest suite
data/          sample dataset
examples/      sample rubric and replay file
```

## AI assistance disclosure

This project was built with help from Claude (Anthropic) for code and guidance. I ran the tests and the tooling myself and I am responsible for the project.

## License

MIT, see [LICENSE](LICENSE).
