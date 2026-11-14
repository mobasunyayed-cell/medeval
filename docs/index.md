# medeval

A small, dependency-free Python toolkit for evaluating language-model answers on medical
questions. It reports accuracy with confidence intervals, compares two models with an exact
paired test, measures agreement with Cohen's kappa, and survives long runs with retries and
checkpoints.

!!! warning "Not a medical device"
    medeval measures how a model answers test questions. It says nothing about clinical safety.

## Install

```bash
pip install -e .
medeval --help
```

## Five-minute tour

```bash
medeval validate data/sample_mcq.jsonl
medeval run --data data/sample_mcq.jsonl --model simulated:0.8 --out runs/a.json
medeval report runs/a.json
```

Continue with [Usage](usage.md) for every command, or [Statistics](statistics.md) for what the
numbers mean.
