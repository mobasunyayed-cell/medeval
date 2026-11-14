# Usage

## Dataset format

One JSON object per line. Lines starting with `#` are comments.

```json
{"id": "anat-001", "question": "Which nerve ...?", "options": {"A": "Ulnar", "B": "Median", "C": "Radial", "D": "Axillary"}, "answer": "C", "specialty": "anatomy", "difficulty": "easy"}
```

## Commands

| Command | Purpose |
|---|---|
| `medeval validate DATA` | Check a dataset file |
| `medeval run --data DATA --model SPEC --out RUN` | Evaluate a model |
| `medeval report RUN [--ci wilson\|bootstrap] [--out FILE]` | Markdown report |
| `medeval compare RUN_A RUN_B` | Paired comparison with an exact McNemar test |
| `medeval agreement RUN_A RUN_B` | Cohen's kappa between two runs |
| `medeval rubric --rubric R --response FILE` | Score free text against a checklist |

## Models

| Spec | Meaning |
|---|---|
| `random[:seed]` | Chance baseline |
| `simulated:<p>` | Knows the answer, right with probability `p` (demos and tests) |
| `replay:<file.jsonl>` | Answers you collected elsewhere |
| `anthropic:<model-id>` | Anthropic API (needs the `anthropic` extra and `ANTHROPIC_API_KEY`) |

## Long runs

```bash
medeval run --data data/sample_mcq.jsonl --model anthropic:claude-sonnet-5-5 \
  --out runs/claude.json --workers 4 --retries 3 --checkpoint runs/claude.ckpt.jsonl
```

Rerun the same command after an interruption and only missing or failed questions are asked.

## Docker

```bash
docker build -t medeval .
docker run --rm medeval validate data/sample_mcq.jsonl
```
