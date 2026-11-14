# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-11-14

### Added
- JSONL dataset loader with validation.
- Answer-letter extraction from free-form model output.
- Model clients: random baseline, simulated, replay and Anthropic.
- Accuracy with Wilson confidence intervals, broken down by specialty and difficulty.
- Paired comparison of two runs with an exact McNemar test.
- Weighted-checklist rubric scorer for free-text answers.
- Markdown reports and a `medeval` command-line interface.
- Test suite, strict typing, linting and GitHub Actions CI.
