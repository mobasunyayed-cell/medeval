# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- pre-commit hooks, Dependabot, CodeQL scanning and a tag-driven release workflow.
- MkDocs documentation site published with GitHub Pages.
- Dockerfile, package build check and image smoke test in CI.
- Issue and pull request templates, security policy, code of conduct and CITATION.cff.

## [0.2.0] - 2026-11-14

### Added
- Retries with exponential backoff, concurrent workers and resumable checkpoints for long runs.
- Bootstrap confidence intervals (`medeval report --ci bootstrap`).
- Cohen's kappa with linear and quadratic weights, and a `medeval agreement` command.
- Property-based tests (Hypothesis) and cross-checks against scipy and scikit-learn.

### Fixed
- Datasets, replay files and checkpoints are split on newline only, so U+2028 and similar
  characters inside a JSON string can no longer break a line.

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
