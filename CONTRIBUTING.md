# Contributing

1. Fork the repository and create a branch: `git switch -c feature/short-name`.
2. Install the development tools: `pip install -e ".[dev]"`.
3. Make your change and add or update tests.
4. Run the same checks as CI:
   ```bash
   ruff check . && ruff format --check .
   mypy
   pytest --cov=medeval --cov-fail-under=90
   ```
5. Open a pull request that says what changed and why.

## Ground rules

- Keep the core free of third-party dependencies. Provider SDKs belong in optional extras and are imported lazily.
- New statistics need a test against a known reference value.
- Medical content in datasets must be correct and reviewed. If you are unsure, say so in the pull request.
