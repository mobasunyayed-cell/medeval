# Security policy

## Supported versions

Only the latest release receives fixes.

## Reporting a vulnerability

Please do not open a public issue. Use GitHub's private reporting instead:
**Security tab, then "Report a vulnerability"**. You will get a first reply within a few days.

## Scope notes

medeval sends the question text of your dataset to whichever model provider you choose.
Do not evaluate datasets that contain patient-identifiable information. API keys are read
from environment variables only and are never written to run files.
