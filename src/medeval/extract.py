"""Pull the chosen option letter out of free-form model output."""

from __future__ import annotations

import re

# "Answer: B", "The correct answer is (C)", "Final answer - D". The keyword is matched
# case-insensitively but the letter must be a capital, so "the answer is a nerve" is ignored.
_KEYWORD = re.compile(
    r"(?i:final answer|correct answer|correct option|correct choice|answer)"
    r"\s*(?:is|:|=|-)?\s*[\(\[]?([A-Z])[\)\]]?(?![A-Za-z])"
)
_BARE = re.compile(r"^[\(\[]?([A-Za-z])[\)\]\.:]?$")
_LEADING = re.compile(r"^[\(\[]?([A-Z])[\)\]\.:]\s")


def extract_choice(text: str | None, valid: str = "ABCDE") -> str | None:
    """Return the option letter the response commits to, or None if it cannot be found.

    Order of preference: the last "answer ..." statement, a reply that is only a letter,
    then a reply that starts with "B." or "(B)". Letters outside ``valid`` are ignored.
    """
    if not text:
        return None
    cleaned = text.replace("*", "").replace("`", "").strip()
    allowed = set(valid.upper())

    hits = [m.group(1) for m in _KEYWORD.finditer(cleaned) if m.group(1) in allowed]
    if hits:
        return hits[-1]

    for pattern in (_BARE, _LEADING):
        match = pattern.match(cleaned)
        if match and match.group(1).upper() in allowed:
            return match.group(1).upper()
    return None
