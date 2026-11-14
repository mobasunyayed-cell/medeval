"""Score free-text answers against a weighted checklist.

This is a transparent keyword baseline. It checks whether required ideas are mentioned and
whether known-bad statements appear. It cannot judge reasoning quality, so treat it as a first
filter next to human or model-based grading, not a replacement.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class RubricError(ValueError):
    """Raised when a rubric file is invalid."""


@dataclass(frozen=True)
class Criterion:
    name: str
    any_of: tuple[str, ...]
    weight: float = 1.0


@dataclass(frozen=True)
class Penalty:
    name: str
    any_of: tuple[str, ...]
    amount: float = 1.0


@dataclass(frozen=True)
class Rubric:
    id: str
    criteria: tuple[Criterion, ...]
    penalties: tuple[Penalty, ...] = ()


@dataclass(frozen=True)
class RubricScore:
    score: float
    earned: float
    possible: float
    matched: tuple[str, ...]
    missed: tuple[str, ...]
    violations: tuple[str, ...]


def _phrases(raw: Any, where: str) -> tuple[str, ...]:
    if not isinstance(raw, list) or not raw or not all(isinstance(p, str) and p for p in raw):
        raise RubricError(f"{where}: 'any_of' must be a non-empty list of strings")
    return tuple(raw)


def parse_rubric(data: Any) -> Rubric:
    try:
        criteria = tuple(
            Criterion(
                name=str(item["name"]),
                any_of=_phrases(item["any_of"], str(item["name"])),
                weight=float(item.get("weight", 1.0)),
            )
            for item in data["criteria"]
        )
        penalties = tuple(
            Penalty(
                name=str(item["name"]),
                any_of=_phrases(item["any_of"], str(item["name"])),
                amount=float(item.get("amount", 1.0)),
            )
            for item in data.get("penalties", [])
        )
        rubric = Rubric(id=str(data["id"]), criteria=criteria, penalties=penalties)
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        if isinstance(exc, RubricError):
            raise
        raise RubricError(f"malformed rubric: {exc!r}") from exc
    if not rubric.criteria:
        raise RubricError("a rubric needs at least one criterion")
    if any(c.weight <= 0 for c in rubric.criteria):
        raise RubricError("criterion weights must be positive")
    return rubric


def load_rubric(path: str | Path) -> Rubric:
    path = Path(path)
    if not path.is_file():
        raise RubricError(f"rubric not found: {path}")
    try:
        return parse_rubric(json.loads(path.read_text(encoding="utf-8")))
    except json.JSONDecodeError as exc:
        raise RubricError(f"{path}: invalid JSON ({exc.msg})") from exc


def _mentions(text: str, phrases: tuple[str, ...]) -> bool:
    return any(re.search(rf"\b{re.escape(p)}\b", text, re.IGNORECASE) for p in phrases)


def score_response(text: str, rubric: Rubric) -> RubricScore:
    """Score = (weights of matched criteria - penalties) / total weight, floored at 0."""
    matched = [c for c in rubric.criteria if _mentions(text, c.any_of)]
    missed = [c for c in rubric.criteria if c not in matched]
    violated = [p for p in rubric.penalties if _mentions(text, p.any_of)]

    possible = sum(c.weight for c in rubric.criteria)
    earned = sum(c.weight for c in matched) - sum(p.amount for p in violated)
    return RubricScore(
        score=max(0.0, earned) / possible,
        earned=earned,
        possible=possible,
        matched=tuple(c.name for c in matched),
        missed=tuple(c.name for c in missed),
        violations=tuple(p.name for p in violated),
    )
