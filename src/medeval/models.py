"""Model clients. Anything with a ``name`` and a ``complete`` method can be evaluated."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Protocol

from medeval.dataset import Question


class ModelError(RuntimeError):
    """Raised when a model cannot be created or fails to answer."""


class PermanentError(ModelError):
    """A failure that retrying cannot fix, such as a missing recorded answer."""


class ModelClient(Protocol):
    name: str

    def complete(self, question: Question, prompt: str) -> str:
        """Return the model's raw text answer for one question."""
        ...


class RandomModel:
    """Guesses a letter. Deterministic per (seed, question id). Use it as a chance baseline."""

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self.name = f"random:{seed}"

    def complete(self, question: Question, prompt: str) -> str:
        rng = random.Random(f"{self.seed}:{question.id}")
        return f"Answer: {rng.choice(sorted(question.options))}"


class SimulatedModel:
    """Knows the gold answer and gets it right with a fixed probability.

    Useful for demos and for testing the pipeline, not for measuring anything real.
    """

    def __init__(self, accuracy: float = 0.8, seed: int = 0) -> None:
        if not 0.0 <= accuracy <= 1.0:
            raise ModelError("simulated accuracy must be between 0 and 1")
        self.accuracy = accuracy
        self.seed = seed
        self.name = f"simulated:{accuracy:g}"

    def complete(self, question: Question, prompt: str) -> str:
        rng = random.Random(f"{self.seed}:{question.id}")
        if rng.random() < self.accuracy:
            return f"Answer: {question.answer}"
        wrong = sorted(set(question.options) - {question.answer})
        return f"Answer: {rng.choice(wrong)}"


class ReplayModel:
    """Replays answers collected elsewhere (for example pasted from a chat) from a JSONL file.

    Each line is ``{"id": "<question id>", "response": "<text>"}``.
    """

    def __init__(self, path: str | Path) -> None:
        path = Path(path)
        if not path.is_file():
            raise ModelError(f"replay file not found: {path}")
        self.name = f"replay:{path.stem}"
        self._responses: dict[str, str] = {}
        for number, raw in enumerate(path.read_text(encoding="utf-8").split("\n"), start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
                self._responses[str(record["id"])] = str(record["response"])
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ModelError(f"{path}, line {number}: expected {{id, response}}") from exc

    def complete(self, question: Question, prompt: str) -> str:
        try:
            return self._responses[question.id]
        except KeyError:
            raise PermanentError(f"no recorded response for question {question.id!r}") from None


class AnthropicModel:
    """Calls the Anthropic Messages API. Needs ``pip install medeval[anthropic]`` and a key."""

    def __init__(self, model_id: str, max_tokens: int = 512) -> None:
        try:
            import anthropic
        except ImportError as exc:
            raise ModelError("install the extra first: pip install 'medeval[anthropic]'") from exc
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise ModelError("set the ANTHROPIC_API_KEY environment variable")
        self._client = anthropic.Anthropic()
        self._model_id = model_id
        self._max_tokens = max_tokens
        self.name = f"anthropic:{model_id}"

    def complete(self, question: Question, prompt: str) -> str:
        reply = self._client.messages.create(
            model=self._model_id,
            max_tokens=self._max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(
            block.text for block in reply.content if getattr(block, "type", "") == "text"
        )


def make_model(spec: str) -> ModelClient:
    """Build a model from a short spec such as ``random``, ``simulated:0.8`` or ``replay:file``."""
    kind, _, arg = spec.partition(":")
    try:
        if kind == "random":
            return RandomModel(seed=int(arg) if arg else 0)
        if kind == "simulated":
            return SimulatedModel(accuracy=float(arg) if arg else 0.8)
    except ValueError as exc:
        raise ModelError(f"bad model spec {spec!r}: {exc}") from exc
    if kind == "replay":
        if not arg:
            raise ModelError("replay needs a path, e.g. replay:answers.jsonl")
        return ReplayModel(arg)
    if kind == "anthropic":
        if not arg:
            raise ModelError("anthropic needs a model id, e.g. anthropic:claude-sonnet-5-5")
        return AnthropicModel(arg)
    raise ModelError(f"unknown model spec {spec!r} (use random, simulated, replay or anthropic)")
