import json
import sys
import types

import pytest

from medeval.dataset import Question
from medeval.models import (
    AnthropicModel,
    ModelError,
    RandomModel,
    ReplayModel,
    SimulatedModel,
    make_model,
)

Q = Question(id="q1", question="?", options={"A": "x", "B": "y", "C": "z"}, answer="B")


def test_random_model_is_deterministic_and_valid():
    a, b = RandomModel(seed=1), RandomModel(seed=1)
    assert a.complete(Q, "") == b.complete(Q, "")
    assert a.complete(Q, "").split(": ")[1] in "ABC"


def test_simulated_model_extremes():
    assert SimulatedModel(accuracy=1.0).complete(Q, "") == "Answer: B"
    assert SimulatedModel(accuracy=0.0).complete(Q, "") != "Answer: B"


def test_simulated_rejects_bad_accuracy():
    with pytest.raises(ModelError):
        SimulatedModel(accuracy=1.5)


def test_replay_model(tmp_path):
    path = tmp_path / "r.jsonl"
    path.write_text(json.dumps({"id": "q1", "response": "Answer: A"}) + "\n", encoding="utf-8")
    model = ReplayModel(path)
    assert model.complete(Q, "") == "Answer: A"
    other = Question(id="zzz", question="?", options={"A": "x", "B": "y"}, answer="A")
    with pytest.raises(ModelError, match="no recorded response"):
        model.complete(other, "")


def test_replay_model_rejects_bad_files(tmp_path):
    with pytest.raises(ModelError, match="not found"):
        ReplayModel(tmp_path / "nope.jsonl")
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{oops", encoding="utf-8")
    with pytest.raises(ModelError, match="expected"):
        ReplayModel(bad)


@pytest.mark.parametrize(
    "spec, name",
    [("random", "random:0"), ("random:7", "random:7"), ("simulated:0.5", "simulated:0.5")],
)
def test_make_model(spec, name):
    assert make_model(spec).name == name


@pytest.mark.parametrize("spec", ["", "gpt", "random:x", "simulated:abc", "replay", "anthropic"])
def test_make_model_rejects_bad_specs(spec):
    with pytest.raises(ModelError):
        make_model(spec)


def test_anthropic_requires_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ModelError):
        make_model("anthropic:some-model")


def test_anthropic_model_with_fake_sdk(monkeypatch):
    calls = {}

    class FakeMessages:
        def create(self, **kwargs):
            calls.update(kwargs)
            block = types.SimpleNamespace(type="text", text="Answer: B")
            ignored = types.SimpleNamespace(type="thinking", text="hidden")
            return types.SimpleNamespace(content=[ignored, block])

    class FakeClient:
        messages = FakeMessages()

    fake = types.SimpleNamespace(Anthropic=lambda: FakeClient())
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    model = make_model("anthropic:some-model")
    assert isinstance(model, AnthropicModel)
    assert model.name == "anthropic:some-model"
    assert model.complete(Q, "prompt text") == "Answer: B"
    assert calls["model"] == "some-model"
    assert calls["messages"] == [{"role": "user", "content": "prompt text"}]


def test_anthropic_model_without_sdk(monkeypatch):
    monkeypatch.setitem(sys.modules, "anthropic", None)  # makes `import anthropic` fail
    with pytest.raises(ModelError, match="pip install"):
        AnthropicModel("x")
