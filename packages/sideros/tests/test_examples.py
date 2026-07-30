import importlib.util
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from sideros import Chat, GenerationOptions, Image, LanguagePrompt, Text
from sideros.suppress import Segment

ROOT = Path(__file__).parents[3]


def load_example(name: str) -> ModuleType:
    path = ROOT / "examples" / f"{name}.py"
    assert path.exists()
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RecordingModel:
    def __init__(self) -> None:
        self.calls: list[tuple[Chat | Text | LanguagePrompt, GenerationOptions]] = []

    def stream(
        self,
        prompt: Chat | Text | LanguagePrompt,
        options: GenerationOptions,
    ) -> Iterator[Segment]:
        self.calls.append((prompt, options))
        yield Segment("content", "generated")


def test_basic_usage_generates_text_without_network(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    example = load_example("basic_usage")
    model = RecordingModel()
    loads: list[tuple[str, dict[str, object]]] = []

    def load_model(repository: str, **options: object) -> RecordingModel:
        loads.append((repository, options))
        return model

    def obsolete_download(*_args: object, **_options: object) -> str:
        raise AssertionError("the example must use sideros.load")

    monkeypatch.setattr(example, "load", load_model, raising=False)
    monkeypatch.setattr(example, "snapshot_download", obsolete_download, raising=False)

    example.main()

    assert loads == [("mlx-community/Qwen3.6-35B-A3B-6bit", {"local_files_only": True})]
    prompt, options = model.calls[0]
    # A conversation, not a raw prompt: an instruct model handed loose text continues the
    # document instead of answering it.
    assert prompt == Chat(({"role": "user", "content": "A simple autoencoder in Python"},))
    assert options == GenerationOptions(max_tokens=2048)
    assert capsys.readouterr().out == "generated\n"


def test_describe_image_generates_from_an_rgb_image(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    example = load_example("describe_image")
    model = RecordingModel()
    loads: list[tuple[str, dict[str, object]]] = []

    def load_model(repository: str, **options: object) -> RecordingModel:
        loads.append((repository, options))
        return model

    def obsolete_download(*_args: object, **_options: object) -> str:
        raise AssertionError("the example must use sideros.load")

    monkeypatch.setattr(example, "load", load_model, raising=False)
    monkeypatch.setattr(example, "snapshot_download", obsolete_download, raising=False)

    example.main()

    assert loads == [("Qwen/Qwen3.5-0.8B", {"local_files_only": True})]
    prompt, options = model.calls[0]
    assert isinstance(prompt, LanguagePrompt)
    assert isinstance(prompt.parts[0], Image)
    assert prompt.parts[0].pixels.shape == (256, 256, 3)
    assert prompt.parts[0].pixels.dtype == np.uint8
    assert prompt.parts[1] == Text("Describe only the colors and their positions in one sentence.")
    assert options == GenerationOptions(max_tokens=50)
    assert capsys.readouterr().out == "generated\n"
