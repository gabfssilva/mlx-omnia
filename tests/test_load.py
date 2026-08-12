import json
from collections.abc import Iterator
from pathlib import Path
from typing import TypeIs

import mlx.core as mx
import mlx.nn as nn
import pytest

import mlx_omnia
import mlx_omnia.engine.task as task
from mlx_omnia import (
    TEXT,
    CompositeModel,
    GenerationOptions,
    ModelInput,
    ModelSignature,
    Text,
)
from mlx_omnia.engine.checkpoint import Checkpoint
from mlx_omnia.engine.parsers import Segment


class TextBackend:
    @property
    def native_signature(self) -> ModelSignature:
        return ModelSignature(frozenset({TEXT}), frozenset({TEXT}))

    def accepts(self, input: ModelInput) -> TypeIs[Text]:
        return isinstance(input, Text)

    def stream(self, input: Text, options: GenerationOptions) -> Iterator[Segment]:
        yield Segment("content", input.value)


def _no_tree(directory: Path, dtype: mx.Dtype | None) -> nn.Module:
    """`load` never reaches for the tree: it only ever asks for the task-level model."""
    raise AssertionError("load must not build the tree itself")


def test_load_is_public() -> None:
    assert callable(getattr(mlx_omnia, "load", None))


def test_load_dispatches_from_the_model_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "custom"}))
    backend = TextBackend()

    def load_custom(directory: Path, dtype: mx.Dtype | None) -> mlx_omnia.LanguageModel[ModelInput]:
        assert directory == tmp_path
        assert dtype is None
        return CompositeModel(backend, [])

    monkeypatch.setitem(task._MODEL_SPECS, "custom", Checkpoint((), _no_tree, load_custom))

    loaded = mlx_omnia.load(tmp_path)

    assert isinstance(loaded, CompositeModel)
    assert loaded.model is backend


def test_load_uses_a_local_directory_without_the_hub(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "gpt2"}))
    backend = TextBackend()
    calls: list[tuple[Path, mx.Dtype | None]] = []

    def load_gpt2(directory: Path, dtype: mx.Dtype | None) -> mlx_omnia.LanguageModel[ModelInput]:
        calls.append((directory, dtype))
        return CompositeModel(backend, [])

    def unexpected_hub_call(*_args: object, **_options: object) -> str:
        raise AssertionError("local paths must not use the Hugging Face Hub")

    monkeypatch.setitem(task._MODEL_SPECS, "gpt2", Checkpoint((), _no_tree, load_gpt2))
    monkeypatch.setattr(task, "_download_config", unexpected_hub_call)

    loaded = mlx_omnia.load(tmp_path, dtype=mx.float16)

    assert isinstance(loaded, CompositeModel)
    assert loaded.model is backend
    assert calls == [(tmp_path, mx.float16)]


def test_load_resolves_a_repository_in_the_default_hugging_face_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"model_type": "qwen3_5"}))
    backend = TextBackend()
    hub_calls: list[tuple[str, str, dict[str, object]]] = []
    snapshot_calls: list[tuple[str, dict[str, object]]] = []

    def download_config(
        repository: str,
        **options: object,
    ) -> Path:
        hub_calls.append((repository, "config.json", options))
        return config

    def download_snapshot(repository: str, **options: object) -> Path:
        snapshot_calls.append((repository, options))
        return tmp_path

    def load_qwen(directory: Path, dtype: mx.Dtype | None) -> mlx_omnia.LanguageModel[ModelInput]:
        assert directory == tmp_path
        assert dtype is None
        return CompositeModel(backend, [])

    monkeypatch.setattr(task, "_download_config", download_config)
    monkeypatch.setattr(task, "_download_snapshot", download_snapshot)
    monkeypatch.setitem(task._MODEL_SPECS, "qwen3_5", Checkpoint((), _no_tree, load_qwen))

    loaded = mlx_omnia.load(
        "Qwen/Qwen3.5-0.8B",
        revision="main",
        local_files_only=True,
    )

    assert isinstance(loaded, CompositeModel)
    assert loaded.model is backend
    assert hub_calls == [
        (
            "Qwen/Qwen3.5-0.8B",
            "config.json",
            {"revision": "main", "local_files_only": True},
        )
    ]
    assert snapshot_calls[0][0] == "Qwen/Qwen3.5-0.8B"
    assert snapshot_calls[0][1]["revision"] == "main"
    assert snapshot_calls[0][1]["local_files_only"] is True
    assert "cache_dir" not in snapshot_calls[0][1]


def test_load_rejects_an_unsupported_architecture(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "bert"}))

    with pytest.raises(ValueError, match="unsupported model_type 'bert'"):
        mlx_omnia.load(tmp_path)
