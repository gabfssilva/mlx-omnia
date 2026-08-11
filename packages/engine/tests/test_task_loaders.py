import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from conftest import local_snapshot, relative_diff
from huggingface_hub import snapshot_download

import mlx_omnia.task as task
from mlx_omnia import (
    RGB_IMAGE,
    TEXT,
    Chat,
    ChatMessage,
    CompositeModel,
    GenerationOptions,
    TextLanguageModel,
    UnsupportedInput,
    load,
)
from mlx_omnia.models import gpt2, qwen3_5
from mlx_omnia.quant.quantization import Affine
from mlx_omnia.task import _MODEL_SPECS
from mlx_omnia.vision import Image

HELLO: ChatMessage = {"role": "user", "content": "Capital of France?"}


@pytest.fixture(scope="module")
def gpt2_directory() -> Path:
    return Path(snapshot_download("gpt2", allow_patterns=list(gpt2.CHECKPOINT.patterns)))


@pytest.fixture(scope="module")
def qwen_directory() -> Path:
    """Pulled with the checkpoint's own patterns: whether the chat template arrives at all
    is part of what the loader tests decide."""
    return Path(
        snapshot_download("Qwen/Qwen3.5-0.8B", allow_patterns=list(qwen3_5.CHECKPOINT.patterns))
    )


def test_gpt2_loads_as_a_text_language_model(gpt2_directory: Path) -> None:
    model = load(gpt2_directory)
    assert model.native_signature.inputs == frozenset({TEXT})
    assert model.native_signature.outputs == frozenset({TEXT})


def test_qwen_declares_loaded_visual_components(qwen_directory: Path) -> None:
    model = load(qwen_directory)
    assert model.native_signature.inputs == frozenset({TEXT, RGB_IMAGE})
    assert model.native_signature.outputs == frozenset({TEXT})


def test_a_conversation_is_accepted_when_the_template_travelled(qwen_directory: Path) -> None:
    model = load(qwen_directory)
    assert model.accepts(Chat((HELLO,)))
    picture: ChatMessage = {
        "role": "user",
        "content": [{"type": "image", "image": Image(np.zeros((4, 4, 3), dtype=np.uint8))}],
    }
    assert model.accepts(Chat((picture,)))


def test_gpt2_without_a_template_does_not_accept_a_conversation(gpt2_directory: Path) -> None:
    """A base model gets no guessed template, so the conversation has nowhere to go."""
    model = load(gpt2_directory)
    assert not model.accepts(Chat((HELLO,)))
    with pytest.raises(UnsupportedInput):
        list(model.stream(Chat((HELLO,)), GenerationOptions(max_tokens=1)))


def _logits(model: object, ids: mx.array) -> mx.array:
    assert isinstance(model, CompositeModel)
    backend = model.model
    assert isinstance(backend, TextLanguageModel)
    return backend.model(ids)


def test_gpt2_quantize_on_load_writes_an_entry_that_reloads_identically(
    gpt2_directory: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """gpt2's preparation is not idempotent (Conv1D transpose, default fp32 cast): the
    entry the quantizing load writes is already prepared, and the reload has to attach it
    without preparing again. Identical logits are the check that it did — same bits on
    both sides, so the comparison is exact."""
    monkeypatch.setattr(task, "_CACHE", tmp_path / "quantized")
    ids = mx.array([[3, 1, 4, 1, 5]])

    memory = load(gpt2_directory, quantize=Affine(group_size=64, bits=8), cache=False)
    cached = load(gpt2_directory, quantize=Affine(group_size=64, bits=8))

    assert relative_diff(_logits(cached, ids), _logits(memory, ids)) == 0.0


def test_qwen_without_processor_declares_only_text(
    qwen_directory: Path,
    tmp_path: Path,
) -> None:
    names = ["config.json", "tokenizer.json"]
    names.extend(path.name for path in qwen_directory.glob("model*.safetensors"))
    for name in names:
        (tmp_path / name).symlink_to(qwen_directory / name)

    model = load(tmp_path)
    assert model.native_signature.inputs == frozenset({TEXT})
    assert not model.accepts(Chat((HELLO,)))


@pytest.mark.parametrize("model_type", sorted(_MODEL_SPECS))
def test_every_checkpoint_pulls_the_chat_template(model_type: str) -> None:
    """Both shapes, on every architecture: a template left out of `patterns` is a
    checkpoint that chats on this machine and not on a clean one."""
    patterns = set(_MODEL_SPECS[model_type].patterns)
    assert {"tokenizer_config.json", "chat_template.jinja"} <= patterns


@pytest.mark.parametrize("model_type", sorted(_MODEL_SPECS))
def test_every_checkpoint_pulls_the_generation_config(model_type: str) -> None:
    """The file is the only place some checkpoints spell the id that ends a turn, and the
    only place any of them says how it wants to be sampled. Left out of `patterns` it is
    not carried into a quantized entry, which then stops on `config.json`'s eos alone:
    Nemotron-3.5 declares `2` there and ends its turns on `<|im_end|>` (11), so the entry
    answers correctly and then repeats `<|im_end|>` until `max_tokens`."""
    assert "generation_config.json" in set(_MODEL_SPECS[model_type].patterns)


def test_the_loader_hands_over_every_eos_the_checkpoint_declares(tmp_path: Path) -> None:
    """Checkpoints declare more than one (Qwen3.6: `<|im_end|>` and `<|endoftext|>`), and
    the facade has to receive all of them — taking the first is what left a raw prompt
    running to `max_tokens`."""
    repo = "mlx-community/Qwen3-0.6B-4bit"
    directory = local_snapshot(repo)
    if directory is None:
        pytest.skip(f"{repo} not in the local HF cache")
    for path in directory.iterdir():
        if path.name != "config.json":
            (tmp_path / path.name).symlink_to(path)
    config = json.loads((directory / "config.json").read_text())
    (tmp_path / "config.json").write_text(json.dumps({**config, "eos_token_id": [7, 9]}))

    model = load(tmp_path)
    assert isinstance(model, CompositeModel)
    backend = model.model
    assert isinstance(backend, TextLanguageModel)
    assert tuple(backend.stop) == (7, 9)
