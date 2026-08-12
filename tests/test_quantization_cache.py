import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TypeIs

import mlx.core as mx
import mlx.nn as nn
import pytest
from conftest import relative_diff

import mlx_omnia
import mlx_omnia.engine.task as task
from mlx_omnia import (
    TEXT,
    CompositeModel,
    GenerationOptions,
    LanguageModel,
    ModelInput,
    ModelSignature,
    Text,
)
from mlx_omnia.engine.checkpoint import Checkpoint, checkpoint, load_checkpoint, save_quantized
from mlx_omnia.engine.parsers import Segment
from mlx_omnia.engine.quant.quantization import (
    Affine,
    ByPath,
    QuantizationPlan,
    expand_plan,
    inventory,
    quantize_weights,
)

_COARSE = Affine(group_size=64, bits=4)
_FINE = Affine(group_size=64, bits=8)
_IDS = mx.array([[3, 1, 4, 1, 5]])


class _Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(32, 64)
        self.attn = nn.Linear(64, 64, bias=False)
        self.mlp = nn.Linear(64, 64, bias=False)
        self.head = nn.Linear(64, 32, bias=False)

    def __call__(self, ids: mx.array) -> mx.array:
        return self.head(nn.silu(self.mlp(self.attn(self.embed(ids)))))


@dataclass(frozen=True)
class _Spine:
    tie_word_embeddings: bool


class _Backend:
    def __init__(self, model: _Tiny) -> None:
        self.model = model

    @property
    def native_signature(self) -> ModelSignature:
        return ModelSignature(frozenset({TEXT}), frozenset({TEXT}))

    def accepts(self, input: ModelInput) -> TypeIs[Text]:
        return isinstance(input, Text)

    def stream(self, input: Text, options: GenerationOptions) -> Iterator[Segment]:
        yield Segment("content", input.value)


def _dense() -> dict[str, mx.array]:
    return {f"{leaf.path}.weight": mx.random.normal(leaf.shape) for leaf in inventory(_Tiny())}


def _spec(reads: list[Path]) -> Checkpoint[_Tiny]:
    """The registry entry of a checkpoint small enough to build in the test, derived by the
    same `checkpoint` factory every architecture registers through. `reads` records every
    directory the weights were prepared from, which is what tells a hit from a miss without
    timing anything — the prepared entry loads without running `weights` at all."""

    def weights(directory: Path, config: None, dtype: mx.Dtype | None) -> dict[str, mx.array]:
        reads.append(directory)
        loaded = mx.load(str(directory / "model.safetensors"))
        assert isinstance(loaded, dict)
        if dtype is None:
            return loaded
        return {name: value.astype(dtype) for name, value in loaded.items()}

    return checkpoint(
        ("config.json", "model.safetensors", "tokenizer.json", "chat_template.jinja"),
        lambda path: None,
        lambda config: _Tiny(),
        weights,
        lambda directory, model: CompositeModel(_Backend(model), []),
    )


@dataclass(frozen=True)
class _Source:
    directory: Path
    cache: Path
    reads: list[Path]


@pytest.fixture
def source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Source:
    mx.random.seed(0)
    directory = tmp_path / "checkpoint"
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps({"model_type": "tiny", "hidden_size": 64}))
    (directory / "tokenizer.json").write_text("{}")
    (directory / "chat_template.jinja").write_text("{{ messages[0]['content'] }}")
    mx.save_safetensors(str(directory / "model.safetensors"), _dense())

    reads: list[Path] = []
    cache = tmp_path / "cache"
    monkeypatch.setattr(task, "_CACHE", cache)
    monkeypatch.setitem(task._MODEL_SPECS, "tiny", _spec(reads))
    return _Source(directory, cache, reads)


def _entries(source: _Source) -> list[Path]:
    root = source.cache / source.directory.name
    return sorted(path for path in root.iterdir() if path.is_dir()) if root.is_dir() else []


def _logits(loaded: LanguageModel[ModelInput]) -> mx.array:
    assert isinstance(loaded, CompositeModel)
    backend = loaded.model
    assert isinstance(backend, _Backend)
    return backend.model(_IDS)


def test_a_second_load_reuses_the_entry_instead_of_quantizing_again(source: _Source) -> None:
    # mutação: forçar o miss (`if not entry.is_dir()` → `if True`) quebra — a origem passa
    # a ser lida duas vezes.
    first = mlx_omnia.load(source.directory, quantize=_COARSE)
    (entry,) = _entries(source)
    written = (entry / "model.safetensors").stat().st_mtime_ns

    second = mlx_omnia.load(source.directory, quantize=_COARSE)

    assert source.reads.count(source.directory) == 1
    assert (entry / "model.safetensors").stat().st_mtime_ns == written
    assert relative_diff(_logits(second), _logits(first)) == 0.0

    config = json.loads((entry / "config.json").read_text())
    assert config["mlx_omnia"]["method"] == "rtn"
    assert config["mlx_omnia"]["source"]["directory"] == str(source.directory.resolve())
    assert config["quantization"]["leaves"]["head"] == {"group_size": 64, "bits": 4}
    assert (entry / "tokenizer.json").exists()


def test_selections_that_expand_to_the_same_plan_share_the_entry(source: _Source) -> None:
    # mutação: digerir `repr(quantize)` no lugar do plano expandido quebra — as quatro
    # seleções são objetos distintos e produziriam quatro entradas.
    mlx_omnia.load(source.directory, quantize=_COARSE)
    mlx_omnia.load(source.directory, quantize=ByPath(_COARSE, {}))
    mlx_omnia.load(source.directory, quantize=ByPath(_COARSE, {"attn": _FINE, "head": _FINE}))
    mlx_omnia.load(source.directory, quantize=ByPath(_COARSE, {"head": _FINE, "attn": _FINE}))

    assert len(_entries(source)) == 2


def test_a_different_plan_or_format_version_produces_a_new_entry(
    source: _Source,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # mutação: tirar "format_version" do payload de `_digest` quebra — o bump volta a cair
    # na entrada gravada pela versão anterior.
    mlx_omnia.load(source.directory, quantize=_COARSE)
    mlx_omnia.load(source.directory, quantize=_FINE)
    mlx_omnia.load(source.directory, quantize=ByPath(_COARSE, {"mlp": _FINE}))
    assert len(_entries(source)) == 3

    monkeypatch.setattr(task, "_FORMAT_VERSION", task._FORMAT_VERSION + 1)
    mlx_omnia.load(source.directory, quantize=_COARSE)

    assert len(_entries(source)) == 4


def test_quantizing_an_already_quantized_checkpoint_raises(source: _Source) -> None:
    # mutação: apagar a varredura de `_quantized` quebra — quem estoura passa a ser
    # `quantize_weights`, com outra mensagem.
    packed = quantize_weights(_dense(), expand_plan(_Tiny(), _COARSE))
    mx.save_safetensors(str(source.directory / "model.safetensors"), packed)

    with pytest.raises(ValueError, match="quantize= cannot be applied to a quantized checkpoint"):
        mlx_omnia.load(source.directory, quantize=_FINE)


def test_every_registered_architecture_has_a_quantizing_load() -> None:
    assert all(spec.quantize is not None for spec in task._MODEL_SPECS.values())


def test_quantizing_an_architecture_without_a_quantizing_load_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # mutação: cair no `spec.quantizer` sem a checagem de `None` quebra — o erro vira um
    # TypeError de chamada a None, depois de resolver o diretório.
    def _no_tree(directory: Path, dtype: mx.Dtype | None) -> _Tiny:
        raise AssertionError

    def _no_task(directory: Path, dtype: mx.Dtype | None) -> LanguageModel[ModelInput]:
        raise AssertionError

    monkeypatch.setitem(task._MODEL_SPECS, "bare", Checkpoint((), _no_tree, _no_task))
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "bare"}))

    with pytest.raises(ValueError, match="quantize= is not supported for model_type 'bare'"):
        mlx_omnia.load(tmp_path, quantize=_COARSE)


def test_a_dtype_cast_is_rejected_by_the_tensors_and_not_by_the_config_block() -> None:
    # mutação: voltar `_reject_dtype_cast` a chavear no formato declarado quebra — este
    # checkpoint carrega `.scales` e não declara bloco nenhum.
    packed = quantize_weights(_dense(), expand_plan(_Tiny(), _COARSE))

    with pytest.raises(ValueError, match="dtype= cannot be applied to a quantized checkpoint"):
        load_checkpoint(_Tiny(), _Spine(False), packed, [], mx.float16)


def test_cache_false_quantizes_without_writing_anything(source: _Source) -> None:
    # mutação: no ramo `cache=False`, passar `pending.weights()` direto ao attach (sem
    # `quantize_weights`) quebra — os logits passam a ser os do modelo denso.
    memory = mlx_omnia.load(source.directory, quantize=_COARSE, cache=False)
    assert not source.cache.exists()

    cached = mlx_omnia.load(source.directory, quantize=_COARSE)

    assert relative_diff(_logits(memory), _logits(cached)) == 0.0


def test_an_interrupted_write_leaves_no_entry_the_next_load_would_accept(
    source: _Source,
) -> None:
    # mutação: `write_entry` gravando direto em `entry` (sem staging + rename) quebra —
    # a interrupção passa a deixar a entrada de pé, com o nome que o hit procura.
    written: list[Path] = []

    def killed(
        directory: Path,
        config: Mapping[str, object],
        weights: Mapping[str, mx.array],
        plan: QuantizationPlan,
    ) -> None:
        save_quantized(directory, config, weights, plan)
        written.append(directory)
        raise KeyboardInterrupt

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("mlx_omnia.engine.task.save_quantized", killed)
        with pytest.raises(KeyboardInterrupt):
            mlx_omnia.load(source.directory, quantize=_COARSE)

    assert written[0].name.startswith(".tmp-")
    assert _entries(source) == []

    mlx_omnia.load(source.directory, quantize=_COARSE)

    assert len(_entries(source)) == 1


def test_the_entry_carries_what_the_loader_reads_besides_the_weights(source: _Source) -> None:
    """A quantized entry has to load on its own: a symlink into the hub's snapshot dangles
    the day it is collected, so everything in `patterns` that is not weights or config is
    copied — the chat template included."""
    mlx_omnia.load(source.directory, quantize=_COARSE)
    (entry,) = _entries(source)
    assert (entry / "chat_template.jinja").read_text() == "{{ messages[0]['content'] }}"
    assert (entry / "tokenizer.json").exists()
