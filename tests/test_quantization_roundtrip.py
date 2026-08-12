import json
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from conftest import relative_diff

from mlx_omnia.engine.checkpoint import load_checkpoint, save_quantized
from mlx_omnia.engine.quant.quantization import (
    Affine,
    ByPath,
    QuantizationPlan,
    expand_plan,
    inventory,
    quantize_weights,
)

_COARSE = Affine(group_size=32, bits=4)
_FINE = Affine(group_size=32, bits=8)
_SOURCE: dict[str, object] = {"model_type": "tiny", "hidden_size": 64}


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
class _Config:
    tie_word_embeddings: bool


def _quantized(model: nn.Module, plan: QuantizationPlan) -> dict[str, mx.array]:
    dense = {f"{leaf.path}.weight": mx.random.normal(leaf.shape) for leaf in inventory(model)}
    return quantize_weights(dense, plan)


def _tensors(directory: Path) -> dict[str, mx.array]:
    loaded = mx.load(str(directory / "model.safetensors"))
    assert isinstance(loaded, dict)
    return loaded


def _reloaded(directory: Path) -> _Tiny:
    """Nothing of the config's block reaches the load: the width of each leaf comes off
    its own tensors."""
    return load_checkpoint(_Tiny(), _Config(False), _tensors(directory), [], None)


def _roundtrip(directory: Path) -> tuple[mx.array, mx.array]:
    """The same tensors reaching the same tree twice — once from memory, once through the
    file — so any difference in the logits belongs to the file."""
    mx.random.seed(0)
    plan = expand_plan(_Tiny(), _COARSE)
    weights = _quantized(_Tiny(), plan)
    memory = load_checkpoint(_Tiny(), _Config(False), dict(weights), [], None)
    save_quantized(directory, _SOURCE, weights, plan)

    ids = mx.array([[3, 1, 4, 1, 5]])
    return _reloaded(directory)(ids), memory(ids)


def test_a_saved_checkpoint_gives_back_the_logits_of_the_model_in_memory(tmp_path: Path) -> None:
    # mutação: gravar só as chaves `.weight` (writer iterando o plano no lugar do dict)
    # quebra — o load estoura por `scales` ausente.
    ours, memory = _roundtrip(tmp_path)

    assert relative_diff(ours, memory) == 0.0


def test_the_reloaded_checkpoint_predicts_the_same_greedy_ids(tmp_path: Path) -> None:
    # mutação: gravar as `scales` também sob a chave `.biases` (mesmo shape, carrega sem
    # erro) quebra — os pesos desquantizados mudam e os argmax deixam de casar.
    ours, memory = _roundtrip(tmp_path)

    assert mx.array_equal(mx.argmax(ours, axis=-1), mx.argmax(memory, axis=-1)).item()


def test_a_heterogeneous_config_reloads_each_leaf_at_its_own_width(tmp_path: Path) -> None:
    # mutação: gravar o formato padrão como bloco global no lugar do plano expandido
    # quebra — o bloco perde a chave `leaves` e o mapa por caminho junto; os `bits` das
    # folhas continuam vindo dos tensores, então a asserção do config é a que discrimina.
    mx.random.seed(0)
    plan = expand_plan(_Tiny(), ByPath(default=_COARSE, overrides={"mlp": _FINE}))
    save_quantized(tmp_path, _SOURCE, _quantized(_Tiny(), plan), plan)

    config = json.loads((tmp_path / "config.json").read_text())

    assert config["model_type"] == "tiny"
    assert config["quantization"]["leaves"] == {
        "attn": {"group_size": 32, "bits": 4},
        "embed": {"group_size": 32, "bits": 4},
        "head": {"group_size": 32, "bits": 4},
        "mlp": {"group_size": 32, "bits": 8},
    }
    assert _bits(_reloaded(tmp_path)) == {"attn": 4, "embed": 4, "head": 4, "mlp": 8}


def _bits(model: nn.Module) -> dict[str, int]:
    found: dict[str, int] = {}

    def visit(path: str, module: nn.Module) -> None:
        if isinstance(module, nn.QuantizedLinear | nn.QuantizedEmbedding):
            found[path] = module.bits

    model.apply_to_modules(visit)
    return found
