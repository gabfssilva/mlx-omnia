from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_omnia.engine.checkpoint import fuse_qkv, load_checkpoint
from tests.conftest import relative_diff

_ATTENTION = "model.layers.0.self_attn."


class _Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qkv_proj = nn.Linear(128, 32, bias=False)


class _Layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _Attention()


class _Trunk(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = [_Layer()]


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _Trunk()


@dataclass(frozen=True)
class _Config:
    tie_word_embeddings: bool


class _Mixed(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.packed = nn.Linear(128, 16, bias=False)
        self.dense = nn.Linear(128, 16, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.dense(self.packed(x))


def _affine(out: int, inp: int, *, group_size: int, bits: int) -> dict[str, mx.array]:
    weight, scales, biases = mx.quantize(
        mx.random.normal((out, inp)), group_size=group_size, bits=bits
    )
    return {"weight": weight, "scales": scales, "biases": biases}


def test_a_format_tensor_left_without_its_weight_names_the_leaf() -> None:
    # mutação: remover `_reject_orphan_formats` de attach_weights quebra — o erro passa a
    # vir do load_weights, sem o caminho da folha que perdeu as escalas.
    mx.random.seed(0)
    weights = {
        f"{_ATTENTION}{name}_proj.{suffix}": tensor
        for name in ("q", "k", "v")
        for suffix, tensor in _affine(16, 128, group_size=64, bits=4).items()
    }
    # v ships its scales and biases without the packed weight, so the fusion cannot move
    # the trio and leaves the two format tensors orphaned. (v with weight alone is not a
    # broken checkpoint any more — it is q/k packed next to a dense v, which loads
    # segmented.)
    del weights[f"{_ATTENTION}v_proj.weight"]

    with pytest.raises(ValueError, match=rf"{_ATTENTION}v_proj carries biases without weight"):
        load_checkpoint(
            _Model(), _Config(False), weights, [lambda w: fuse_qkv(w, 1)], None
        )


def test_a_dense_leaf_inside_a_quantized_checkpoint_stays_dense() -> None:
    # mutação: quantizar toda folha (predicate constante `True`) quebra — `dense` viraria
    # QuantizedLinear e o load_weights estouraria por scales ausentes. A referência sai do
    # tensor denso do dicionário, não do módulo, então a mutação move só um lado.
    mx.random.seed(0)
    dense = mx.random.normal((16, 128))
    weights = {
        f"packed.{suffix}": tensor
        for suffix, tensor in _affine(16, 128, group_size=64, bits=4).items()
    }
    weights["dense.weight"] = dense

    model = load_checkpoint(_Mixed(), _Config(False), weights, [], None)

    assert isinstance(model.packed, nn.QuantizedLinear)
    assert not isinstance(model.dense, nn.QuantizedLinear)
    assert isinstance(model.dense, nn.Linear)
    assert relative_diff(model.dense.weight, dense) == 0.0


def test_an_mxfp4_leaf_loads_in_its_own_mode() -> None:
    # mutação: fixar `mode="affine"` no dict que `_quantization` devolve quebra — o módulo
    # passa a desempacotar os mesmos bits como affine, enquanto a referência continua saindo
    # de `mx.dequantize(..., mode="mxfp4")`.
    mx.random.seed(0)
    source = mx.random.normal((16, 128))
    # mxfp4 returns no biases; the bundled stub always types three.
    packed, scales, *_ = mx.quantize(source, group_size=32, bits=4, mode="mxfp4")
    weights = {
        "packed.weight": packed,
        "packed.scales": scales,
        "dense.weight": mx.random.normal((16, 128)),
    }
    x = mx.random.normal((1, 4, 128))

    model = load_checkpoint(_Mixed(), _Config(False), weights, [], None)

    quantized = model.packed
    assert isinstance(quantized, nn.QuantizedLinear)
    assert quantized.mode == "mxfp4"
    reference = x @ mx.dequantize(packed, scales, group_size=32, bits=4, mode="mxfp4").T
    floor = relative_diff(x @ source.T, reference)
    assert floor > 0
    assert relative_diff(quantized(x), reference) < floor
