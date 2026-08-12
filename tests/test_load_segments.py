"""The qkv fusion when the three leaves do not share a format: q in 4 bits, k in 8 and v
dense have no common matrix, so the loader keeps three physical leaves under the fused
name and the module concatenates their outputs."""

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
from conftest import relative_diff

from mlx_omnia.engine.checkpoint import fuse_qkv, load_checkpoint
from mlx_omnia.engine.core.layers import SegmentedQKV

_ATTENTION = "model.layers.0.self_attn."
_HIDDEN = 128
_QUERIES = 32
_KEY_VALUES = 16


class _Attention(nn.Module):
    def __init__(self, *, bias: bool) -> None:
        super().__init__()
        self.qkv_proj = nn.Linear(_HIDDEN, _QUERIES + 2 * _KEY_VALUES, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.qkv_proj(x)


class _Layer(nn.Module):
    def __init__(self, *, bias: bool) -> None:
        super().__init__()
        self.self_attn = _Attention(bias=bias)


class _Trunk(nn.Module):
    def __init__(self, *, bias: bool) -> None:
        super().__init__()
        self.layers = [_Layer(bias=bias)]


class _Model(nn.Module):
    def __init__(self, *, bias: bool = False) -> None:
        super().__init__()
        self.model = _Trunk(bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.model.layers[0].self_attn(x)


@dataclass(frozen=True)
class _Config:
    tie_word_embeddings: bool


def _sources() -> tuple[mx.array, mx.array, mx.array]:
    return (
        mx.random.normal((_QUERIES, _HIDDEN)),
        mx.random.normal((_KEY_VALUES, _HIDDEN)),
        mx.random.normal((_KEY_VALUES, _HIDDEN)),
    )


def _packed(source: mx.array, *, bits: int) -> dict[str, mx.array]:
    weight, scales, biases = mx.quantize(source, group_size=64, bits=bits)
    return {"weight": weight, "scales": scales, "biases": biases}


def _mixed_weights(
    sources: tuple[mx.array, mx.array, mx.array],
) -> dict[str, mx.array]:
    """q at 4 bits, k at 8, v dense — three formats no concatenation can hold."""
    q, k, v = sources
    return {
        f"{_ATTENTION}{name}_proj.{suffix}": tensor
        for name, leaf in (
            ("q", _packed(q, bits=4)),
            ("k", _packed(k, bits=8)),
            ("v", {"weight": v}),
        )
        for suffix, tensor in leaf.items()
    }


def _load(model: _Model, weights: dict[str, mx.array]) -> _Model:
    return load_checkpoint(
        model, _Config(False), weights, [lambda w: fuse_qkv(w, 1)], None
    )


def _dequantized(weights: dict[str, mx.array], name: str, *, bits: int) -> mx.array:
    return mx.dequantize(
        weights[f"{_ATTENTION}qkv_proj.{name}_proj.weight"],
        weights[f"{_ATTENTION}qkv_proj.{name}_proj.scales"],
        weights[f"{_ATTENTION}qkv_proj.{name}_proj.biases"],
        group_size=64,
        bits=bits,
    )


def test_a_mixed_qkv_matches_the_three_projections_run_separately() -> None:
    # mutação: trocar a ordem do concat em SegmentedQKV (q, v, k) quebra; usar os bits de
    # q para o segmento k também. A referência sai de mx.dequantize + matmul denso, e o
    # valor sai do qmm de cada folha — caminhos diferentes sobre os mesmos tensores, então
    # a mutação move só um lado.
    mx.random.seed(0)
    sources = _sources()
    weights = _mixed_weights(sources)
    x = mx.random.normal((1, 4, _HIDDEN))

    model = _load(_Model(), weights)

    dense = [
        _dequantized(weights, "q", bits=4),
        _dequantized(weights, "k", bits=8),
        weights[f"{_ATTENTION}qkv_proj.v_proj.weight"],
    ]
    reference = mx.concatenate([x @ weight.T for weight in dense], axis=-1)
    # The floor is the rounding the quantization itself cost, measured here from the same
    # leaves: the loaded module has to land closer to the dequantized matmul than the
    # fp32 sources do.
    floor = relative_diff(mx.concatenate([x @ w.T for w in sources], axis=-1), reference)
    assert floor > 0

    assert relative_diff(model(x), reference) < floor


def test_a_mixed_qkv_keeps_one_physical_leaf_per_segment() -> None:
    # mutação: apagar `_build_segments` de attach_weights quebra — o load_weights estoura
    # nos nomes segmentados em vez de construir o módulo.
    mx.random.seed(0)
    weights = _mixed_weights(_sources())

    model = _load(_Model(), weights)

    projection = model.model.layers[0].self_attn.qkv_proj
    assert isinstance(projection, SegmentedQKV)
    q, k, v = projection.q_proj, projection.k_proj, projection.v_proj
    assert isinstance(q, nn.QuantizedLinear)
    assert isinstance(k, nn.QuantizedLinear)
    assert q.bits == 4
    assert k.bits == 8
    # v declares no format, so it stays dense inside a checkpoint that is not.
    assert not isinstance(v, nn.QuantizedLinear)
    assert isinstance(v, nn.Linear)


def test_a_mixed_qkv_carries_the_projection_bias_per_segment() -> None:
    # mutação: fixar bias=False na construção do SegmentedQKV quebra — o load_weights
    # estoura no nome `bias` que a árvore não declara.
    mx.random.seed(0)
    sources = _sources()
    weights = _mixed_weights(sources)
    biases = {
        name: mx.random.normal((out,))
        for name, out in (("q", _QUERIES), ("k", _KEY_VALUES), ("v", _KEY_VALUES))
    }
    for name, bias in biases.items():
        weights[f"{_ATTENTION}{name}_proj.bias"] = bias
    x = mx.zeros((1, 1, _HIDDEN))

    model = _load(_Model(bias=True), weights)

    # At x = 0 the projection is the bias alone, in the concatenated order.
    expected = mx.concatenate([biases[name] for name in ("q", "k", "v")]).reshape(1, 1, -1)
    assert relative_diff(model(x), expected) == 0.0


def test_a_homogeneous_qkv_still_takes_the_fused_path() -> None:
    # mutação: devolver sempre False em `_same_qkv_format` quebra — o módulo passa a ser
    # SegmentedQKV e a folha fundida some do dicionário.
    mx.random.seed(0)
    weights = {
        f"{_ATTENTION}{name}_proj.{suffix}": tensor
        for name, source in zip(("q", "k", "v"), _sources(), strict=True)
        for suffix, tensor in _packed(source, bits=4).items()
    }

    model = _load(_Model(), weights)

    projection = model.model.layers[0].self_attn.qkv_proj
    assert not isinstance(projection, SegmentedQKV)
    assert isinstance(projection, nn.QuantizedLinear)
    assert projection.weight.shape[0] == _QUERIES + 2 * _KEY_VALUES
