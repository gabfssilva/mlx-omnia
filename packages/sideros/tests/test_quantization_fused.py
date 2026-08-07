import re

import mlx.core as mx
import pytest

from sideros.checkpoint import fuse_qkv, interleave_gate_up
from sideros.models.gpt2.checkpoint import _transpose_conv1d
from sideros.quant.quantization import Affine, QuantizationPlan, quantize_weights

_FORMAT = Affine(group_size=64, bits=4)
_SUFFIXES = ("weight", "scales", "biases")
_ATTENTION = "model.layers.0.self_attn."
_SWITCH = "model.layers.0.mlp.switch_mlp."


def _reference(weight: mx.array) -> tuple[mx.array, mx.array, mx.array]:
    return mx.quantize(weight, group_size=_FORMAT.group_size, bits=_FORMAT.bits)


def _dense(parts: dict[str, mx.array], prefix: str) -> dict[str, mx.array]:
    return {f"{prefix}{name}_proj.weight": weight for name, weight in parts.items()}


def _plan(*paths: str) -> QuantizationPlan:
    return {path: _FORMAT for path in paths}


def test_quantizing_around_the_qkv_fusion_gives_the_same_codes_per_segment() -> None:
    # mutação: concatenar q, v, k (ordem trocada) em fuse_qkv quebra.
    mx.random.seed(0)
    parts = {
        "q": mx.random.normal((16, 128)),
        "k": mx.random.normal((8, 128)),
        "v": mx.random.normal((8, 128)),
    }
    fused = f"{_ATTENTION}qkv_proj"

    before = fuse_qkv(
        quantize_weights(
            _dense(parts, _ATTENTION),
            _plan(*(f"{_ATTENTION}{name}_proj" for name in parts)),
        ),
        1,
    )
    after = quantize_weights(fuse_qkv(_dense(parts, _ATTENTION), 1), _plan(fused))

    for suffix in _SUFFIXES:
        assert mx.array_equal(before[f"{fused}.{suffix}"], after[f"{fused}.{suffix}"]).item()

    row = 0
    for weight in parts.values():
        segment = slice(row, row + weight.shape[0])
        for suffix, expected in zip(_SUFFIXES, _reference(weight), strict=True):
            assert mx.array_equal(after[f"{fused}.{suffix}"][segment], expected).item()
        row += weight.shape[0]


def test_quantizing_around_the_gate_up_interleave_keeps_the_row_pairs() -> None:
    # mutação: axis=2 → axis=1 em interleave_gate_up quebra (blocos no lugar de pares).
    mx.random.seed(0)
    parts = {"gate": mx.random.normal((2, 8, 128)), "up": mx.random.normal((2, 8, 128))}
    fused = f"{_SWITCH}gate_up_proj"

    before = interleave_gate_up(
        quantize_weights(
            _dense(parts, _SWITCH),
            _plan(*(f"{_SWITCH}{name}_proj" for name in parts)),
        ),
        1,
    )
    after = quantize_weights(interleave_gate_up(_dense(parts, _SWITCH), 1), _plan(fused))

    for suffix in _SUFFIXES:
        assert mx.array_equal(before[f"{fused}.{suffix}"], after[f"{fused}.{suffix}"]).item()

    for offset, name in enumerate(("gate", "up")):
        for suffix, expected in zip(_SUFFIXES, _reference(parts[name]), strict=True):
            assert mx.array_equal(after[f"{fused}.{suffix}"][:, offset::2], expected).item()


def test_quantizing_the_expert_stack_matches_quantizing_each_expert() -> None:
    # mutação: em AffineRTN.quantize, passar `weight.swapaxes(-1, -2)` ao mx.quantize quebra.
    mx.random.seed(0)
    stack = mx.random.normal((4, 16, 128))
    path = f"{_SWITCH}down_proj"

    quantized = quantize_weights({f"{path}.weight": stack}, _plan(path))

    for expert in range(stack.shape[0]):
        for suffix, expected in zip(_SUFFIXES, _reference(stack[expert]), strict=True):
            assert mx.array_equal(quantized[f"{path}.{suffix}"][expert], expected).item()


def test_the_conv1d_transpose_lands_before_the_grouping() -> None:
    # mutação: remover o `.T` de _transpose_conv1d quebra.
    mx.random.seed(0)
    raw = mx.random.normal((128, 128))
    path = "h.0.attn.c_attn"

    quantized = quantize_weights(_transpose_conv1d({f"{path}.weight": raw}), _plan(path))

    for suffix, expected in zip(_SUFFIXES, _reference(raw.T), strict=True):
        assert mx.array_equal(quantized[f"{path}.{suffix}"], expected).item()
    # The one fusion that moves the last axis: grouping the raw [in, out] matrix produces
    # other codes at the same shape.
    assert not mx.array_equal(quantized[f"{path}.weight"], _reference(raw)[0]).item()


def test_quantizing_a_leaf_that_is_already_quantized_raises_naming_it() -> None:
    # mutação: remover a checagem `packed` quebra — o RTN estoura sem nomear a folha.
    mx.random.seed(0)
    path = f"{_ATTENTION}qkv_proj"

    quantized = quantize_weights({f"{path}.weight": mx.random.normal((32, 128))}, _plan(path))

    with pytest.raises(ValueError, match=re.escape(f"['{path}']")):
        quantize_weights(quantized, _plan(path))


def test_a_fused_leaf_outside_the_plan_stays_dense_beside_its_neighbours() -> None:
    # mutação: em quantize_weights, iterar sobre `weights` no lugar do plano quebra.
    mx.random.seed(0)
    attention = {name: mx.random.normal((16, 128)) for name in ("q", "k", "v")}
    mlp = {name: mx.random.normal((2, 8, 128)) for name in ("gate", "up")}
    weights = interleave_gate_up(
        fuse_qkv(_dense(attention, _ATTENTION) | _dense(mlp, _SWITCH), 1), 1
    )

    quantized = quantize_weights(weights, _plan(f"{_ATTENTION}qkv_proj"))

    assert quantized[f"{_ATTENTION}qkv_proj.weight"].dtype == mx.uint32
    assert quantized[f"{_SWITCH}gate_up_proj.weight"].dtype == mx.float32
    assert f"{_SWITCH}gate_up_proj.scales" not in quantized
