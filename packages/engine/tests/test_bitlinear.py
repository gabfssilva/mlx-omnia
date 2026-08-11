# pyright: basic
"""The ternary matmul the ``bitlinear`` kernel replaces, against the op chain.

The replica is the matmul the kernel owns — unpack the uint8 ternary weights to
``{-1,0,1}`` floats, dot with the activation, multiply by the scalar ``weight_scale``
— over random activations and random ternary weights. No checkpoint and no
activation quantization: the kernel never sees the per-token int8 fake-quant (the
``BitLinear`` leaf runs it before the dispatch), so the replica does not either.

The kernel accumulates the dot in fp32 and rounds once to ``T`` at the write
(``static_cast<T>(sum * scale)``); the reference does the same (fp32 matmul, then
``astype(T)``). They differ only by accumulation order — the kernel splits the
``in_features`` reduction across 32 lanes, the reference is one blocked matmul — so
the bound is a handful of ``T`` ulps, not bit-exactness. The comparison side runs in
fp32.
"""

import mlx.core as mx
import numpy as np
import pytest
from conftest import relative_diff

from mlx_omnia.core.kernels.qmv.ternary import _SOURCE, bitlinear, bitlinear_applies
from mlx_omnia.core.mxcompat import metal_kernel

# (batch, in_features, out_features): a real q_proj shape (2560->2560) and a small
# shape that exercises the dispatch without a long reduction.
SHAPES = [(1, 2560, 2560), (4, 128, 512)]


def _bound(dtype: mx.Dtype) -> float:
    # fp32: accumulation-order slack across the 32-lane split-K. bf16: the single
    # round-to-T lands within a couple of bf16 ulps of the reference's.
    return 1e-5 if dtype == mx.float32 else 1e-2


def random_packed(out_features: int, in_features: int, seed: int) -> np.ndarray:
    """Random ternary weights packed 4-per-byte, LSB-first, field ``- 1`` -> ``{-1,0,1}``."""
    rng = np.random.default_rng(seed)
    ternary = rng.integers(-1, 2, size=(out_features, in_features), dtype=np.int8)
    packed = np.zeros((out_features // 4, in_features), dtype=np.uint8)
    for field in range(4):
        rows = slice(field * (out_features // 4), (field + 1) * (out_features // 4))
        packed |= (ternary[rows] + 1).astype(np.uint8) << (2 * field)
    return packed


def reference(
    x: mx.array, packed: np.ndarray, weight_scale: np.ndarray, dtype: mx.Dtype
) -> np.ndarray:
    """Unpack to logical order, matmul in fp32, scale, round once to T."""
    fields = [(packed >> (2 * i)) & 3 for i in range(4)]
    w = np.concatenate(fields, axis=0).astype(np.float32) - 1.0  # [out, in]
    y = x.astype(mx.float32) @ mx.array(w.T)
    y = (y * mx.array(weight_scale.astype(np.float32))).astype(dtype)
    return np.array(y.astype(mx.float32))


def _dispatch(
    source: str, name: str, x: mx.array, weight: mx.array, weight_scale: mx.array
) -> mx.array:
    n, in_features = x.shape
    out_features = weight.shape[0] * 4
    return metal_kernel(
        name=name,
        input_names=["x", "packed_weights", "weight_scale"],
        output_names=["out"],
        source=source,
    )(
        inputs=[x, weight, weight_scale],
        template=[
            ("T", weight_scale.dtype),
            ("invert_weight_scales", 0),
            ("in_features", in_features),
            ("out_features", out_features),
        ],
        grid=(32, n * out_features // 4, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(n, out_features)],
        output_dtypes=[weight_scale.dtype],
    )[0]


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("shape", SHAPES)
def test_matches_op_chain(shape: tuple[int, int, int], dtype: mx.Dtype) -> None:
    n, in_features, out_features = shape
    for seed in range(3):
        packed = random_packed(out_features, in_features, seed)
        scale_np = np.random.default_rng(seed + 13).uniform(0.01, 0.1, size=1).astype(np.float32)
        x = mx.array(np.random.default_rng(seed + 7).standard_normal((n, in_features))).astype(
            dtype
        )
        weight = mx.array(packed)
        weight_scale = mx.array(scale_np).astype(dtype)
        ours = bitlinear(x, weight, weight_scale)
        theirs = reference(x, packed, scale_np, dtype)
        assert ours.shape == (n, out_features)
        assert relative_diff(ours, mx.array(theirs)) < _bound(dtype)


def test_applies_predicate() -> None:
    weight = mx.zeros((64, 256), dtype=mx.uint8)
    assert bitlinear_applies(256, 256, weight)
    assert not bitlinear_applies(256, 255, weight)  # in_features not divisible by 128
    assert not bitlinear_applies(254, 256, weight)  # out not divisible by 4
    assert not bitlinear_applies(256, 256, weight.astype(mx.float32))  # not uint8


_MUTATIONS = {
    "drop the ternary offset (field maps to {0,1,2})": (
        "sum[0] += v[j] * ((w & 3) - 1);",
        "sum[0] += v[j] * (w & 3);",
    ),
    "drop the weight scale": (
        "static_cast<T>(sum[i] * scale);",
        "static_cast<T>(sum[i]);",
    ),
    "flip the scale direction": (
        "invert_weight_scales ? 1 / weight_scale[0] : weight_scale[0];",
        "invert_weight_scales ? weight_scale[0] : 1 / weight_scale[0];",
    ),
    "scramble the output write stride": (
        "out[batch_idx * out_features + row_idx + i * out_packs] =",
        "out[batch_idx * out_features + row_idx + i * 4] =",
    ),
}


@pytest.mark.parametrize("mutation", sorted(_MUTATIONS))
def test_mutations_break_the_matmul(mutation: str) -> None:
    old, new = _MUTATIONS[mutation]
    source = _SOURCE.replace(old, new)
    assert source != _SOURCE
    n, in_features, out_features = 4, 128, 512
    packed = random_packed(out_features, in_features, 0)
    scale_np = np.array([0.05], dtype=np.float32)
    x = mx.array(np.random.default_rng(7).standard_normal((n, in_features))).astype(mx.float32)
    weight = mx.array(packed)
    weight_scale = mx.array(scale_np)
    theirs = reference(x, packed, scale_np, mx.float32)
    broken = _dispatch(
        source, f"bitlinear_broken_{sorted(_MUTATIONS).index(mutation)}", x, weight, weight_scale
    )
    assert relative_diff(broken, mx.array(theirs)) > _bound(mx.float32)


def test_unmutated_replica_agrees_with_the_module() -> None:
    """The dispatch replica is only evidence if the untouched source reproduces the
    module's own entry point."""
    n, in_features, out_features = 4, 128, 512
    packed = random_packed(out_features, in_features, 0)
    scale_np = np.array([0.05], dtype=np.float32)
    x = mx.array(np.random.default_rng(7).standard_normal((n, in_features))).astype(mx.float32)
    weight = mx.array(packed)
    weight_scale = mx.array(scale_np)
    ours = bitlinear(x, weight, weight_scale)
    intact = _dispatch(_SOURCE, "bitlinear_intact", x, weight, weight_scale)
    assert relative_diff(ours, intact) == 0
