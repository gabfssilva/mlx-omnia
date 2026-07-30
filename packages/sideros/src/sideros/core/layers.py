"""Shapes every ported architecture repeats, extracted on the second identical use.

Nothing here knows a model: `split_qkv` is a function over arrays (the checkpoint's
leaf names stay in the model file), `sorted_gather` takes the expert call as a
callback, and `SwiGLU` takes its activation. Every one of them reproduces the op
order of the copies it replaced — the parity fixtures are the contract.
"""

import math
from collections.abc import Callable
from typing import Literal

import mlx.core as mx
import mlx.nn as nn

from sideros.core.kernels.segmented_qkv import qkv_step
from sideros.core.mxcompat import gather_mm

# What `mx.quantize` accepts. Kept as primitives here so `core` stays below
# `sideros.quant.quantization`, which is free to import this module.
type QuantizeMode = Literal["affine", "mxfp4", "mxfp8", "nvfp4"]

# The prefill reorder pays for itself once enough rows share an expert; below this
# many routed pairs the two argsorts cost more than the gather saves. Measured once
# for Qwen3-MoE and carried by every routed prefill since.
SORTED_GATHER_MIN = 64


def split_qkv(
    fused: mx.array, *, heads: int, kv_heads: int, head_dim: int
) -> tuple[mx.array, mx.array, mx.array]:
    """A qkv projection fused on the output axis, split back into per-head
    `[1, heads, length, head_dim]` — the layout SDPA and rope both want."""
    length = fused.shape[1]
    query_width = heads * head_dim
    kv_width = kv_heads * head_dim
    q, k, v = (
        part.reshape(1, length, count, head_dim).transpose(0, 2, 1, 3)
        for part, count in zip(
            mx.split(fused, [query_width, query_width + kv_width], axis=-1),
            (heads, kv_heads, kv_heads),
            strict=True,
        )
    )
    return q, k, v


class SegmentedQKV(nn.Module):
    """q/k/v sharing the input and concatenated on the output axis, kept as three
    physical leaves instead of one matrix.

    Concatenating the weights requires a common format; sharing the input does not. Each
    segment carries its own `weight`, `scales`/`biases` and (once quantized) its own
    bits/group_size/mode, so a checkpoint with q in 4 bits, k in 8 and v dense loads
    without either widening or requantizing anything. The interface is `nn.Linear`'s —
    the model splits the concatenated output exactly as it does on the fused path. On a
    single-token step `qkv_step` runs the three segments in one dispatch; the
    concatenation stays as fallback and parity reference.
    """

    def __init__(
        self, input_dims: int, *, queries: int, keys: int, values: int, bias: bool
    ) -> None:
        super().__init__()
        self.q_proj: nn.Linear | nn.QuantizedLinear = nn.Linear(input_dims, queries, bias=bias)
        self.k_proj: nn.Linear | nn.QuantizedLinear = nn.Linear(input_dims, keys, bias=bias)
        self.v_proj: nn.Linear | nn.QuantizedLinear = nn.Linear(input_dims, values, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        if x.shape[1] == 1:
            fused = qkv_step(x, self.q_proj, self.k_proj, self.v_proj)
            if fused is not None:
                return fused
        return mx.concatenate([self.q_proj(x), self.k_proj(x), self.v_proj(x)], axis=-1)


def sorted_gather(
    x: mx.array,
    chosen: mx.array,
    *,
    k: int,
    hidden: int,
    apply: Callable[[mx.array, mx.array], mx.array],
) -> mx.array:
    """The routed MLP over `[1, T, hidden]` with the rows grouped by expert, so the
    gather streams each expert's weight once instead of once per token. A pure
    reorder: `apply` sees the same pairs, and the unsort puts them back before
    anything is summed. Returns `[1, T, k, hidden]`, unweighted."""
    length = x.shape[-2]
    flat = chosen.reshape(-1)
    order = mx.argsort(flat)
    tokens = x.reshape(length, 1, hidden)[order // k]
    out = apply(tokens, flat[order])
    return out[mx.argsort(order)].reshape(1, length, k, hidden)


class SwitchLinear(nn.Module):
    """The experts stacked one matrix per row: [experts, out, in]."""

    def __init__(self, experts: int, input_dims: int, output_dims: int) -> None:
        super().__init__()
        scale = math.sqrt(1.0 / input_dims)
        self.weight = mx.random.uniform(-scale, scale, (experts, output_dims, input_dims))

    def __call__(self, x: mx.array, indices: mx.array, *, sorted_indices: bool = False) -> mx.array:
        return gather_mm(
            x,
            mx.swapaxes(self.weight, -2, -1),
            rhs_indices=indices,
            sorted_indices=sorted_indices,
        )

    def to_quantized(
        self, group_size: int = 64, bits: int = 4, mode: QuantizeMode = "affine"
    ) -> "QuantizedSwitchLinear":
        return QuantizedSwitchLinear(
            self.weight, group_size=group_size, bits=bits, mode=mode
        )


class QuantizedSwitchLinear(nn.Module):
    """`biases` is `None` in every mode but affine: the MXFP/NVFP scales carry the
    exponent alone, and mlx drops a `None` attribute from the parameter tree, so
    `load_weights(strict=True)` stays exact for each mode."""

    def __init__(
        self, weight: mx.array, *, group_size: int, bits: int, mode: QuantizeMode = "affine"
    ) -> None:
        super().__init__()
        packed, scales, *rest = mx.quantize(
            weight, group_size=group_size, bits=bits, mode=mode
        )
        self.weight = packed
        self.scales = scales
        self.biases = rest[0] if rest else None
        self.group_size = group_size
        self.bits = bits
        self.mode = mode

    def __call__(self, x: mx.array, indices: mx.array, *, sorted_indices: bool = False) -> mx.array:
        return mx.gather_qmm(
            x,
            self.weight,
            self.scales,
            self.biases,
            rhs_indices=indices,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
            sorted_indices=sorted_indices,
        )


def swish(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)


class SwiGLU(nn.Module):
    """gate‖up concatenated on the output axis at load, split back by slice; the
    activation is the only thing the architectures disagree on (silu, or gelu-tanh
    on Gemma 3)."""

    def __init__(
        self, hidden: int, inner: int, activation: Callable[[mx.array], mx.array] = swish
    ) -> None:
        super().__init__()
        self.inner = inner
        self.gate_up_proj = nn.Linear(hidden, 2 * inner, bias=False)
        self.down_proj = nn.Linear(inner, hidden, bias=False)
        self._activation = activation

    def __call__(self, x: mx.array) -> mx.array:
        gate, up = mx.split(self.gate_up_proj(x), [self.inner], axis=-1)
        return self.down_proj(self._activation(gate) * up)
