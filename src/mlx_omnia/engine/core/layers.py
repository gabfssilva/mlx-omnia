"""Shapes every ported architecture repeats, extracted on the second identical use.

Nothing here knows a model: `split_qkv` is a function over arrays (the checkpoint's
leaf names stay in the model file), `sorted_gather` takes the expert call as a
callback, and `SwiGLU` takes its activation. Every one of them reproduces the op
order of the copies it replaced — the parity fixtures are the contract.
"""

import math
from collections.abc import Callable, Sequence
from typing import Literal

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.kernels.qkv_rope.segmented import QkvStep, qkv_plan
from mlx_omnia.engine.core.mxcompat import gather_mm

# What `mx.quantize` accepts. Kept as primitives here so `core` stays below
# `mlx_omnia.engine.quant.quantization`, which is free to import this module.
type QuantizeMode = Literal["affine", "mxfp4", "mxfp8", "nvfp4"]

# The prefill reorder pays for itself once enough rows share an expert; below this
# many routed pairs the two argsorts cost more than the gather saves. Measured once
# for Qwen3-MoE and carried by every routed prefill since.
SORTED_GATHER_MIN = 64

_L2_EPS = 1e-6


def split_qkv(
    fused: mx.array, *, heads: int, kv_heads: int, head_dim: int
) -> tuple[mx.array, mx.array, mx.array]:
    """A qkv projection fused on the output axis, split back into per-head
    `[batch, heads, length, head_dim]` — the layout SDPA and rope both want."""
    length = fused.shape[1]
    query_width = heads * head_dim
    kv_width = kv_heads * head_dim
    q, k, v = (
        part.reshape(-1, length, count, head_dim).transpose(0, 2, 1, 3)
        for part, count in zip(
            mx.split(fused, [query_width, query_width + kv_width], axis=-1),
            (heads, kv_heads, kv_heads),
            strict=True,
        )
    )
    return q, k, v


class _Segmented(nn.Module):
    """Projections sharing the input, concatenated on the output axis. Three of them on a
    single-token step go through the planned kernel in one dispatch; the plan resolves
    once, on the first step — after load, when each leaf's format is final — and the
    concatenation stays as fallback and parity reference."""

    def __init__(self) -> None:
        super().__init__()
        self._step: QkvStep | None = None
        self._planned = False

    def _project(
        self, x: mx.array, parts: Sequence[nn.Linear | nn.QuantizedLinear]
    ) -> mx.array:
        if len(parts) == 3 and x.shape[1] == 1:
            if not self._planned:
                self._step = qkv_plan(parts[0], parts[1], parts[2])
                self._planned = True
            if self._step is not None and (fused := self._step(x)) is not None:
                return fused
        return mx.concatenate([part(x) for part in parts], axis=-1)


class SegmentedQKV(_Segmented):
    """q/k/v sharing the input and concatenated on the output axis, kept as three
    physical leaves instead of one matrix.

    Concatenating the weights requires a common format; sharing the input does not. Each
    segment carries its own `weight`, `scales`/`biases` and (once quantized) its own
    bits/group_size/mode, so a checkpoint with q in 4 bits, k in 8 and v dense loads
    without either widening or requantizing anything. The interface is `nn.Linear`'s —
    the model splits the concatenated output exactly as it does on the fused path.
    """

    def __init__(
        self, input_dims: int, *, queries: int, keys: int, values: int, bias: bool
    ) -> None:
        super().__init__()
        self.q_proj: nn.Linear | nn.QuantizedLinear = nn.Linear(input_dims, queries, bias=bias)
        self.k_proj: nn.Linear | nn.QuantizedLinear = nn.Linear(input_dims, keys, bias=bias)
        self.v_proj: nn.Linear | nn.QuantizedLinear = nn.Linear(input_dims, values, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self._project(x, (self.q_proj, self.k_proj, self.v_proj))


class SegmentedLinear(_Segmented):
    """The same split for a projection whose parts the checkpoint does not name q/k/v —
    a mixer that fuses four (qkv‖z‖b‖a), a gate‖up pair — kept as separate leaves when a
    mixed plan gives them no common matrix.

    The parts are numbered, not named: their order is the order the model splits the
    concatenated output in, and the loader that declined to fuse is what writes it down.
    """

    def __init__(self, input_dims: int, outputs: Sequence[int], *, bias: bool) -> None:
        super().__init__()
        self.parts: list[nn.Linear | nn.QuantizedLinear] = [
            nn.Linear(input_dims, output, bias=bias) for output in outputs
        ]

    def __call__(self, x: mx.array) -> mx.array:
        return self._project(x, self.parts)


def sorted_gather(
    x: mx.array,
    chosen: mx.array,
    *,
    k: int,
    hidden: int,
    apply: Callable[[mx.array, mx.array], mx.array],
) -> mx.array:
    """The routed MLP over `[..., T, hidden]` with the rows grouped by expert, so the
    gather streams each expert's weight once instead of once per token. A pure
    reorder: `apply` sees the same pairs, and the unsort puts them back before
    anything is summed. Returns `[..., T, k, hidden]`, unweighted."""
    tokens_count = x.size // hidden
    flat = chosen.reshape(-1)
    order = mx.argsort(flat)
    tokens = x.reshape(tokens_count, 1, hidden)[order // k]
    out = apply(tokens, flat[order])
    # The unsort needs argsort(order), and order is a permutation: its inverse is the
    # scatter inverse[order] = arange — one indexed write instead of a second sort.
    inverse = mx.put_along_axis(
        mx.zeros_like(order), order, mx.arange(order.size, dtype=order.dtype), axis=0
    )
    return out[inverse].reshape(*x.shape[:-1], k, hidden)


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


class MultiLinear(nn.Module):
    """One matrix per head, stacked `[heads, out, in]`, applied to every head at once.

    `SwitchLinear`'s shape without the gather: MLA's `kv_b_proj` split per head is a batch
    of matmuls over the head axis, not a selection of a few rows of it. `transpose=False`
    is the form that reads the stack as `[in, out]` — the key side wants the transpose of
    what the value side wants, and both come from the same split.
    """

    def __init__(self, input_dims: int, output_dims: int, heads: int) -> None:
        super().__init__()
        scale = math.sqrt(1.0 / input_dims)
        self.weight = mx.random.uniform(-scale, scale, (heads, output_dims, input_dims))

    def __call__(self, x: mx.array, *, transpose: bool = True) -> mx.array:
        return x @ (mx.swapaxes(self.weight, -2, -1) if transpose else self.weight)

    def to_quantized(
        self, group_size: int = 64, bits: int = 4, mode: QuantizeMode = "affine"
    ) -> "QuantizedMultiLinear":
        return QuantizedMultiLinear(self.weight, group_size=group_size, bits=bits, mode=mode)


class QuantizedMultiLinear(nn.Module):
    """`biases` is `None` outside affine, for the reason `QuantizedSwitchLinear` says."""

    def __init__(
        self, weight: mx.array, *, group_size: int, bits: int, mode: QuantizeMode = "affine"
    ) -> None:
        super().__init__()
        packed, scales, *rest = mx.quantize(weight, group_size=group_size, bits=bits, mode=mode)
        self.weight = packed
        self.scales = scales
        self.biases = rest[0] if rest else None
        self.group_size = group_size
        self.bits = bits
        self.mode = mode

    def __call__(self, x: mx.array, *, transpose: bool = True) -> mx.array:
        return mx.quantized_matmul(
            x,
            self.weight,
            scales=self.scales,
            biases=self.biases,
            transpose=transpose,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
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
        # Load-time quantization swaps these children for `nn.QuantizedLinear`, which is
        # not a `Linear` subclass — the declaration carries both so the strategies'
        # `isinstance` checks stay real checks.
        self.gate_up_proj: nn.Linear | nn.QuantizedLinear = nn.Linear(hidden, 2 * inner, bias=False)
        self.down_proj: nn.Linear | nn.QuantizedLinear = nn.Linear(inner, hidden, bias=False)
        self._activation = activation

    def __call__(self, x: mx.array) -> mx.array:
        gate, up = mx.split(self.gate_up_proj(x), [self.inner], axis=-1)
        return self.down_proj(self._activation(gate) * up)


class SwitchGLU(nn.Module):
    """Gate and up fused row-interleaved ([g0,u0,g1,u1,…]) at load: one gather reads both."""

    def __init__(self, experts: int, hidden: int, inner: int) -> None:
        super().__init__()
        self.gate_up_proj = SwitchLinear(experts, hidden, 2 * inner)
        self.down_proj = SwitchLinear(experts, inner, hidden)
        self.inner = inner

    def activate(self, fused: mx.array) -> mx.array:
        pairs = fused.reshape(*fused.shape[:-1], self.inner, 2)
        gated = pairs[..., 0]
        return gated * mx.sigmoid(gated) * pairs[..., 1]

    def __call__(self, tokens: mx.array, indices: mx.array, *, sorted_indices: bool) -> mx.array:
        projected = self.gate_up_proj(tokens, indices, sorted_indices=sorted_indices)
        return self.down_proj(self.activate(projected), indices, sorted_indices=sorted_indices)


class SharedMLP(nn.Module):
    """The always-on expert, gate and up kept as two leaves."""

    def __init__(self, hidden: int, inner: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden, inner, bias=False)
        self.up_proj = nn.Linear(hidden, inner, bias=False)
        self.down_proj = nn.Linear(inner, hidden, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swish(self.gate_proj(x)) * self.up_proj(x))


def l2norm(x: mx.array, scale: float = 1.0, *, dtype: mx.Dtype | None = None) -> mx.array:
    """transformers' l2norm: the eps sits inside the sum, not in a mean.

    `scale` folds a rule's `1/sqrt(Dk)` into the query, and `dtype` skips the trip back to
    the model dtype: the gated-delta kernel takes `q` pre-scaled and in float32, which is
    where the ops path does this same arithmetic anyway — rounding it to bfloat16 first is
    a rounding the ops path never pays, and on Qwen3.5-35B it costs 4.3e-2 against a
    3.9e-2 bound.
    """
    lifted = x.astype(mx.float32)
    inv = mx.rsqrt((lifted * lifted).sum(axis=-1, keepdims=True) + _L2_EPS)
    normed = lifted * inv if scale == 1.0 else lifted * inv * scale
    return normed.astype(x.dtype if dtype is None else dtype)
