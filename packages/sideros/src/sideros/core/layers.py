"""Shapes every ported architecture repeats, extracted on the second identical use.

Nothing here knows a model: `split_qkv` is a function over arrays (the checkpoint's
leaf names stay in the model file), `sorted_gather` takes the expert call as a
callback, and `SwiGLU` takes its activation. Every one of them reproduces the op
order of the copies it replaced — the parity fixtures are the contract.
"""

from collections.abc import Callable

import mlx.core as mx
import mlx.nn as nn

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
