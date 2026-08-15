"""The universal gated delta strategy: the recurrence walked in ops, token by token.

`build` accepts every shape, so it registers last and makes the delegator total —
`GatedDelta` always resolves. This is also the parity reference the fused kernel is
held against: `delta_rule` is the same recurrence transformers'
`torch_recurrent_gated_delta_rule` runs, and both it and the strategy share the loop
below, so what the tests measure is what a model without the kernel executes.

The two differ only in convention. `delta_rule` keeps the model-side one — the log
decay, an unscaled `q` with the key heads already repeated, a `[B, Hv, Dk, Dv]` state.
The strategy takes the kernel's — the decay past the exp, `q` pre-scaled, unrepeated
q/k, a `[B, Hv, Dv, Dk]` state — and transposes into the loop and back out.
"""

import math
from dataclasses import dataclass
from typing import Self

import mlx.core as mx

from mlx_omnia.engine.core.kernels.gated_delta.kernel import GatedDeltaStrategy


def _recurrence(
    q32: mx.array,
    k32: mx.array,
    v32: mx.array,
    decay: mx.array,
    beta32: mx.array,
    state: mx.array,
) -> tuple[mx.array, mx.array]:
    """`[B, H, T, D]` inputs at float32, a `[B, H, Dk, Dv]` state, and a decay already
    broadcast to `[B, H, T, Dk|1, 1]` — one number per head or one per key channel."""
    outputs: list[mx.array] = []
    for t in range(q32.shape[2]):
        k_t = k32[:, :, t, :, None]
        state = state * decay[:, :, t]
        delta = (v32[:, :, t] - (state * k_t).sum(axis=-2)) * beta32[:, :, t, None]
        state = state + k_t * delta[..., None, :]
        outputs.append((state * q32[:, :, t, :, None]).sum(axis=-2))
    out = mx.stack(outputs, axis=2) if outputs else mx.zeros(v32.shape)
    return out, state


def delta_rule(
    q: mx.array, k: mx.array, v: mx.array, g: mx.array, beta: mx.array, state: mx.array
) -> tuple[mx.array, mx.array]:
    """The recurrent gated delta rule, token by token, entirely in float32.

    `q`, `k`, `v` are `[1, T, H, D]`, `g` the *log* decay and `beta` the write
    strength, `state` `[1, H, Dk, Dv]`. Same recurrence as transformers'
    `torch_recurrent_gated_delta_rule`, which is also what its chunked prefill path
    computes — the chunking is an associative rewrite, not a different rule.
    """
    scale = 1 / math.sqrt(q.shape[-1])
    q32 = q.astype(mx.float32).transpose(0, 2, 1, 3) * scale
    k32 = k.astype(mx.float32).transpose(0, 2, 1, 3)
    v32 = v.astype(mx.float32).transpose(0, 2, 1, 3)
    decay = mx.exp(g.astype(mx.float32)).transpose(0, 2, 1)
    beta32 = beta.astype(mx.float32).transpose(0, 2, 1)

    out, state = _recurrence(q32, k32, v32, decay[..., None, None], beta32, state)
    return out.transpose(0, 2, 1, 3).astype(v.dtype), state


@dataclass(frozen=True)
class DefaultGatedDelta(GatedDeltaStrategy):
    ratio: int

    @classmethod
    def build(
        cls,
        *,
        key_dim: int,
        key_heads: int,
        value_heads: int,
        value_dim: int,
        enabled: bool,
    ) -> Self:
        return cls(value_heads // key_heads)

    def __call__(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        g: mx.array,
        beta: mx.array,
        state: mx.array,
    ) -> tuple[mx.array, mx.array]:
        if self.ratio > 1:
            q = mx.repeat(q, self.ratio, axis=2)
            k = mx.repeat(k, self.ratio, axis=2)
        q32 = q.astype(mx.float32).transpose(0, 2, 1, 3)
        k32 = k.astype(mx.float32).transpose(0, 2, 1, 3)
        v32 = v.astype(mx.float32).transpose(0, 2, 1, 3)
        beta32 = beta.astype(mx.float32).transpose(0, 2, 1)
        g32 = g.astype(mx.float32)
        decay = (
            g32.transpose(0, 2, 1, 3)[..., None]
            if g32.ndim == 4
            else g32.transpose(0, 2, 1)[..., None, None]
        )
        out, advanced = _recurrence(
            q32, k32, v32, decay, beta32, state.transpose(0, 1, 3, 2)
        )
        return out.transpose(0, 2, 1, 3).astype(q.dtype), advanced.transpose(0, 1, 3, 2)
