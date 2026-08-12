"""The ordinary MLX chain for the T=1 sparse step; always builds.

A transcription of the sparse block's own decode branch — the two gathered
projections with squared ReLU between, the fp32-accumulated combine, the shared
expert's own chain and its add — so the fused strategies have a parity reference whose
rounding boundaries are the model's own.
"""

from dataclasses import dataclass
from typing import Self

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.layers import QuantizedSwitchLinear, SwitchLinear

SharedPair = tuple[nn.Linear | nn.QuantizedLinear, nn.Linear | nn.QuantizedLinear]


def _squared_relu(x: mx.array) -> mx.array:
    activated = mx.maximum(x, 0)
    return activated * activated


@dataclass(frozen=True)
class DefaultMoeStep:
    fc1: SwitchLinear | QuantizedSwitchLinear
    fc2: SwitchLinear | QuantizedSwitchLinear
    shared: SharedPair | None

    @classmethod
    def build(
        cls,
        *,
        fc1: SwitchLinear | QuantizedSwitchLinear,
        fc2: SwitchLinear | QuantizedSwitchLinear,
        hidden: int,
        inner: int,
        shared: SharedPair | None = None,
    ) -> Self | None:
        return cls(fc1, fc2, shared)

    def __call__(self, x: mx.array, chosen: mx.array, weights: mx.array) -> mx.array:
        tokens = x.reshape(1, 1, 1, 1, -1)
        indices = chosen.reshape(1, 1, -1)
        up = self.fc1(tokens, indices, sorted_indices=False)
        routed = self.fc2(_squared_relu(up), indices, sorted_indices=False).squeeze(-2)
        mixed = (routed * weights.reshape(1, 1, -1, 1)).sum(axis=-2).astype(x.dtype)
        combined = mixed.reshape(-1)
        if self.shared is not None:
            up_proj, down_proj = self.shared
            combined = combined + down_proj(_squared_relu(up_proj(x)))
        return combined.reshape(x.shape)
