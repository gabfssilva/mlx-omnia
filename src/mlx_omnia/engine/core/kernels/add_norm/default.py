"""The universal residual-join strategy: the add and the norm as two ops.

`build` accepts every leaf and every declaration, so it registers last and makes the
delegator total — `AddRmsNorm` always resolves, and a model uses it like any other
layer. The sum is materialized between the two ops and `mx.fast.rms_norm` reads it
back; its cost is that round trip and the extra dispatch the fused strategies remove.
"""

from dataclasses import dataclass
from typing import Self

import mlx.core as mx
import mlx.nn as nn


@dataclass(frozen=True)
class DefaultAddRmsNorm:
    weight: mx.array
    eps: float

    @classmethod
    def build(cls, leaf: nn.RMSNorm, *, tokens: int | None) -> Self:
        return cls(leaf.weight, leaf.eps)

    def __call__(self, x: mx.array, projected: mx.array) -> tuple[mx.array, mx.array]:
        added = x + projected
        return added, mx.fast.rms_norm(added, self.weight, self.eps)
