"""The universal MLP strategy: the leaf called as the layer it already is.

`build` accepts every leaf and every declaration, so it registers last and makes the
delegator total — `Mlp` always resolves, and a model uses it like any other layer.
The projection runs through `SwiGLU.__call__` (mlx's gemv — dense or quantized, every
mode, and whatever activation the leaf was built with), then the residual add in ops.
What defines it is universality, not the absence of a kernel; its cost is the
op-boundary rounding and the dispatches the specialized strategies fuse away.
"""

from dataclasses import dataclass
from typing import Self

import mlx.core as mx

from mlx_omnia.engine.core.kernels.mlp.kernel import Activation
from mlx_omnia.engine.core.layers import SwiGLU


@dataclass(frozen=True)
class DefaultMlp:
    leaf: SwiGLU

    @classmethod
    def build(cls, leaf: SwiGLU, *, hidden: int, inner: int, activation: Activation) -> Self:
        return cls(leaf)

    def __call__(self, row: mx.array, residual: mx.array) -> mx.array:
        return self.leaf(row[None]).reshape(-1) + residual
