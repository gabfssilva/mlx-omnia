"""The universal one-row projection: the leaf called as the layer it already is.

`build` accepts every leaf and every declaration, so it registers last and makes the
delegator total. The projection runs through the leaf's own `__call__` (mlx's dense and
quantized matmuls, every mode), then the declared epilogue in ops: the per-head gate
product before the projection, or softplus in fp32 after it. A ternary leaf is the one
exception — its `__call__` wraps the projection in the activation fake-quant and may
itself be routed through `Qmv`, so this arm runs the unpack-matmul-scale op chain the
kernel replaces instead of calling back into the leaf.
"""

from dataclasses import dataclass
from typing import Self

import mlx.core as mx
import mlx.nn as nn

from sideros.core.kernels.qmv.kernel import Epilogue, QmvLeaf, is_ternary
from sideros.core.kernels.qmv.ternary import ternary_matmul


@dataclass(frozen=True)
class DefaultQmv:
    leaf: QmvLeaf
    epilogue: Epilogue
    heads: int | None

    @classmethod
    def build(
        cls,
        leaf: QmvLeaf,
        *,
        kdim: int,
        rows: int,
        epilogue: Epilogue,
        heads: int | None,
    ) -> Self:
        return cls(leaf, epilogue, heads)

    def __call__(self, x: mx.array, gate: mx.array | None = None) -> mx.array:
        row = x
        if self.epilogue == "gate":
            assert gate is not None and self.heads is not None
            head_dim = x.shape[-1] // self.heads
            row = (x.reshape(*x.shape[:-1], self.heads, head_dim) * gate[..., None]).reshape(
                x.shape
            )
        if is_ternary(self.leaf):
            flat = row.reshape(-1, self.leaf.in_features)
            out = ternary_matmul(flat, self.leaf.weight, self.leaf.weight_scale).reshape(
                *row.shape[:-1], self.leaf.out_features
            )
        else:
            out = self.leaf(row)
        if self.epilogue == "softplus":
            return nn.softplus(out.astype(mx.float32)).astype(x.dtype)
        return out
