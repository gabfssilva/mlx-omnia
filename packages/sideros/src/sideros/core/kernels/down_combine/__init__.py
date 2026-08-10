"""The T=1 routed down/combine primitive: one kernel module per format, one delegator.

`GateUp`'s sibling. The declaration carries what varies on this half: the affine
kernel takes an optional shared expert's down leaf riding the spare slot (its tensors
must match the routed stack's bits and group), and the mxfp4 kernel takes the
mandatory per-row projection bias. The nvfp4 kernel carries neither; its packed sibling
requires the shared leaf, whose down projection its unrouted slot always runs. `default.py`
serves everything else through the leaf's own `__call__`, so the delegator is total:
a model uses `DownCombine` like any other layer.

`layout` is the same declaration `GateUp` takes and the two halves state it together: under
`blocked` the activation block arrives un-activated, `[n, 2*inner]`, and the SwiGLU is this
half's.
"""

import mlx.core as mx
import mlx.nn as nn

from sideros.core.kernels.down_combine.affine import AffineDownCombine
from sideros.core.kernels.down_combine.default import DefaultDownCombine
from sideros.core.kernels.down_combine.dense import DenseDownCombine
from sideros.core.kernels.down_combine.kernel import DownCombineStrategy, Layout
from sideros.core.kernels.down_combine.mxfp4 import Mxfp4DownCombine
from sideros.core.kernels.down_combine.nvfp4 import Nvfp4DownCombine
from sideros.core.kernels.down_combine.nvfp4_packed import Nvfp4PackedDownCombine
from sideros.core.layers import QuantizedSwitchLinear, SwitchLinear

__all__ = [
    "AffineDownCombine",
    "DefaultDownCombine",
    "DenseDownCombine",
    "DownCombine",
    "DownCombineStrategy",
    "Layout",
    "Mxfp4DownCombine",
    "Nvfp4DownCombine",
    "Nvfp4PackedDownCombine",
]

# Order is preference: the first build that returns an instance wins; the default
# accepts everything, so resolution never fails.
_BUILDS = (
    Nvfp4PackedDownCombine.build,
    Nvfp4DownCombine.build,
    Mxfp4DownCombine.build,
    AffineDownCombine.build,
    DenseDownCombine.build,
    DefaultDownCombine.build,
)


class DownCombine:
    """Resolves the strategy at construction and delegates; itself a
    `DownCombineStrategy`."""

    def __init__(
        self,
        leaf: SwitchLinear | QuantizedSwitchLinear,
        *,
        hidden: int,
        inner: int,
        bias: mx.array | None = None,
        shared: nn.Linear | nn.QuantizedLinear | None = None,
        layout: Layout = "interleaved",
    ) -> None:
        self.strategy: DownCombineStrategy = next(
            built
            for build in _BUILDS
            if (
                built := build(
                    leaf, hidden=hidden, inner=inner, bias=bias, shared=shared, layout=layout
                )
            )
            is not None
        )

    def __call__(
        self, act: mx.array, chosen: mx.array, weights: mx.array, residual: mx.array
    ) -> mx.array:
        return self.strategy(act, chosen, weights, residual)
