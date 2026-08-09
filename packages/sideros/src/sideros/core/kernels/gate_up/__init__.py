"""The T=1 routed gate/up primitive: one kernel module per format, one delegator.

A model block declares the primitive — activation, clamp, projection bias — and
`GateUp` binds the specialization the leaf's quantization format admits for that
declaration, or none, at construction time. The model never names a format; a new
format is a new module here, registered in `_BUILDS`, and every family engages it.

The resolution table is (format x activation): affine serves silu (with the optional
clamp-before-activation order), nvfp4 serves unclamped silu, mxfp4 serves only
swiglu_oai. `default.py` serves everything else through the leaf's own `__call__`,
so the delegator is total: a model uses `GateUp` like any other layer.
"""

import mlx.core as mx

from sideros.core.kernels.gate_up.affine import AffineGateUp
from sideros.core.kernels.gate_up.default import DefaultGateUp
from sideros.core.kernels.gate_up.kernel import Activation, GateUpStrategy
from sideros.core.kernels.gate_up.mxfp4 import Mxfp4GateUp
from sideros.core.kernels.gate_up.nvfp4 import Nvfp4GateUp
from sideros.core.layers import QuantizedSwitchLinear, SwitchLinear

__all__ = [
    "Activation",
    "AffineGateUp",
    "DefaultGateUp",
    "GateUp",
    "GateUpStrategy",
    "Mxfp4GateUp",
    "Nvfp4GateUp",
]

# Order is preference: the first build that returns an instance wins; the default
# accepts everything, so resolution never fails.
_BUILDS = (Nvfp4GateUp.build, Mxfp4GateUp.build, AffineGateUp.build, DefaultGateUp.build)


class GateUp:
    """Resolves the strategy at construction and delegates; itself a
    `GateUpStrategy`."""

    def __init__(
        self,
        leaf: SwitchLinear | QuantizedSwitchLinear,
        *,
        hidden: int,
        inner: int,
        activation: Activation = "silu",
        limit: float | None = None,
        bias: mx.array | None = None,
    ) -> None:
        self.strategy: GateUpStrategy = next(
            built
            for build in _BUILDS
            if (
                built := build(
                    leaf,
                    hidden=hidden,
                    inner=inner,
                    activation=activation,
                    limit=limit,
                    bias=bias,
                )
            )
            is not None
        )

    def __call__(self, row: mx.array, chosen: mx.array) -> mx.array:
        return self.strategy(row, chosen)
