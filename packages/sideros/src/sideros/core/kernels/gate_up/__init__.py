"""The T=1 routed gate/up primitive: one kernel module per format, one delegator.

A model block declares the primitive — activation, clamp, projection bias — and
`GateUp` binds the specialization the leaf's quantization format admits for that
declaration, or none, at construction time. The model never names a format; a new
format is a new module here, registered in `_BUILDS`, and every family engages it.

The resolution table is (format x activation): affine serves silu (with the optional
clamp-before-activation order), nvfp4 serves unclamped silu, mxfp4 serves only
swiglu_oai, dense serves the `blocked` layout, and packed nvfp4 serves the declaration
that hands routing to the kernel. `default.py` serves everything else through the leaf's
own `__call__`, so the delegator is total: a model uses `GateUp` like any other layer.

Three declarations widen the primitive rather than the epilogue. `layout` says how the leaf
stacks gate and up, and `blocked` moves the activation to the down/combine half. `routing`,
when it is an `OrdinalRouting`, says the second `__call__` argument carries the router's
ordinal keys instead of expert indices, which is what lets a strategy select the experts
inside its own kernel. `shared` is a leaf beside the stack — a shared expert the last
chosen slot addresses, in its own quantization format — served by the affine kernel's
spare slot and by the default strategy's own layer call; every other format refuses it.
"""

import mlx.core as mx
import mlx.nn as nn

from sideros.core.kernels.gate_up.affine import AffineGateUp
from sideros.core.kernels.gate_up.default import DefaultGateUp
from sideros.core.kernels.gate_up.dense import DenseGateUp
from sideros.core.kernels.gate_up.kernel import (
    Activation,
    GateUpStrategy,
    Layout,
    OrdinalRouting,
)
from sideros.core.kernels.gate_up.mxfp4 import Mxfp4GateUp
from sideros.core.kernels.gate_up.nvfp4 import Nvfp4GateUp
from sideros.core.kernels.gate_up.nvfp4_packed import Nvfp4PackedGateUp
from sideros.core.layers import QuantizedSwitchLinear, SwitchLinear

__all__ = [
    "Activation",
    "AffineGateUp",
    "DefaultGateUp",
    "DenseGateUp",
    "GateUp",
    "GateUpStrategy",
    "Layout",
    "Mxfp4GateUp",
    "Nvfp4GateUp",
    "Nvfp4PackedGateUp",
    "OrdinalRouting",
]

# Order is preference: the first build that returns an instance wins; the default
# accepts everything, so resolution never fails.
_BUILDS = (
    Nvfp4PackedGateUp.build,
    Nvfp4GateUp.build,
    Mxfp4GateUp.build,
    AffineGateUp.build,
    DenseGateUp.build,
    DefaultGateUp.build,
)


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
        layout: Layout = "interleaved",
        routing: OrdinalRouting | None = None,
        shared: nn.Linear | nn.QuantizedLinear | None = None,
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
                    layout=layout,
                    routing=routing,
                    shared=shared,
                )
            )
            is not None
        )

    def __call__(self, row: mx.array, chosen: mx.array) -> mx.array:
        return self.strategy(row, chosen)
