"""The T=1 dense projection: one kernel module per format, one delegator.

The purest kernel-as-layer case — the facade behaves like calling the leaf. A model
declares the primitive (the leaf, its logical shape, and the epilogue: nothing, a
per-head gate product, or softplus) and `Qmv` binds the specialization the leaf's
quantization format admits for that declaration, or none, at construction time. The
model never names a format; a new format is a new module here, registered in `_BUILDS`,
and every family engages it.

The resolution table is (format x epilogue): ternary serves the bare projection off a
packed 1.58-bit leaf, nvfp4 serves the bare projection and the gate product off a
group-16 leaf, affine INT8 serves both of those and softplus. `default.py` serves
everything else through the leaf's own `__call__`, so the delegator is total: a model
uses `Qmv` like any other layer.

Two contracts a caller has to know. The gate is already activated — the gated kernels
multiply it in as it arrives, and the delegator's `preactivated` is fixed True; a model
that holds gate logits applies its activation first (or declares `softplus` on the
projection that produces them). And the kernel arms carry no additive bias, so a leaf
with one only ever resolves to the default.
"""

import mlx.core as mx

from sideros.core.kernels.qmv.default import DefaultQmv
from sideros.core.kernels.qmv.gated_nvfp4 import GatedNvfp4Qmv
from sideros.core.kernels.qmv.int8 import Int8Qmv
from sideros.core.kernels.qmv.kernel import Epilogue, QmvLeaf, QmvStrategy, TernaryLinear
from sideros.core.kernels.qmv.nvfp4 import Nvfp4Qmv
from sideros.core.kernels.qmv.softplus import SoftplusQmv
from sideros.core.kernels.qmv.ternary import TernaryQmv

__all__ = [
    "DefaultQmv",
    "Epilogue",
    "GatedNvfp4Qmv",
    "Int8Qmv",
    "Nvfp4Qmv",
    "Qmv",
    "QmvLeaf",
    "QmvStrategy",
    "SoftplusQmv",
    "TernaryLinear",
    "TernaryQmv",
]

# Order is preference: the first build that returns an instance wins; the default
# accepts everything, so resolution never fails.
_BUILDS = (
    TernaryQmv.build,
    GatedNvfp4Qmv.build,
    Nvfp4Qmv.build,
    SoftplusQmv.build,
    Int8Qmv.build,
    DefaultQmv.build,
)


class Qmv:
    """Resolves the strategy at construction and delegates; itself a `QmvStrategy`."""

    def __init__(
        self,
        leaf: QmvLeaf,
        *,
        kdim: int,
        rows: int,
        epilogue: Epilogue = "none",
        heads: int | None = None,
    ) -> None:
        self.strategy: QmvStrategy = next(
            built
            for build in _BUILDS
            if (built := build(leaf, kdim=kdim, rows=rows, epilogue=epilogue, heads=heads))
            is not None
        )

    def __call__(self, x: mx.array, gate: mx.array | None = None) -> mx.array:
        return self.strategy(x, gate)
