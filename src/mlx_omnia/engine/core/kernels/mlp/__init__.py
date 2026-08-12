"""The T=1 unrouted SwiGLU MLP: one kernel module per format, one delegator.

A model block declares the primitive — the leaf and its activation — and `Mlp` binds
the specialization that leaf's weight format admits, or none, at construction time.
The model never names a format; a new format is a new module here, registered in
`_BUILDS`, and every family engages it.

The resolution table is (format x activation): the bf16 dense kernels serve silu in
two dispatches, gate‖up + SwiGLU then down + residual; the nvfp4 kernel serves the
gate/up half of a silu leaf over a halved scale plane and leaves the down projection
to the leaf. `default.py` serves everything else through the leaf's own `__call__`,
so the delegator is total: a model uses `Mlp` like any other layer.
"""

import mlx.core as mx

from mlx_omnia.engine.core.kernels.mlp.default import DefaultMlp
from mlx_omnia.engine.core.kernels.mlp.dense import DenseMlp
from mlx_omnia.engine.core.kernels.mlp.kernel import Activation, MlpStrategy
from mlx_omnia.engine.core.kernels.mlp.nvfp4 import Nvfp4Mlp
from mlx_omnia.engine.core.layers import SwiGLU

__all__ = [
    "Activation",
    "DefaultMlp",
    "DenseMlp",
    "Mlp",
    "MlpStrategy",
    "Nvfp4Mlp",
]

# Order is preference: the first build that returns an instance wins; the default
# accepts everything, so resolution never fails.
_BUILDS = (Nvfp4Mlp.build, DenseMlp.build, DefaultMlp.build)


class Mlp:
    """Resolves the strategy at construction and delegates; itself an `MlpStrategy`."""

    def __init__(
        self,
        leaf: SwiGLU,
        *,
        hidden: int,
        inner: int,
        activation: Activation = "silu",
    ) -> None:
        self.strategy: MlpStrategy = next(
            built
            for build in _BUILDS
            if (built := build(leaf, hidden=hidden, inner=inner, activation=activation))
            is not None
        )

    def __call__(self, row: mx.array, residual: mx.array) -> mx.array:
        return self.strategy(row, residual)
