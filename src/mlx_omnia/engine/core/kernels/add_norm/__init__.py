"""The residual join — add plus the rms_norm that reads it — one module per fusion,
one delegator.

A model block declares the norm leaf and how many tokens it will hand over, and
`AddRmsNorm` binds the fusion that declaration admits, or none, at construction time.
The model never names a kernel; a new fusion is a new module here, registered in
`_BUILDS`, and every family engages it.

The resolution table is (declaration x hidden): `fused.py` serves the single token
whose whole vector fits one threadgroup, `rows.py` serves any number of rows whose
`hidden / 4` threads tile, `default.py` serves everything else through the two ops, so
the delegator is total. The two kernels are separate numerical paths, not refinements
of one another — see `rows.py` — so which one a block gets follows from its
declaration, not from a tolerance.
"""

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.kernels.add_norm.default import DefaultAddRmsNorm
from mlx_omnia.engine.core.kernels.add_norm.fused import FusedAddRmsNorm
from mlx_omnia.engine.core.kernels.add_norm.kernel import AddRmsNormStrategy
from mlx_omnia.engine.core.kernels.add_norm.rows import RowsAddRmsNorm

__all__ = [
    "AddRmsNorm",
    "AddRmsNormStrategy",
    "DefaultAddRmsNorm",
    "FusedAddRmsNorm",
    "RowsAddRmsNorm",
]

# Order is preference: the first build that returns an instance wins; the default
# accepts everything, so resolution never fails.
_BUILDS = (FusedAddRmsNorm.build, RowsAddRmsNorm.build, DefaultAddRmsNorm.build)


class AddRmsNorm:
    """Resolves the strategy at construction and delegates; itself an
    `AddRmsNormStrategy`."""

    def __init__(self, leaf: nn.RMSNorm, *, tokens: int | None = None) -> None:
        self.strategy: AddRmsNormStrategy = next(
            built
            for build in _BUILDS
            if (built := build(leaf, tokens=tokens)) is not None
        )

    def __call__(self, x: mx.array, projected: mx.array) -> tuple[mx.array, mx.array]:
        return self.strategy(x, projected)
