"""The residual-join primitive's contract: what a strategy is and what a model declares.

The primitive: the residual add and the rms_norm that reads it —
(x [..., hidden], projected [..., hidden]) -> (their sum, its norm), both keeping the
input's shape and dtype. One module per fusion implements it; the `AddRmsNorm`
delegator in `__init__.py` resolves which one serves a given norm leaf and
declaration, once, at construction.
"""

from typing import Protocol

import mlx.core as mx


class AddRmsNormStrategy(Protocol):
    """(x [..., hidden], projected [..., hidden]) -> (x + projected, its rms_norm)."""

    def __call__(self, x: mx.array, projected: mx.array) -> tuple[mx.array, mx.array]: ...
