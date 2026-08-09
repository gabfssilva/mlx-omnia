"""The gate/up primitive's contract: what a strategy is and what a model declares.

The primitive: the declared activation over the chosen experts' gate‖up stacks,
(row [hidden], chosen [k]) -> [k, inner]. One module per format implements it; the
`GateUp` delegator in `__init__.py` resolves which one serves a given leaf and
declaration, once, at construction.
"""

from typing import Literal, Protocol

import mlx.core as mx

Activation = Literal["silu", "swiglu_oai"]


class GateUpStrategy(Protocol):
    """The declared activation over the chosen experts' gate‖up stacks:
    (row [hidden], chosen [k]) -> [k, inner]."""

    def __call__(self, row: mx.array, chosen: mx.array) -> mx.array: ...
