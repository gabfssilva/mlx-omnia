"""The gate/up primitive's contract: what a strategy is and what a model declares.

The primitive: the declared activation over the chosen experts' gate‖up stacks,
(row [hidden], chosen [k]) -> [k, inner]. One module per format implements it; the
`GateUp` delegator in `__init__.py` resolves which one serves a given leaf and
declaration, once, at construction.
"""

from dataclasses import dataclass
from typing import Literal, Protocol

import mlx.core as mx

Activation = Literal["silu", "swiglu_oai"]

Layout = Literal["interleaved", "blocked"]
"""How the leaf stacks gate and up. `interleaved` is `[g0, u0, g1, u1, ...]` on the output
axis and the primitive is the whole step: the activation is this half's. `blocked` is
`[gate ‖ up]`, and the pair block travels un-activated to the down/combine half, which owns
the activation — the two halves are declared together or the block is read as a result."""


@dataclass(frozen=True)
class OrdinalRouting:
    """Declares that the second `__call__` argument carries the router's ordinal keys
    ([experts] uint32, ascending order being descending routing score) instead of expert
    indices, and that the strategy decides which `top_k` experts win."""

    top_k: int

    def indices(self, keys: mx.array) -> mx.array:
        """The keys resolved to expert indices, best first. `mx.argsort` is stable and
        ascending, so equal ordinals go to the lower expert index — the tournament's rule."""
        return mx.argsort(keys)[: self.top_k].astype(mx.uint32)


class GateUpStrategy(Protocol):
    """The declared activation over the chosen experts' gate‖up stacks:
    (row [hidden], chosen [k]) -> [k, inner]."""

    def __call__(self, row: mx.array, chosen: mx.array) -> mx.array: ...
