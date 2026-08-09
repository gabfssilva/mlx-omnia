"""The down/combine primitive's contract — `GateUp`'s sibling.

The primitive: each activation row down-projected by its expert, routing-weighted,
summed and residual-added in one dispatch — (act [n, inner], chosen [n], weights [n],
residual [hidden]) -> [hidden]. `n` is the routed top-k, plus one when a shared
expert rides the affine kernel's spare slot; growing `chosen`/`weights` by that row
is the caller's routing, not this package's. The down side rounds `(T)((T)dot·wt)`
per expert and accumulates in fp32, adding the residual at T.
"""

from typing import Protocol

import mlx.core as mx


class DownCombineStrategy(Protocol):
    """(act [n, inner], chosen [n], weights [n], residual [hidden]) -> [hidden]."""

    def __call__(
        self, act: mx.array, chosen: mx.array, weights: mx.array, residual: mx.array
    ) -> mx.array: ...
