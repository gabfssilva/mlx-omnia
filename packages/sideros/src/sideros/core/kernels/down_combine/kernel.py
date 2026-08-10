"""The down/combine primitive's contract — `GateUp`'s sibling.

The primitive: each activation row down-projected by its expert, routing-weighted,
summed and residual-added in one dispatch — (act [n, inner], chosen [n], weights [n],
residual [hidden]) -> [hidden]. `n` is the routed top-k, plus one when a shared
expert rides the affine kernel's spare slot; growing `chosen`/`weights` by that row
is the caller's routing, not this package's. The down side rounds `(T)((T)dot·wt)`
per expert and accumulates in fp32, adding the residual at T.
"""

from typing import Literal, Protocol

import mlx.core as mx

Layout = Literal["interleaved", "blocked"]
"""How the gate/up half handed its result over. `interleaved` means `act` is already
silu(gate)·up, `[n, inner]`. `blocked` means it is the un-activated `[gate ‖ up]` block,
`[n, 2*inner]`, and the activation is this half's — the two halves declare the same
layout."""


class DownCombineStrategy(Protocol):
    """(act [n, inner], chosen [n], weights [n], residual [hidden]) -> [hidden]."""

    def __call__(
        self, act: mx.array, chosen: mx.array, weights: mx.array, residual: mx.array
    ) -> mx.array: ...
