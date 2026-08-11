"""The qkv prologue's contract: what a strategy is and what a model declares.

The primitive: a hidden state to the heads attention consumes — the concatenated q/k/v
projection, the per-head q/k RMSNorm when the architecture has one, and the rotary —
`(x [batch, length, hidden], offset) -> (q [batch, heads, length, head_dim], k, v
[batch, kv_heads, length, head_dim])`. One module per fusion implements it; the
`QkvRope` delegator in `__init__.py` resolves which one serves a given declaration,
once, at construction.

The declaration is deliberately split between what produces the concatenated tensor
(`projection`, plus `segments` when the three leaves are physically separate) and what
rotates it (`rope` in ops, `base` for the in-kernel angle derivation, `angles` for the
table-driven one). A strategy reads only the parts its fusion needs and its `build`
returns None when a part it needs is missing.
"""

from collections.abc import Callable
from typing import Protocol

import mlx.core as mx
import mlx.nn as nn

type Leaf = nn.Linear | nn.QuantizedLinear
type Projection = Callable[[mx.array], mx.array]
type Rotation = Callable[[mx.array, int | mx.array], mx.array]
type Angles = Callable[[int | mx.array, int], mx.array]


class QkvRopeStrategy(Protocol):
    """`(x [batch, length, hidden], offset) -> (q, k, v)`, head-major."""

    def __call__(
        self, x: mx.array, offset: int | mx.array
    ) -> tuple[mx.array, mx.array, mx.array]: ...
