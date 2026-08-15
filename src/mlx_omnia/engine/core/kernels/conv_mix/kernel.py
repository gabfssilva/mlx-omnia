"""The gated short conv's T=1 step contract: what a strategy is and what a model declares.

The primitive: one token through in_proj's B, C, x split, the causal depthwise conv
over B*x against the cached window, and C's gate —
(x [hidden], weights [3*hidden, hidden], taps [hidden*kernel], window [kernel-1, hidden])
-> (gated [hidden], window [kernel-1, hidden]). The `ConvMix` delegator in `__init__.py`
resolves which module serves a given shape and declaration, once, at construction.
"""

from typing import Protocol, runtime_checkable

import mlx.core as mx


@runtime_checkable
class ConvMixStrategy(Protocol):
    """One token through the gated short conv, returning the gate's output and the
    slid window."""

    def __call__(
        self, x: mx.array, weights: mx.array, taps: mx.array, window: mx.array
    ) -> tuple[mx.array, mx.array]: ...
