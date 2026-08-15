"""The Mamba2 mixer's T=1 middle — conv, dt, selective step, gated norm — as one
primitive.

The mixer's decode step between its two projections is a chain of small dependent
ops: the causal conv over the cached window, SiLU, the split into hidden/B/C, the
fp32 softplus of dt, the SSD recurrence against the cached state, and the grouped
gated RMSNorm. The primitive takes the in_proj row and both cache tensors and returns
the normed row ready for out_proj plus the slid window and advanced state —
(proj [inner + conv_dim + heads], window [kernel-1, conv_dim], state [H, Dh, Ds] fp32)
-> (normed [inner], window, state). The `MambaStep` delegator in `__init__.py`
resolves which module serves a given configuration, once, at construction.
"""

from typing import Protocol, runtime_checkable

import mlx.core as mx


@runtime_checkable
class MambaStepStrategy(Protocol):
    """(proj, window, state) -> (normed row for out_proj, slid window, new state)."""

    def __call__(
        self, proj: mx.array, window: mx.array, state: mx.array
    ) -> tuple[mx.array, mx.array, mx.array]: ...
