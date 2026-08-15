"""The dense MLP primitive's contract: what a strategy is and what a model declares.

The primitive: one token through an unrouted SwiGLU leaf with its residual added —
(row [hidden], residual [hidden]) -> [hidden]. One module per format implements it;
the `Mlp` delegator in `__init__.py` resolves which one serves a given leaf and
declaration, once, at construction. The activation is the only thing the
architectures disagree on, and the kernels bake silu.
"""

from typing import Literal, Protocol, runtime_checkable

import mlx.core as mx

Activation = Literal["silu", "gelu_tanh"]


@runtime_checkable
class MlpStrategy(Protocol):
    """One token through the unrouted SwiGLU leaf, residual added:
    (row [hidden], residual [hidden]) -> [hidden]."""

    def __call__(self, row: mx.array, residual: mx.array) -> mx.array: ...
