"""The one-row projection primitive's contract: what a strategy is and what a model
declares.

The primitive: a single activation row through the leaf's weight stack, x [..., kdim]
-> [..., rows], plus the epilogue the model declares — none, a per-head gate product
(the gate arrives already activated), or softplus over the projection. One module per
format implements it; the `Qmv` delegator in `__init__.py` resolves which one serves a
given leaf and declaration, once, at construction.

A ternary leaf is not an `nn.Linear`: its weight is a packed uint8 table and its
`__call__` wraps the projection in the activation fake-quant, so the facade stands for
the matmul inside that leaf, not for the leaf. Every other leaf's facade is the leaf.
"""

from typing import Literal, Protocol, TypeGuard

import mlx.core as mx
import mlx.nn as nn

Epilogue = Literal["none", "gate", "softplus"]


class TernaryLinear(Protocol):
    """A packed 1.58-bit leaf: `weight` [rows // 4, kdim] uint8 and a scalar
    `weight_scale`."""

    in_features: int
    out_features: int
    weight: mx.array
    weight_scale: mx.array

    def __call__(self, x: mx.array) -> mx.array: ...


QmvLeaf = nn.Linear | nn.QuantizedLinear | TernaryLinear


def is_ternary(leaf: QmvLeaf) -> TypeGuard[TernaryLinear]:
    """`isinstance` against the protocol is unusable here: mlx's `Module` keeps its
    attributes in the dict it subclasses, and the runtime protocol check reads them
    statically. The union has three arms, so excluding the two mlx ones decides it."""
    return not isinstance(leaf, (nn.Linear, nn.QuantizedLinear))


class QmvStrategy(Protocol):
    """x [..., kdim] through the leaf's stack, with the declared epilogue:
    (x, gate) -> [..., rows]. `gate` is read only by the gate epilogue."""

    def __call__(self, x: mx.array, gate: mx.array | None = None) -> mx.array: ...
