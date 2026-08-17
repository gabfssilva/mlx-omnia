"""The universal screened-head strategy: the stock projection, unscreened.

`build` accepts every projection and every geometry, so it registers last and makes
the delegator total — `ScreenedHead` always resolves. The true logits row is trivially
its own screen: its argmax is its argmax. What defines it is universality, not the
absence of a kernel.
"""

from dataclasses import dataclass
from typing import Self

import mlx.core as mx

from mlx_omnia.engine.core.kernels.lm_head.kernel import HeadProjection, ScreenedHeadStrategy


@dataclass(frozen=True)
class DefaultScreenedHead(ScreenedHeadStrategy):
    projection: HeadProjection

    @classmethod
    def build(
        cls,
        projection: HeadProjection,
        *,
        weight: mx.array | None,
        refine: bool,
    ) -> Self:
        return cls(projection)

    def __call__(self, x: mx.array) -> mx.array:
        return self.projection(x)
