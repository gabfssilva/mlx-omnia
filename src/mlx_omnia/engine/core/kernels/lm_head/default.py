"""The universal greedy-head strategy: the stock projection followed by `argmax`.

`build` accepts every projection and every geometry, so it registers last and makes
the delegator total — `GreedyHead` always resolves. It reads the whole `[vocab,
hidden]` weight to produce a row it then reduces to one index; what defines it is
universality, not the absence of a kernel.
"""

from dataclasses import dataclass
from typing import Self

import mlx.core as mx

from mlx_omnia.engine.core.kernels.lm_head.kernel import HeadProjection


@dataclass(frozen=True)
class DefaultGreedyHead:
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
        return mx.argmax(self.projection(x), axis=-1)
