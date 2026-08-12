"""The greedy output head: one strategy per head shape, one delegator.

The primitive is the greedy STEP — `(x [..., hidden]) -> token id [...]` — not a logits
row. `ArgmaxGreedyHead` serves it with the screened int5 chain, which reads the full bf16
weight only for the rows a certificate cannot rule out; `default.py` serves it with the
stock projection followed by `argmax`, accepts everything, and so makes the delegator
total. Both compute the same function of `x`, which is what lets the model call
`GreedyHead` without knowing which one it got.

Sampling is not this primitive. Logprobs, temperature, top-p, top-k, speculative
acceptance and row penalties all read logits the pruned chain never computes, and belong
on the head layer itself.
"""

import mlx.core as mx

from mlx_omnia.engine.core.kernels.lm_head.argmax import ArgmaxGreedyHead
from mlx_omnia.engine.core.kernels.lm_head.default import DefaultGreedyHead
from mlx_omnia.engine.core.kernels.lm_head.kernel import GreedyHeadStrategy, HeadProjection

__all__ = [
    "ArgmaxGreedyHead",
    "DefaultGreedyHead",
    "GreedyHead",
    "GreedyHeadStrategy",
    "HeadProjection",
]

# Order is preference: the first build that returns an instance wins; the default
# accepts everything, so resolution never fails.
_BUILDS = (ArgmaxGreedyHead.build, DefaultGreedyHead.build)


class GreedyHead:
    """Resolves the strategy at construction and delegates; itself a
    `GreedyHeadStrategy`."""

    def __init__(
        self,
        projection: HeadProjection,
        *,
        weight: mx.array | None = None,
        refine: bool = True,
    ) -> None:
        self.strategy: GreedyHeadStrategy = next(
            built
            for build in _BUILDS
            if (built := build(projection, weight=weight, refine=refine)) is not None
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.strategy(x)
