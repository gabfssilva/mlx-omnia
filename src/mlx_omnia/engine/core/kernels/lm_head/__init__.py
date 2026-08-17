"""The screened output head: one strategy per head shape, one delegator.

The primitive is the SCREENED ROW — `(x [..., hidden]) -> [..., vocab]`, a row whose
argmax along the last axis is the stock head projection's, and whose other slots a
caller must not read. It is the altitude `core.api.Screened` consumes.
`ArgmaxScreenedHead` serves it with the screened int5 chain, which reads the full bf16
weight only for the rows a certificate cannot rule out; `default.py` serves it with the
stock projection, accepts everything, and so makes the delegator total. Both agree on
the argmax, which is what lets a model call `ScreenedHead` without knowing which one it
got.

Sampling is not this primitive. Logprobs, temperature, top-p, top-k, speculative
acceptance and row penalties all read logits the pruned chain never computes, and belong
on the head layer itself.
"""

import mlx.core as mx

from mlx_omnia.engine.core.kernels.lm_head.argmax import ArgmaxScreenedHead
from mlx_omnia.engine.core.kernels.lm_head.default import DefaultScreenedHead
from mlx_omnia.engine.core.kernels.lm_head.kernel import HeadProjection, ScreenedHeadStrategy
from mlx_omnia.engine.core.kernels.resolve import resolve

__all__ = [
    "ArgmaxScreenedHead",
    "DefaultScreenedHead",
    "HeadProjection",
    "ScreenedHead",
    "ScreenedHeadStrategy",
]

# Order is preference: the first strategy that builds wins; the default accepts
# everything, so resolution never fails.
_STRATEGIES = (ArgmaxScreenedHead, DefaultScreenedHead)


class ScreenedHead(ScreenedHeadStrategy):
    """Resolves the strategy at construction and delegates; itself a
    `ScreenedHeadStrategy`."""

    def __init__(
        self,
        projection: HeadProjection,
        *,
        weight: mx.array | None = None,
        refine: bool = True,
    ) -> None:
        self.strategy: ScreenedHeadStrategy = resolve(
            _STRATEGIES,
            projection,
            weight=weight,
            refine=refine,
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.strategy(x)
