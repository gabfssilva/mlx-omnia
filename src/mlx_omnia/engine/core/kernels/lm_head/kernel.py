"""The screened head primitive's contract: what a strategy is and what a model declares.

The primitive is the SCREENED ROW: (x [..., hidden]) -> [..., vocab], a row whose
argmax along the last axis is the stock head projection's argmax — and, outside that
argmax, not the same numbers. That is the altitude `core.api.Screened` consumes, and
it is what makes the delegator total: the pruned argmax-exact chain and the stock
projection agree on the one thing a caller may read.

Anything that is not a pure greedy pick — logprobs, softmax, temperature or top-p
sampling, a top-k set, a speculative acceptance ratio, a penalty over the row — is
outside this primitive and must go through the head layer itself.
"""

from typing import Protocol, runtime_checkable

import mlx.core as mx


@runtime_checkable
class HeadProjection(Protocol):
    """The stock head: (x [..., hidden]) -> logits [..., vocab]."""

    def __call__(self, x: mx.array) -> mx.array: ...


@runtime_checkable
class ScreenedHeadStrategy(Protocol):
    """(x [..., hidden]) -> [..., vocab]: the stock projection's argmax, and outside
    it a row a caller must read nothing else from."""

    def __call__(self, x: mx.array) -> mx.array: ...
