"""The greedy head primitive's contract: what a strategy is and what a model declares.

The primitive is the GREEDY STEP, not a projection: (x [..., hidden]) -> token id
[...]. The output is the id the stock head projection's argmax would select; no logit
row leaves a strategy. That is what makes the delegator total — the pruned
argmax-exact chain and a full matmul followed by `argmax` compute the same function,
and a caller cannot accidentally read a slot the pruned chain never computed.

Anything that is not a pure greedy pick — logprobs, softmax, temperature or top-p
sampling, a top-k set, a speculative acceptance ratio, a penalty over the row — is
outside this primitive and must go through the head layer itself.
"""

from typing import Protocol

import mlx.core as mx


class HeadProjection(Protocol):
    """The stock head: (x [..., hidden]) -> logits [..., vocab]."""

    def __call__(self, x: mx.array) -> mx.array: ...


class GreedyHeadStrategy(Protocol):
    """The greedy step: (x [..., hidden]) -> token id [...]."""

    def __call__(self, x: mx.array) -> mx.array: ...
