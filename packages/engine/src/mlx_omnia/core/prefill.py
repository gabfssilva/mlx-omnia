"""Prefill in blocks, so a long prompt costs a block instead of itself.

A forward over the whole prompt allocates two things that scale with its length: the trunk's
transients, and the `T x vocab` logits the head writes for positions nobody reads — 9.3 GiB
at 32k over a 152k vocabulary, whatever the model weighs. Feeding the prompt in blocks and
dropping what each block returns removes both. Dropping is what does it: with nothing
referencing a block's logits, the head is a graph MLX never evaluates, and evaluating the
caches between blocks is what keeps the trunk's own graph from accumulating across them.

The last block is the caller's, because the logits it reads are the ones that exist. It is
handed back whole rather than cut down to the single row the sampler needs: a one-row
forward is the decode regime, not the prefill one, and swapping regimes at the last position
moves more numerically than shortening the block does.
"""

from collections.abc import Callable, Iterator, Sequence
from itertools import islice

import mlx.core as mx

from mlx_omnia.core.cache import LayerCache

BLOCK = 2048
"""Rows per block — mlx-lm's `prefill_step_size` and oMLX's default, both 2048. A memory
knob and not a speed one: the prefill saturates bandwidth well below this, and the blocks
that are not the last cost nothing but the trunk they were going to run anyway."""

ARRIVING = 64
"""Rows per block for a prompt that is still being written, where the block is a latency knob
instead of a memory one. At 2048 a prompt of a few hundred ids is one block, which is fed
only once its last id exists — the whole point, missed. Small enough that the producer's
first pieces reach the GPU while it is still writing the rest, large enough that a prompt of
this size costs single-digit forwards; the prefill saturates bandwidth well below either."""


def prefill(
    feed: Callable[[slice], object],
    length: int,
    cache: Sequence[LayerCache],
    *,
    block: int = BLOCK,
) -> slice:
    """Feed every block but the last through `feed`; return the last for the caller's own
    forward.

    `feed` may return whatever the model returns — it is dropped here, which is the point.
    Between blocks only `LayerCache.tensors` is evaluated, so what is forced is the trunk
    and never the head.
    """
    at = 0
    while length - at > block:
        feed(slice(at, at + block))
        mx.eval([tensor for layer in cache for tensor in layer.tensors])
        # Back to the system rather than into MLX's own pool: a block's transients are the
        # peak this split exists to bound, and a pool that keeps them holds the peak too.
        mx.clear_cache()
        at += block
    return slice(at, length)


def pulled(
    feed: Callable[[list[int]], object],
    ids: Iterator[int],
    cache: Sequence[LayerCache],
    *,
    block: int = BLOCK,
) -> list[int]:
    """The same split over a prompt that arrives instead of one that is already there: feed
    every block but the last as its ids turn up, and hand the last back.

    Pulling is what makes the overlap. A block's forward is queued before the ids of the next
    one exist, so a prompt still being written — a caption coming out of a second model — is
    produced against this model's blocks rather than before them. A prompt already complete
    goes through the same loop and notices nothing: the first pull hands back a whole block.

    The last block cannot be fed here for the reason the indexed version gives, and here it is
    also the only one this can recognize: a block is last because the pull after it came back
    empty, which is knowable only once it has.
    """
    held: list[int] = []
    while chunk := list(islice(ids, block)):
        if held:
            feed(held)
            mx.eval([tensor for layer in cache for tensor in layer.tensors])
            mx.clear_cache()
        held = chunk
    return held
