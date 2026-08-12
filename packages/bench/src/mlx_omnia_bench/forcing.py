"""The one module here that imports mlx, and it exists for a single trick.

Teacher forcing means pinning the stream, but a sampler that simply returns the next scripted
id makes the forward dead code: mlx is lazy, nothing downstream depends on the logits, and the
graph is never evaluated. The bench then reports a model that ran three times too fast — 582
tok/s with the dependency and 1809 without, measured on gpt2.

So the argmax still runs, because it is part of the step being timed, and its result is added
in scaled to zero: the returned id is exactly the scripted one and still carries a data
dependency on the logits.
"""

from collections.abc import Callable, Sequence

import mlx.core as mx


def forced(script: Sequence[int]) -> Callable[[mx.array], mx.array]:
    """A sampler pinned to `script`. The index clamps at the last id because a decode loop
    queues one step past the last one it emits."""
    ids = mx.array(list(script), dtype=mx.uint32)
    last = len(script) - 1
    drawn = -1

    def sample(logits: mx.array) -> mx.array:
        nonlocal drawn
        drawn = min(drawn + 1, last)
        return ids[drawn : drawn + 1] + mx.argmax(logits, axis=-1) * 0

    return sample
