"""The T=1 attention step: one kernel module per fused variant, one delegator.

The facade is drawn at the widest fusion available, not at the narrowest: the sliding
kernel swallows the Q/K norm, the rotation and the ring write along with the softmax, so
the declaration carries the raw q/k/v projections, the norm gains, the rotary geometry
and the cache — and the strategies that fuse less (`sink.py`, which starts at the roped
q/k) run those stages in ops themselves. A model declares the layer, never a kernel.

The resolution table is (cache x layer): a promoted ring resolves to the sliding kernel,
a fixed capacity cache to the full kernel, a layer with sink logits to the split-K
flash-decode, everything else to `default.py` — the stock chain of `rms_norm`, `rope`,
`update_and_fetch` and `mx.fast.scaled_dot_product_attention`. `sdpa.py` is not a
strategy: it is the lifted decode kernel that stands in for that op, shared by whoever
installs its patch.

One branch survives resolution, in every specialized strategy: prefill. `T != 1` is not
the shape any of these kernels is written for, and the same layer object serves both
phases, so a step whose call is not a decode step defers to its `fallback`.
"""

import mlx.core as mx

from mlx_omnia.engine.core.kernels.attention.default import DefaultAttentionStep
from mlx_omnia.engine.core.kernels.attention.full import FullAttentionStep
from mlx_omnia.engine.core.kernels.attention.kernel import (
    Angles,
    AttentionCache,
    AttentionStepStrategy,
)
from mlx_omnia.engine.core.kernels.attention.sink import SinkAttentionStep
from mlx_omnia.engine.core.kernels.attention.sliding import SlidingAttentionStep

__all__ = [
    "Angles",
    "AttentionCache",
    "AttentionStep",
    "AttentionStepStrategy",
    "DefaultAttentionStep",
    "FullAttentionStep",
    "SinkAttentionStep",
    "SlidingAttentionStep",
]

# Order is preference: the first build that returns an instance wins; the default
# accepts everything, so resolution never fails.
_BUILDS = (
    SlidingAttentionStep.build,
    FullAttentionStep.build,
    SinkAttentionStep.build,
    DefaultAttentionStep.build,
)


class AttentionStep:
    """Resolves the strategy at construction and delegates; itself an
    `AttentionStepStrategy`.

    The cache is part of the declaration, not of the call: which kernel serves a layer is
    decided by the cache's own shape and clock, and the fused kernels write into it in
    place. A step is therefore bound to one cache and is rebuilt when that cache is
    replaced or promoted.
    """

    def __init__(
        self,
        cache: AttentionCache,
        *,
        heads: int,
        kv_heads: int,
        head_dim: int,
        scale: float,
        dtype: mx.Dtype,
        query_weight: mx.array | None = None,
        key_weight: mx.array | None = None,
        eps: float = 1e-6,
        rotary_pairs: int = 0,
        mscale: float = 1.0,
        angles: Angles | None = None,
        freqs: mx.array | None = None,
        base: float = 0.0,
        sinks: mx.array | None = None,
    ) -> None:
        self.strategy: AttentionStepStrategy = next(
            built
            for build in _BUILDS
            if (
                built := build(
                    cache,
                    heads=heads,
                    kv_heads=kv_heads,
                    head_dim=head_dim,
                    scale=scale,
                    dtype=dtype,
                    query_weight=query_weight,
                    key_weight=key_weight,
                    eps=eps,
                    rotary_pairs=rotary_pairs,
                    mscale=mscale,
                    angles=angles,
                    freqs=freqs,
                    base=base,
                    sinks=sinks,
                )
            )
            is not None
        )

    def __call__(
        self,
        raw_queries: mx.array,
        raw_keys: mx.array,
        raw_values: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        return self.strategy(raw_queries, raw_keys, raw_values, mask)
