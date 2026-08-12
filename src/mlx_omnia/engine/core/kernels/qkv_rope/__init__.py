"""The qkv prologue: one kernel module per fusion, one delegator.

An attention block declares the prologue — the projection, the head geometry, the
per-head norm weights, and how the rotary is described — and `QkvRope` binds the
fusion that declaration admits, or none, at construction time. The model never names a
kernel; a new fusion is a new module here, registered in `_BUILDS`, and every family
engages it.

The resolution table is (rotary description x projection shape): `angles` plus
`rotary_pairs` serve the table-driven norm+rotary of `qk_norm_rope.py` (partial rotary,
frequency tables, the YaRN pre-scale); a bare `base` with q/k norms serves the
full-rotary decode epilogue of `epilogue.py`; three separate leaves serve the fused
projection of `segmented.py`. `default.py` serves everything else — projection, then
`fast.rms_norm` and the declared rotation in ops — so the delegator is total: a model
uses `QkvRope` like any other layer.

A declaration that admits both a fused epilogue and a fused projection resolves to the
epilogue: it is the deeper fusion, and no family ships both today.
"""

import mlx.core as mx

from mlx_omnia.engine.core.kernels.qkv_rope.default import DefaultQkvRope
from mlx_omnia.engine.core.kernels.qkv_rope.epilogue import RopeEpilogue
from mlx_omnia.engine.core.kernels.qkv_rope.kernel import (
    Angles,
    Leaf,
    Projection,
    QkvRopeStrategy,
    Rotation,
)
from mlx_omnia.engine.core.kernels.qkv_rope.qk_norm_rope import QkNormRope
from mlx_omnia.engine.core.kernels.qkv_rope.segmented import SegmentedQkv

__all__ = [
    "Angles",
    "DefaultQkvRope",
    "Leaf",
    "Projection",
    "QkNormRope",
    "QkvRope",
    "QkvRopeStrategy",
    "RopeEpilogue",
    "Rotation",
    "SegmentedQkv",
]

# Order is preference: the first build that returns an instance wins; the default
# accepts everything, so resolution never fails.
_BUILDS = (
    QkNormRope.build,
    RopeEpilogue.build,
    SegmentedQkv.build,
    DefaultQkvRope.build,
)


class QkvRope:
    """Resolves the strategy at construction and delegates; itself a
    `QkvRopeStrategy`."""

    def __init__(
        self,
        projection: Projection,
        *,
        heads: int,
        kv_heads: int,
        head_dim: int,
        rope: Rotation,
        eps: float,
        q_norm: mx.array | None = None,
        k_norm: mx.array | None = None,
        base: float | None = None,
        angles: Angles | None = None,
        rotary_pairs: int | None = None,
        mscale: float = 1.0,
        segments: tuple[Leaf, Leaf, Leaf] | None = None,
    ) -> None:
        self.strategy: QkvRopeStrategy = next(
            built
            for build in _BUILDS
            if (
                built := build(
                    projection,
                    heads=heads,
                    kv_heads=kv_heads,
                    head_dim=head_dim,
                    rope=rope,
                    eps=eps,
                    q_norm=q_norm,
                    k_norm=k_norm,
                    base=base,
                    angles=angles,
                    rotary_pairs=rotary_pairs,
                    mscale=mscale,
                    segments=segments,
                )
            )
            is not None
        )

    def __call__(
        self, x: mx.array, offset: int | mx.array
    ) -> tuple[mx.array, mx.array, mx.array]:
        return self.strategy(x, offset)
