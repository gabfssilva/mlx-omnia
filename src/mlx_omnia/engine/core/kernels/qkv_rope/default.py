"""The universal qkv prologue: the projection called as the layer it already is, then
`fast.rms_norm` and the model's own rotation in ops.

`build` accepts every declaration, so it registers last and makes the delegator total —
`QkvRope` always resolves, and a model uses it like any other layer. The chain is the
one every attention block already ran: the concatenated projection, the split into
head-major q/k/v, the per-head norm when the architecture has one, and `rope` on q and
k. What defines it is universality, not the absence of a kernel; its cost is the four
dispatches (two norms, two rotations) the specialized strategies fuse into the
projection's epilogue, and it is what those strategies fall back to on the lengths and
batches their kernels do not serve.
"""

from dataclasses import dataclass
from typing import Self

import mlx.core as mx

from mlx_omnia.engine.core.kernels.qkv_rope.kernel import Angles, Leaf, Projection, Rotation


def split_heads(
    fused: mx.array, *, heads: int, kv_heads: int, head_dim: int
) -> tuple[mx.array, mx.array, mx.array]:
    """The concatenated projection as head-major q/k/v."""
    batch, length = fused.shape[0], fused.shape[1]
    query_width = heads * head_dim
    kv_width = kv_heads * head_dim
    parts = mx.split(fused, [query_width, query_width + kv_width], axis=-1)
    q, k, v = (
        part.reshape(batch, length, count, head_dim).transpose(0, 2, 1, 3)
        for part, count in zip(parts, (heads, kv_heads, kv_heads), strict=True)
    )
    return q, k, v


@dataclass(frozen=True)
class DefaultQkvRope:
    projection: Projection
    heads: int
    kv_heads: int
    head_dim: int
    rope: Rotation
    eps: float
    q_norm: mx.array | None
    k_norm: mx.array | None

    @classmethod
    def build(
        cls,
        projection: Projection,
        *,
        heads: int,
        kv_heads: int,
        head_dim: int,
        rope: Rotation,
        eps: float,
        q_norm: mx.array | None,
        k_norm: mx.array | None,
        base: float | None,
        angles: Angles | None,
        rotary_pairs: int | None,
        mscale: float,
        segments: tuple[Leaf, Leaf, Leaf] | None,
    ) -> Self:
        return cls(projection, heads, kv_heads, head_dim, rope, eps, q_norm, k_norm)

    def epilogue(
        self, fused: mx.array, offset: int | mx.array
    ) -> tuple[mx.array, mx.array, mx.array]:
        """The norm and the rotation over an already projected token or segment."""
        q, k, v = split_heads(
            fused, heads=self.heads, kv_heads=self.kv_heads, head_dim=self.head_dim
        )
        if self.q_norm is not None and self.k_norm is not None:
            q = mx.fast.rms_norm(q, self.q_norm, self.eps)
            k = mx.fast.rms_norm(k, self.k_norm, self.eps)
        return self.rope(q, offset), self.rope(k, offset), v

    def __call__(
        self, x: mx.array, offset: int | mx.array
    ) -> tuple[mx.array, mx.array, mx.array]:
        return self.epilogue(self.projection(x), offset)
