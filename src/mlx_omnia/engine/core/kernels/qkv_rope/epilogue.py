"""Per-head q/k RMSNorm + rope rotation of a one-token qkv projection in one dispatch.

The op chain needs four kernels (two `fast.rms_norm`, two `fast.rope`); v needs
neither and stays a free slice of the fused projection. One threadgroup per q or k
head, `head_dim` threads.

Two roundings are copied from the ops it replaces, and they are what makes the
kernel bit-exact instead of merely close:

- the norm writes `w * (T)(x * rsqrt)` — mlx's `rms_norm` rounds the normalized
  value to T *before* the weight multiply, and the tensor between norm and rope
  is in T;
- the angle is `offset * exp2(-2i/head_dim * log2(base))` closed with
  `metal::fast::cos/sin`, which is how `mx.fast.rope` computes it. A host-side
  `base ** (-2i/d)` table with `metal::cos` is a different number.
"""

import math
from dataclasses import dataclass
from typing import Self

import mlx.core as mx

from mlx_omnia.engine.core.kernels.qkv_rope.default import DefaultQkvRope
from mlx_omnia.engine.core.kernels.qkv_rope.kernel import Angles, Leaf, Projection, Rotation
from mlx_omnia.engine.core.mxcompat import metal_kernel

_SOURCE = """
    uint head = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint sg = tid / 32;
    uint lane = tid % 32;
    bool isq = head < (uint)QHEADS;
    uint base = head * HDIM;

    threadgroup float part[HDIM / 32];
    float v = (float)F[base + tid];
    float g = simd_sum(v * v);
    if (lane == 0) part[sg] = g;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sumsq = 0.0f;
    for (uint j = 0; j < HDIM / 32; j++) sumsq += part[j];
    float hr = metal::rsqrt(sumsq / (float)HDIM + EPS[0]);

    threadgroup float normed[HDIM];
    const device T* nw = isq ? QNW : KNW;
    float vn = (float)(T)((float)nw[tid] * (float)(T)(v * hr));
    normed[tid] = vn;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint i = tid % (HDIM / 2);
    bool hi = tid >= (uint)HDIM / 2;
    float other = normed[hi ? tid - HDIM / 2 : tid + HDIM / 2];
    float d = (float)i / (float)(HDIM / 2);
    float angle = (float)OFF[0] * metal::exp2(-d * LOGBASE[0]);
    float c = metal::fast::cos(angle), sn = metal::fast::sin(angle);
    float rot = hi ? (other * sn + vn * c) : (vn * c - other * sn);
    if (isq) Q[head * HDIM + tid] = (T)rot;
    else K[(head - (uint)QHEADS) * HDIM + tid] = (T)rot;
"""

_KERNEL = metal_kernel(
    name="rope_epilogue",
    input_names=["F", "QNW", "KNW", "OFF", "LOGBASE", "EPS"],
    output_names=["Q", "K"],
    source=_SOURCE,
)


def rope_epilogue_applies(head_dim: int) -> bool:
    """One threadgroup per head: head_dim threads, reduced over whole simdgroups."""
    return head_dim % 32 == 0 and head_dim <= 1024


def rope_epilogue(
    fused: mx.array,
    *,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    q_norm: mx.array,
    k_norm: mx.array,
    offset: int | mx.array,
    base: float,
    eps: float,
) -> tuple[mx.array, mx.array]:
    """One token's qkv tail: `fused` [(q+2kv)·head_dim] -> normed+rotated q and k.

    Returns q [query_heads·head_dim] and k [kv_heads·head_dim], head-major — a
    reshape away from attention's layout. The v slice of `fused` is untouched.
    """
    assert rope_epilogue_applies(head_dim)
    assert fused.size == (query_heads + 2 * kv_heads) * head_dim
    out = _KERNEL(
        inputs=[
            fused.reshape(-1),
            q_norm,
            k_norm,
            mx.array([offset], dtype=mx.int32),
            mx.array([math.log2(base)], dtype=mx.float32),
            mx.array([eps], dtype=mx.float32),
        ],
        template=[("T", fused.dtype), ("QHEADS", query_heads), ("HDIM", head_dim)],
        grid=((query_heads + kv_heads) * head_dim, 1, 1),
        threadgroup=(head_dim, 1, 1),
        output_shapes=[(query_heads * head_dim,), (kv_heads * head_dim,)],
        output_dtypes=[fused.dtype, fused.dtype],
    )
    return out[0], out[1]


@dataclass(frozen=True)
class RopeEpilogue:
    """The one-token prologue whose norm and rotation ride the projection's tail.

    Serves the full-rotary, q/k-normed decode of a fused projection: the angle is
    derived in-kernel from `offset` and `log2(base)`, so the declaration must carry
    `base` and nothing else — a model with a frequency table or a partial rotary
    declares `angles` instead and lands on `QkNormRope`. Everything the kernel does
    not serve (a segment, a batch) goes to the fallback, which is the chain the
    kernel replaces.
    """

    projection: Projection
    heads: int
    kv_heads: int
    head_dim: int
    base: float
    eps: float
    q_norm: mx.array
    k_norm: mx.array
    fallback: DefaultQkvRope

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
    ) -> Self | None:
        if base is None or q_norm is None or k_norm is None:
            return None
        if not rope_epilogue_applies(head_dim):
            return None
        return cls(
            projection,
            heads,
            kv_heads,
            head_dim,
            base,
            eps,
            q_norm,
            k_norm,
            DefaultQkvRope.build(
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
            ),
        )

    def __call__(
        self, x: mx.array, offset: int | mx.array
    ) -> tuple[mx.array, mx.array, mx.array]:
        fused = self.projection(x)
        if x.shape[0] != 1 or x.shape[1] != 1:
            return self.fallback.epilogue(fused, offset)
        queries, keys = rope_epilogue(
            fused,
            query_heads=self.heads,
            kv_heads=self.kv_heads,
            head_dim=self.head_dim,
            q_norm=self.q_norm,
            k_norm=self.k_norm,
            offset=offset,
            base=self.base,
            eps=self.eps,
        )
        values = fused[..., (self.heads + self.kv_heads) * self.head_dim :]
        return (
            queries.reshape(1, self.heads, 1, self.head_dim),
            keys.reshape(1, self.kv_heads, 1, self.head_dim),
            values.reshape(1, self.kv_heads, 1, self.head_dim),
        )
