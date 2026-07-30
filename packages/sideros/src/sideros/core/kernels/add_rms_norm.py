"""Residual add + the rms_norm that reads it, in one dispatch.

The norm needs the whole vector, so a single threadgroup carries it: `VPT` values
per thread, the sum of squares reduced through threadgroup memory. Both outputs
leave the kernel — the sum (the stream the block's output adds to) and its normed
view (what the next projection reads). The add rounds to T before the squares, as
the materialized tensor between the two ops would.

`VPT` is chosen so the thread count `HDIM/VPT` stays a whole number of simdgroups:
4 for the usual 128-aligned hidden, 5 for a 2880-wide one (576 threads, 18
simdgroups). At VPT=4 the arithmetic is identical to the original.
"""

from typing import NamedTuple

import mlx.core as mx

from sideros.core.kernels import MetalDispatch, MetalKernel

_SOURCE = """
    uint tid = thread_position_in_threadgroup.x;
    uint sg = tid / 32;
    threadgroup float vals[HDIM];
    threadgroup float part[HDIM / (VPT * 32)];
    float local = 0.0f;
    for (uint j = 0; j < VPT; j++) {
        uint i = tid * VPT + j;
        float a = (float)(T)((float)X[i] + (float)Op[i]);
        vals[i] = a;
        local += a * a;
    }
    float s = simd_sum(local);
    if (tid % 32 == 0) part[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sumsq = 0.0f;
    for (uint j = 0; j < HDIM / (VPT * 32); j++) sumsq += part[j];
    float r = metal::rsqrt(sumsq / (float)HDIM + EPS);
    for (uint j = 0; j < VPT; j++) {
        uint i = tid * VPT + j;
        A[i] = (T)vals[i];
        HN[i] = (T)(vals[i] * r * (float)NW[i]);
    }
"""


class _Inputs(NamedTuple):
    X: mx.array
    Op: mx.array
    NW: mx.array
    EPS: float


class _Outputs(NamedTuple):
    A: mx.array
    HN: mx.array


def _dispatch(input: _Inputs) -> MetalDispatch:
    hidden = input.NW.size
    vpt = 4 if hidden % 128 == 0 else 5
    return MetalDispatch(
        template=(("T", input.X.dtype), ("HDIM", hidden), ("VPT", vpt)),
        grid=(hidden // vpt, 1, 1),
        threadgroup=(hidden // vpt, 1, 1),
        output_shapes=((hidden,), (hidden,)),
        output_dtypes=(input.X.dtype, input.X.dtype),
    )


_KERNEL = MetalKernel[_Inputs, _Outputs](
    name="add_rmsnorm",
    source=_SOURCE,
    launch=_dispatch,
)


_MAX_THREADGROUP = 1024


def add_rms_norm_applies(hidden: int) -> bool:
    """The single threadgroup has to tile: whole simdgroups, and its `hidden // VPT`
    threads within the device's 1024-thread limit (4096 at VPT=4, 5120 at VPT=5) —
    the dispatch below asks for exactly that many."""
    vpt = 4 if hidden % 128 == 0 else 5
    return (
        hidden % vpt == 0
        and (hidden // vpt) % 32 == 0
        and hidden // vpt <= _MAX_THREADGROUP
    )


def add_rms_norm(
    x: mx.array, projected: mx.array, weight: mx.array, eps: float
) -> tuple[mx.array, mx.array]:
    """One token's residual join: returns (x + projected, its rms_norm).

    `x` and `projected` are [hidden] (or any shape of that many elements); the
    outputs keep `x`'s shape and dtype.
    """
    hidden = weight.size
    assert x.size == hidden and projected.size == hidden
    assert x.dtype == projected.dtype == weight.dtype
    assert add_rms_norm_applies(hidden)
    out = _KERNEL(
        _Inputs(
            x.reshape(hidden),
            projected.reshape(hidden),
            weight,
            eps,
        )
    )
    return out.A.reshape(x.shape), out.HN.reshape(x.shape)
