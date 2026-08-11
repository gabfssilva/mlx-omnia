"""Residual add + the rms_norm that reads it, in one dispatch.

The norm needs the whole vector, so a single threadgroup carries it: `VPT` values
per thread, the sum of squares reduced through threadgroup memory. Both outputs
leave the kernel — the sum (the stream the block's output adds to) and its normed
view (what the next projection reads). The add rounds to T before the squares, as
the materialized tensor between the two ops would.

`VPT` is chosen so the thread count `HDIM/VPT` stays a whole number of simdgroups:
4 for the usual 128-aligned hidden, 5 for a 2880-wide one (576 threads, 18
simdgroups). At VPT=4 the arithmetic is identical to the original.

One token per call: `build` takes it only when the declaration says so.
"""

from dataclasses import dataclass
from typing import NamedTuple, Self

import mlx.core as mx
import mlx.nn as nn

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


def applies(hidden: int) -> bool:
    """The single threadgroup has to tile: whole simdgroups, and its `hidden // VPT`
    threads within the device's 1024-thread limit (4096 at VPT=4, 5120 at VPT=5) —
    the dispatch below asks for exactly that many."""
    vpt = 4 if hidden % 128 == 0 else 5
    return (
        hidden % vpt == 0
        and (hidden // vpt) % 32 == 0
        and hidden // vpt <= _MAX_THREADGROUP
    )


@dataclass(frozen=True)
class FusedAddRmsNorm:
    weight: mx.array
    eps: float

    @classmethod
    def build(cls, leaf: nn.RMSNorm, *, tokens: int | None) -> Self | None:
        if tokens != 1 or not applies(leaf.weight.size):
            return None
        return cls(leaf.weight, leaf.eps)

    def __call__(self, x: mx.array, projected: mx.array) -> tuple[mx.array, mx.array]:
        hidden = self.weight.size
        assert x.size == hidden and projected.size == hidden
        assert x.dtype == projected.dtype == self.weight.dtype
        out = _KERNEL(
            _Inputs(
                x.reshape(hidden),
                projected.reshape(hidden),
                self.weight,
                self.eps,
            )
        )
        return out.A.reshape(x.shape), out.HN.reshape(x.shape)
