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

import mlx.core as mx

from sideros.core.mxcompat import metal_kernel

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
    float r = metal::rsqrt(sumsq / (float)HDIM + EPS[0]);
    for (uint j = 0; j < VPT; j++) {
        uint i = tid * VPT + j;
        A[i] = (T)vals[i];
        HN[i] = (T)(vals[i] * r * (float)NW[i]);
    }
"""

_KERNEL = metal_kernel(
    name="add_rmsnorm",
    input_names=["X", "Op", "NW", "EPS"],
    output_names=["A", "HN"],
    source=_SOURCE,
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
    vpt = 4 if hidden % 128 == 0 else 5
    out = _KERNEL(
        inputs=[
            x.reshape(hidden),
            projected.reshape(hidden),
            weight,
            mx.array([eps], dtype=mx.float32),
        ],
        template=[("T", x.dtype), ("HDIM", hidden), ("VPT", vpt)],
        grid=(hidden // vpt, 1, 1),
        threadgroup=(hidden // vpt, 1, 1),
        output_shapes=[(hidden,), (hidden,)],
        output_dtypes=[x.dtype, x.dtype],
    )
    return out[0].reshape(x.shape), out[1].reshape(x.shape)
