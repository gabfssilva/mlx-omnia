"""The routed *dense* (unquantized) expert MLP of a one-token step in three dispatches.

`gather_mm` streams a T=1 routed expert MLP at ~450 GB/s on the M5 Max; a row-pair
gemv carrying the expert indirection itself sustains ~700 — the tile kernel dispatched
at m=1 wastes half the bandwidth. The second dispatch computes silu(gate)·up while
streaming the down weights, and folds the routing weight into the output, so nothing
elementwise runs between them. The third piece is routing: the unfused chain (gemv,
sigmoid, argpartition, take_along_axis, renormalize) is ~10 tiny dependent kernels.

The stacks are the checkpoint layout `[experts, out, in]`, rows contiguous, gate‖up
**block**-concatenated (`[w1 ‖ w3]` on the output axis, LFM2.5's load-time fusion) —
not the row interleave `moe_gemv.py` wants.

Two rounding differences against the op chain, both documented by the tests: the router
score is sigmoid of an f32 dot (the op chain rounds the gemv and the sigmoid to T), and
the renormalization sum accumulates in f32 in descending-score order. Selection matches:
ties go to the higher expert index, as mlx's stable ascending argpartition does.
"""

import mlx.core as mx

from sideros.core.mxcompat import metal_kernel

_GATE_UP_SOURCE = """
    uint pair = thread_position_in_grid.y;
    uint expert = thread_position_in_grid.z;
    uint lane = thread_position_in_threadgroup.x;
    uint n0 = pair * 2;
    uint n = (uint)N;
    uint kd = (uint)KD;
    uint k4 = kd / 4;
    if (n0 >= n) return;

    size_t base = (size_t)IDX[expert] * n * kd;
    const device vec<T, 4>* w0 = (const device vec<T, 4>*)(W + base + (size_t)n0 * kd);
    const device vec<T, 4>* w1 = (const device vec<T, 4>*)(W + base + (size_t)(n0 + 1) * kd);
    const device vec<T, 4>* xv = (const device vec<T, 4>*)X;

    float4 acc0 = float4(0.0f);
    float4 acc1 = float4(0.0f);
    for (uint i = lane; i < k4; i += 32) {
        float4 xm = float4(xv[i]);
        acc0 += xm * float4(w0[i]);
        acc1 += xm * float4(w1[i]);
    }
    float t0 = simd_sum(acc0.x + acc0.y + acc0.z + acc0.w);
    float t1 = simd_sum(acc1.x + acc1.y + acc1.z + acc1.w);
    if (lane == 0) {
        Y[(size_t)expert * n + n0] = (T)t0;
        Y[(size_t)expert * n + n0 + 1] = (T)t1;
    }
"""

_DOWN_SOURCE = """
    uint pair = thread_position_in_grid.y;
    uint expert = thread_position_in_grid.z;
    uint lane = thread_position_in_threadgroup.x;
    uint n0 = pair * 2;
    uint n = (uint)N;
    uint kd = (uint)KD;
    uint k4 = kd / 4;
    if (n0 >= n) return;

    size_t base = (size_t)IDX[expert] * n * kd;
    const device vec<T, 4>* w0 = (const device vec<T, 4>*)(W + base + (size_t)n0 * kd);
    const device vec<T, 4>* w1 = (const device vec<T, 4>*)(W + base + (size_t)(n0 + 1) * kd);
    const device vec<T, 4>* gv = (const device vec<T, 4>*)(GU + (size_t)expert * 2 * kd);
    const device vec<T, 4>* uv = (const device vec<T, 4>*)(GU + ((size_t)expert * 2 + 1) * kd);

    float4 acc0 = float4(0.0f);
    float4 acc1 = float4(0.0f);
    for (uint i = lane; i < k4; i += 32) {
        float4 g = float4(gv[i]);
        float4 xm = g / (1.0f + metal::exp(-g)) * float4(uv[i]);
        acc0 += xm * float4(w0[i]);
        acc1 += xm * float4(w1[i]);
    }
    float t0 = simd_sum(acc0.x + acc0.y + acc0.z + acc0.w);
    float t1 = simd_sum(acc1.x + acc1.y + acc1.z + acc1.w);
    if (lane == 0) {
        float scale = (float)WS[expert];
        Y[(size_t)expert * n + n0] = (T)(t0 * scale);
        Y[(size_t)expert * n + n0 + 1] = (T)(t1 * scale);
    }
"""

_ROUTE_SOURCE = """
    uint sg = thread_position_in_threadgroup.x / 32;
    uint lane = thread_position_in_threadgroup.x % 32;
    uint k4 = (uint)KD / 4;

    threadgroup float scores[E];
    const device vec<T, 4>* w = (const device vec<T, 4>*)(RW + (size_t)sg * (uint)KD);
    const device vec<T, 4>* xv = (const device vec<T, 4>*)X;
    float4 acc = float4(0.0f);
    for (uint i = lane; i < k4; i += 32) {
        acc += float4(xv[i]) * float4(w[i]);
    }
    float dot = simd_sum(acc.x + acc.y + acc.z + acc.w);
    if (lane == 0) {
        scores[sg] = 1.0f / (1.0f + metal::exp(-dot));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg != 0) return;

    float score = scores[lane];
    float selector = score + BIAS[lane];
    float picked[TOPK];
    float total = 0.0f;
    for (uint j = 0; j < TOPK; j++) {
        float best = simd_max(selector);
        int winner = simd_max(selector == best ? (int)lane : -1);
        float s = simd_shuffle(score, (ushort)winner);
        picked[j] = s;
        total += s;
        if ((int)lane == winner) selector = -INFINITY;
        if (lane == 0) OI[j] = (uint)winner;
    }
    if (lane < TOPK) {
        float w = picked[lane];
        OW[lane] = (T)((NORM ? w / (total + 1e-6f) : w) * SCALE);
    }
"""

_ROUTE_KERNEL = metal_kernel(
    name="moe_route_sigmoid",
    input_names=["X", "RW", "BIAS", "SCALE", "KD"],
    output_names=["OI", "OW"],
    source=_ROUTE_SOURCE,
)

_GATE_UP_KERNEL = metal_kernel(
    name="moe_dense_gate_up",
    input_names=["X", "W", "IDX", "N", "KD"],
    output_names=["Y"],
    source=_GATE_UP_SOURCE,
)

_DOWN_KERNEL = metal_kernel(
    name="moe_dense_down",
    input_names=["GU", "W", "IDX", "WS", "N", "KD"],
    output_names=["Y"],
    source=_DOWN_SOURCE,
)


def moe_gemv_dense_applies(hidden: int, inner: int, experts: int, k: int) -> bool:
    """Every reduction is a float4 lane sweep, and routing is one threadgroup of E
    simdgroups (Metal's 1024-thread limit caps E at 32)."""
    return (
        hidden % 4 == 0
        and inner % 4 == 0
        and hidden % 2 == 0
        and (2 * inner) % 2 == 0
        and 0 < k <= 32
        and 0 < experts <= 32
    )


def _i32(value: int) -> mx.array:
    return mx.array(value, dtype=mx.int32)


def moe_route_sigmoid(
    x: mx.array,
    router: mx.array,
    bias: mx.array,
    scale: mx.array,
    k: int,
    *,
    normalized: bool,
) -> tuple[mx.array, mx.array]:
    """One token's routing: x [hidden], router [experts, hidden], bias [experts] float32,
    scale [] float32 -> (indices [k] uint32, weights [k] T). The bias shifts selection
    only; the weight is the bias-free sigmoid score."""
    experts, kdim = router.shape
    assert experts <= 32 and kdim % 4 == 0 and 0 < k <= 32
    out = _ROUTE_KERNEL(
        inputs=[x, router, bias, scale, _i32(kdim)],
        template=[
            ("T", x.dtype),
            ("E", experts),
            ("TOPK", k),
            ("NORM", int(normalized)),
        ],
        grid=(experts * 32, 1, 1),
        threadgroup=(experts * 32, 1, 1),
        output_shapes=[(k,), (k,)],
        output_dtypes=[mx.uint32, x.dtype],
    )
    return out[0], out[1]


def moe_dense_gate_up(x: mx.array, weight: mx.array, indices: mx.array) -> mx.array:
    """x [hidden] through the routed gate‖up stack [experts, 2·inner, hidden] -> [k, 2·inner]."""
    _, n, kdim = weight.shape
    assert kdim % 4 == 0 and n % 2 == 0
    return _GATE_UP_KERNEL(
        inputs=[x, weight, indices, _i32(n), _i32(kdim)],
        template=[("T", x.dtype)],
        grid=(32, n // 2, indices.shape[0]),
        threadgroup=(32, 1, 1),
        output_shapes=[(indices.shape[0], n)],
        output_dtypes=[x.dtype],
    )[0]


def moe_dense_down(
    gate_up: mx.array, weight: mx.array, indices: mx.array, routing: mx.array
) -> mx.array:
    """silu(gate)·up fused in the epilogue of the routed down projection: gate_up
    [k, 2·inner], weight [experts, hidden, inner], routing [k] -> [k, hidden], ready to
    sum over k."""
    _, n, kdim = weight.shape
    assert kdim % 4 == 0 and n % 2 == 0
    return _DOWN_KERNEL(
        inputs=[gate_up, weight, indices, routing, _i32(n), _i32(kdim)],
        template=[("T", gate_up.dtype)],
        grid=(32, n // 2, indices.shape[0]),
        threadgroup=(32, 1, 1),
        output_shapes=[(indices.shape[0], n)],
        output_dtypes=[gate_up.dtype],
    )[0]
