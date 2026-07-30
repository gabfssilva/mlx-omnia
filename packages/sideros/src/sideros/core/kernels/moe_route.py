"""softmax -> top-k -> renormalize in one dispatch, bit-exact with the op chain.

The logits still come from the stock quantized matmul — recomputing them in a fused
kernel rounds differently and flips which expert wins a near-tie (measured: 1 in ~11
tokens). Ties go to the higher index, matching mlx's stable ascending argpartition.
"""

import mlx.core as mx

from sideros.core.mxcompat import metal_kernel

_SOURCE = """
    uint lane = thread_position_in_threadgroup.x;
    constexpr uint per_lane = EXPERTS / 32;

    float p[per_lane];
    float m = -INFINITY;
    for (uint i = 0; i < per_lane; i++) {
        p[i] = (float)L[lane + i * 32];
        m = metal::max(m, p[i]);
    }
    m = simd_max(m);
    float e = 0.0f;
    for (uint i = 0; i < per_lane; i++) {
        p[i] = metal::exp(p[i] - m);
        e += p[i];
    }
    float denom = simd_sum(e);
    for (uint i = 0; i < per_lane; i++) {
        p[i] = (float)(T)(p[i] / denom);
    }

    float pick[TOPK];
    for (uint j = 0; j < TOPK; j++) {
        float local = -1.0f;
        for (uint i = 0; i < per_lane; i++) {
            local = metal::max(local, p[i]);
        }
        float best = simd_max(local);
        int slot = -1;
        for (int i = (int)per_lane - 1; i >= 0; i--) {
            if (slot < 0 && p[i] == best) slot = i;
        }
        int cand = slot >= 0 ? (int)(lane + (uint)slot * 32) : -1;
        int winner = simd_max(cand);
        pick[j] = best;
        for (uint i = 0; i < per_lane; i++) {
            if (winner == cand && (int)i == slot) p[i] = -1.0f;
        }
        if (lane == 0) OI[j] = (uint)winner;
    }
    float total = 0.0f;
    for (int j = TOPK - 1; j >= 0; j--) {
        total = (float)(T)(total + pick[j]);
    }
    if (lane < TOPK) OW[lane] = (T)(pick[lane] / total);

    if (SHARED && lane == 0) {
        T sx = L[EXPERTS];
        auto sy = 1 / (1 + metal::exp(metal::abs(sx)));
        OI[TOPK] = (uint)EXPERTS;
        OW[TOPK] = (sx < 0) ? sy : 1 - sy;
    }
"""

_KERNEL = metal_kernel(
    name="moe_softmax_topk", input_names=["L"], output_names=["OI", "OW"], source=_SOURCE
)

_SIGMOID_SOURCE = """
    uint lane = thread_position_in_threadgroup.x;
    constexpr uint per_lane = EXPERTS / 32;

    float s[per_lane];
    float b[per_lane];
    for (uint i = 0; i < per_lane; i++) {
        float x = (float)L[lane + i * 32];
        s[i] = 1.0f / (1.0f + metal::exp(-x));
        b[i] = s[i] + B[lane + i * 32];
    }

    float pick[TOPK];
    for (uint j = 0; j < TOPK; j++) {
        float local = -INFINITY;
        for (uint i = 0; i < per_lane; i++) {
            local = metal::max(local, b[i]);
        }
        float best = simd_max(local);
        int slot = -1;
        for (int i = (int)per_lane - 1; i >= 0; i--) {
            if (slot < 0 && b[i] == best) slot = i;
        }
        int cand = slot >= 0 ? (int)(lane + (uint)slot * 32) : -1;
        int winner = simd_max(cand);
        uint wslot = (uint)winner / 32;
        pick[j] = simd_broadcast(s[wslot], (ushort)((uint)winner % 32));
        if (winner == cand && slot >= 0) b[(uint)slot] = -INFINITY;
        if (lane == 0) OI[j] = (uint)winner;
    }
    float total = 0.0f;
    for (int j = TOPK - 1; j >= 0; j--) {
        total = total + pick[j];
    }
    if (lane < TOPK) {
        T w = (T)(pick[lane] / total);
        OW[lane] = (T)((float)w * SC);
    }
"""

_SIGMOID_KERNEL = metal_kernel(
    name="moe_sigmoid_topk",
    input_names=["L", "B", "SC"],
    output_names=["OI", "OW"],
    source=_SIGMOID_SOURCE,
)


def softmax_topk_applies(experts: int, k: int) -> bool:
    """One simdgroup owns the whole row: `experts / 32` entries per lane, and the k
    winners are written by the first k lanes, so k can never exceed the 32 of them."""
    return experts >= 32 and experts % 32 == 0 and 0 < k <= 32


def softmax_topk(
    logits: mx.array, k: int, *, shared: bool = False
) -> tuple[mx.array, mx.array]:
    """One token's routing pick: logits [experts] -> (indices [k], renormalized weights)."""
    experts = logits.size - (1 if shared else 0)
    slots = k + (1 if shared else 0)
    assert softmax_topk_applies(experts, k)
    out = _KERNEL(
        inputs=[logits],
        template=[("T", logits.dtype), ("TOPK", k), ("EXPERTS", experts), ("SHARED", int(shared))],
        grid=(32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(slots,), (slots,)],
        output_dtypes=[mx.uint32, logits.dtype],
    )
    return out[0], out[1]


def sigmoid_topk(
    logits: mx.array, bias: mx.array, k: int, *, scale: float
) -> tuple[mx.array, mx.array]:
    """One token's sigmoid routing pick (DeepSeek-V3 style): selection by
    `sigmoid(logits) + bias`, weights from the unbiased scores, renormalized and
    scaled. Sigmoid and renorm run in fp32; the weight rounds to T before the
    scale multiplies, matching the stock chain's `astype` placement."""
    experts = logits.size
    assert softmax_topk_applies(experts, k)
    out = _SIGMOID_KERNEL(
        inputs=[logits, bias.astype(mx.float32), mx.array(scale, dtype=mx.float32)],
        template=[("T", logits.dtype), ("TOPK", k), ("EXPERTS", experts)],
        grid=(32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(k,), (k,)],
        output_dtypes=[mx.uint32, logits.dtype],
    )
    return out[0], out[1]
