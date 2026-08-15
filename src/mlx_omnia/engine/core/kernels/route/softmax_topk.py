"""softmax -> top-k -> renormalize in one dispatch, bit-exact with the op chain.

The logits still come from the stock quantized matmul — recomputing them in a fused
kernel rounds differently and flips which expert wins a near-tie (measured: 1 in ~11
tokens). Ties go to the higher index, matching mlx's stable ascending argpartition.

One kernel family, four sources: the declaration's scoring (and its hash table) picks
which one `SoftmaxTopkRoute` dispatches. They share the shape the predicate below
states — one simdgroup owns the whole row — and nothing else.
"""

from dataclasses import dataclass
from typing import Self

import mlx.core as mx

from mlx_omnia.engine.core.kernels.route.kernel import RouteStrategy, Routing, gate_logits
from mlx_omnia.engine.core.mxcompat import metal_kernel

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

_SOFTPLUS_HEADER = """
inline float mlx_log1p(float x) {
    float xp1 = 1.0f + x;
    if (xp1 == 1.0f) return x;
    return x * (metal::log(xp1) / (xp1 - 1.0f));
}
inline float mlx_softplus(float v) {
    float maxval = metal::max(v, 0.0f);
    float minval = metal::min(v, 0.0f);
    return maxval + mlx_log1p(metal::exp(minval - maxval));
}
"""

_SOFTPLUS_SOURCE = """
    uint lane = thread_position_in_threadgroup.x;
    constexpr uint per_lane = EXPERTS / 32;

    float s[per_lane];
    float b[per_lane];
    for (uint i = 0; i < per_lane; i++) {
        float v = (float)L[lane + i * 32];
        s[i] = metal::precise::sqrt(mlx_softplus(v));
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
        float w = NORM ? pick[lane] / (total + 1e-20f) : pick[lane];
        OW[lane] = w * SC;
    }
"""

_SOFTPLUS_KERNEL = metal_kernel(
    name="moe_softplus_topk",
    input_names=["L", "B", "SC"],
    output_names=["OI", "OW"],
    source=_SOFTPLUS_SOURCE,
    header=_SOFTPLUS_HEADER,
)

_SOFTPLUS_GATHER_SOURCE = """
    uint lane = thread_position_in_threadgroup.x;
    float s = 0.0f;
    if (lane < TOPK) {
        float v = (float)L[C[lane]];
        s = metal::precise::sqrt(mlx_softplus(v));
    }
    float total = simd_sum(s);
    if (lane < TOPK) {
        float w = NORM ? s / (total + 1e-20f) : s;
        OW[lane] = w * SC;
    }
"""

_SOFTPLUS_GATHER_KERNEL = metal_kernel(
    name="moe_softplus_gather",
    input_names=["L", "C", "SC"],
    output_names=["OW"],
    source=_SOFTPLUS_GATHER_SOURCE,
    header=_SOFTPLUS_HEADER,
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


def softplus_topk(
    logits: mx.array, bias: mx.array, k: int, *, scale: float, norm: bool
) -> tuple[mx.array, mx.array]:
    """One token's `sqrt(softplus)` routing pick (DeepSeek-V4 style): selection by
    `score + bias`, weights from the unbiased scores, renormalized when `norm` and
    scaled. The score replicates mlx's lowering exactly — LogAddExp as
    `max + log1p(exp(min - max))` with the Goldberg `log1p`, `metal::precise::sqrt` —
    so a selection flip against the op chain needs a tie at fp32 resolution."""
    experts = logits.size
    assert softmax_topk_applies(experts, k)
    out = _SOFTPLUS_KERNEL(
        inputs=[logits, bias.astype(mx.float32), mx.array(scale, dtype=mx.float32)],
        template=[("T", logits.dtype), ("TOPK", k), ("EXPERTS", experts), ("NORM", int(norm))],
        grid=(32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(k,), (k,)],
        output_dtypes=[mx.uint32, mx.float32],
    )
    return out[0], out[1]


def softplus_gather(
    logits: mx.array, chosen: mx.array, *, scale: float, norm: bool
) -> mx.array:
    """The hash-routed variant: the experts are given, only their `sqrt(softplus)`
    weights are computed, renormalized when `norm` and scaled."""
    k = chosen.size
    assert 0 < k <= 32
    (out,) = _SOFTPLUS_GATHER_KERNEL(
        inputs=[logits, chosen, mx.array(scale, dtype=mx.float32)],
        template=[("T", logits.dtype), ("TOPK", k), ("NORM", int(norm))],
        grid=(32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(k,)],
        output_dtypes=[mx.float32],
    )
    return out


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


_SOFTPLUS_NORM_EPS = 1e-20


@dataclass(frozen=True)
class SoftmaxTopkRoute(RouteStrategy):
    """The one-simdgroup routing family: softmax, sigmoid or `sqrt(softplus)` scoring,
    top-k and the declared renormalization in a single dispatch over the router row."""

    gate: mx.array | None
    routing: Routing

    @classmethod
    def build(cls, gate: mx.array | None, *, routing: Routing) -> Self | None:
        if routing.softcap != 0.0:
            return None
        if not softmax_topk_applies(routing.experts, routing.k):
            return None
        if routing.hash_table is not None:
            # The gather variant computes the weights of experts it is handed.
            return (
                cls(gate, routing)
                if routing.scoring == "sqrt_softplus"
                and not routing.shared
                and routing.norm_eps == (_SOFTPLUS_NORM_EPS if routing.normalize else 0.0)
                else None
            )
        if routing.scoring == "softmax":
            # The kernel renormalizes, carries no bias and no scale.
            return (
                cls(gate, routing)
                if routing.bias is None
                and routing.normalize
                and routing.norm_eps == 0.0
                and routing.scale == 1.0
                else None
            )
        if routing.shared or routing.bias is None:
            return None
        if routing.scoring == "sigmoid":
            # The kernel always renormalizes, with no epsilon.
            return (
                cls(gate, routing)
                if routing.normalize and routing.norm_eps == 0.0
                else None
            )
        return (
            cls(gate, routing)
            if routing.norm_eps == (_SOFTPLUS_NORM_EPS if routing.normalize else 0.0)
            else None
        )

    def __call__(
        self,
        row: mx.array,
        *,
        logits: mx.array | None = None,
        ids: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        routing = self.routing
        values = gate_logits(self.gate, row, logits)
        table = routing.hash_table
        if table is not None:
            assert ids is not None
            chosen = table[ids].reshape(-1).astype(mx.uint32)
            return chosen, softplus_gather(
                values, chosen, scale=routing.scale, norm=routing.normalize
            )
        if routing.scoring == "softmax":
            return softmax_topk(values, routing.k, shared=routing.shared)
        bias = routing.bias
        assert bias is not None
        if routing.scoring == "sigmoid":
            return sigmoid_topk(values, bias, routing.k, scale=routing.scale)
        return softplus_topk(
            values, bias, routing.k, scale=routing.scale, norm=routing.normalize
        )
