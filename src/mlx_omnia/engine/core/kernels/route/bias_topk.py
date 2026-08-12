"""softmax -> bias-to-scores -> top-k (no renorm) in one dispatch.

The Longcat Flash router adds `e_score_correction_bias` to the softmax *scores*
(not logits), selects the top-k from the biased scores, but pulls the routing
weights from the raw (unbiased) scores and multiplies by `routed_scaling_factor`
without renormalizing. Neither existing ``softmax_topk`` (which renormalizes and
takes no bias) nor ``sigmoid_topk`` (which is a different family) matches.

Ties go to the higher index, matching mlx's stable ascending argpartition — the
same rule ``softmax_topk`` and ``sigmoid_topk`` already follow.
"""

from dataclasses import dataclass
from typing import Self

import mlx.core as mx

from mlx_omnia.engine.core.kernels.route.kernel import Routing, gate_logits
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

    float b[per_lane];
    for (uint i = 0; i < per_lane; i++) {
        b[i] = (float)p[i] + (float)B[lane + i * 32];
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
        pick[j] = simd_broadcast(p[wslot], (ushort)((uint)winner % 32));
        if (winner == cand && slot >= 0) b[(uint)slot] = -INFINITY;
        if (lane == 0) OI[j] = (uint)winner;
    }

    if (lane < TOPK) {
        T w = (T)pick[lane];
        OW[lane] = (T)((float)w * SC);
    }
"""

_KERNEL = metal_kernel(
    name="softmax_bias_topk",
    input_names=["L", "B", "SC"],
    output_names=["OI", "OW"],
    source=_SOURCE,
)


def softmax_bias_topk_applies(experts: int, k: int) -> bool:
    """One simdgroup owns the whole row: ``experts / 32`` entries per lane, and
    the k winners are written by the first k lanes, so k can never exceed 32."""
    return experts >= 32 and experts % 32 == 0 and 0 < k <= 32


def softmax_bias_topk(
    logits: mx.array, bias: mx.array, k: int, *, scale: float
) -> tuple[mx.array, mx.array]:
    """One token's routing pick: logits [experts] + bias [experts] -> (indices [k],
    weights [k]).

    Softmax runs in fp32 and rounds to T before the bias is added; the bias is
    fp32. Selection is on the biased scores (ties to higher index); the weight
    is the raw softmax score (unbiased) multiplied by ``scale``, with no
    renormalization.
    """
    experts = logits.size
    assert softmax_bias_topk_applies(experts, k)
    out = _KERNEL(
        inputs=[logits, bias.astype(mx.float32), mx.array(scale, dtype=mx.float32)],
        template=[("T", logits.dtype), ("TOPK", k), ("EXPERTS", experts)],
        grid=(32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(k,), (k,)],
        output_dtypes=[mx.uint32, logits.dtype],
    )
    return out[0], out[1]


@dataclass(frozen=True)
class BiasTopkRoute:
    """Softmax scoring with the correction bias on the *scores*, selection on the biased
    scores and no renormalization — the Longcat Flash router."""

    gate: mx.array | None
    routing: Routing

    @classmethod
    def build(cls, gate: mx.array | None, *, routing: Routing) -> Self | None:
        if routing.scoring != "softmax" or routing.bias is None:
            return None
        if routing.normalize or routing.norm_eps != 0.0:
            return None
        if routing.shared or routing.hash_table is not None or routing.softcap != 0.0:
            return None
        if not softmax_bias_topk_applies(routing.experts, routing.k):
            return None
        return cls(gate, routing)

    def __call__(
        self,
        row: mx.array,
        *,
        logits: mx.array | None = None,
        ids: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        bias = self.routing.bias
        assert bias is not None
        return softmax_bias_topk(
            gate_logits(self.gate, row, logits),
            bias,
            self.routing.k,
            scale=self.routing.scale,
        )
