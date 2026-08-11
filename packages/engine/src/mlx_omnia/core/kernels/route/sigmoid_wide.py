"""sigmoid -> bias-to-scores -> top-k -> renormalize, for expert counts past 32.

`sigmoid_topk.py` fuses the router gemv and caps at 32 experts (one simdgroup per
expert inside one threadgroup); this family keeps the gemv outside — one mx matmul,
exactly the op chain's — and runs the rest of the chain in one dispatch: one simdgroup,
``experts / 32`` scores per lane. The DeepSeek-style sigmoid router: scores are the
fp32 sigmoid of the logits, the correction bias shifts selection only, the kept weights
are the raw scores renormalized by their sum plus ``norm_eps`` and scaled.

Ties go to the higher index, matching mlx's stable ascending argpartition. One
documented rounding difference against the op chain: the renormalizing sum accumulates
in descending-score order, where `take_along_axis` sums in argpartition's partition
order — same set, different fp32 order.
"""

from dataclasses import dataclass
from typing import Self

import mlx.core as mx

from mlx_omnia.core.kernels.route.kernel import Routing, gate_logits
from mlx_omnia.core.mxcompat import metal_kernel

_SOURCE = """
    uint lane = thread_position_in_threadgroup.x;
    constexpr uint per_lane = EXPERTS / 32;

    float p[per_lane];
    for (uint i = 0; i < per_lane; i++) {
        float x = (float)L[lane + i * 32];
        float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(x)));
        p[i] = (x < 0.0f) ? y : 1.0f - y;
    }

    float b[per_lane];
    for (uint i = 0; i < per_lane; i++) {
        b[i] = p[i] + (float)B[lane + i * 32];
    }

    float pick[TOPK];
    float total = 0.0f;
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
        total += pick[j];
        if (winner == cand && slot >= 0) b[(uint)slot] = -INFINITY;
        if (lane == 0) OI[j] = (uint)winner;
    }

    if (lane < TOPK) {
        float w = pick[lane];
        OW[lane] = (NORM ? w / (total + EPS) : w) * SC;
    }
"""

_KERNEL = metal_kernel(
    name="sigmoid_wide_topk",
    input_names=["L", "B", "SC", "EPS"],
    output_names=["OI", "OW"],
    source=_SOURCE,
)

_ROWS_SOURCE = _SOURCE.replace(
    "uint lane = thread_position_in_threadgroup.x;",
    """uint lane = thread_position_in_threadgroup.x;
    uint row_t = threadgroup_position_in_grid.z;
    const device T* Lr = L + row_t * (uint)EXPERTS;
    device uint* OIr = OI + row_t * (uint)TOPK;
    device float* OWr = OW + row_t * (uint)TOPK;""",
).replace("L[lane", "Lr[lane").replace("OI[j]", "OIr[j]").replace("OW[lane]", "OWr[lane]")

_ROWS_KERNEL = metal_kernel(
    name="sigmoid_wide_topk_rows",
    input_names=["L", "B", "SC", "EPS"],
    output_names=["OI", "OW"],
    source=_ROWS_SOURCE,
)



def sigmoid_wide_applies(experts: int, k: int) -> bool:
    """One simdgroup owns the whole row: ``experts / 32`` scores per lane, and the k
    winners are written by the first k lanes, so k can never exceed 32."""
    return experts > 32 and experts % 32 == 0 and 0 < k <= 32


@dataclass(frozen=True)
class SigmoidWideRoute:
    """The unfused-gemv sigmoid pick for wide routers: the logits arrive from the op
    chain's own matmul, and everything after it is one dispatch."""

    gate: mx.array | None
    routing: Routing
    scale: mx.array
    eps: mx.array

    @classmethod
    def build(cls, gate: mx.array | None, *, routing: Routing) -> Self | None:
        if routing.scoring != "sigmoid" or routing.bias is None:
            return None
        if routing.shared or routing.hash_table is not None or routing.softcap != 0.0:
            return None
        if not sigmoid_wide_applies(routing.experts, routing.k):
            return None
        return cls(
            gate,
            routing,
            mx.array(routing.scale, dtype=mx.float32),
            mx.array(routing.norm_eps, dtype=mx.float32),
        )

    def __call__(
        self,
        row: mx.array,
        *,
        logits: mx.array | None = None,
        ids: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        bias = self.routing.bias
        assert bias is not None
        out = _KERNEL(
            inputs=[
                gate_logits(self.gate, row, logits),
                bias.astype(mx.float32),
                self.scale,
                self.eps,
            ],
            template=[
                ("T", row.dtype),
                ("TOPK", self.routing.k),
                ("EXPERTS", self.routing.experts),
                ("NORM", int(self.routing.normalize and self.routing.k > 1)),
            ],
            grid=(32, 1, 1),
            threadgroup=(32, 1, 1),
            output_shapes=[(self.routing.k,), (self.routing.k,)],
            output_dtypes=[mx.uint32, mx.float32],
        )
        return out[0], out[1]

    def rows(self, logits: mx.array) -> tuple[mx.array, mx.array]:
        """The same pick over T logit rows in one dispatch — one threadgroup per row,
        each running exactly the single-row kernel's chain, so the outputs match the
        single-row calls row for row."""
        bias = self.routing.bias
        assert bias is not None
        tokens = logits.shape[0]
        out = _ROWS_KERNEL(
            inputs=[logits, bias.astype(mx.float32), self.scale, self.eps],
            template=[
                ("T", logits.dtype),
                ("TOPK", self.routing.k),
                ("EXPERTS", self.routing.experts),
                ("NORM", int(self.routing.normalize and self.routing.k > 1)),
            ],
            grid=(32, 1, tokens),
            threadgroup=(32, 1, 1),
            output_shapes=[(tokens, self.routing.k), (tokens, self.routing.k)],
            output_dtypes=[mx.uint32, mx.float32],
        )
        return out[0], out[1]
