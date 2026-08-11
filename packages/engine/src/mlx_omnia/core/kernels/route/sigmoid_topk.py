"""The routed sigmoid pick with the router gemv fused into the same dispatch.

One threadgroup, one simdgroup per expert: each computes its own dot product of the
token row against its router weight row, the sigmoid lands in threadgroup memory, and
simdgroup 0 runs the top-k scan over it. The routing chain the stock path pays for —
gemv, sigmoid, argpartition, take_along_axis, renormalize — is ~10 tiny dependent
kernels; this is one. Metal's 1024-thread limit caps the expert count at 32.

Two rounding differences against the op chain, both documented by the tests: the score
is the sigmoid of an f32 dot (the op chain rounds the gemv and the sigmoid to T), and
the renormalization sum accumulates in f32 in descending-score order. Selection matches:
ties go to the higher expert index, as mlx's stable ascending argpartition does.
"""

from dataclasses import dataclass
from typing import Self

import mlx.core as mx

from mlx_omnia.core.kernels.route.kernel import Routing
from mlx_omnia.core.mxcompat import metal_kernel


def _i32(value: int) -> mx.array:
    return mx.array(value, dtype=mx.int32)


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


_NORM_EPS = 1e-6


@dataclass(frozen=True)
class SigmoidTopkRoute:
    """The fused-gemv sigmoid pick. Alone among the strategies it reads the token row
    rather than a logit row, so it rejects a caller-supplied `logits` operand."""

    gate: mx.array
    routing: Routing
    scale: mx.array

    @classmethod
    def build(cls, gate: mx.array | None, *, routing: Routing) -> Self | None:
        if routing.scoring != "sigmoid" or routing.bias is None:
            return None
        if routing.shared or routing.hash_table is not None or routing.softcap != 0.0:
            return None
        if routing.norm_eps != (_NORM_EPS if routing.normalize else 0.0):
            return None
        if gate is None:
            return None
        experts, kdim = gate.shape
        if experts != routing.experts or experts > 32 or kdim % 4 != 0:
            return None
        if not 0 < routing.k <= 32:
            return None
        return cls(gate, routing, mx.array(routing.scale, dtype=mx.float32))

    def __call__(
        self,
        row: mx.array,
        *,
        logits: mx.array | None = None,
        ids: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        assert logits is None
        bias = self.routing.bias
        assert bias is not None
        return moe_route_sigmoid(
            row.reshape(-1),
            self.gate,
            bias,
            self.scale,
            self.routing.k,
            normalized=self.routing.normalize,
        )
