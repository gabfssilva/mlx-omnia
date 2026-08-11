"""The dense T=1 MoE kernels against LFM2.5's op path, on a replica of the whole step.

The replica is the routed MLP of `LFM2SparseMLP` at the checkpoint's shapes (hidden
2048, moe_intermediate 1792, 32 experts, 4 per token, sigmoid routing with a float32
`expert_bias` used only for selection and `norm_topk_prob` eps 1e-6) and at the
checkpoint's magnitudes: rows scaled 1/sqrt(fan_in), so the dots land where bf16 ulps
actually separate candidates. The fp32 case runs the real kernel through its `T` template
with 8 experts (a full-width fp32 pair of stacks is 1.4 GB); the bf16 case carries all 32.
"""

import mlx.core as mx
import numpy as np
import pytest
from conftest import relative_diff
from sideros.core.kernels.moe_gemv_dense import (
    _DOWN_SOURCE,
    _GATE_UP_SOURCE,
    moe_dense_down,
    moe_dense_gate_up,
    moe_gemv_dense_applies,
    moe_route_sigmoid,
)

from sideros.core.mxcompat import gather_mm, metal_kernel

HIDDEN, INNER, EXPERTS, TOPK = 2048, 1792, 32, 4


class Replica:
    """One token through router, gate‖up and down, in the shapes the checkpoint has."""

    def __init__(self, experts: int, seed: int, dtype: mx.Dtype) -> None:
        rng = np.random.default_rng(seed)

        def normal(*shape: int, fan_in: int) -> mx.array:
            data = rng.standard_normal(shape) / np.sqrt(fan_in)
            return mx.array(data.astype(np.float32)).astype(dtype)

        self.experts = experts
        self.x = normal(HIDDEN, fan_in=HIDDEN)
        self.router = normal(experts, HIDDEN, fan_in=HIDDEN)
        self.w13 = normal(experts, 2 * INNER, HIDDEN, fan_in=HIDDEN)
        self.w2 = normal(experts, HIDDEN, INNER, fan_in=INNER)
        self.bias = mx.array((rng.standard_normal(experts) * 0.02).astype(np.float32))
        self.scale = mx.array(1.0, dtype=mx.float32)

    def route(self) -> tuple[mx.array, mx.array]:
        """`LFM2SparseMLP.route`, verbatim: bias shifts selection, not the weight."""
        scores = mx.sigmoid(self.x[None] @ self.router.T)
        selector = scores.astype(mx.float32) + self.bias
        chosen = mx.argpartition(selector, kth=self.experts - TOPK, axis=-1)[
            ..., self.experts - TOPK :
        ]
        weights = mx.take_along_axis(scores, chosen, axis=-1)
        weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-6)
        return chosen.reshape(-1), weights.reshape(-1)

    def mlp(self, indices: mx.array, weights: mx.array) -> mx.array:
        """The gather_mm path, fed the indices under test so only arithmetic is compared."""
        tokens = self.x.reshape(1, 1, 1, HIDDEN)
        fused = gather_mm(
            tokens, mx.swapaxes(self.w13, -2, -1), rhs_indices=indices[None], sorted_indices=False
        ).squeeze((0, 2))
        gated = fused[:, :INNER]
        activated = gated * mx.sigmoid(gated) * fused[:, INNER:]
        routed = gather_mm(
            activated[None, :, None], mx.swapaxes(self.w2, -2, -1), rhs_indices=indices[None]
        ).squeeze((0, 2))
        return (routed * weights.astype(routed.dtype)[:, None]).sum(axis=0)

    def kernels(self) -> tuple[mx.array, mx.array, mx.array]:
        indices, weights = moe_route_sigmoid(
            self.x, self.router, self.bias, self.scale, TOPK, normalized=True
        )
        gate_up = moe_dense_gate_up(self.x, self.w13, indices)
        return indices, weights, moe_dense_down(gate_up, self.w2, indices, weights).sum(axis=0)


def test_applies_covers_the_checkpoint_and_rejects_odd_shapes() -> None:
    assert moe_gemv_dense_applies(HIDDEN, INNER, EXPERTS, TOPK)
    assert not moe_gemv_dense_applies(HIDDEN + 2, INNER, EXPERTS, TOPK)
    assert not moe_gemv_dense_applies(HIDDEN, INNER + 1, EXPERTS, TOPK)
    assert not moe_gemv_dense_applies(HIDDEN, INNER, 64, TOPK)


def test_kernels_match_the_op_path_fp32() -> None:
    """fp32 template of the real kernels against gather_mm: 1e-5, the house bound."""
    replica = Replica(8, seed=3, dtype=mx.float32)
    indices, weights, out = replica.kernels()
    ref_indices, ref_weights = replica.route()

    assert set(np.array(indices).tolist()) == set(np.array(ref_indices).tolist())
    order = {int(i): w for i, w in zip(np.array(indices), np.array(weights), strict=True)}
    theirs = {int(i): w for i, w in zip(np.array(ref_indices), np.array(ref_weights), strict=True)}
    assert max(abs(order[i] - theirs[i]) for i in theirs) < 1e-6

    assert relative_diff(out, replica.mlp(indices, weights)) < 1e-5


def test_kernels_match_the_op_path_bf16() -> None:
    """bf16 at full width. The floor is the op path's own bf16 rounding, measured here
    against the same replica in fp32 — the kernel accumulates in f32, so it lands inside."""
    replica = Replica(EXPERTS, seed=11, dtype=mx.bfloat16)
    indices, weights, out = replica.kernels()
    ref_indices, ref_weights = replica.route()
    assert set(np.array(indices).tolist()) == set(np.array(ref_indices).tolist())
    # The op chain renormalizes in bf16; the kernel in f32, rounded once on the way out.
    ours = {int(i): w for i, w in zip(np.array(indices), np.array(weights.astype(mx.float32)),
                                      strict=True)}
    theirs = {int(i): w for i, w in zip(np.array(ref_indices),
                                        np.array(ref_weights.astype(mx.float32)), strict=True)}
    assert max(abs(ours[i] - theirs[i]) for i in theirs) < 4e-3

    ops = replica.mlp(ref_indices, ref_weights)
    exact = Replica(EXPERTS, seed=11, dtype=mx.float32)
    reference = exact.mlp(ref_indices, ref_weights.astype(mx.float32))

    floor = relative_diff(ops, reference)
    assert 0.0 < floor < 0.1
    assert relative_diff(out, reference) < 3 * floor


def test_selector_ties_go_to_the_higher_index_like_argpartition() -> None:
    """Five experts with an identical router row and no bias spread: the tie straddles
    the top-4 cutoff, and mlx's stable ascending argpartition keeps the higher indices."""
    replica = Replica(EXPERTS, seed=5, dtype=mx.float32)
    row = replica.router[0]
    tied = list(range(3, 8))
    replica.router = mx.concatenate(
        [row[None] if e in tied else replica.router[e][None] for e in range(EXPERTS)]
    )
    replica.bias = mx.zeros((EXPERTS,), dtype=mx.float32)
    scores = mx.sigmoid(replica.x[None] @ replica.router.T)
    assert float(scores[0, 3].item()) == float(mx.max(scores).item())

    indices, _ = moe_route_sigmoid(
        replica.x, replica.router, replica.bias, replica.scale, TOPK, normalized=True
    )
    chosen, _ = replica.route()
    assert sorted(np.array(indices).tolist()) == sorted(np.array(chosen).tolist()) == tied[1:]


_DOWN_MUTATIONS = {
    "drop the expert indirection": ("(size_t)IDX[expert] * n * kd", "(size_t)expert * n * kd"),
    "swap gate and up": ("float4 g = float4(gv[i]);", "float4 g = float4(uv[i]);"),
    "drop the routing weight": ("float scale = (float)WS[expert];", "float scale = 1.0f;"),
}


def _run_down(source: str, name: str, gate_up: mx.array, replica: Replica,
              indices: mx.array, weights: mx.array) -> mx.array:
    kernel = metal_kernel(
        name=name,
        input_names=["GU", "W", "IDX", "WS", "N", "KD"],
        output_names=["Y"],
        source=source,
    )
    return kernel(
        inputs=[
            gate_up, replica.w2, indices, weights,
            mx.array(HIDDEN, dtype=mx.int32), mx.array(INNER, dtype=mx.int32),
        ],
        template=[("T", mx.float32)],
        grid=(32, HIDDEN // 2, TOPK),
        threadgroup=(32, 1, 1),
        output_shapes=[(TOPK, HIDDEN)],
        output_dtypes=[mx.float32],
    )[0].sum(axis=0)


@pytest.mark.parametrize("mutation", sorted(_DOWN_MUTATIONS))
def test_down_mutations_break_parity(mutation: str) -> None:
    """Each documented break is compiled as its own kernel and must fail the fp32 bound."""
    old, new = _DOWN_MUTATIONS[mutation]
    replica = Replica(8, seed=3, dtype=mx.float32)
    indices, weights, out = replica.kernels()
    reference = replica.mlp(indices, weights)
    assert relative_diff(out, reference) < 1e-5

    source = _DOWN_SOURCE.replace(old, new)
    assert source != _DOWN_SOURCE
    broken = _run_down(source, f"down_broken_{sorted(_DOWN_MUTATIONS).index(mutation)}",
                       moe_dense_gate_up(replica.x, replica.w13, indices), replica,
                       indices, weights)
    assert relative_diff(broken, reference) > 1e-2


def test_gate_up_mutation_breaks_parity() -> None:
    """Dropping the expert indirection makes the gate‖up gemv read the wrong stack."""
    replica = Replica(8, seed=3, dtype=mx.float32)
    indices, _, _ = replica.kernels()
    source = _GATE_UP_SOURCE.replace("(size_t)IDX[expert] * n * kd", "(size_t)expert * n * kd")
    assert source != _GATE_UP_SOURCE
    broken = metal_kernel(
        name="gate_up_broken",
        input_names=["X", "W", "IDX", "N", "KD"],
        output_names=["Y"],
        source=source,
    )(
        inputs=[
            replica.x, replica.w13, indices,
            mx.array(2 * INNER, dtype=mx.int32), mx.array(HIDDEN, dtype=mx.int32),
        ],
        template=[("T", mx.float32)],
        grid=(32, INNER, TOPK),
        threadgroup=(32, 1, 1),
        output_shapes=[(TOPK, 2 * INNER)],
        output_dtypes=[mx.float32],
    )[0]
    assert relative_diff(broken, moe_dense_gate_up(replica.x, replica.w13, indices)) > 1e-2
