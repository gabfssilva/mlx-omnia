"""The MoE tail of a prefill step, reading the expert-sorted rows where they lie.

A sorted-MoE prefill leaves the expert outputs in expert order, and the stock tail
un-sorts them into token order — a full copy of `tokens x top_k x hidden` — before the
router weights can be applied. This reads through the permutation instead:
`inverse_order[t, s]` is the row of the sorted buffer that the un-sort would have written
into token `t`'s slot `s`, so the gather costs an index load per slot and the copy
disappears. The eight row addresses are resolved once per thread and reused across the
four columns the thread owns.

The arithmetic is unchanged and stays bf16 in the stock order: slot 0 through slot
`top_k - 1`, each product rounded before it accumulates, the routed scaling factor applied
to the total, then the unrouted stack's output, then the residual. The scaling factor is
the model's, so it arrives as an input.
"""

import mlx.core as mx

from sideros.core.mxcompat import metal_kernel

_SOURCE = """
constexpr uint hidden = HIDDEN;
constexpr uint experts = TOPK;
constexpr uint n_cols = 4;

uint row = thread_position_in_grid.y;
uint col = thread_position_in_grid.x * n_cols;
const device float* weight_row = router_weights + row * experts;

bfloat expert_weights[experts];
uint sorted_rows[experts];
for (uint e = 0; e < experts; ++e) {
    expert_weights[e] = bfloat(weight_row[e]);
    sorted_rows[e] = inverse_order[row * experts + e];
}

for (uint i = 0; i < n_cols; ++i) {
    bfloat total = bfloat(0);
    for (uint e = 0; e < experts; ++e) {
        bfloat product = bfloat(
            sorted_expert_outputs[sorted_rows[e] * hidden + col + i] *
            expert_weights[e]);
        total = bfloat(product + total);
    }
    bfloat scaled = bfloat(total * bfloat(routed_scaling[0]));
    bfloat r2 = bfloat(scaled + shared_output[row * hidden + col + i]);
    output[row * hidden + col + i] =
        bfloat(residual[row * hidden + col + i] + r2);
}
"""

_KERNEL = metal_kernel(
    name="moe_sorted_tail",
    input_names=[
        "sorted_expert_outputs",
        "inverse_order",
        "router_weights",
        "shared_output",
        "residual",
        "routed_scaling",
    ],
    output_names=["output"],
    source=_SOURCE,
)


def moe_sorted_tail_applies(hidden: int) -> bool:
    """A thread owns four adjacent columns of one token."""
    return hidden % 4 == 0


def moe_sorted_tail(
    sorted_expert_outputs: mx.array,
    inverse_order: mx.array,
    router_weights: mx.array,
    shared_output: mx.array,
    residual: mx.array,
    routed_scaling: float,
) -> mx.array:
    """`sorted_expert_outputs` [tokens*top_k, hidden] bf16 in expert-sorted order, gathered
    through `inverse_order` [tokens, top_k] uint32, weighted by `router_weights` [tokens,
    top_k] fp32, scaled by `routed_scaling` and added to `shared_output` and `residual`
    (both [tokens, hidden] bf16) -> [tokens, hidden] bf16."""
    tokens = router_weights.shape[0]
    top_k = router_weights.shape[1]
    hidden = residual.shape[-1]
    assert moe_sorted_tail_applies(hidden)
    assert sorted_expert_outputs.dtype == mx.bfloat16
    assert sorted_expert_outputs.size == tokens * top_k * hidden
    assert inverse_order.dtype == mx.uint32 and inverse_order.size == tokens * top_k
    assert router_weights.dtype == mx.float32
    assert shared_output.dtype == mx.bfloat16 and shared_output.size == tokens * hidden
    assert residual.dtype == mx.bfloat16 and residual.size == tokens * hidden
    return _KERNEL(
        inputs=[
            sorted_expert_outputs,
            inverse_order,
            router_weights,
            shared_output,
            residual,
            mx.array([routed_scaling], dtype=mx.float32),
        ],
        template=[("HIDDEN", hidden), ("TOPK", top_k)],
        grid=(hidden // 4, tokens, 1),
        threadgroup=(min(256, hidden // 4), 1, 1),
        output_shapes=[(tokens, hidden)],
        output_dtypes=[mx.bfloat16],
    )[0]
