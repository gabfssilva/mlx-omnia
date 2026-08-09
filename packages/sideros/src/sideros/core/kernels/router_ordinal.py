"""Top-k routing by an order-preserving integer key, sorted by a bitonic tournament.

A routing pick is an ordering problem, not an arithmetic one, so the payload carried
through the sorting network is not the float score but its *ordinal*: the bit pattern of
the fp32 key remapped so that unsigned integer order reproduces float order (sign bit
flipped for positives, all bits flipped for negatives, +-0 collapsed to one value, NaN
sent above everything). Comparisons become integer compares, the payload halves, and the
tie rule becomes explicit — equal ordinals resolve to the *lower* index, which is a
stable descending sort by `score + bias` and the opposite of what `moe_route`'s
`simd_max` scan promises.

The key sorted is `-(score + bias)`, so ascending ordinal order is descending score
order and the winners land in the low lanes.

`router_tournament` sorts a whole row of experts in two phases. Phase one is a bitonic
sort inside each 32-lane block, alternating direction so `simd_shuffle_xor` alone carries
every stage. A block can contribute at most `k` of the global top-k, so only its best `k`
survive: `(experts / 32) * k` candidates, written to threadgroup memory at the one
barrier the kernel pays. Phase two sorts that candidate set — again simdgroup-local up to
sequence 32, then a single 64-crossing merge stage through threadgroup memory — and lanes
`0..k-1` hold the winners. Only the sorting payload crosses the barrier; the sigmoid
scores stay in a per-row table and are read back by index at the end, so the weights are
the original bytes and never a re-derivation.

`ORDINAL_HEADER` is shared with `residual_rms_router`, which emits the same keys straight
out of the fused router gemv so the sort has nothing left to compute.
"""

import mlx.core as mx

from sideros.core.mxcompat import metal_kernel

ORDINAL_HEADER = """
METAL_FUNC uint router_key_ordinal(float key) {
    uint bits = as_type<uint>(key);
    uint magnitude = bits & 0x7FFFFFFFu;
    if (magnitude > 0x7F800000u) {
        return 0xFFFFFFFFu;
    }
    if (magnitude == 0u) {
        return 0x80000000u;
    }
    return (bits & 0x80000000u) != 0u ? ~bits : (bits ^ 0x80000000u);
}

METAL_FUNC bool router_ordinal_before(
    uint a, uint a_index, uint b, uint b_index) {
    if (a < b) {
        return true;
    }
    if (b < a) {
        return false;
    }
    return a_index < b_index;
}
"""

_SOURCE = """
constexpr uint experts = EXPERTS;
constexpr uint top_k = TOPK;
constexpr uint candidates = (experts / 32) * top_k;

uint lane = thread_position_in_threadgroup.x;
uint row = threadgroup_position_in_grid.y;

threadgroup uint xchg_ordinals[candidates];
threadgroup uint xchg_indices[candidates];
threadgroup uint candidate_ordinals[candidates];
threadgroup uint candidate_indices[candidates];
threadgroup float original_scores[experts];

float x = float(logits[row * experts + lane]);
float y = 1.0f / (1.0f + metal::exp(metal::abs(x)));
float score = x < 0.0f ? y : 1.0f - y;
original_scores[lane] = score;
float key = -(score + float(correction_bias[lane]));
uint my_ordinal = router_key_ordinal(key);
uint my_index = lane;

for (uint sequence = 2; sequence <= 32; sequence <<= 1) {
    for (uint stride = sequence >> 1; stride > 0; stride >>= 1) {
        uint other_ordinal = simd_shuffle_xor(my_ordinal, ushort(stride));
        uint other_index = simd_shuffle_xor(my_index, ushort(stride));

        bool is_lower = (lane & stride) == 0;
        bool lower_wants_better = (lane & sequence) == 0;
        bool want_better = lower_wants_better == is_lower;
        bool other_before_my = router_ordinal_before(
            other_ordinal, other_index, my_ordinal, my_index);
        bool take_other = want_better ? other_before_my : !other_before_my;
        if (take_other) {
            my_ordinal = other_ordinal;
            my_index = other_index;
        }
    }
}

uint block = lane >> 5;
uint within_block = lane & 31;
bool block_ascending = (block & 1) == 0;
uint rank_in_block = block_ascending ? within_block : (31 - within_block);
bool is_local_top8 = block_ascending ? (within_block < top_k)
                                     : (within_block >= 32 - top_k);
if (is_local_top8) {
    candidate_ordinals[block * top_k + rank_in_block] = my_ordinal;
    candidate_indices[block * top_k + rank_in_block] = my_index;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

uint my_ordinal2 = 0u;
uint my_index2 = 0u;
if (lane < candidates) {
    my_ordinal2 = candidate_ordinals[lane];
    my_index2 = candidate_indices[lane];
}

// Sort one 64-candidate set instead of four duplicate copies. Sequences up to
// 32 are simdgroup-local, so inactive simdgroups can skip them entirely.
if (lane < candidates) {
for (uint sequence = 2; sequence <= 32; sequence <<= 1) {
    for (uint stride = sequence >> 1; stride > 0; stride >>= 1) {
        uint other_ordinal = simd_shuffle_xor(my_ordinal2, ushort(stride));
        uint other_index = simd_shuffle_xor(my_index2, ushort(stride));

        bool is_lower = (lane & stride) == 0;
        bool lower_wants_better = (lane & sequence) == 0;
        bool want_better = lower_wants_better == is_lower;
        bool other_before_my = router_ordinal_before(
            other_ordinal, other_index, my_ordinal2, my_index2);
        bool take_other = want_better ? other_before_my : !other_before_my;
        if (take_other) {
            my_ordinal2 = other_ordinal;
            my_index2 = other_index;
        }
    }
}
}

// The first stage of sequence 64 crosses the two active simdgroups. All 256
// threads reach the barrier, but only the 64 live candidates touch memory or
// execute the comparator.
if (lane < candidates) {
    xchg_ordinals[lane] = my_ordinal2;
    xchg_indices[lane] = my_index2;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (lane < candidates) {
    uint partner = lane ^ 32u;
    uint other_ordinal = xchg_ordinals[partner];
    uint other_index = xchg_indices[partner];
    bool is_lower = (lane & 32u) == 0;
    bool other_before_my = router_ordinal_before(
        other_ordinal, other_index, my_ordinal2, my_index2);
    bool take_other = is_lower ? other_before_my : !other_before_my;
    if (take_other) {
        my_ordinal2 = other_ordinal;
        my_index2 = other_index;
    }

    for (uint stride = 16; stride > 0; stride >>= 1) {
        other_ordinal = simd_shuffle_xor(my_ordinal2, ushort(stride));
        other_index = simd_shuffle_xor(my_index2, ushort(stride));
        is_lower = (lane & stride) == 0;
        other_before_my = router_ordinal_before(
            other_ordinal, other_index, my_ordinal2, my_index2);
        take_other = is_lower ? other_before_my : !other_before_my;
        if (take_other) {
            my_ordinal2 = other_ordinal;
            my_index2 = other_index;
        }
    }
}

float my_score2 = lane < top_k ? original_scores[my_index2] : 0.0f;
float total = 0.0f;
for (uint i = 0; i < top_k; ++i) {
    total = simd_shuffle(my_score2, ushort(i)) + total;
}
if (lane < top_k) {
    router_indices[row * top_k + lane] = my_index2;
    router_scores[row * top_k + lane] = my_score2 / total;
}
"""

_KERNEL = metal_kernel(
    name="router_tournament_ordinal",
    input_names=["logits", "correction_bias"],
    output_names=["router_indices", "router_scores"],
    source=_SOURCE,
    header=ORDINAL_HEADER,
)

_CANDIDATES = 64


def router_tournament_applies(experts: int, k: int) -> bool:
    """One thread per expert, one 32-lane block per bitonic run, and a candidate set the
    second phase merges with a single 64-crossing stage.

    The `lane ^ 32` exchange and the `stride = 16 .. 1` tail that follow it are that
    merge written out for exactly two simdgroups, so the candidate count — `k` survivors
    from each of the `experts / 32` blocks — has to be 64. `k <= 32` keeps the winners
    (and the reduction that renormalizes them) inside simdgroup 0.
    """
    return (
        experts % 32 == 0
        and 32 <= experts <= 1024
        and 0 < k <= 32
        and (experts // 32) * k == _CANDIDATES
    )


def router_tournament(
    logits: mx.array, correction_bias: mx.array, k: int
) -> tuple[mx.array, mx.array]:
    """Rows of router logits -> (top-`k` expert indices, renormalized sigmoid weights).

    Selection is on `sigmoid(logit) + correction_bias` descending, ties to the lower
    index; the weights are the *unbiased* sigmoid scores of the winners, renormalized
    over themselves. `logits` is `(..., experts)` and both outputs come back
    `(..., k)`.
    """
    experts = logits.shape[-1]
    rows = logits.size // experts
    assert router_tournament_applies(experts, k)
    assert correction_bias.shape == (experts,)
    assert correction_bias.dtype == mx.float32
    out = _KERNEL(
        inputs=[logits, correction_bias],
        template=[("EXPERTS", experts), ("TOPK", k)],
        grid=(experts, rows, 1),
        threadgroup=(experts, 1, 1),
        output_shapes=[(rows, k), (rows, k)],
        output_dtypes=[mx.uint32, mx.float32],
    )
    shape = (*logits.shape[:-1], k)
    return out[0].reshape(shape), out[1].reshape(shape)
