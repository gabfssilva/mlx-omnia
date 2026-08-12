"""The sorted-MoE permutation — gathered rows, sorted keys, inverse order — in one pass.

Grouping tokens by expert is a counting sort over a tiny alphabet: the keys *are* the
expert ids, so `argsort` is doing comparison work it never needs to. One threadgroup per
tile of the key stream, one thread per expert id: the threadgroup counts every key in the
stream (`tg_total`) and every key strictly before its own tile (`tg_before`) in the same
cooperative pass, a simd exclusive prefix over the totals gives the global base of each
key, and adding the before-count places the tile's slice. Counts are commutative integer
adds, so no accumulation order changes the tables — the histogram and scan dispatches the
stock chain pays for collapse into this one.

The tile's slice is then walked in input order, which is what makes the result stable and
identical to a stable `argsort` of the keys.

At that write point every downstream index product is already known, so all three leave
together: `idx / top_k` is the gathered row, the tested key is `keys[order]`, and the
write offset is the inverse permutation entry for `idx`. The `floorDivide`, the
`keys[order]` take and the inverse-permutation dispatch all disappear from the serial
sort -> gather chain, with the same integer values by construction.
"""

import mlx.core as mx

from mlx_omnia.engine.core.mxcompat import metal_kernel

_SOURCE = """
constexpr uint TILE = TILE_SIZE;
constexpr uint M = TOPK;
constexpr uint keys_count = NKEYS;
uint t = threadgroup_position_in_grid.x;
uint k = thread_position_in_threadgroup.x;
uint simd_id = k / 32;
uint lane = k % 32;
uint n = keys_shape[0];
// In-threadgroup histograms replace both the standalone hist
// dispatch and the scan dispatch: one cooperative pass counts
// every key (totals) and every key in earlier tiles (before),
// then a simd exclusive prefix over the totals yields the
// base table. Counts and sums are commutative integer adds, so
// any accumulation order produces the byte-identical tables.
threadgroup atomic_uint tg_total[keys_count];
threadgroup atomic_uint tg_before[keys_count];
atomic_store_explicit(&tg_total[k], 0u, memory_order_relaxed);
atomic_store_explicit(&tg_before[k], 0u, memory_order_relaxed);
threadgroup_barrier(mem_flags::mem_threadgroup);
// Split at the before-limit boundary so the tail segment
// carries no branch; identical counters, identical adds.
uint before_limit = t * TILE;
uint idx = k;
for (; idx < before_limit; idx += keys_count) {
    uint key = keys[idx];
    atomic_fetch_add_explicit(
        &tg_total[key], 1u, memory_order_relaxed);
    atomic_fetch_add_explicit(
        &tg_before[key], 1u, memory_order_relaxed);
}
for (; idx < n; idx += keys_count) {
    atomic_fetch_add_explicit(
        &tg_total[keys[idx]], 1u, memory_order_relaxed);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
uint total = atomic_load_explicit(&tg_total[k], memory_order_relaxed);
uint lane_excl = simd_prefix_exclusive_sum(total);
threadgroup uint simd_totals[keys_count / 32];
if (lane == 31) {
    simd_totals[simd_id] = lane_excl + total;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
uint simd_base = 0;
for (uint s = 0; s < simd_id; ++s) {
    simd_base += simd_totals[s];
}
// Rank base for key k in tile t: global base + earlier tiles.
uint off = simd_base + lane_excl +
    atomic_load_explicit(&tg_before[k], memory_order_relaxed);
// Walk this tile's slice in input order: stability by
// construction, exactly the stock scatter's write order.
for (uint i = 0; i < TILE; ++i) {
    uint idx = t * TILE + i;
    if (keys[idx] == k) {
        row_order[off] = idx / M;
        sorted_keys[off] = k;
        inverse_order[idx] = off;
        ++off;
    }
}
"""

_KERNEL = metal_kernel(
    name="route_counting_sort_fused",
    input_names=["keys"],
    output_names=["row_order", "sorted_keys", "inverse_order"],
    source=_SOURCE,
)

_TILE = 128


def route_counting_sort_applies(count: int, keys: int, top_k: int) -> bool:
    """One thread per key value and one threadgroup per fixed-size tile.

    The threadgroup is `keys` threads wide — it zeroes and reads one histogram slot per
    thread, and the exclusive prefix that turns the totals into bases runs over exactly
    those threads — so `keys` has to be a whole number of simdgroups within the device's
    1024-thread limit. The stream is walked in `TILE`-sized slices with no tail, so its
    length has to be a multiple of 128, and `top_k` only has to be the fan-out the flat
    stream was built with.
    """
    return (
        count > 0
        and count % _TILE == 0
        and keys % 32 == 0
        and 32 <= keys <= 1024
        and top_k > 0
    )


def route_counting_sort(
    keys: mx.array, experts: int, top_k: int
) -> tuple[mx.array, mx.array, mx.array]:
    """Group a flat `(tokens * top_k,)` stream of expert ids by expert.

    Returns `(row_order, sorted_keys, inverse_order)`: the token row each sorted slot
    gathers (`order // top_k`), the expert id it belongs to (`keys[order]`), and the
    position each input slot ended up at (`argsort(order)`) — the same three arrays the
    stable-`argsort` chain produces, and the same permutation.
    """
    count = keys.size
    assert keys.dtype == mx.uint32
    assert route_counting_sort_applies(count, experts, top_k)
    out = _KERNEL(
        inputs=[keys.reshape(count)],
        template=[("TILE_SIZE", _TILE), ("TOPK", top_k), ("NKEYS", experts)],
        grid=((count // _TILE) * experts, 1, 1),
        threadgroup=(experts, 1, 1),
        output_shapes=[(count,), (count,), (count,)],
        output_dtypes=[mx.uint32, mx.uint32, mx.uint32],
    )
    return out[0], out[1], out[2]
