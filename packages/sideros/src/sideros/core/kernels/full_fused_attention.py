"""One decode step of a full-attention layer into a growable cache, in a single dispatch.

The sibling of `sliding_fused_attention` for a layer that attends to everything written
so far rather than to a fixed ring: Q/K RMSNorm, partial RoPE with a folded YaRN mscale,
the append into a cache that has spare backing capacity, and the flash-decode over rows
`[0, write_idx]` — one dispatch where the stock composition spends six.

Transcribed from `laguna_full_fused_attn_grow_v1` in the mlxfast tree. It carries the
same mechanisms as the sliding kernel — head pairing so a threadgroup's two query heads
share one normed K and staged V, the phase-1 split across simdgroups 0..3, the
`simd_shuffle` that fetches the partner half of the rotation, the read-back of row
`write_idx` out of threadgroup memory because no barrier orders simdgroup 0's device
write against the other simdgroups' reads, the `ONLINE_RESCALE` guard that keeps an
unmoved max at exactly `1.0f`, the two-deep key pipeline, and the bank-padded transposed
combine through `outputs[]` in two rounds of two planes. Three things are its own:

- **runtime length.** The row count is `params[1]`, not a compile-time constant, so the
  pipelined loop is followed by a single-row tail for the odd row a 2-wide stride leaves
  over. A simdgroup whose first row is already past the end contributes a `lowest()` max
  and a zero denominator, which the global combine rescales to nothing.
- **partial rotary with a passthrough tail.** Only `[0, 2 * rotary_pairs)` is rotated;
  lanes at or past `rotary_pairs / 2` copy their four normalized elements through
  untouched. Lanes between `rotary_pairs / 4` and `rotary_pairs / 2` write nothing — they
  exist to be the shuffle source for the rotation's second half.
- **the folded mscale.** The YaRN length factor is rounded to bfloat16 once and
  multiplied into both halves of the rotated block, each product rounded to bfloat16
  again, before the rotation runs in float32. It multiplies the rotated block only; the
  passthrough tail is left as the norm produced it. Both roundings are load-bearing —
  they are what the fused kernel has to reproduce to match the unfused chain it replaces.

Scores, the online softmax and the accumulators are float32; only the context and the
cache rows round back to bfloat16.

The kernel writes the new K/V row into `k_cache` / `v_cache` **in place**, casting away
the const mlx puts on an input pointer. Both must therefore already be row-contiguous —
`ensure_row_contiguous` would otherwise hand the kernel a copy and the write would be
lost — and the caller advances its own offset afterwards.

`head_dim` is a template parameter but the thread mapping pins it to 128: a lane holds
four elements (`vec<bfloat, 4>` loads, unrolled 4-wide dot products) and there are 32
lanes. `full_fused_attention_applies` states that, and the rest of the geometry, as the
contract.
"""

import hashlib
from functools import cache
from string import Template
from typing import TYPE_CHECKING

import mlx.core as mx

from sideros.core.mxcompat import metal_kernel

if TYPE_CHECKING:
    from sideros.core.mxcompat import MetalKernel

_HEAD_DIM = 128
_THREADS = 1024

_INPUTS = [
    "raw_queries",
    "raw_keys",
    "raw_values",
    "query_weight",
    "key_weight",
    "angles",
    "k_cache",
    "v_cache",
    "params",
    "scale_arr",
]

_HEADER = r"""
#define ONLINE_RESCALE(dst, delta_expr)         \
  do {                                          \
    const float db_delta_ = (delta_expr);       \
    if (as_type<uint>(db_delta_) == 0u) {       \
      dst = float(1.0f);                        \
    } else {                                    \
      dst = metal::fast::exp(db_delta_);        \
    }                                           \
  } while (false)

#define T_LOAD_K(dst, substitute, ptr)                     \
  do {                                                     \
    if (substitute) {                                      \
      dst[0] = tg_k[lane * qk_per_thread + 0];             \
      dst[1] = tg_k[lane * qk_per_thread + 1];             \
      dst[2] = tg_k[lane * qk_per_thread + 2];             \
      dst[3] = tg_k[lane * qk_per_thread + 3];             \
    } else {                                               \
      const vec<bfloat, 4> v_ =                            \
          *reinterpret_cast<const device vec<bfloat, 4>*>( \
              ptr);                                        \
      dst[0] = v_.x;                                       \
      dst[1] = v_.y;                                       \
      dst[2] = v_.z;                                       \
      dst[3] = v_.w;                                       \
    }                                                      \
  } while (false)

#define T_LOAD_V(d0, d1, d2, d3, substitute, ptr)          \
  do {                                                     \
    if (substitute) {                                      \
      d0 = tg_v[lane * v_per_thread + 0];                  \
      d1 = tg_v[lane * v_per_thread + 1];                  \
      d2 = tg_v[lane * v_per_thread + 2];                  \
      d3 = tg_v[lane * v_per_thread + 3];                  \
    } else {                                               \
      const vec<bfloat, 4> v_ =                            \
          *reinterpret_cast<const device vec<bfloat, 4>*>( \
              ptr);                                        \
      d0 = v_.x;                                           \
      d1 = v_.y;                                           \
      d2 = v_.z;                                           \
      d3 = v_.w;                                           \
    }                                                      \
  } while (false)
"""

# Every geometry constant the source baked in is either an mlx template parameter
# (integers) or a `$name` interpolated before compilation (floats, which mlx templates do
# not take). Nothing below names a model.
_SOURCE = """
constexpr uint head_dim = HEAD_DIM;
constexpr uint gqa = GQA;
constexpr int BN = 32;
constexpr int BD = 32;
constexpr int BDP = BD + 1;
constexpr int qk_per_thread = 4;
constexpr int v_per_thread = 4;
constexpr uint rotary_pairs = ROTARY_PAIRS;
constexpr float yarn_mscale = $mscale;

typedef float U;

uint pair_tg = threadgroup_position_in_grid.x;
uint head0 = pair_tg * 2;
uint head1 = head0 + 1;
uint kv_head = head0 / gqa;
uint sg = simdgroup_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
uint widx = params[0];
int N = int(params[1]);
uint capacity = params[2];
float scale = scale_arr[0];

threadgroup bfloat tg_q0[head_dim];
threadgroup bfloat tg_q1[head_dim];
threadgroup bfloat tg_k[head_dim];
threadgroup bfloat tg_v[head_dim];

if (sg < 3) {
    const device bfloat* input =
        sg == 0 ? raw_queries + head0 * head_dim
        : sg == 1 ? raw_queries + head1 * head_dim
                  : raw_keys + kv_head * head_dim;
    const device bfloat* weight =
        sg == 2 ? key_weight : query_weight;
    threadgroup bfloat* outrow =
        sg == 0 ? tg_q0 : sg == 1 ? tg_q1 : tg_k;

    uint base = lane * 4;
    thread bfloat normalized[4];
    float sum = 0.0f;
    for (uint i = 0; i < 4; ++i) {
        float value = float(input[base + i]);
        sum += value * value;
    }
    sum = simd_sum(sum);
    float inverse_rms = metal::precise::rsqrt(sum / float(head_dim) + $eps);
    for (uint i = 0; i < 4; ++i) {
        normalized[i] =
            weight[base + i] *
            bfloat(float(input[base + i]) * inverse_rms);
    }
    thread float paired[4];
    for (uint i = 0; i < 4; ++i) {
        paired[i] = simd_shuffle(float(normalized[i]), lane ^ (rotary_pairs / 4));
    }
    if (lane < rotary_pairs / 4) {
        bfloat rounded_mscale = bfloat(yarn_mscale);
        for (uint i = 0; i < 4; ++i) {
            uint pair = base + i;
            float first =
                float(bfloat(normalized[i] * rounded_mscale));
            float second =
                float(bfloat(bfloat(paired[i]) * rounded_mscale));
            float cosine = angles[pair];
            float sine = angles[pair + rotary_pairs];
            outrow[pair] = bfloat(first * cosine - second * sine);
            outrow[pair + rotary_pairs] =
                bfloat(first * sine + second * cosine);
        }
    } else if (lane >= rotary_pairs / 2) {
        for (uint i = 0; i < 4; ++i) {
            outrow[base + i] = normalized[i];
        }
    }
} else if (sg == 3) {
    const device bfloat* vin = raw_values + kv_head * head_dim;
    for (uint i = lane; i < head_dim; i += 32) {
        tg_v[i] = vin[i];
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);

if ((head0 % gqa) == 0 && sg == 0) {
    device bfloat* kc = (device bfloat*)k_cache +
        (size_t)kv_head * (capacity * head_dim) +
        (size_t)widx * head_dim;
    device bfloat* vc = (device bfloat*)v_cache +
        (size_t)kv_head * (capacity * head_dim) +
        (size_t)widx * head_dim;
    for (uint i = lane; i < head_dim; i += 32) {
        kc[i] = tg_k[i];
        vc[i] = tg_v[i];
    }
}

threadgroup U outputs[4 * BN * BDP];
threadgroup U max_scores[2 * BN];
threadgroup U sum_exp_scores[2 * BN];

const device bfloat* pair_keys = k_cache +
    (size_t)kv_head * (capacity * head_dim) +
    (size_t)sg * head_dim + lane * qk_per_thread;
const device bfloat* pair_values = v_cache +
    (size_t)kv_head * (capacity * head_dim) +
    (size_t)sg * head_dim + lane * v_per_thread;
const int inner_k_stride = BN * int(head_dim);
const int inner_v_stride = BN * int(head_dim);

thread U pair_q0[qk_per_thread];
thread U pair_q1[qk_per_thread];
thread U pair_k[qk_per_thread];
thread U pair_o0[v_per_thread];
thread U pair_o1[v_per_thread];

for (int j = 0; j < qk_per_thread; ++j) {
    pair_q0[j] =
        static_cast<U>(scale) * tg_q0[lane * qk_per_thread + j];
    pair_q1[j] =
        static_cast<U>(scale) * tg_q1[lane * qk_per_thread + j];
}
for (int j = 0; j < v_per_thread; ++j) {
    pair_o0[j] = 0;
    pair_o1[j] = 0;
}

U pair_max0 = metal::numeric_limits<U>::lowest();
U pair_max1 = metal::numeric_limits<U>::lowest();
U pair_sum0 = 0;
U pair_sum1 = 0;

int i = sg;
for (; i + BN < N; i += 2 * BN) {
    const device bfloat* pipe_keys_b = pair_keys + inner_k_stride;
    const device bfloat* pipe_values_b = pair_values + inner_v_stride;
    const bool sub_a = uint(i) == widx;
    const bool sub_b = uint(i + BN) == widx;
    U pipe_ka[4];
    U pipe_kb[4];
    T_LOAD_K(pipe_ka, sub_a, pair_keys);
    T_LOAD_K(pipe_kb, sub_b, pipe_keys_b);
    bfloat pipe_va0, pipe_va1, pipe_va2, pipe_va3;
    bfloat pipe_vb0, pipe_vb1, pipe_vb2, pipe_vb3;
    T_LOAD_V(pipe_va0, pipe_va1, pipe_va2, pipe_va3, sub_a,
        pair_values);
    T_LOAD_V(pipe_vb0, pipe_vb1, pipe_vb2, pipe_vb3, sub_b,
        pipe_values_b);

    U pair_score0 = 0;
    U pair_score1 = 0;
    pair_score0 += pair_q0[0] * pipe_ka[0];
    pair_score1 += pair_q1[0] * pipe_ka[0];
    pair_score0 += pair_q0[1] * pipe_ka[1];
    pair_score1 += pair_q1[1] * pipe_ka[1];
    pair_score0 += pair_q0[2] * pipe_ka[2];
    pair_score1 += pair_q1[2] * pipe_ka[2];
    pair_score0 += pair_q0[3] * pipe_ka[3];
    pair_score1 += pair_q1[3] * pipe_ka[3];
    pair_score0 = simd_sum(pair_score0);
    pair_score1 = simd_sum(pair_score1);

    U pair_new_max0 = metal::max(pair_max0, pair_score0);
    U pair_new_max1 = metal::max(pair_max1, pair_score1);
    U pair_factor0;
    U pair_factor1;
    ONLINE_RESCALE(pair_factor0, pair_max0 - pair_new_max0);
    ONLINE_RESCALE(pair_factor1, pair_max1 - pair_new_max1);
    U pair_exp0 = metal::fast::exp(pair_score0 - pair_new_max0);
    U pair_exp1 = metal::fast::exp(pair_score1 - pair_new_max1);

    pair_max0 = pair_new_max0;
    pair_max1 = pair_new_max1;
    pair_sum0 = pair_sum0 * pair_factor0 + pair_exp0;
    pair_sum1 = pair_sum1 * pair_factor1 + pair_exp1;

    pair_o0[0] = pair_o0[0] * pair_factor0 + pair_exp0 * pipe_va0;
    pair_o1[0] = pair_o1[0] * pair_factor1 + pair_exp1 * pipe_va0;
    pair_o0[1] = pair_o0[1] * pair_factor0 + pair_exp0 * pipe_va1;
    pair_o1[1] = pair_o1[1] * pair_factor1 + pair_exp1 * pipe_va1;
    pair_o0[2] = pair_o0[2] * pair_factor0 + pair_exp0 * pipe_va2;
    pair_o1[2] = pair_o1[2] * pair_factor1 + pair_exp1 * pipe_va2;
    pair_o0[3] = pair_o0[3] * pair_factor0 + pair_exp0 * pipe_va3;
    pair_o1[3] = pair_o1[3] * pair_factor1 + pair_exp1 * pipe_va3;

    U pipeb_score0 = 0;
    U pipeb_score1 = 0;
    pipeb_score0 += pair_q0[0] * pipe_kb[0];
    pipeb_score1 += pair_q1[0] * pipe_kb[0];
    pipeb_score0 += pair_q0[1] * pipe_kb[1];
    pipeb_score1 += pair_q1[1] * pipe_kb[1];
    pipeb_score0 += pair_q0[2] * pipe_kb[2];
    pipeb_score1 += pair_q1[2] * pipe_kb[2];
    pipeb_score0 += pair_q0[3] * pipe_kb[3];
    pipeb_score1 += pair_q1[3] * pipe_kb[3];
    pipeb_score0 = simd_sum(pipeb_score0);
    pipeb_score1 = simd_sum(pipeb_score1);

    U pipeb_new_max0 = metal::max(pair_max0, pipeb_score0);
    U pipeb_new_max1 = metal::max(pair_max1, pipeb_score1);
    U pipeb_factor0;
    U pipeb_factor1;
    ONLINE_RESCALE(pipeb_factor0, pair_max0 - pipeb_new_max0);
    ONLINE_RESCALE(pipeb_factor1, pair_max1 - pipeb_new_max1);
    U pipeb_exp0 = metal::fast::exp(pipeb_score0 - pipeb_new_max0);
    U pipeb_exp1 = metal::fast::exp(pipeb_score1 - pipeb_new_max1);

    pair_max0 = pipeb_new_max0;
    pair_max1 = pipeb_new_max1;
    pair_sum0 = pair_sum0 * pipeb_factor0 + pipeb_exp0;
    pair_sum1 = pair_sum1 * pipeb_factor1 + pipeb_exp1;

    pair_o0[0] = pair_o0[0] * pipeb_factor0 + pipeb_exp0 * pipe_vb0;
    pair_o1[0] = pair_o1[0] * pipeb_factor1 + pipeb_exp1 * pipe_vb0;
    pair_o0[1] = pair_o0[1] * pipeb_factor0 + pipeb_exp0 * pipe_vb1;
    pair_o1[1] = pair_o1[1] * pipeb_factor1 + pipeb_exp1 * pipe_vb1;
    pair_o0[2] = pair_o0[2] * pipeb_factor0 + pipeb_exp0 * pipe_vb2;
    pair_o1[2] = pair_o1[2] * pipeb_factor1 + pipeb_exp1 * pipe_vb2;
    pair_o0[3] = pair_o0[3] * pipeb_factor0 + pipeb_exp0 * pipe_vb3;
    pair_o1[3] = pair_o1[3] * pipeb_factor1 + pipeb_exp1 * pipe_vb3;

    pair_keys += 2 * inner_k_stride;
    pair_values += 2 * inner_v_stride;
}
if (i < N) {
    const bool sub_t = uint(i) == widx;
    T_LOAD_K(pair_k, sub_t, pair_keys);
    bfloat pipe_va0, pipe_va1, pipe_va2, pipe_va3;
    T_LOAD_V(pipe_va0, pipe_va1, pipe_va2, pipe_va3, sub_t,
        pair_values);

    U pair_score0 = 0;
    U pair_score1 = 0;
    pair_score0 += pair_q0[0] * pair_k[0];
    pair_score1 += pair_q1[0] * pair_k[0];
    pair_score0 += pair_q0[1] * pair_k[1];
    pair_score1 += pair_q1[1] * pair_k[1];
    pair_score0 += pair_q0[2] * pair_k[2];
    pair_score1 += pair_q1[2] * pair_k[2];
    pair_score0 += pair_q0[3] * pair_k[3];
    pair_score1 += pair_q1[3] * pair_k[3];
    pair_score0 = simd_sum(pair_score0);
    pair_score1 = simd_sum(pair_score1);

    U pair_new_max0 = metal::max(pair_max0, pair_score0);
    U pair_new_max1 = metal::max(pair_max1, pair_score1);
    U pair_factor0;
    U pair_factor1;
    ONLINE_RESCALE(pair_factor0, pair_max0 - pair_new_max0);
    ONLINE_RESCALE(pair_factor1, pair_max1 - pair_new_max1);
    U pair_exp0 = metal::fast::exp(pair_score0 - pair_new_max0);
    U pair_exp1 = metal::fast::exp(pair_score1 - pair_new_max1);

    pair_max0 = pair_new_max0;
    pair_max1 = pair_new_max1;
    pair_sum0 = pair_sum0 * pair_factor0 + pair_exp0;
    pair_sum1 = pair_sum1 * pair_factor1 + pair_exp1;

    pair_o0[0] = pair_o0[0] * pair_factor0 + pair_exp0 * pipe_va0;
    pair_o1[0] = pair_o1[0] * pair_factor1 + pair_exp1 * pipe_va0;
    pair_o0[1] = pair_o0[1] * pair_factor0 + pair_exp0 * pipe_va1;
    pair_o1[1] = pair_o1[1] * pair_factor1 + pair_exp1 * pipe_va1;
    pair_o0[2] = pair_o0[2] * pair_factor0 + pair_exp0 * pipe_va2;
    pair_o1[2] = pair_o1[2] * pair_factor1 + pair_exp1 * pipe_va2;
    pair_o0[3] = pair_o0[3] * pair_factor0 + pair_exp0 * pipe_va3;
    pair_o1[3] = pair_o1[3] * pair_factor1 + pair_exp1 * pipe_va3;
}

constexpr int pair_planes = 2;
constexpr int pair_plane_size = BN * BDP;
if (lane == 0) {
    max_scores[sg] = pair_max0;
    max_scores[BN + sg] = pair_max1;
    sum_exp_scores[sg] = pair_sum0;
    sum_exp_scores[BN + sg] = pair_sum1;
}
for (int p = 0; p < pair_planes; ++p) {
    outputs[p * pair_plane_size + lane * BDP + sg] = pair_o0[p];
    outputs[
        (pair_planes + p) * pair_plane_size + lane * BDP + sg] =
        pair_o1[p];
}
threadgroup_barrier(mem_flags::mem_threadgroup);

pair_max0 = max_scores[lane];
pair_max1 = max_scores[BN + lane];
U pair_global_max0 = simd_max(pair_max0);
U pair_global_max1 = simd_max(pair_max1);
U pair_global_factor0 = metal::fast::exp(pair_max0 - pair_global_max0);
U pair_global_factor1 = metal::fast::exp(pair_max1 - pair_global_max1);
pair_sum0 = simd_sum(sum_exp_scores[lane] * pair_global_factor0);
pair_sum1 = simd_sum(sum_exp_scores[BN + lane] * pair_global_factor1);

for (int p = 0; p < pair_planes; ++p) {
    U acc0 = simd_sum(
        outputs[p * pair_plane_size + sg * BDP + lane] *
        pair_global_factor0);
    U acc1 = simd_sum(
        outputs[
            (pair_planes + p) * pair_plane_size + sg * BDP + lane] *
        pair_global_factor1);
    pair_o0[p] = pair_sum0 == 0 ? acc0 : (acc0 / pair_sum0);
    pair_o1[p] = pair_sum1 == 0 ? acc1 : (acc1 / pair_sum1);
}

threadgroup_barrier(mem_flags::mem_threadgroup);
for (int p = 0; p < pair_planes; ++p) {
    outputs[p * pair_plane_size + lane * BDP + sg] =
        pair_o0[pair_planes + p];
    outputs[
        (pair_planes + p) * pair_plane_size + lane * BDP + sg] =
        pair_o1[pair_planes + p];
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (int p = 0; p < pair_planes; ++p) {
    U acc0 = simd_sum(
        outputs[p * pair_plane_size + sg * BDP + lane] *
        pair_global_factor0);
    U acc1 = simd_sum(
        outputs[
            (pair_planes + p) * pair_plane_size + sg * BDP + lane] *
        pair_global_factor1);
    pair_o0[pair_planes + p] =
        pair_sum0 == 0 ? acc0 : (acc0 / pair_sum0);
    pair_o1[pair_planes + p] =
        pair_sum1 == 0 ? acc1 : (acc1 / pair_sum1);
}

if (lane == 0) {
    device bfloat* pair_out0 =
        attended + head0 * head_dim + sg * v_per_thread;
    device bfloat* pair_out1 =
        attended + head1 * head_dim + sg * v_per_thread;
    for (int p = 0; p < v_per_thread; ++p) {
        pair_out0[p] = static_cast<bfloat>(pair_o0[p]);
        pair_out1[p] = static_cast<bfloat>(pair_o1[p]);
    }
}
"""


def _metal_float(value: float) -> str:
    return f"{value!r}f"


@cache
def _build(source: str, header: str) -> "MetalKernel":
    """The kernel name carries a digest of the text: mlx caches a compiled library by
    name, so two variants of this source must not answer to the same one. Parameterized
    by source and header so a mutation test can rebuild a broken variant."""
    digest = hashlib.blake2b((header + source).encode(), digest_size=6).hexdigest()
    return metal_kernel(
        name=f"full_fused_attn_grow_{digest}",
        input_names=_INPUTS,
        output_names=["attended"],
        source=source,
        header=header,
    )


@cache
def _kernel(eps: float, mscale: float) -> "MetalKernel":
    source = Template(_SOURCE).substitute(
        eps=_metal_float(eps), mscale=_metal_float(mscale)
    )
    return _build(source, _HEADER)


def _rotary_pairs(angles: mx.array) -> int:
    """`angles` is `[cos(0..r-1), sin(0..r-1)]`, so it is twice the pair count."""
    return angles.size // 2


def full_fused_attention_applies(
    raw_queries: mx.array,
    raw_keys: mx.array,
    raw_values: mx.array,
    query_weight: mx.array,
    key_weight: mx.array,
    angles: mx.array,
    k_cache: mx.array,
    v_cache: mx.array,
    write_idx: int | mx.array,
) -> bool:
    """Everything bfloat16 but `angles` (float32); `k_cache`/`v_cache` a
    `[1, kv_heads, capacity, 128]` buffer with room for row `write_idx`, kept
    row-contiguous by the caller.

    `head_dim == 128` because a lane holds four elements across 32 lanes. An even head
    count so the pair mapping exists, and an even `gqa` so both heads of a pair share a kv
    head. `rotary_pairs = angles.size / 2` has to land on a lane boundary from both sides:
    a multiple of four (the rotated half ends at lane `rotary_pairs / 4`) whose quarter is
    a power of two (the partner lane is reached by XOR, not by addition), and the rotated
    block `2 * rotary_pairs` must fit inside the head.

    Any row count works — the tail iteration covers what the 2-wide stride leaves — down
    to a cache holding only row 0.
    """
    if k_cache.ndim != 4 or v_cache.shape != k_cache.shape:
        return False
    batch, kv_heads, capacity, head_dim = k_cache.shape
    if batch != 1 or head_dim != _HEAD_DIM:
        return False
    if isinstance(write_idx, int):
        if not 0 <= write_idx < capacity:
            return False
    elif write_idx.size != 1:
        return False
    if raw_queries.ndim != 3 or raw_queries.shape[:2] != (1, 1):
        return False
    heads, remainder = divmod(raw_queries.shape[2], head_dim)
    if remainder != 0 or heads % 2 != 0 or heads % kv_heads != 0:
        return False
    if (heads // kv_heads) % 2 != 0:
        return False
    if raw_keys.shape != (1, 1, kv_heads * head_dim):
        return False
    if raw_values.shape != raw_keys.shape:
        return False
    if query_weight.size != head_dim or key_weight.size != head_dim:
        return False
    if angles.dtype != mx.float32 or angles.size % 2 != 0:
        return False
    pairs = _rotary_pairs(angles)
    quarter = pairs // 4
    if pairs == 0 or pairs % 4 != 0 or quarter & (quarter - 1) != 0:
        return False
    if 2 * pairs > head_dim:
        return False
    return all(
        array.dtype == mx.bfloat16
        for array in (
            raw_queries,
            raw_keys,
            raw_values,
            query_weight,
            key_weight,
            k_cache,
            v_cache,
        )
    )


def full_fused_attention(
    raw_queries: mx.array,
    raw_keys: mx.array,
    raw_values: mx.array,
    query_weight: mx.array,
    key_weight: mx.array,
    angles: mx.array,
    k_cache: mx.array,
    v_cache: mx.array,
    write_idx: int | mx.array,
    scale: float,
    eps: float,
    mscale: float,
) -> mx.array:
    """A whole full-attention decode step: normalize, rotate, append, attend.

    `raw_queries` [1, 1, heads*128] and `raw_keys`/`raw_values` [1, 1, kv_heads*128] are
    the projection outputs, unnormalized and unrotated. `query_weight`/`key_weight` are
    the per-head RMSNorm gains [128]. `angles` is [cos(0..r-1), sin(0..r-1)] float32 for
    this one position, `r` pairs covering the head's first `2r` dimensions; the rest is
    passed through unrotated. `mscale` is the YaRN length factor, folded into the rotated
    block only. `k_cache`/`v_cache` are the [1, kv_heads, capacity, 128] buffer —
    **mutated in place** at row `write_idx` with the new K/V — and rows `[0, write_idx]`
    are the ones attended.

    Returns the context [1, heads, 1, 128], the shape
    `mx.fast.scaled_dot_product_attention` returns.
    """
    assert full_fused_attention_applies(
        raw_queries,
        raw_keys,
        raw_values,
        query_weight,
        key_weight,
        angles,
        k_cache,
        v_cache,
        write_idx,
    )
    kv_heads, capacity, head_dim = k_cache.shape[1:]
    heads = raw_queries.shape[2] // head_dim
    return _kernel(eps, mscale)(
        inputs=[
            raw_queries,
            raw_keys,
            raw_values,
            query_weight,
            key_weight,
            angles,
            k_cache,
            v_cache,
            mx.array([write_idx, write_idx + 1, capacity], dtype=mx.uint32)
            if isinstance(write_idx, int)
            else mx.concatenate(
                [
                    write_idx.astype(mx.uint32),
                    write_idx.astype(mx.uint32) + 1,
                    mx.array([capacity], dtype=mx.uint32),
                ]
            ),
            mx.array([scale], dtype=mx.float32),
        ],
        template=[
            ("HEAD_DIM", head_dim),
            ("GQA", heads // kv_heads),
            ("ROTARY_PAIRS", _rotary_pairs(angles)),
        ],
        grid=((heads // 2) * _THREADS, 1, 1),
        threadgroup=(_THREADS, 1, 1),
        output_shapes=[(1, heads, 1, head_dim)],
        output_dtypes=[mx.bfloat16],
    )[0]
