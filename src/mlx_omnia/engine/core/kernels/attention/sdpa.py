"""Decode-path scaled-dot-product attention on the mlxfast-challenge record kernel.

A lift, not a reimplementation. Source: `sdpa_vector.h` of Layr-Labs/mlxfast-challenge
(MIT), whose baseline at commit 38fcfff9c865ff45244cb491629108d913e35d77 is byte-identical
to the `sdpa_vector.h` the mlx 0.32.0 wheel ships, and whose patched form at commit
c5b0a13c5cc032b485022db41bcd745792316714 is the text `_SDPA_VECTOR` carries.

What the lifted text contains, in the submitters' terms:

- *the widened output-transpose exchange plane* ("SDPA vector: widen the output-transpose
  exchange plane (8 barriers -> 1)", their reported score delta +0.0216). The combine loop
  transposed the per-simdgroup output partials through one recycled threadgroup plane, so
  every element paid a RAW barrier after its write and a WAR barrier before the next write.
  One plane per element removes both, and the single surviving rendezvous absorbs the
  max/sum publish. At head_dim 128 (four output elements per thread) that is 8 barriers
  down to 1 for 16 KiB of threadgroup memory. `PLANES` is their shipped 4;
- *K/V reads shared across adjacent GQA heads* (PR 249, their reported +0.0315). Two
  adjacent query heads run in one threadgroup, each keeping its own query registers,
  online-softmax state, output accumulator and the stock 32-simdgroup reduction tree, and
  sharing every key and value load. The grid is unchanged; its upper half returns before
  touching memory. Runtime-restricted to head_dim 128, GQA factor 6 or 8, one query row,
  unmasked, non-causal, sink-free, and K/V strides and base pointers aligned for 8-byte
  loads. Anything else falls through to the per-head path;
- *paired-GQA load pipelining* (PR 725, the first of its two parts): a two-deep software
  pipeline that hoists both positions' K and V rows to the top of the trip, 8-byte
  `vec<T,4>` K/V loads in place of four scalar loads, and a manual full unroll of the
  four-element score loop;
- *ALPHASKIP*, a bit-exact elision on the online-softmax rescale. When the running max does
  not advance the delta is bit-exactly +0.0f and `fast::exp` returns 1.0f, so the branch
  substitutes that constant and leaves every downstream expression character-identical.
  Their own A/B calls the timing a null; it is here because it arrived with the file and
  costs nothing.

What is left behind:

- PR 1670's KV-native two-group schedule (two groups of three or four query heads per KV
  head, six exchange planes) is not in the file at c5b0a13 — that revision still pairs
  heads two at a time and combines through four planes. Lifting it means lifting a later
  revision of the header;
- PR 725's second part, the NVFP4 nibble unpack split, edits the submission's own model
  file, not the attention kernel;
- `sdpa_vector_2pass_1`/`_2`. Their two-pass plane count ships at 1, i.e. upstream, and
  nothing here dispatches the two-pass split;
- their AOT measurement scaffolding (the `_planes1/2/4` named variants and the environment
  variable that appends the suffix), which only exists to A/B one vendored Swift metallib.

The edits to the vendored text, all mechanical:

- the seven `[[function_constant(n)]]` globals become template parameters of the same
  names, so the body is untouched and each arm is chosen at instantiation.
  `mx.fast.metal_kernel` has no way to set function constants;
- `[[kernel]]` and the `[[buffer(n)]]` / `[[threadgroup_position_in_grid]]`-style parameter
  attributes come off. A Metal kernel cannot be called, so this is a plain function
  template and the entry point mlx generates from `source=` instantiates it — that
  instantiation is what their `.metal` wrapper used to provide;
- the three `threadgroup` arrays become `threadgroup float*` parameters, declared in the
  entry point instead. Metal only allows a `threadgroup` variable to be declared inside a
  kernel function, and this is no longer one. Indexing is unchanged;
- `const constant X&` scalar parameters become by-value `X`;
- `DEFAULT_ENTRY` is dropped (it only kept two entry points from colliding) and `PLANES` is
  passed explicitly instead of defaulted from their macro;
- their file-level prose about the vendored Swift build system and its measurement protocol
  is dropped. The in-function comments, which argue exactness, are kept verbatim.

`mx.fast.metal_kernel` prepends `mlx/backend/metal/kernels/utils.h` to every JIT source
(`metal::utils() + source_`, `backend/metal/custom_kernel.cpp`), so `Limits<>`,
`bfloat16_t` and `using namespace metal` are already in scope: nothing is spliced in.
"""

from collections.abc import Callable

import mlx.core as mx

from mlx_omnia.engine.core.mxcompat import metal_kernel
from mlx_omnia.engine.core.patch import Patch

_HEAD_DIM = 128
_THREADS = 1024
# mlx passes an input with fewer than 8 elements in the `constant` address space; the
# vendored mask parameter is a `device` pointer.
_MIN_MASK_ELEMENTS = 8

_SDPA_VECTOR = r"""
using namespace metal;

#ifndef DARKBLOOM_GQA_PAIR_HEADS
#define DARKBLOOM_GQA_PAIR_HEADS 2
#endif

#ifndef DARKBLOOM_ALPHASKIP
#define DARKBLOOM_ALPHASKIP 1
#endif

#if DARKBLOOM_ALPHASKIP
#define DARKBLOOM_RESCALE_FACTOR(dst, delta_expr)   \
  do {                                              \
    const U db_delta_ = (delta_expr);               \
    if (as_type<uint>(db_delta_) == 0u) {           \
      dst = U(1.0f);                                \
    } else {                                        \
      dst = fast::exp(db_delta_);                   \
    }                                               \
  } while (false)
#else
#define DARKBLOOM_RESCALE_FACTOR(dst, delta_expr) \
  do {                                            \
    dst = fast::exp(delta_expr);                  \
  } while (false)
#endif

template <
    typename T,
    int D,
    int V,
    int PLANES,
    bool has_mask,
    bool query_transposed,
    bool do_causal,
    bool bool_mask,
    bool float_mask,
    bool has_sinks>
inline void sdpa_vector(
    const device T* queries,
    const device T* keys,
    const device T* values,
    device T* out,
    int gqa_factor,
    int N,
    size_t k_head_stride,
    size_t k_seq_stride,
    size_t v_head_stride,
    size_t v_seq_stride,
    float scale,
    const device bool* bmask,
    const device T* fmask,
    int mask_kv_seq_stride,
    int mask_q_seq_stride,
    int mask_head_stride,
    const device T* sinks,
    int num_q_heads,
    uint3 tid,
    uint3 tpg,
    uint simd_gid,
    uint simd_lid,
    threadgroup float* outputs,
    threadgroup float* max_scores,
    threadgroup float* sum_exp_scores) {
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = V / BD;
  int inner_k_stride = BN * int(k_seq_stride);
  int inner_v_stride = BN * int(v_seq_stride);

  typedef float U;

  // Clamp: more planes than elements would allocate threadgroup memory nobody
  // writes. The clamp is also the compile-safety net for D = V = 256, where
  // v_per_thread = 8 and eight 4 KiB planes (32768 B, plus max_scores and
  // sum_exp_scores) would exceed the 32 KiB per-threadgroup limit - a hard
  // metallib compile error, not a slow kernel. PLANES never exceeds 4.
  constexpr int v_planes = PLANES < v_per_thread ? PLANES : v_per_thread;
  constexpr int exchange_planes =
      (D == 128 && V == 128 && DARKBLOOM_GQA_PAIR_HEADS == 2)
      ? 4
      : v_planes;
  // outputs / max_scores / sum_exp_scores are parameters here; see the module docstring.

  // DARKBLOOM_GQA_PAIR_HEADS: preserve each head's exact key order and
  // reduction tree while sharing the K/V device reads across adjacent heads.
  if constexpr (D == 128 && V == 128 && DARKBLOOM_GQA_PAIR_HEADS == 2) {
  // Vector-load eligibility: the pair path issues 8-byte vec<T,4> K/V loads
  // (T is 2 bytes at D == V == 128), so every element offset in the K/V
  // indexing must be a multiple of 4 elements and the base pointers 8-byte
  // aligned. simd_lid * qk_per_thread is always a multiple of 4 here; the
  // strides and bases are checked at runtime. Ineligible layouts fall back
  // to the stock per-head path below, which the pair path reproduces
  // bit-for-bit by construction (same key order, same reduction tree), so
  // the fallback changes nothing but speed.
  const bool pair_vec_aligned =
      (((k_seq_stride | v_seq_stride | k_head_stride | v_head_stride) & 3) ==
       0) &&
      ((reinterpret_cast<uintptr_t>(keys) |
        reinterpret_cast<uintptr_t>(values)) &
       7) == 0;
  const bool use_gqa_pair =
      (gqa_factor == 8 || gqa_factor == 6) &&
      tpg.y == 1 && (tpg.x % 2) == 0 &&
      pair_vec_aligned &&
      !has_mask && !do_causal && !has_sinks;
  if (use_gqa_pair) {
    const int pair_idx = tid.x;
    const int q_head0 = 2 * pair_idx;
    if (q_head0 >= int(tpg.x)) {
      return;
    }
    const int q_head1 = q_head0 + 1;
    const int kv_head_idx = q_head0 / gqa_factor;

    const device T* pair_query0 =
        queries + q_head0 * D + simd_lid * qk_per_thread;
    const device T* pair_query1 =
        queries + q_head1 * D + simd_lid * qk_per_thread;
    const device T* pair_keys =
        keys + kv_head_idx * k_head_stride + simd_gid * k_seq_stride +
        simd_lid * qk_per_thread;
    const device T* pair_values =
        values + kv_head_idx * v_head_stride + simd_gid * v_seq_stride +
        simd_lid * v_per_thread;
    device T* pair_out0 =
        out + q_head0 * V + simd_gid * v_per_thread;
    device T* pair_out1 =
        out + q_head1 * V + simd_gid * v_per_thread;

    thread U pair_q0[qk_per_thread];
    thread U pair_q1[qk_per_thread];
    thread U pair_k[qk_per_thread];
    thread U pair_o0[v_per_thread];
    thread U pair_o1[v_per_thread];

    for (int j = 0; j < qk_per_thread; ++j) {
      pair_q0[j] = static_cast<U>(scale) * pair_query0[j];
      pair_q1[j] = static_cast<U>(scale) * pair_query1[j];
    }
    for (int j = 0; j < v_per_thread; ++j) {
      pair_o0[j] = 0;
      pair_o1[j] = 0;
    }

    U pair_max0 = Limits<U>::finite_min;
    U pair_max1 = Limits<U>::finite_min;
    U pair_sum0 = 0;
    U pair_sum1 = 0;

    // Two-deep software pipeline, loads hoisted: both positions' K and V
    // rows are read at the top of the trip, then the two positions are
    // accumulated strictly in order (i before i+BN). Per-position FP
    // sequence is character-identical to stock; only load placement moved,
    // and loads have no side effects (no device stores occur in the loop).
    int i = simd_gid;
    for (; i + BN < N; i += 2 * BN) {
      const device T* pipe_keys_b = pair_keys + inner_k_stride;
      const device T* pipe_values_b = pair_values + inner_v_stride;
      // 8-byte vec<T,4> loads (alignment certified by pair_vec_aligned at
      // entry). Identical elements in identical order to the four scalar
      // loads each replaces; the T -> U conversion points are unchanged, so
      // the FP sequence is character-identical.
      const vec<T, 4> vec_ka =
          *reinterpret_cast<const device vec<T, 4>*>(pair_keys);
      const vec<T, 4> vec_kb =
          *reinterpret_cast<const device vec<T, 4>*>(pipe_keys_b);
      U pipe_ka[4];
      U pipe_kb[4];
      pipe_ka[0] = vec_ka.x;
      pipe_ka[1] = vec_ka.y;
      pipe_ka[2] = vec_ka.z;
      pipe_ka[3] = vec_ka.w;
      pipe_kb[0] = vec_kb.x;
      pipe_kb[1] = vec_kb.y;
      pipe_kb[2] = vec_kb.z;
      pipe_kb[3] = vec_kb.w;
      const vec<T, 4> vec_va =
          *reinterpret_cast<const device vec<T, 4>*>(pair_values);
      const vec<T, 4> vec_vb =
          *reinterpret_cast<const device vec<T, 4>*>(pipe_values_b);
      const T pipe_va0 = vec_va.x;
      const T pipe_va1 = vec_va.y;
      const T pipe_va2 = vec_va.z;
      const T pipe_va3 = vec_va.w;
      const T pipe_vb0 = vec_vb.x;
      const T pipe_vb1 = vec_vb.y;
      const T pipe_vb2 = vec_vb.z;
      const T pipe_vb3 = vec_vb.w;
      // Manual full unroll of the qk_per_thread == 4 element loop. Same
      // loads, same fmuladd chain order per score: identical FP sequence.

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

      U pair_new_max0 = max(pair_max0, pair_score0);
      U pair_new_max1 = max(pair_max1, pair_score1);
      U pair_factor0;
      U pair_factor1;
      DARKBLOOM_RESCALE_FACTOR(pair_factor0, pair_max0 - pair_new_max0);
      DARKBLOOM_RESCALE_FACTOR(pair_factor1, pair_max1 - pair_new_max1);
      U pair_exp0 = fast::exp(pair_score0 - pair_new_max0);
      U pair_exp1 = fast::exp(pair_score1 - pair_new_max1);

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

      // Manual full unroll of the qk_per_thread == 4 element loop. Same
      // loads, same fmuladd chain order per score: identical FP sequence.

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

      U pipeb_new_max0 = max(pair_max0, pipeb_score0);
      U pipeb_new_max1 = max(pair_max1, pipeb_score1);
      U pipeb_factor0;
      U pipeb_factor1;
      DARKBLOOM_RESCALE_FACTOR(pipeb_factor0, pair_max0 - pipeb_new_max0);
      DARKBLOOM_RESCALE_FACTOR(pipeb_factor1, pair_max1 - pipeb_new_max1);
      U pipeb_exp0 = fast::exp(pipeb_score0 - pipeb_new_max0);
      U pipeb_exp1 = fast::exp(pipeb_score1 - pipeb_new_max1);

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
      const vec<T, 4> vec_kt =
          *reinterpret_cast<const device vec<T, 4>*>(pair_keys);
      const vec<T, 4> vec_vt =
          *reinterpret_cast<const device vec<T, 4>*>(pair_values);
      pair_k[0] = vec_kt.x;
      pair_k[1] = vec_kt.y;
      pair_k[2] = vec_kt.z;
      pair_k[3] = vec_kt.w;
      const T pipe_va0 = vec_vt.x;
      const T pipe_va1 = vec_vt.y;
      const T pipe_va2 = vec_vt.z;
      const T pipe_va3 = vec_vt.w;
      // Manual full unroll of the qk_per_thread == 4 element loop. Same
      // loads, same fmuladd chain order per score: identical FP sequence.

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

      U pair_new_max0 = max(pair_max0, pair_score0);
      U pair_new_max1 = max(pair_max1, pair_score1);
      U pair_factor0;
      U pair_factor1;
      DARKBLOOM_RESCALE_FACTOR(pair_factor0, pair_max0 - pair_new_max0);
      DARKBLOOM_RESCALE_FACTOR(pair_factor1, pair_max1 - pair_new_max1);
      U pair_exp0 = fast::exp(pair_score0 - pair_new_max0);
      U pair_exp1 = fast::exp(pair_score1 - pair_new_max1);

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

    // Each head keeps the promoted two-plane combine shape. The additive
    // head-bank offset changes no producer/consumer pairing or simd_sum tree.
    constexpr int pair_planes = 2;
    constexpr int pair_plane_size = BN * BD;
    if (simd_lid == 0) {
      max_scores[simd_gid] = pair_max0;
      max_scores[BN + simd_gid] = pair_max1;
      sum_exp_scores[simd_gid] = pair_sum0;
      sum_exp_scores[BN + simd_gid] = pair_sum1;
    }
    for (int i = 0; i < pair_planes; ++i) {
      outputs[i * pair_plane_size + simd_lid * BD + simd_gid] = pair_o0[i];
      outputs[
          (pair_planes + i) * pair_plane_size + simd_lid * BD + simd_gid] =
          pair_o1[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    pair_max0 = max_scores[simd_lid];
    pair_max1 = max_scores[BN + simd_lid];
    U pair_global_max0 = simd_max(pair_max0);
    U pair_global_max1 = simd_max(pair_max1);
    U pair_global_factor0 = fast::exp(pair_max0 - pair_global_max0);
    U pair_global_factor1 = fast::exp(pair_max1 - pair_global_max1);
    pair_sum0 =
        simd_sum(sum_exp_scores[simd_lid] * pair_global_factor0);
    pair_sum1 =
        simd_sum(sum_exp_scores[BN + simd_lid] * pair_global_factor1);

    for (int i = 0; i < pair_planes; ++i) {
      U acc0 = simd_sum(
          outputs[i * pair_plane_size + simd_gid * BD + simd_lid] *
          pair_global_factor0);
      U acc1 = simd_sum(
          outputs[
              (pair_planes + i) * pair_plane_size +
              simd_gid * BD + simd_lid] *
          pair_global_factor1);
      pair_o0[i] = pair_sum0 == 0 ? acc0 : (acc0 / pair_sum0);
      pair_o1[i] = pair_sum1 == 0 ? acc1 : (acc1 / pair_sum1);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = 0; i < pair_planes; ++i) {
      outputs[i * pair_plane_size + simd_lid * BD + simd_gid] =
          pair_o0[pair_planes + i];
      outputs[
          (pair_planes + i) * pair_plane_size + simd_lid * BD + simd_gid] =
          pair_o1[pair_planes + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = 0; i < pair_planes; ++i) {
      U acc0 = simd_sum(
          outputs[i * pair_plane_size + simd_gid * BD + simd_lid] *
          pair_global_factor0);
      U acc1 = simd_sum(
          outputs[
              (pair_planes + i) * pair_plane_size +
              simd_gid * BD + simd_lid] *
          pair_global_factor1);
      pair_o0[pair_planes + i] =
          pair_sum0 == 0 ? acc0 : (acc0 / pair_sum0);
      pair_o1[pair_planes + i] =
          pair_sum1 == 0 ? acc1 : (acc1 / pair_sum1);
    }

    if (simd_lid == 0) {
      for (int i = 0; i < v_per_thread; ++i) {
        pair_out0[i] = static_cast<T>(pair_o0[i]);
        pair_out1[i] = static_cast<T>(pair_o1[i]);
      }
    }
    return;
  }
  }

  thread U q[qk_per_thread];
  thread U k[qk_per_thread];
  thread U o[v_per_thread];

  // Adjust positions
  const int q_batch_head_idx = tid.x;
  const int q_seq_idx = tid.y;
  const int kv_head_idx = q_batch_head_idx / gqa_factor;
  const int o_offset = q_batch_head_idx * tpg.y + q_seq_idx;
  const int q_offset =
      query_transposed ? tpg.x * q_seq_idx + q_batch_head_idx : o_offset;
  queries += q_offset * D + simd_lid * qk_per_thread;
  keys += kv_head_idx * k_head_stride + simd_gid * k_seq_stride +
      simd_lid * qk_per_thread;
  values += kv_head_idx * v_head_stride + simd_gid * v_seq_stride +
      simd_lid * v_per_thread;
  if (bool_mask) {
    bmask += q_batch_head_idx * mask_head_stride +
        simd_gid * mask_kv_seq_stride + q_seq_idx * mask_q_seq_stride;
  }
  if (float_mask) {
    fmask += q_batch_head_idx * mask_head_stride +
        simd_gid * mask_kv_seq_stride + q_seq_idx * mask_q_seq_stride;
  }

  out += o_offset * V + simd_gid * v_per_thread;

  // Read the query and 0 the output accumulator
  for (int i = 0; i < qk_per_thread; i++) {
    q[i] = static_cast<U>(scale) * queries[i];
  }
  for (int i = 0; i < v_per_thread; i++) {
    o[i] = 0;
  }

  U max_score = Limits<U>::finite_min;
  U sum_exp_score = 0;
  if (has_sinks && simd_gid == 0) {
    max_score = static_cast<U>(sinks[q_batch_head_idx % num_q_heads]);
    sum_exp_score = 1;
  }

  // For each key
  for (int i = simd_gid; i < N; i += BN) {
    bool use_key = true;
    if (do_causal) {
      use_key = i <= (N - int(tpg.y) + int(q_seq_idx));
    } else if (bool_mask) {
      use_key = bmask[0];
    } else if (float_mask) {
      use_key = (fmask[0] >= Limits<T>::finite_min);
    }
    if (use_key) {
      // Read the key
      for (int j = 0; j < qk_per_thread; j++) {
        k[j] = keys[j];
      }

      // Compute the i-th score
      U score = 0;
      for (int j = 0; j < qk_per_thread; j++) {
        score += q[j] * k[j];
      }
      score = simd_sum(score);
      if (float_mask) {
        score += static_cast<U>(fmask[0]);
      }

      // Update the accumulators
      U new_max = max(max_score, score);
      U factor;
      DARKBLOOM_RESCALE_FACTOR(factor, max_score - new_max);
      U exp_score = fast::exp(score - new_max);

      max_score = new_max;
      sum_exp_score = sum_exp_score * factor + exp_score;

      // Update the output accumulator
      for (int j = 0; j < v_per_thread; j++) {
        o[j] = o[j] * factor + exp_score * values[j];
      }
    }

    // Move the pointers to the next kv
    keys += inner_k_stride;
    values += inner_v_stride;
    if (bool_mask) {
      bmask += BN * mask_kv_seq_stride;
    }
    if (float_mask) {
      fmask += BN * mask_kv_seq_stride;
    }
  }

  // Each thread has a partial part of the output so we need to combine them.

  // `v_planes` is a compile-time constant, so this selector is folded and only
  // one arm survives in each specialization; the barriers inside it are never
  // reached divergently.
  U factor;
  if (v_planes > 1) {
    // Widened exchange. Elements are combined in groups of `v_planes`; within
    // a group every element owns its own plane, so neither the RAW hazard
    // (write then read) nor the WAR hazard (read then rewrite) recurs inside
    // the group. The group's single rendezvous also absorbs the max/sum
    // publish, because the plane stores depend only on registers (`o[]` is
    // final here) and target a threadgroup array disjoint from
    // max_scores/sum_exp_scores, so hoisting them above that barrier
    // introduces no dependency.
    //
    // At D = V = 128 (v_per_thread = 4): 8 barriers -> 3 at PLANES=2 (two
    // groups of two), -> 1 at PLANES=4 (one group of four).
    //
    // Exactness, per lane. Baseline: lane l of simdgroup g writes o[i] to
    // outputs[l * BD + g] and reads outputs[g * BD + l], so the simd_sum over
    // the 32 lanes of simdgroup g reduces, in lane order l = 0..31, the o[i]
    // produced by lane g of simdgroup l. Widened: the same lane writes
    // o[base+p] to outputs[p * (BN * BD) + l * BD + g] and reads
    // outputs[p * (BN * BD) + g * BD + l]. The plane base p * (BN * BD) is the
    // same additive constant for the writer and the reader of a given element,
    // so the producer/consumer pairing, the lane ordering and the reduction
    // tree are identical; only the base address differs. `factor`, the divide
    // and the `sum_exp_score == 0` guard are untouched. No reassociation, no
    // rounding boundary moved.
    if (simd_lid == 0) {
      max_scores[simd_gid] = max_score;
      sum_exp_scores[simd_gid] = sum_exp_score;
    }
    for (int i = 0; i < v_planes; i++) {
      outputs[i * (BN * BD) + simd_lid * BD + simd_gid] = o[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    max_score = max_scores[simd_lid];
    U new_max = simd_max(max_score);
    factor = fast::exp(max_score - new_max);
    sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);

    for (int i = 0; i < v_planes; i++) {
      U acc =
          simd_sum(outputs[i * (BN * BD) + simd_gid * BD + simd_lid] * factor);
      o[i] = sum_exp_score == 0 ? acc : (acc / sum_exp_score);
    }
    // Only entered when v_per_thread exceeds v_planes (PLANES=2 at D >= 96,
    // PLANES=4 at D = 256). Each extra group costs one WAR plus one RAW.
    for (int base = v_planes; base < v_per_thread; base += v_planes) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (int i = 0; i < v_planes && base + i < v_per_thread; i++) {
        outputs[i * (BN * BD) + simd_lid * BD + simd_gid] = o[base + i];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (int i = 0; i < v_planes && base + i < v_per_thread; i++) {
        U acc = simd_sum(
            outputs[i * (BN * BD) + simd_gid * BD + simd_lid] * factor);
        o[base + i] = sum_exp_score == 0 ? acc : (acc / sum_exp_score);
      }
    }
  } else {
    // Upstream shape, preserved verbatim so PLANES=1 is a true control.
    // First let's communicate the max and sum_exp
    if (simd_lid == 0) {
      max_scores[simd_gid] = max_score;
      sum_exp_scores[simd_gid] = sum_exp_score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    max_score = max_scores[simd_lid];
    U new_max = simd_max(max_score);
    factor = fast::exp(max_score - new_max);
    sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);

    // Now we need to aggregate all the outputs. The trailing barrier only
    // protects the exchange plane's reuse by the NEXT iteration, so the last
    // iteration's trailing barrier guards nothing and is skipped; every
    // exchanged value, slot, and reduction is unchanged.
    for (int i = 0; i < v_per_thread; i++) {
      outputs[simd_lid * BD + simd_gid] = o[i];
      threadgroup_barrier(mem_flags::mem_threadgroup);
      o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor);
      o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
      if (i + 1 < v_per_thread) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
    }
  }

  // And write the output
  if (simd_lid == 0) {
    for (int i = 0; i < v_per_thread; i++) {
      out[i] = static_cast<T>(o[i]);
    }
  }
}
"""

_UNMASKED_SOURCE = r"""
    // Sized to match the vendored `exchange_planes * BN * BD` and
    // `DARKBLOOM_GQA_PAIR_HEADS * BN` at D = V = 128, PLANES = 4.
    threadgroup float sdpa_outputs[4 * 32 * 32];
    threadgroup float sdpa_max_scores[DARKBLOOM_GQA_PAIR_HEADS * 32];
    threadgroup float sdpa_sum_exp_scores[DARKBLOOM_GQA_PAIR_HEADS * 32];
    static_assert(DARKBLOOM_GQA_PAIR_HEADS == 2, "the plane count above is 2-head");
    sdpa_vector<T, 128, 128, 4, false, false, false, false, false, false>(
        Q, K, V, OUT,
        GQA, N,
        (size_t)K_strides[1], (size_t)K_strides[2],
        (size_t)V_strides[1], (size_t)V_strides[2],
        SCALE,
        (const device bool*)nullptr, (const device T*)nullptr,
        0, 0, 0,
        (const device T*)nullptr, 0,
        threadgroup_position_in_grid, threadgroups_per_grid,
        simdgroup_index_in_threadgroup, thread_index_in_simdgroup,
        sdpa_outputs, sdpa_max_scores, sdpa_sum_exp_scores);
"""

_MASKED_SOURCE = r"""
    // Sized to match the vendored `exchange_planes * BN * BD` and
    // `DARKBLOOM_GQA_PAIR_HEADS * BN` at D = V = 128, PLANES = 4.
    threadgroup float sdpa_outputs[4 * 32 * 32];
    threadgroup float sdpa_max_scores[DARKBLOOM_GQA_PAIR_HEADS * 32];
    threadgroup float sdpa_sum_exp_scores[DARKBLOOM_GQA_PAIR_HEADS * 32];
    static_assert(DARKBLOOM_GQA_PAIR_HEADS == 2, "the plane count above is 2-head");
    sdpa_vector<T, 128, 128, 4, true, false, false, true, false, false>(
        Q, K, V, OUT,
        GQA, N,
        (size_t)K_strides[1], (size_t)K_strides[2],
        (size_t)V_strides[1], (size_t)V_strides[2],
        SCALE,
        MASK, (const device T*)nullptr,
        1, 0, 0,
        (const device T*)nullptr, 0,
        threadgroup_position_in_grid, threadgroups_per_grid,
        simdgroup_index_in_threadgroup, thread_index_in_simdgroup,
        sdpa_outputs, sdpa_max_scores, sdpa_sum_exp_scores);
"""

SdpaDecode = Callable[[mx.array, mx.array, mx.array, mx.array | None, float], mx.array]


def _build(name: str, masked: bool) -> SdpaDecode:
    """One kernel per mask form: the mask buffer is a parameter of the generated entry
    point, so its presence changes the signature and not just a template argument."""
    kernel = metal_kernel(
        name=name,
        input_names=["Q", "K", "V", *(["MASK"] if masked else []), "GQA", "N", "SCALE"],
        output_names=["OUT"],
        source=_MASKED_SOURCE if masked else _UNMASKED_SOURCE,
        header=_SDPA_VECTOR,
        ensure_row_contiguous=False,
    )

    def run(
        queries: mx.array,
        keys: mx.array,
        values: mx.array,
        band: mx.array | None,
        scale: float,
    ) -> mx.array:
        heads, kv_heads, key_length = queries.shape[1], keys.shape[1], keys.shape[2]
        dtype = queries.dtype
        inputs = [queries.reshape(heads, _HEAD_DIM), keys, values]
        if band is not None:
            inputs.append(band)
        inputs += [
            mx.array(heads // kv_heads, dtype=mx.int32),
            mx.array(key_length, dtype=mx.int32),
            mx.array(scale, dtype=mx.float32),
        ]
        out = kernel(
            inputs=inputs,
            template=[("T", dtype)],
            grid=(_THREADS * heads, 1, 1),
            threadgroup=(_THREADS, 1, 1),
            output_shapes=[(heads, _HEAD_DIM)],
            output_dtypes=[dtype],
        )
        return out[0].reshape(1, heads, 1, _HEAD_DIM)

    return run


_UNMASKED = _build("sdpa_decode_vector", masked=False)
_MASKED = _build("sdpa_decode_vector_bool_mask", masked=True)


def _band(mask: mx.array | str | None) -> mx.array | None:
    """A single query row sees every key a causal mask would allow, so `"causal"` and
    `None` are the same call — which is also what mlx's own host does
    (`do_causal = do_causal_ && q.shape(2) > 1`). Anything else is the boolean band,
    flattened to one entry per key."""
    if mask is None or isinstance(mask, str):
        return None
    return mask.reshape(-1)


def _mask_applies(mask: mx.array | str | None, key_length: int) -> bool:
    if mask is None:
        return True
    if isinstance(mask, str):
        return mask == "causal"
    return mask.dtype == mx.bool_ and mask.size == key_length and key_length >= _MIN_MASK_ELEMENTS


def sdpa_decode_applies(
    queries: mx.array, keys: mx.array, values: mx.array, mask: mx.array | str | None
) -> bool:
    """One query row per head over a whole cache, head_dim 128, bfloat16, batch 1, GQA by
    whole groups, and either no mask or one boolean entry per key shared by every head.

    head_dim 128 is what the paired path is compiled for and the only width the widened
    exchange plane was measured at. bfloat16 is load-bearing rather than cosmetic: the
    paired path certifies its `vec<T,4>` loads with a 4-element stride test and an 8-byte
    base test, which is the right alignment only for a 2-byte element. Batch 1 because the
    kernel derives the KV head from the query head alone. The mask floor exists because
    mlx passes an array of fewer than 8 elements in the `constant` address space, which
    does not bind to the vendored `device` parameter.

    Beyond the shape: mlx's host routes to `sdpa_vector_2pass` from key_length 1024 on this
    device class, so above that this replaces a two-pass baseline with a one-pass kernel —
    a different reduction, and a trade the record never measured (their timed window peaks
    at 640 keys).
    """
    if queries.ndim != 4 or keys.ndim != 4 or values.ndim != 4:
        return False
    heads, kv_heads, key_length = queries.shape[1], keys.shape[1], keys.shape[2]
    shaped = (
        queries.shape[0] == 1
        and keys.shape[0] == 1
        and values.shape[0] == 1
        and queries.shape[2] == 1
        and queries.shape[3] == _HEAD_DIM
        and keys.shape[3] == _HEAD_DIM
        and values.shape[3] == _HEAD_DIM
        and values.shape[1] == kv_heads
        and values.shape[2] == key_length
        and kv_heads > 0
        and key_length > 0
        and heads % kv_heads == 0
    )
    typed = (
        queries.dtype == mx.bfloat16 and keys.dtype == mx.bfloat16 and values.dtype == mx.bfloat16
    )
    return shaped and typed and _mask_applies(mask, key_length)


def sdpa_decode(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    *,
    scale: float,
    mask: mx.array | str | None = None,
) -> mx.array:
    """`mx.fast.scaled_dot_product_attention` for the one-query step, on the lifted kernel.

    `queries` [1, heads, 1, 128], `keys`/`values` [1, kv_heads, key_length, 128] with the
    head dimension packed — they may be strided slices of a block buffer, the head and row
    strides are read from what mlx passes. Returns the context [1, heads, 1, 128].

    The paired path engages only at GQA factor 6 or 8 with no mask; every other accepted
    shape runs the per-head path, which still carries the widened exchange plane and the
    rescale elision.
    """
    assert sdpa_decode_applies(queries, keys, values, mask)
    band = _band(mask)
    if band is None:
        return _UNMASKED(queries, keys, values, None, scale)
    return _MASKED(queries, keys, values, band, scale)


def _stands_in_for_sdpa(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    /,
    scale: float = 1.0,
    mask: mx.array | str | None = None,
    **_: object,
) -> bool:
    """The stock op's contract is far wider than this kernel's; anything past the decode
    step it was lifted for stays on mlx."""
    return sdpa_decode_applies(queries, keys, values, mask)


def _as_sdpa(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    /,
    scale: float = 1.0,
    mask: mx.array | str | None = None,
    **_: object,
) -> mx.array:
    return sdpa_decode(queries, keys, values, scale=scale, mask=mask)


SCALED_DOT_PRODUCT_ATTENTION = Patch(
    module=mx.fast,
    name="scaled_dot_product_attention",
    applies=_stands_in_for_sdpa,
    replacement=_as_sdpa,
)
