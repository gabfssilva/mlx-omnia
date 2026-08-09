"""A pruned output head that returns an ARGMAX-EXACT row -- NOT a logits tensor.

**Read this before wiring it into anything.** The row this chain returns is a full
`[vocab]` bf16 buffer, but only the slots that survived the screen hold the value a stock
head projection would produce. Every other slot holds an int5 *approximation* of its
logit, certified only to sit strictly below the winner. The chain therefore guarantees
exactly one thing:

    argmax(assembled) == argmax(stock_bf16_head(x))

and nothing else. It does not guarantee the second-best token, a top-k set, a softmax, a
logprob, a temperature-scaled sample, a speculative-decoding acceptance ratio, a
repetition penalty applied to the row, or any statistic that reads a non-winning slot.
Sideros serves logprobs and temperature sampling, so **this is not a drop-in replacement
for the head projection** on any request that is not pure greedy argmax. That is what
`lm_head_argmax_applies(..., argmax_only=...)` refuses: pass the caller's honest answer to
"is `argmax` the only thing I will do with this row", and the predicate declines the whole
family when it is False, whatever the shapes say.

Transcribed from the mlxfast-challenge record tree (Layr Labs, MIT), where the ablation
measured the family at 12.9% of decode. Six kernels, four dispatches, two arms:

  1. COARSE. One fused gemv over an init-time planar int5 copy of the head weight
     (`int5_planes`), emitting each row's coarse logit `c_i` and a certified error bound
     `delta_i` rounded UP to bf16. Two forms:
     `lm_head_int5_coarse` reads the nibble plane, the residual bit plane and the scales
     (`hidden * 5/8 + hidden/32` bytes per row) and certifies the half-cell bound
     `d_i * (1 + 61*gamma)`; `lm_head_int5_base_coarse` reads only the nibble plane and the
     scales, decodes the cell midpoint (`q0 = 2H - 15.5`), and certifies the full-cell
     bound `d_i * (1 + 32*gamma)`. `gamma = 2^-15` covers every float rounding on both
     paths. `coarse` stays fp32: it would have to round DOWN for the threshold path and UP
     for the candidate test, and one buffer cannot do both.
  2. ARGMAX stage one (`lm_head_coarse_argmax_partials`). A (value, index) reduction over
     `coarse` alone -- `delta` is not read here -- into 128 partials. Lowest index wins a
     tie, so the selected row is deterministic. Which row wins only affects the candidate
     count, i.e. speed: step 3 is sound for any row index.
  3. THRESHOLD (`lm_head_exact_winner_threshold`). Finishes the argmax over the 128
     partials, runs the stock single-row gemv for the winning row `r`, and emits the fp32
     midpoint just below `bfloat(e_r)`. Sound for ANY `r` because `e_r <= e_winner`, which
     is what removes the winner's own quantization delta from the threshold -- the one
     number that made a pure int5 head untenable.
  4. EXACT. Each simdgroup owns a fixed four-row block and runs a full bf16 gemv over that
     block only when `coarse[r] + float(delta[r]) >= threshold` for one of its rows,
     writing `bfloat(coarse[r])` otherwise. The per-row arithmetic is a TEXTUAL replica of
     mlx's own `gemv_al_bfloat16` (`gemv.h`, `bm8_bn1_sm1_sn32_tm4_tn4_nc0_axpby0`, the
     aligned path) -- same lane partition, same sequential fp32 order, same vec4 loads,
     same simd tree, same bf16 cast -- so a candidate row's logit is bit-identical to the
     stock full gemv's. Every slot is written by exactly one lane on exactly one path.
     `lm_head_refined_exact_assemble` is the three-level arm that pairs with the base
     coarse pass: a surviving block re-reads the residual bit plane for its live rows only,
     restores the exact int5 value and the tighter half-cell bound, and re-screens before
     touching the bf16 weight.

Deviations from the source, all forced and all shape-only: the vocabulary and hidden sizes
are template parameters instead of the record's literals, and so is everything the source
derived from them by hand -- the three plane strides (`hidden/2`, `hidden/8`, `hidden/32`),
the argmax partition width (`vocab/128`), the gemv replica's block count (`hidden/128`) and
a lane's group count (`hidden/1024`). Each substitution evaluates to the literal it
replaced. The e8m0 decoder is renamed off its model prefix. Nothing else moved.
"""

from typing import NamedTuple

import mlx.core as mx

from sideros.core.mxcompat import metal_kernel

_HEADER = """
    // e8m0 decode, identical to fp8.h:70-77 (bits<<7 as bf16; bits==0 ->
    // 0x40 as bf16 = 2^-127). Exponent-bit construction, exact.
    static inline float e8m0_decode(uint8_t b) {
        if (b == 0u) {
            return as_type<float>(0x00400000u);  // 2^-127
        }
        return as_type<float>(uint(b) << 23);
    }
"""

_INT5_COARSE_SOURCE = """
    constexpr float GAMMA = 0x1p-15f;
    constexpr uint KDIM = (uint)K;
    constexpr uint GROUPS_PER_LANE = KDIM / 1024u;

    uint row = threadgroup_position_in_grid.x * 16 +
        simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;

    const device uint8_t* lorow = codes_lo + size_t(row) * (KDIM / 2);
    const device uint8_t* hirow = codes_hi + size_t(row) * (KDIM / 8);
    const device uint8_t* srow = scales + size_t(row) * (KDIM / 32);

    float c_acc = 0.0f;
    float d_acc = 0.0f;
    for (uint gg = 0; gg < GROUPS_PER_LANE; ++gg) {
        uint g = GROUPS_PER_LANE * lane + gg;
        float sd = e8m0_decode(srow[g]);
        uint4 lo4 = ((const device uint4*)(lorow + g * 16))[0];
        uint hb = ((const device uint*)(hirow + g * 4))[0];
        const device ushort4* xrow = (const device ushort4*)(x + g * 32);
        float cg = 0.0f;
        float ag = 0.0f;
        #pragma clang loop unroll(full)
        for (uint w = 0; w < 4; ++w) {
            // Word w: elements 8w..8w+7 of the group. Nibble plane byte
            // b holds elements 2b (low) / 2b+1 (high); 1-bit plane bit j
            // of the group's word holds element j's residual bit.
            uint lw = lo4[w];
            uint hw = hb >> (8u * w);
            uint4 ne = (uint4(lw) >> uint4(0u, 8u, 16u, 24u)) & 15u;
            uint4 no = (uint4(lw) >> uint4(4u, 12u, 20u, 28u)) & 15u;
            uint4 he = (uint4(hw) >> uint4(0u, 2u, 4u, 6u)) & 1u;
            uint4 ho = (uint4(hw) >> uint4(1u, 3u, 5u, 7u)) & 1u;
            // The nibble stores floor(q/2)+8 and the bit plane stores
            // q-2*floor(q/2), so joining them rebuilds u = q + 16 in
            // [1, 31]; offset-binary decode is exact.
            float4 ve = float4((ne << 1u) | he) - 16.0f;
            float4 vo = float4((no << 1u) | ho) - 16.0f;
            // bf16 -> f32 is exactly bits<<16 for every value class.
            float4 xa = as_type<float4>(uint4(xrow[2 * w]) << 16);
            float4 xb = as_type<float4>(uint4(xrow[2 * w + 1]) << 16);
            float4 xe = float4(xa.x, xa.z, xb.x, xb.z);
            float4 xo = float4(xa.y, xa.w, xb.y, xb.w);
            float4 axe = metal::abs(xe);
            float4 axo = metal::abs(xo);
            #pragma clang loop unroll(full)
            for (uint k = 0; k < 4; ++k) {
                cg += xe[k] * ve[k];
                cg += xo[k] * vo[k];
                ag += axe[k];
                ag += axo[k];
            }
        }
        c_acc += sd * cg;
        d_acc += (0.5f * sd) * ag;
    }
    c_acc = simd_sum(c_acc);
    d_acc = simd_sum(d_acc);
    if (lane == 0) {
        coarse[row] = c_acc;
        // FP32 bound, then rounded UP to BF16 (mask-and-bump, sign clear).
        float d_up = d_acc * (1.0f + 61.0f * GAMMA);
        uint dbits = as_type<uint>(d_up);
        uint dtrunc = dbits & 0xFFFF0000u;
        if (dtrunc != dbits) {
            dtrunc += 0x00010000u;
        }
        delta[row] = as_type<bfloat>(ushort(dtrunc >> 16));
    }
"""

_INT5_BASE_COARSE_SOURCE = """
    constexpr float GAMMA = 0x1p-15f;
    constexpr uint KDIM = (uint)K;
    constexpr uint GROUPS_PER_LANE = KDIM / 1024u;

    uint row = threadgroup_position_in_grid.x * 16 +
        simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;

    const device uint8_t* crow = codes_base + size_t(row) * (KDIM / 2);
    const device uint8_t* srow = scales + size_t(row) * (KDIM / 32);

    float c_acc = 0.0f;
    float d_acc = 0.0f;
    for (uint gg = 0; gg < GROUPS_PER_LANE; ++gg) {
        uint g = GROUPS_PER_LANE * lane + gg;
        float sd = e8m0_decode(srow[g]);
        uint4 c4 = ((const device uint4*)(crow + g * 16))[0];
        const device ushort4* xrow = (const device ushort4*)(x + g * 32);
        float cg = 0.0f;
        float ag = 0.0f;
        #pragma clang loop unroll(full)
        for (uint w = 0; w < 4; ++w) {
            uint lw = c4[w];
            uint4 ne = (uint4(lw) >> uint4(0u, 8u, 16u, 24u)) & 15u;
            uint4 no = (uint4(lw) >> uint4(4u, 12u, 20u, 28u)) & 15u;
            float4 ve = float4(ne << 1u) - 15.5f;
            float4 vo = float4(no << 1u) - 15.5f;
            float4 xa = as_type<float4>(uint4(xrow[2 * w]) << 16);
            float4 xb = as_type<float4>(uint4(xrow[2 * w + 1]) << 16);
            float4 xe = float4(xa.x, xa.z, xb.x, xb.z);
            float4 xo = float4(xa.y, xa.w, xb.y, xb.w);
            float4 axe = metal::abs(xe);
            float4 axo = metal::abs(xo);
            #pragma clang loop unroll(full)
            for (uint k = 0; k < 4; ++k) {
                cg += xe[k] * ve[k];
                cg += xo[k] * vo[k];
                ag += axe[k];
                ag += axo[k];
            }
        }
        c_acc += sd * cg;
        d_acc += sd * ag;
    }
    c_acc = simd_sum(c_acc);
    d_acc = simd_sum(d_acc);
    if (lane == 0) {
        coarse[row] = c_acc;
        float d_up = d_acc * (1.0f + 32.0f * GAMMA);
        uint dbits = as_type<uint>(d_up);
        uint dtrunc = dbits & 0xFFFF0000u;
        if (dtrunc != dbits) {
            dtrunc += 0x00010000u;
        }
        delta[row] = as_type<bfloat>(ushort(dtrunc >> 16));
    }
"""

_ARGMAX_STAGE1_SOURCE = """
    constexpr uint ROW_SIZE = (uint)VOCAB / 128u;
    constexpr uint READS = 4;
    constexpr uint ACTIVE_THREADS = ROW_SIZE / READS;
    constexpr uint SIMD_GROUPS = (ACTIVE_THREADS + 31u) / 32u;

    uint row = threadgroup_position_in_grid.y;
    uint lid = thread_position_in_threadgroup.x;
    uint simd_lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    threadgroup float shared_vals[32];
    threadgroup uint shared_idxs[32];

    float best = -metal::numeric_limits<float>::infinity();
    uint best_idx = 0xFFFFFFFFu;
    if (lid < ACTIVE_THREADS) {
        uint base = row * ROW_SIZE + lid * READS;
        #pragma clang loop unroll(full)
        for (uint i = 0; i < READS; ++i) {
            float v = coarse[base + i];
            if (v > best || (v == best && base + i < best_idx)) {
                best = v;
                best_idx = base + i;
            }
        }
    }

    #pragma clang loop unroll(full)
    for (ushort sn = 16; sn >= 1; sn >>= 1) {
        float ov = simd_shuffle_down(best, sn);
        uint oi = simd_shuffle_down(best_idx, sn);
        if (ov > best || (ov == best && oi < best_idx)) {
            best = ov;
            best_idx = oi;
        }
    }
    if (simd_lane == 0) {
        shared_vals[simd_group] = best;
        shared_idxs[simd_group] = best_idx;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    best = lid < SIMD_GROUPS
        ? shared_vals[lid]
        : -metal::numeric_limits<float>::infinity();
    best_idx = lid < SIMD_GROUPS ? shared_idxs[lid] : 0xFFFFFFFFu;
    #pragma clang loop unroll(full)
    for (ushort sn = 16; sn >= 1; sn >>= 1) {
        float ov = simd_shuffle_down(best, sn);
        uint oi = simd_shuffle_down(best_idx, sn);
        if (ov > best || (ov == best && oi < best_idx)) {
            best = ov;
            best_idx = oi;
        }
    }
    if (lid == 0) {
        partial_max[row] = best;
        partial_idx[row] = best_idx;
    }
"""

_THRESHOLD_SOURCE = """
    constexpr uint READS = 4;
    uint lid = thread_position_in_threadgroup.x;
    threadgroup uint winner_row[1];

    // Verbatim final argmax over the retained 128 partials.
    float best = -metal::numeric_limits<float>::infinity();
    uint best_idx = 0xFFFFFFFFu;
    uint base = lid * READS;
    #pragma clang loop unroll(full)
    for (uint i = 0; i < READS; ++i) {
        float v = partial_max[base + i];
        uint idx = partial_idx[base + i];
        if (v > best || (v == best && idx < best_idx)) {
            best = v;
            best_idx = idx;
        }
    }
    #pragma clang loop unroll(full)
    for (ushort sn = 16; sn >= 1; sn >>= 1) {
        float ov = simd_shuffle_down(best, sn);
        uint oi = simd_shuffle_down(best_idx, sn);
        if (ov > best || (ov == best && oi < best_idx)) {
            best = ov;
            best_idx = oi;
        }
    }
    if (lid == 0) {
        winner_row[0] = metal::min(best_idx, uint(VOCAB - 1));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint r = winner_row[0];

    // --- stock gemv_al replica begin (single row r; gemv.h:151-289) ---
    float result = 0.0f;
    thread bfloat inter[4];
    thread float v_coeff[4];
    uint bn = lid * 4;
    const device bfloat* mrow = lm_head + size_t(r) * (uint)K;
    for (uint i = 0; i < (uint)K / 128u; ++i) {
        vec<bfloat, 4> xv =
            *((const device vec<bfloat, 4>*)(x + bn));
        v_coeff[0] = float(xv.x);
        v_coeff[1] = float(xv.y);
        v_coeff[2] = float(xv.z);
        v_coeff[3] = float(xv.w);
        vec<bfloat, 4> mv =
            *((const device vec<bfloat, 4>*)(mrow + bn));
        inter[0] = mv.x;
        inter[1] = mv.y;
        inter[2] = mv.z;
        inter[3] = mv.w;
        result += inter[0] * v_coeff[0];
        result += inter[1] * v_coeff[1];
        result += inter[2] * v_coeff[2];
        result += inter[3] * v_coeff[3];
        bn += 128;
    }
    #pragma unroll
    for (ushort sn = 16; sn >= 1; sn >>= 1) {
        result += simd_shuffle_down(result, sn);
    }
    // --- stock gemv_al replica end ---
    if (lid == 0) {
        bfloat rounded = bfloat(result);
        // Expand through the numeric BF16->FP32 conversion, whose bits are
        // exactly `bf16_bits << 16`; do not reinterpret the Metal wrapper.
        ushort bits = ushort(as_type<uint>(float(rounded)) >> 16);
        ushort magnitude = bits & 0x7FFFu;
        ushort predecessor_bits;
        if (magnitude == 0u) {
            predecessor_bits = 0x8001u;  // predecessor of either zero
        } else if ((bits & 0x8000u) == 0u) {
            predecessor_bits = bits - 1u;
        } else {
            predecessor_bits = bits + 1u;
        }
        float predecessor =
            as_type<float>(uint(predecessor_bits) << 16);
        float rounded_value = as_type<float>(uint(bits) << 16);
        threshold[0] = predecessor + (rounded_value - predecessor) * 0.5f;
    }
"""

_EXACT_SOURCE = """
    constexpr uint VOCAB_SIZE = (uint)VOCAB;
    constexpr uint KDIM = (uint)K;

    uint tgid = threadgroup_position_in_grid.x;
    uint sgid = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;

    // This simdgroup's fixed four output rows. The grid tiles the vocabulary
    // exactly; the bounds test is belt-and-braces.
    uint base = tgid * 32 + sgid * 4;

    // The predicate is simdgroup-uniform, so lane 0 forms it once and
    // broadcasts the four row decisions. Reusing the mask below removes
    // the same coarse/delta/threshold reads from the final write path.
    uint candidate_mask = 0;
    if (lane == 0) {
        #pragma unroll
        for (uint tm = 0; tm < 4; ++tm) {
            uint r = base + tm;
            if (r < VOCAB_SIZE && coarse[r] + float(delta[r]) >= thr[0]) {
                candidate_mask |= 1u << tm;
            }
        }
    }
    candidate_mask = simd_broadcast(candidate_mask, 0);

    if (candidate_mask == 0) {
        if (lane < 4 && base + lane < VOCAB_SIZE) {
            assembled[base + lane] = bfloat(coarse[base + lane]);
        }
        return;
    }

    // --- stock gemv_al replica begin (gemv.h:151-289) ---
    thread float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    thread bfloat inter[4];
    thread float v_coeff[4];
    uint bn = lane * 4;
    for (uint i = 0; i < KDIM / 128u; ++i) {
        vec<bfloat, 4> xv =
            *((const device vec<bfloat, 4>*)(x + bn));
        v_coeff[0] = float(xv.x);
        v_coeff[1] = float(xv.y);
        v_coeff[2] = float(xv.z);
        v_coeff[3] = float(xv.w);
        #pragma unroll
        for (uint tm = 0; tm < 4; ++tm) {
            const device bfloat* mrow = lm_head + size_t(base + tm) * KDIM;
            vec<bfloat, 4> mv =
                *((const device vec<bfloat, 4>*)(mrow + bn));
            inter[0] = mv.x;
            inter[1] = mv.y;
            inter[2] = mv.z;
            inter[3] = mv.w;
            result[tm] += inter[0] * v_coeff[0];
            result[tm] += inter[1] * v_coeff[1];
            result[tm] += inter[2] * v_coeff[2];
            result[tm] += inter[3] * v_coeff[3];
        }
        bn += 128;
    }
    #pragma unroll
    for (uint tm = 0; tm < 4; ++tm) {
        #pragma unroll
        for (ushort sn = 16; sn >= 1; sn >>= 1) {
            result[tm] += simd_shuffle_down(result[tm], sn);
        }
    }
    // --- stock gemv_al replica end ---
    if (lane == 0) {
        #pragma unroll
        for (uint tm = 0; tm < 4; ++tm) {
            uint r = base + tm;
            if (r < VOCAB_SIZE) {
                assembled[r] = (candidate_mask & (1u << tm)) != 0
                    ? bfloat(result[tm])
                    : bfloat(coarse[r]);
            }
        }
    }
"""

_REFINED_EXACT_SOURCE = """
    constexpr uint VOCAB_SIZE = (uint)VOCAB;
    constexpr uint KDIM = (uint)K;
    constexpr uint GROUPS_PER_LANE = KDIM / 1024u;

    uint tgid = threadgroup_position_in_grid.x;
    uint sgid = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;

    uint base = tgid * 32 + sgid * 4;

    uint base_mask = 0;
    if (lane == 0) {
        #pragma unroll
        for (uint tm = 0; tm < 4; ++tm) {
            uint r = base + tm;
            if (r < VOCAB_SIZE && coarse[r] + float(delta[r]) >= thr[0]) {
                base_mask |= 1u << tm;
            }
        }
    }
    base_mask = simd_broadcast(base_mask, 0);

    if (base_mask == 0) {
        if (lane < 4 && base + lane < VOCAB_SIZE) {
            assembled[base + lane] = bfloat(coarse[base + lane]);
        }
        return;
    }

    // Per-thread scratch: every lane holds a copy, only lane 0's is ever
    // written or consumed.
    thread float refined_coarse[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    uint refined_mask = 0;
    #pragma clang loop unroll(disable)
    for (uint tm = 0; tm < 4; ++tm) {
        if ((base_mask & (1u << tm)) == 0) {
            continue;
        }
        uint r = base + tm;
        const device uint8_t* hirow = codes_bit + size_t(r) * (KDIM / 8);
        const device uint8_t* srow = scales + size_t(r) * (KDIM / 32);
        float correction = 0.0f;
        for (uint gg = 0; gg < GROUPS_PER_LANE; ++gg) {
            uint g = GROUPS_PER_LANE * lane + gg;
            float sd = e8m0_decode(srow[g]);
            uint hb = ((const device uint*)(hirow + g * 4))[0];
            const device ushort4* xrow =
                (const device ushort4*)(x + g * 32);
            float cg = 0.0f;
            #pragma clang loop unroll(full)
            for (uint w = 0; w < 4; ++w) {
                uint hw = hb >> (8u * w);
                uint4 he = (uint4(hw) >> uint4(0u, 2u, 4u, 6u)) & 1u;
                uint4 ho = (uint4(hw) >> uint4(1u, 3u, 5u, 7u)) & 1u;
                float4 ve = float4(he) - 0.5f;
                float4 vo = float4(ho) - 0.5f;
                float4 xa = as_type<float4>(uint4(xrow[2 * w]) << 16);
                float4 xb = as_type<float4>(uint4(xrow[2 * w + 1]) << 16);
                float4 xe = float4(xa.x, xa.z, xb.x, xb.z);
                float4 xo = float4(xa.y, xa.w, xb.y, xb.w);
                #pragma clang loop unroll(full)
                for (uint k = 0; k < 4; ++k) {
                    cg += xe[k] * ve[k];
                    cg += xo[k] * vo[k];
                }
            }
            correction += sd * cg;
        }
        correction = simd_sum(correction);
        if (lane == 0) {
            float c_refined = coarse[r] + correction;
            refined_coarse[tm] = c_refined;
            float d_up = float(delta[r]) * 0x1.005p-1f;
            uint dbits = as_type<uint>(d_up);
            uint dtrunc = dbits & 0xFFFF0000u;
            if (dtrunc != dbits) {
                dtrunc += 0x00010000u;
            }
            float delta_up =
                float(as_type<bfloat>(ushort(dtrunc >> 16)));
            if (r < VOCAB_SIZE && c_refined + delta_up >= thr[0]) {
                refined_mask |= 1u << tm;
            }
        }
    }
    refined_mask = simd_broadcast(refined_mask, 0);

    if (refined_mask == 0) {
        if (lane == 0) {
            #pragma unroll
            for (uint tm = 0; tm < 4; ++tm) {
                uint r = base + tm;
                if (r < VOCAB_SIZE) {
                    assembled[r] = (base_mask & (1u << tm)) != 0
                        ? bfloat(refined_coarse[tm])
                        : bfloat(coarse[r]);
                }
            }
        }
        return;
    }

    // --- stock gemv_al replica begin (gemv.h:151-289) ---
    thread float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    thread bfloat inter[4];
    thread float v_coeff[4];
    uint bn = lane * 4;
    for (uint i = 0; i < KDIM / 128u; ++i) {
        vec<bfloat, 4> xv =
            *((const device vec<bfloat, 4>*)(x + bn));
        v_coeff[0] = float(xv.x);
        v_coeff[1] = float(xv.y);
        v_coeff[2] = float(xv.z);
        v_coeff[3] = float(xv.w);
        #pragma unroll
        for (uint tm = 0; tm < 4; ++tm) {
            const device bfloat* mrow = lm_head + size_t(base + tm) * KDIM;
            vec<bfloat, 4> mv =
                *((const device vec<bfloat, 4>*)(mrow + bn));
            inter[0] = mv.x;
            inter[1] = mv.y;
            inter[2] = mv.z;
            inter[3] = mv.w;
            result[tm] += inter[0] * v_coeff[0];
            result[tm] += inter[1] * v_coeff[1];
            result[tm] += inter[2] * v_coeff[2];
            result[tm] += inter[3] * v_coeff[3];
        }
        bn += 128;
    }
    #pragma unroll
    for (uint tm = 0; tm < 4; ++tm) {
        #pragma unroll
        for (ushort sn = 16; sn >= 1; sn >>= 1) {
            result[tm] += simd_shuffle_down(result[tm], sn);
        }
    }
    // --- stock gemv_al replica end ---
    if (lane == 0) {
        #pragma unroll
        for (uint tm = 0; tm < 4; ++tm) {
            uint r = base + tm;
            if (r < VOCAB_SIZE) {
                assembled[r] = (refined_mask & (1u << tm)) != 0
                    ? bfloat(result[tm])
                    : ((base_mask & (1u << tm)) != 0
                        ? bfloat(refined_coarse[tm])
                        : bfloat(coarse[r]));
            }
        }
    }
"""

_INT5_COARSE_KERNEL = metal_kernel(
    name="lm_head_int5_coarse_ratio_bound_delta",
    input_names=["x", "codes_lo", "codes_hi", "scales"],
    output_names=["coarse", "delta"],
    source=_INT5_COARSE_SOURCE,
    header=_HEADER,
)

_INT5_BASE_COARSE_KERNEL = metal_kernel(
    name="lm_head_int5_base_coarse_delta",
    input_names=["x", "codes_base", "scales"],
    output_names=["coarse", "delta"],
    source=_INT5_BASE_COARSE_SOURCE,
    header=_HEADER,
)

_ARGMAX_STAGE1_KERNEL = metal_kernel(
    name="lm_head_coarse_argmax_stage1",
    input_names=["coarse"],
    output_names=["partial_max", "partial_idx"],
    source=_ARGMAX_STAGE1_SOURCE,
)

_THRESHOLD_KERNEL = metal_kernel(
    name="lm_head_exact_winner_midpoint_threshold",
    input_names=["partial_max", "partial_idx", "lm_head", "x"],
    output_names=["threshold"],
    source=_THRESHOLD_SOURCE,
)

_EXACT_KERNEL = metal_kernel(
    name="lm_head_exact_inline_mask_block_delta",
    input_names=["coarse", "delta", "thr", "lm_head", "x"],
    output_names=["assembled"],
    source=_EXACT_SOURCE,
)

_REFINED_EXACT_KERNEL = metal_kernel(
    name="lm_head_exact_fused_int5_sparse_refine",
    input_names=["coarse", "delta", "thr", "lm_head", "x", "codes_bit", "scales"],
    output_names=["assembled"],
    source=_REFINED_EXACT_SOURCE,
    header=_HEADER,
)

# The launch geometry the record tree ships, all of it structural rather than
# model-derived: 16 rows and 512 threads per coarse threadgroup, 32 rows and 256
# threads per exact threadgroup, and a 128-partial argmax whose second stage is
# one simdgroup reading four partials per lane.
_COARSE_ROWS_PER_TG = 16
_COARSE_THREADS = 512
_EXACT_ROWS_PER_TG = 32
_EXACT_THREADS = 256
_PARTIALS = 128
_READS = 4
_GROUP = 32


class Int5Planes(NamedTuple):
    """The init-time planar int5 copy of the head weight, `hidden * 5/8 + hidden/32` bytes
    per row.

    `codes_lo` `[vocab, hidden/2]` is the nibble plane holding `u >> 1` (byte j carries
    element 2j in its low nibble and 2j+1 in its high nibble); `codes_hi`
    `[vocab, hidden/8]` is the residual bit plane holding `u & 1` (element j of a
    32-element group at bit j of the group's little-endian uint32 word); `scales`
    `[vocab, hidden/32]` is one e8m0 power-of-two byte per group. The nibble plane and the
    scales alone are a self-contained 2x-coarse code, which is what the three-level decode
    arm reads.
    """

    codes_lo: mx.array
    codes_hi: mx.array
    scales: mx.array


def _merge_nibble_pairs(pairs: mx.array) -> mx.array:
    """Two uint16-adjacent low nibbles folded into one byte: the low byte's nibble stays
    put, the high byte's moves to bits 4-7, and the uint8 cast drops the rest."""
    low = pairs & mx.array(0x000F, dtype=mx.uint16)
    high = (pairs >> 4) & mx.array(0x00F0, dtype=mx.uint16)
    return (low | high).astype(mx.uint8)


def int5_planes(weight: mx.array) -> Int5Planes | None:
    """Builds the planar int5 copy of a `[vocab, hidden]` bf16 head weight (untimed init).

    Scale rule: per 32-element group with `gmax = max|w|`, `sd = 2^e` with
    `e = floor_exp(gmax) - 3`, bumped by one when the `gmax` mantissa is `>= 1.9375`, so
    that `gmax/sd < 15.5` EXACTLY in both cases. Then `q = round(w/sd)` -- the quotient is
    exact, `sd` being a power of two -- satisfies `|q| <= 15` and `|w - sd*q| <= sd/2`
    exactly, the flat half-cell the coarse kernel's d-term uses.

    Returns None when the `|q| <= 15` guard fails on the actual tensor. That guard is
    load-bearing, not defensive: the coarse kernel's `m <= 30*d` ratio bound assumes
    `u = q + 16` lands in `[1, 31]`, and were `u = 0` reachable the ratio would be 32. A
    decline must leave the caller on the stock head rather than reach a weaker certificate.

    One caveat carried over from the source rather than fixed: an all-zero 32-element group
    decodes its scale to the fp32 subnormal `2^-127`, and `w / sd` is then `0 / 0` wherever
    mlx's fast-math division flushes that subnormal. A real head weight has no such group;
    a synthetic one does, and the guard reads NaN as passing.
    """
    vocab, hidden = weight.shape
    w = weight.astype(mx.float32).reshape(vocab, hidden // _GROUP, _GROUP)
    gmax = mx.abs(w).max(axis=2)
    gbits = gmax.view(mx.uint32)
    biased_e = (gbits >> 23).astype(mx.int32)
    mant = gbits & mx.array(0x007FFFFF, dtype=mx.uint32)
    # bump when mantissa >= 0.9375 * 2^23 (i.e. m >= 15.5/8).
    bump = (mant >= mx.array(0x780000, dtype=mx.uint32)).astype(mx.int32)
    sd_byte = mx.clip(biased_e - 3 + bump, 0, 255)
    sd = mx.where(
        sd_byte == 0,
        mx.array([0x00400000], dtype=mx.uint32).view(mx.float32),  # 2^-127, e8m0 semantics
        (sd_byte.astype(mx.uint32) << 23).view(mx.float32),
    )
    q = mx.round(w / sd[..., None])
    max_code = mx.abs(q).max().item()
    assert isinstance(max_code, float)
    if max_code > 15.0:
        return None
    # Offset-binary u = q + 16 in [1, 31], split so the nibble plane is a
    # self-contained 2x-coarse code: the nibble holds u >> 1 (the HIGH four
    # bits, floor(q/2) + 8) and the 1-bit plane holds u & 1 (q - 2*floor(q/2)).
    u = (q + 16).astype(mx.uint8).reshape(vocab, hidden)
    base = u >> 1
    u16 = base.view(mx.uint16)  # [V, hidden/2]: elem 2b low byte
    lo = _merge_nibble_pairs(u16)
    # 1-bit plane: bit 0 of each code; element j of each 32-element group
    # lands at bit j of the group's little-endian uint32 word. Step 1: per
    # uint32 word of u (4 codes), gather the four bit-0s into one low nibble.
    u32 = u.view(mx.uint32)  # [V, hidden/4]: elem 4t..4t+3
    nib = (
        (u32 & mx.array(0x01, dtype=mx.uint32))
        | ((u32 >> 7) & mx.array(0x02, dtype=mx.uint32))
        | ((u32 >> 14) & mx.array(0x04, dtype=mx.uint32))
        | ((u32 >> 21) & mx.array(0x08, dtype=mx.uint32))
    ).astype(mx.uint8)
    # Step 2: merge nibble pairs into bytes (byte s = elements 8s..8s+7).
    nib16 = nib.view(mx.uint16)  # [V, hidden/8]
    hi = _merge_nibble_pairs(nib16)
    planes = Int5Planes(lo, hi, sd_byte.astype(mx.uint8))
    mx.eval(planes.codes_lo, planes.codes_hi, planes.scales)
    return planes


def lm_head_argmax_applies(
    vocab: int,
    hidden: int,
    *,
    rows: int,
    dtype: mx.Dtype,
    argmax_only: bool,
) -> bool:
    """Whether the whole six-kernel chain may stand in for a head projection.

    `argmax_only` is the caller's assertion that `argmax` over the returned row is the ONLY
    thing it will do with it. Logprobs, a softmax, temperature or top-p sampling, a top-k
    set, a speculative acceptance ratio, a penalty applied to the row, or simply handing
    the row back to a client -- all of those read slots the chain does not compute, and all
    of them must pass False. The predicate then refuses regardless of geometry, and that
    refusal is the point of this function: nothing else in the module can tell an argmax
    request from a logits request.

    The geometry conditions, once past that gate: one row at a time (a fused single-token
    head, so no batch and no prefill block), bf16 on both sides (the exact pass is a
    literal transcription of mlx's bf16 gemv), a hidden size that is a whole number of
    1024-element lane passes, and a vocabulary that tiles the coarse (16 rows), exact (32
    rows) and argmax (128 partitions of 4 reads per lane) launches at once -- which is
    `vocab % 512 == 0` -- with the second argmax stage still fitting one simdgroup's worth
    of partitions.
    """
    if not argmax_only:
        return False
    if rows != 1 or dtype != mx.bfloat16:
        return False
    partition = vocab // (_PARTIALS * _READS)
    return (
        hidden % 1024 == 0
        and vocab % (_PARTIALS * _READS) == 0
        and vocab % _COARSE_ROWS_PER_TG == 0
        and vocab % _EXACT_ROWS_PER_TG == 0
        and (partition + 31) // 32 <= 32
    )


def _assert_chain(vocab: int, hidden: int, dtype: mx.Dtype) -> None:
    assert lm_head_argmax_applies(vocab, hidden, rows=1, dtype=dtype, argmax_only=True)


def lm_head_int5_coarse(x: mx.array, planes: Int5Planes) -> tuple[mx.array, mx.array]:
    """The one-pass coarse gemv: both int5 planes and the scales -> `(coarse, delta)`.

    `coarse` is fp32, `delta` a bf16 bound rounded toward +inf. Reads
    `hidden * 5/8 + hidden/32` bytes per row and certifies the half-cell
    `d_i * (1 + 61*gamma)`.
    """
    vocab, half = planes.codes_lo.shape
    hidden = half * 2
    _assert_chain(vocab, hidden, x.dtype)
    coarse, delta = _INT5_COARSE_KERNEL(
        inputs=[x, planes.codes_lo, planes.codes_hi, planes.scales],
        template=[("K", hidden)],
        grid=(vocab // _COARSE_ROWS_PER_TG * _COARSE_THREADS, 1, 1),
        threadgroup=(_COARSE_THREADS, 1, 1),
        output_shapes=[(vocab,), (vocab,)],
        output_dtypes=[mx.float32, mx.bfloat16],
    )
    return coarse, delta


def lm_head_int5_base_coarse(x: mx.array, planes: Int5Planes) -> tuple[mx.array, mx.array]:
    """Level one of the three-level screen: the nibble plane and the scales only.

    `hidden * 1/2 + hidden/32` bytes per row -- a quarter of the code bytes fewer than the
    one-pass form -- decoding the cell midpoint `q0 = 2H - 15.5` and certifying the
    full-cell `d_i * (1 + 32*gamma)`. Pairs with `lm_head_refined_exact_assemble`, which
    re-reads the dropped residual plane for surviving blocks only.
    """
    vocab, half = planes.codes_lo.shape
    hidden = half * 2
    _assert_chain(vocab, hidden, x.dtype)
    coarse, delta = _INT5_BASE_COARSE_KERNEL(
        inputs=[x, planes.codes_lo, planes.scales],
        template=[("K", hidden)],
        grid=(vocab // _COARSE_ROWS_PER_TG * _COARSE_THREADS, 1, 1),
        threadgroup=(_COARSE_THREADS, 1, 1),
        output_shapes=[(vocab,), (vocab,)],
        output_dtypes=[mx.float32, mx.bfloat16],
    )
    return coarse, delta


def _argmax_threads(vocab: int) -> int:
    """The stage-one threadgroup: one lane per four coarse values of a partition, rounded
    up to whole simdgroups."""
    active = vocab // (_PARTIALS * _READS)
    return ((active + 31) // 32) * 32


def lm_head_coarse_argmax_partials(coarse: mx.array) -> tuple[mx.array, mx.array]:
    """128 partial `(value, index)` maxima of `coarse`, lowest index winning a tie.

    Reads no `delta`: the exact-winner threshold never forms the coarse lower bound
    `coarse - delta`, so this stage touches one buffer. NaN coarse values lose every `>`
    and are skipped, and correctness does not depend on which row wins -- the threshold is
    sound for any index, so the argmax quality only moves the candidate count.
    """
    (vocab,) = coarse.shape
    threads = _argmax_threads(vocab)
    partial_max, partial_idx = _ARGMAX_STAGE1_KERNEL(
        inputs=[coarse],
        template=[("VOCAB", vocab)],
        grid=(threads, _PARTIALS, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(_PARTIALS,), (_PARTIALS,)],
        output_dtypes=[mx.float32, mx.uint32],
    )
    return partial_max, partial_idx


def lm_head_exact_winner_threshold(
    partial_max: mx.array,
    partial_idx: mx.array,
    weight: mx.array,
    x: mx.array,
) -> mx.array:
    """Finishes the argmax, runs the stock single-row gemv for the winning row `r`, and
    emits the fp32 midpoint between `bfloat(e_r)` and its bf16 predecessor.

    Sound for ANY `r` because `e_r <= e_winner`: a non-candidate has
    `coarse_i + delta_i < p`, so `bfloat(coarse_i) <= p < bfloat(e_r) <= bfloat(e_winner)`,
    while the true winner stays a candidate because its certified upper bound is at least
    `e_winner`. Thresholding against the winner's own exact logit is what removes the
    winner's quantization delta from the comparison.
    """
    vocab, hidden = weight.shape
    _assert_chain(vocab, hidden, x.dtype)
    return _THRESHOLD_KERNEL(
        inputs=[partial_max, partial_idx, weight, x],
        template=[("VOCAB", vocab), ("K", hidden)],
        grid=(_GROUP, 1, 1),
        threadgroup=(_GROUP, 1, 1),
        output_shapes=[(1,)],
        output_dtypes=[mx.float32],
    )[0]


def lm_head_exact_assemble(
    coarse: mx.array,
    delta: mx.array,
    threshold: mx.array,
    weight: mx.array,
    x: mx.array,
) -> mx.array:
    """The one-pass exact pass: a fixed four-row block per simdgroup, the stock bf16 gemv
    replica for candidate rows and `bfloat(coarse[r])` for the rest.

    The stored `delta` was rounded UP, so `coarse[r] + float(delta[r])` is at least the
    fp32 bound's sum while `thr[0]` moved DOWN, and every row an fp32 delta would call a
    candidate is still one here. Newly admitted rows get the stock-exact value in place of
    their certified-below coarse one, which cannot move the argmax.
    """
    vocab, hidden = weight.shape
    _assert_chain(vocab, hidden, x.dtype)
    return _EXACT_KERNEL(
        inputs=[coarse, delta, threshold, weight, x],
        template=[("VOCAB", vocab), ("K", hidden)],
        grid=(vocab // _EXACT_ROWS_PER_TG * _EXACT_THREADS, 1, 1),
        threadgroup=(_EXACT_THREADS, 1, 1),
        output_shapes=[(vocab,)],
        output_dtypes=[mx.bfloat16],
    )[0]


def lm_head_refined_exact_assemble(
    coarse: mx.array,
    delta: mx.array,
    threshold: mx.array,
    weight: mx.array,
    x: mx.array,
    planes: Int5Planes,
) -> mx.array:
    """Levels two and three fused into the exact pass, for the base-plane coarse arm.

    A block failing the base screen writes `bfloat(coarse[r])` and returns. A surviving
    block re-reads the residual bit plane and the scales for its live rows only, adds
    `sum_g sd_g * sum_j x_j*(b_j - 0.5)` to the midpoint value -- reconstructing `sd*q`
    exactly -- and re-screens against `delta[r] * 0x1.005p-1f`, the half-cell bound with
    7*gamma of margin. Only rows surviving THAT read the bf16 weight.
    """
    vocab, hidden = weight.shape
    _assert_chain(vocab, hidden, x.dtype)
    return _REFINED_EXACT_KERNEL(
        inputs=[coarse, delta, threshold, weight, x, planes.codes_hi, planes.scales],
        template=[("VOCAB", vocab), ("K", hidden)],
        grid=(vocab // _EXACT_ROWS_PER_TG * _EXACT_THREADS, 1, 1),
        threadgroup=(_EXACT_THREADS, 1, 1),
        output_shapes=[(vocab,)],
        output_dtypes=[mx.bfloat16],
    )[0]


def lm_head_argmax_row(
    x: mx.array,
    weight: mx.array,
    planes: Int5Planes,
    *,
    refine: bool = True,
) -> mx.array:
    """The four dispatches, composed: a `[vocab]` bf16 row whose ARGMAX is the stock head's
    and whose non-winning slots are an int5 approximation, NOT logits.

    Do not soften that in a caller. Feeding this row to a softmax, a temperature sample, a
    top-k, or a logprob response returns numbers that were never computed. See the module
    docstring and `lm_head_argmax_applies`.

    `refine` selects the three-level decode arm (base plane for every row, residual plane
    for surviving blocks only); False runs the one-pass int5 arm, which is what prefill's
    already-sliced final row uses.
    """
    vocab, hidden = weight.shape
    _assert_chain(vocab, hidden, x.dtype)
    row = x.reshape(hidden)
    coarse, delta = (
        lm_head_int5_base_coarse(row, planes) if refine else lm_head_int5_coarse(row, planes)
    )
    partial_max, partial_idx = lm_head_coarse_argmax_partials(coarse)
    threshold = lm_head_exact_winner_threshold(partial_max, partial_idx, weight, row)
    if refine:
        return lm_head_refined_exact_assemble(coarse, delta, threshold, weight, row, planes)
    return lm_head_exact_assemble(coarse, delta, threshold, weight, row)
