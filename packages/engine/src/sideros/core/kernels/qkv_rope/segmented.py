"""The three segments of a mixed qkv projection in one dispatch at T=1.

The segmented module pays three matmul launches plus a concat per step where the
fused projection pays one. This kernel reads the three leaves exactly as loaded —
one decoder per format (dense read, affine packed, mxfp4), selected per segment by
template constant — and writes the concatenated output directly. The code is linear
in the number of formats; a combination is template instantiation, JIT-compiled for
the one the checkpoint actually uses.

`qkv_step` gates applicability and returns None otherwise; the module's concat path
stays as fallback and parity reference. Outside the gate today: linear bias,
affine bits other than 4/8, mxfp8/nvfp4, and geometry that doesn't tile.
"""

from dataclasses import dataclass
from typing import NamedTuple, Self, assert_never

import mlx.core as mx
import mlx.nn as nn

from sideros.core.kernels import MetalDispatch, MetalKernel
from sideros.core.kernels.qkv_rope.default import DefaultQkvRope
from sideros.core.kernels.qkv_rope.kernel import Angles, Leaf, Projection, Rotation

_VALUES_PER_LANE = 8
_BLOCK = _VALUES_PER_LANE * 32
_ROWS_PER_TILE = 4
# One simdgroup per 4-row tile starved the GPU: 32 threads per threadgroup measured
# 38.6% of the bandwidth ceiling against 49.1% for the 3-matmul fallback. Packing 8
# simdgroups per threadgroup (each with its own tile, no cross-simd reduction) reaches
# 49.8% — parity with the fallback per byte, minus three dispatches per step.
_SIMDS_PER_GROUP = 8


def qkv_step(
    x: mx.array,
    queries: nn.Linear | nn.QuantizedLinear,
    keys: nn.Linear | nn.QuantizedLinear,
    values: nn.Linear | nn.QuantizedLinear,
) -> mx.array | None:
    """The concatenated projection of a single token, or None off the gate."""
    if x.ndim != 3 or x.shape[0] != 1 or x.shape[1] != 1:
        return None
    kdim = x.shape[-1]
    if kdim % _BLOCK:
        return None
    q, k, v = (_segment(module) for module in (queries, keys, values))
    if q is None or k is None or v is None:
        return None
    if not all(_applies(segment, x.dtype) for segment in (q, k, v)):
        return None
    out = _kernel(q, k, v, kdim)(
        _Inputs(x.reshape(-1), *_tensors(q), *_tensors(k), *_tensors(v))
    )
    return out.reshape(1, 1, -1)


@dataclass(frozen=True)
class _Dense:
    weight: mx.array


@dataclass(frozen=True)
class _AffinePacked:
    bits: int
    group: int
    weight: mx.array
    scales: mx.array
    biases: mx.array


@dataclass(frozen=True)
class _Mxfp4:
    weight: mx.array
    scales: mx.array


type _Segment = _Dense | _AffinePacked | _Mxfp4


class _Inputs(NamedTuple):
    x: mx.array
    wq: mx.array
    sq: mx.array
    bq: mx.array
    wk: mx.array
    sk: mx.array
    bk: mx.array
    wv: mx.array
    sv: mx.array
    bv: mx.array


def _segment(module: nn.Linear | nn.QuantizedLinear) -> _Segment | None:
    if "bias" in module:
        return None
    match module:
        case nn.QuantizedLinear(
            mode="affine",
            bits=bits,
            group_size=group,
            weight=weight,
            scales=scales,
            biases=biases,
        ) if isinstance(biases, mx.array):
            return _AffinePacked(bits, group, weight, scales, biases)
        case nn.QuantizedLinear(mode="mxfp4", weight=weight, scales=scales):
            return _Mxfp4(weight, scales)
        case nn.QuantizedLinear():
            return None
        case nn.Linear():
            return _Dense(module.weight)
        case _:
            assert_never(module)


def _applies(segment: _Segment, dtype: mx.Dtype) -> bool:
    if segment.weight.shape[0] % _ROWS_PER_TILE:
        return False
    match segment:
        case _Dense(weight=weight):
            return weight.dtype == dtype
        case _AffinePacked(bits=bits, group=group, scales=scales):
            # 2/3/5/6-bit affine would take the generic unpack path unexercised by the
            # parity grid; they stay on the fallback until the grid covers them.
            return (
                bits in (4, 8)
                and not group % _VALUES_PER_LANE
                and not _BLOCK % group
                and scales.dtype == dtype
            )
        case _Mxfp4():
            return True
        case _:
            assert_never(segment)


def _template(segment: _Segment, name: str) -> tuple[tuple[str, int], ...]:
    match segment:
        case _Dense():
            fmt, bits, group = 0, 4, 8
        case _AffinePacked(bits=bits, group=group):
            fmt = 1
        case _Mxfp4():
            fmt, bits, group = 2, 4, 32
        case _:
            assert_never(segment)
    return ((f"F{name}", fmt), (f"B{name}", bits), (f"G{name}", group))


def _tensors(segment: _Segment) -> tuple[mx.array, mx.array, mx.array]:
    """The three kernel buffers of a segment; a format without a tensor repeats the
    weight, which the matching decoder never reads."""
    match segment:
        case _Dense(weight=weight):
            return weight, weight, weight
        case _AffinePacked(weight=weight, scales=scales, biases=biases):
            return weight, scales, biases
        case _Mxfp4(weight=weight, scales=scales):
            return weight, scales, weight
        case _:
            assert_never(segment)


_HEADER = """
    constant float SEG_FP4[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
    };

    template <int bits, int vpt>
    inline float segPackedDot(const device uint8_t* w, const thread float* x) {
        float d = 0.0f;
        if (bits == 4) {
            const device uint* wq = (const device uint*)w;
            for (uint p = 0; p < vpt / 8; p++) {
                uint q = wq[p];
                d += x[p*8+0] * (float)(q & 0xF) + x[p*8+1] * (float)((q >> 4) & 0xF)
                   + x[p*8+2] * (float)((q >> 8) & 0xF) + x[p*8+3] * (float)((q >> 12) & 0xF)
                   + x[p*8+4] * (float)((q >> 16) & 0xF) + x[p*8+5] * (float)((q >> 20) & 0xF)
                   + x[p*8+6] * (float)((q >> 24) & 0xF) + x[p*8+7] * (float)((q >> 28) & 0xF);
            }
        } else {
            constexpr uint mask = (1u << bits) - 1;
            for (uint j = 0; j < vpt; j++) {
                uint bit = j * bits;
                uint shift = bit & 7;
                uint raw = w[bit >> 3];
                if (shift + bits > 8) raw |= (uint)w[(bit >> 3) + 1] << 8;
                d += x[j] * (float)((raw >> shift) & mask);
            }
        }
        return d;
    }

    template <int vpt>
    inline float segMxfp4Dot(const device uint8_t* w, const thread float* x) {
        float d = 0.0f;
        for (uint p = 0; p < vpt / 2; p++) {
            uint8_t q = w[p];
            d += x[p*2] * SEG_FP4[q & 0xF] + x[p*2+1] * SEG_FP4[q >> 4];
        }
        return d;
    }

    template <typename T, int fmt, int bits, int gsize>
    inline void segRows(
        const device uint8_t* w,
        const device uint8_t* s,
        const device uint8_t* b,
        const device T* x,
        device T* y,
        uint row0,
        uint lane,
        uint kdim
    ) {
        constexpr uint vpt = 8;
        constexpr uint block = vpt * 32;
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        if (fmt == 0) {
            const device T* wt = (const device T*)w + (size_t)row0 * kdim + lane * vpt;
            const device T* xv = x + lane * vpt;
            for (uint k = 0; k < kdim; k += block) {
                float xt[vpt];
                for (uint j = 0; j < vpt; j++) xt[j] = (float)xv[j];
                for (uint row = 0; row < 4; row++) {
                    const device T* wr = wt + row * kdim;
                    float d = 0.0f;
                    for (uint j = 0; j < vpt; j++) d += xt[j] * (float)wr[j];
                    acc[row] += d;
                }
                wt += block;
                xv += block;
            }
        } else {
            constexpr uint bytes_per_thread = vpt * bits / 8;
            constexpr uint lanes_per_group = gsize / vpt;
            uint kw_bytes = kdim * bits / 8;
            uint kg = kdim / gsize;
            const device uint8_t* ws = w + (size_t)row0 * kw_bytes + lane * bytes_per_thread;
            const device T* xv = x + lane * vpt;
            uint group = lane / lanes_per_group;
            if (fmt == 1) {
                const device T* sl = (const device T*)s + (size_t)row0 * kg + group;
                const device T* bl = (const device T*)b + (size_t)row0 * kg + group;
                for (uint k = 0; k < kdim; k += block) {
                    float xt[vpt];
                    float xsum = 0.0f;
                    for (uint j = 0; j < vpt; j++) {
                        xt[j] = (float)xv[j];
                        xsum += xt[j];
                    }
                    for (uint row = 0; row < 4; row++) {
                        float sc = (float)sl[row * kg];
                        float bi = (float)bl[row * kg];
                        float d = segPackedDot<bits, vpt>(ws + row * kw_bytes, xt);
                        acc[row] += d * sc + xsum * bi;
                    }
                    ws += block * bits / 8;
                    sl += block / gsize;
                    bl += block / gsize;
                    xv += block;
                }
            } else {
                const device uint8_t* sl = s + (size_t)row0 * kg + group;
                for (uint k = 0; k < kdim; k += block) {
                    float xt[vpt];
                    for (uint j = 0; j < vpt; j++) xt[j] = (float)xv[j];
                    for (uint row = 0; row < 4; row++) {
                        float sc = metal::exp2((float)sl[row * kg] - 127.0f);
                        acc[row] += segMxfp4Dot<vpt>(ws + row * kw_bytes, xt) * sc;
                    }
                    ws += block * bits / 8;
                    sl += block / gsize;
                    xv += block;
                }
            }
        }

        for (uint row = 0; row < 4; row++) {
            float total = simd_sum(acc[row]);
            if (lane == 0) y[row0 + row] = (T)total;
        }
    }
"""

_SOURCE = """
    uint tid = thread_position_in_threadgroup.x;
    uint tgy = threadgroup_position_in_grid.y;
    uint lane = tid & 31;
    uint row0 = (tgy * SIMDS + (tid >> 5)) * 4;
    uint rq = (uint)RQ;
    uint rk = (uint)RK;
    uint kdim = (uint)KD;
    if (row0 >= (uint)RT) return;

    if (row0 < rq) {
        segRows<T, FQ, BQ, GQ>(
            (const device uint8_t*)wq, (const device uint8_t*)sq, (const device uint8_t*)bq,
            x, output, row0, lane, kdim);
    } else if (row0 < rq + rk) {
        segRows<T, FK, BK, GK>(
            (const device uint8_t*)wk, (const device uint8_t*)sk, (const device uint8_t*)bk,
            x, output + rq, row0 - rq, lane, kdim);
    } else {
        segRows<T, FV, BV, GV>(
            (const device uint8_t*)wv, (const device uint8_t*)sv, (const device uint8_t*)bv,
            x, output + rq + rk, row0 - rq - rk, lane, kdim);
    }
"""

_KERNELS: dict[tuple[tuple[str, int], ...], MetalKernel[_Inputs, mx.array]] = {}


def _kernel(
    q: _Segment, k: _Segment, v: _Segment, kdim: int
) -> MetalKernel[_Inputs, mx.array]:
    queries, keys, values = (segment.weight.shape[0] for segment in (q, k, v))
    template = (
        *_template(q, "Q"),
        *_template(k, "K"),
        *_template(v, "V"),
        ("RQ", queries),
        ("RK", keys),
        ("RT", queries + keys + values),
        ("KD", kdim),
        ("SIMDS", _SIMDS_PER_GROUP),
    )
    found = _KERNELS.get(template)
    if found is None:
        rows = queries + keys + values
        tiles = rows // _ROWS_PER_TILE
        groups = (tiles + _SIMDS_PER_GROUP - 1) // _SIMDS_PER_GROUP
        width = 32 * _SIMDS_PER_GROUP

        def launch(input: _Inputs) -> MetalDispatch:
            return MetalDispatch(
                template=(("T", input.x.dtype), *template),
                grid=(width, groups, 1),
                threadgroup=(width, 1, 1),
                output_shapes=((rows,),),
                output_dtypes=(input.x.dtype,),
            )

        found = MetalKernel[_Inputs, mx.array](
            name="segmented_qkv", source=_SOURCE, header=_HEADER, launch=launch
        )
        _KERNELS[template] = found
    return found


@dataclass(frozen=True)
class SegmentedQkv:
    """The prologue of a projection kept as three physical leaves.

    Only the projection fuses here — `qkv_step` reads the three leaves in whatever
    formats the checkpoint mixed and writes the concatenated output in one dispatch —
    and the norm and the rotation run in ops, exactly as the fallback runs them. Off the
    gate (a segment, a format the kernel does not decode) the projection itself falls
    back to the declared one, which is the concatenation of the three leaf calls.
    """

    projection: Projection
    segments: tuple[Leaf, Leaf, Leaf]
    fallback: DefaultQkvRope

    @classmethod
    def build(
        cls,
        projection: Projection,
        *,
        heads: int,
        kv_heads: int,
        head_dim: int,
        rope: Rotation,
        eps: float,
        q_norm: mx.array | None,
        k_norm: mx.array | None,
        base: float | None,
        angles: Angles | None,
        rotary_pairs: int | None,
        mscale: float,
        segments: tuple[Leaf, Leaf, Leaf] | None,
    ) -> Self | None:
        if segments is None:
            return None
        return cls(
            projection,
            segments,
            DefaultQkvRope.build(
                projection,
                heads=heads,
                kv_heads=kv_heads,
                head_dim=head_dim,
                rope=rope,
                eps=eps,
                q_norm=q_norm,
                k_norm=k_norm,
                base=base,
                angles=angles,
                rotary_pairs=rotary_pairs,
                mscale=mscale,
                segments=segments,
            ),
        )

    def __call__(
        self, x: mx.array, offset: int | mx.array
    ) -> tuple[mx.array, mx.array, mx.array]:
        fused = qkv_step(x, *self.segments) if x.shape[1] == 1 else None
        if fused is None:
            fused = self.projection(x)
        return self.fallback.epilogue(fused, offset)
