"""A one-row affine INT8 projection with softplus applied, in one dispatch.

Transcribed from the mlxfast-challenge (Layr Labs, MIT) record tree's
`laguna_gate_sp_*_v1`. Every line is the record's, down to its dense formatting; the only
edit is the contraction width and the group size becoming template arguments. The op
chain it replaces is `mx.quantized_matmul(...,
transpose=True, group_size=32, bits=8)` followed by `nn.softplus` -- and it is the same
chain, not an approximation of it: the contraction is `qmv_fast`'s geometry (32 lanes,
eight codes each, `s * dot + xsum * bias` per group), the fp32 accumulator is rounded to
bf16 before the activation exactly where the two-dispatch chain's projection output is,
and the activation is `mx.logaddexp`'s own body -- `hi + log1p(exp(lo - hi))` with the
infinity mask and an explicit NaN arm.

The output is small (one value per head) and the projection is narrow, so the win is one
dispatch and one round trip through memory for the logits, not arithmetic.
"""

from dataclasses import dataclass
from typing import Self

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.kernels.qmv.kernel import Epilogue, QmvLeaf
from mlx_omnia.engine.core.mxcompat import metal_kernel

_VALUES_PER_THREAD = 8
_BLOCK_SIZE = 256
_ROWS_PER_THREADGROUP = 8
_BITS = 8

_SOURCE = """
constexpr uint K=(uint)AXIS,GS=(uint)GROUP,V=8;
constexpr uint BK=V*32,R=4,NS=2,KG=K/GS,SS=GS/V;
constexpr uint OUT=(uint)OUT_VEC;
uint tile=threadgroup_position_in_grid.x;
uint sg=simdgroup_index_in_threadgroup;
uint lane=thread_index_in_simdgroup;
uint rows_per_tile=NS*R;
uint tiles_per_input=OUT/rows_per_tile;
uint input_row=tile/tiles_per_input;
uint orow=(tile%tiles_per_input)*rows_per_tile+sg*R;
const device uint8_t* ws=(const device uint8_t*)packed_codes+orow*K+lane*V;
const device bfloat* sc=scales+orow*KG+lane/SS;
const device bfloat* bs=biases+orow*KG+lane/SS;
thread float x[V];
thread float r[R]={0.0f,0.0f,0.0f,0.0f};
uint col=lane*V;
const device bfloat* xp=input+input_row*K;
for(uint k=0;k<K;k+=BK){
    float sum=0.0f;
    for(uint i=0;i<V;++i){
        x[i]=float(xp[col+i]);
        sum+=x[i];
    }
    for(uint row=0;row<R;++row){
        const device uint8_t* wl=ws+row*K;
        float s=float(sc[row*KG]),b=float(bs[row*KG]),a=0.0f;
        for(uint i=0;i<V;++i) a+=x[i]*wl[i];
        r[row]+=s*a+sum*b;
    }
    ws+=BK; sc+=BK/GS; bs+=BK/GS; col+=BK;
}
for(uint row=0;row<R;++row){
    r[row]=simd_sum(r[row]);
    if(lane==0){
        float l=float(bfloat(r[row]));
        float g;
        if(metal::isnan(l)) g=NAN;
        else {
            float hi=metal::max(l,0.0f);
            float lo=metal::min(l,0.0f);
            g=(metal::isinf(lo)||metal::isinf(hi))?hi:hi+log1p(metal::exp(lo-hi));
        }
        gate_values[input_row*OUT+orow+row]=bfloat(g);
    }
}
"""

_KERNEL = metal_kernel(
    name="gate_softplus_int8",
    input_names=["input", "packed_codes", "scales", "biases"],
    output_names=["gate_values"],
    source=_SOURCE,
    header="",
)


def gate_softplus_applies(kdim: int, rows: int, *, group_size: int, bits: int) -> bool:
    """The affine byte-per-code contraction has to tile with nothing bounds-checked.

    `bits` is pinned to 8: one code per byte is what `wl[i]` reads. A lane's eight values
    have to fall inside one quantization group and a block has to cover whole groups, so
    `group_size` must be a multiple of eight that divides the 256-value block. A
    simdgroup steps 256 values and a threadgroup owns eight output rows (two simdgroups,
    four rows each), so both dimensions have to divide exactly.
    """
    return (
        bits == _BITS
        and group_size % _VALUES_PER_THREAD == 0
        and _BLOCK_SIZE % group_size == 0
        and kdim % _BLOCK_SIZE == 0
        and rows % _ROWS_PER_THREADGROUP == 0
    )


def gate_softplus(
    x: mx.array,
    weight: mx.array,
    scales: mx.array,
    biases: mx.array,
    *,
    group_size: int = 32,
) -> mx.array:
    """softplus of x [..., kdim] (one row) against a [rows, kdim] affine INT8 stack,
    as [..., rows] bf16."""
    kdim = weight.shape[-1] * 4
    rows = weight.shape[-2]
    input_rows = x.size // kdim
    assert x.shape[-1] == kdim
    assert scales.dtype == mx.bfloat16 and biases.dtype == mx.bfloat16
    assert gate_softplus_applies(kdim, rows, group_size=group_size, bits=_BITS)
    assert scales.shape == (rows, kdim // group_size) and biases.shape == scales.shape
    return _KERNEL(
        inputs=[x, weight, scales, biases],
        template=[("AXIS", kdim), ("GROUP", group_size), ("OUT_VEC", rows)],
        grid=(input_rows * (rows // _ROWS_PER_THREADGROUP) * 64, 1, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(*x.shape[:-1], rows)],
        output_dtypes=[mx.bfloat16],
    )[0]


@dataclass(frozen=True)
class SoftplusQmv:
    weight: mx.array
    scales: mx.array
    biases: mx.array
    group_size: int

    @classmethod
    def build(
        cls,
        leaf: QmvLeaf,
        *,
        kdim: int,
        rows: int,
        epilogue: Epilogue,
        heads: int | None,
    ) -> Self | None:
        if epilogue != "softplus" or not isinstance(leaf, nn.QuantizedLinear):
            return None
        if "bias" in leaf or leaf.mode != "affine" or leaf.biases is None:
            return None
        if leaf.scales.dtype != mx.bfloat16 or leaf.biases.dtype != mx.bfloat16:
            return None
        if not gate_softplus_applies(kdim, rows, group_size=leaf.group_size, bits=leaf.bits):
            return None
        return cls(leaf.weight, leaf.scales, leaf.biases, leaf.group_size)

    def __call__(self, x: mx.array, gate: mx.array | None = None) -> mx.array:
        projected = gate_softplus(
            x, self.weight, self.scales, self.biases, group_size=self.group_size
        )
        return projected.astype(x.dtype)
