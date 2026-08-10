"""The MXFP4 decode: an e2m1 nibble read through a threadgroup lookup and an e8m0 group
exponent applied as a shift, plus the tiling's shape predicate.

The routed gate/up and down/combine kernels read the same bytes on both sides of the
step, so the header and the predicate live here rather than in either half.
"""

HEADER = """
    // The signed e2m1 lookup: the full nibble (sign bit included) indexes the value
    // directly, so the dot loop is a table load and an fma — no mask, no sign branch.
    // The table lives in threadgroup memory, not `constant`: each lane looks up a
    // different nibble and `constant` memory is broadcast-optimized, so divergent reads
    // serialize it (measured 345 vs 608 GB/s in the Swift era). Seeded once per
    // threadgroup.
    inline void mxfp4Lut(threadgroup float* L, uint tid) {
        if (tid < 16) {
            const float v[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                                 -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
            L[tid] = v[tid];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // Eight e2m1 nibbles from one uint32 word, scaled by their shared group's e8m0
    // exponent — which is a float's exponent field, so 2^(byte-127) is a shift, not a
    // transcendental.
    inline float mxfp4Dot8(uint q, float s, const thread float* x,
                           threadgroup const float* L) {
        float d = 0.0f;
        for (uint n = 0; n < 8; n++) d += x[n] * L[(q >> (n * 4)) & 0xF];
        return d * s;
    }
"""


def applies(hidden: int, inner: int) -> bool:
    """A lane reads one uint32 word (eight aligned values inside one e8m0 group) per
    block, and both contractions have to cover whole 32-value groups."""
    return hidden % 32 == 0 and inner % 32 == 0
