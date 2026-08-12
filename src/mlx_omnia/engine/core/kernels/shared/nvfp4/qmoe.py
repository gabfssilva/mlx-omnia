"""The readable NVFP4 decode: a threadgroup LUT for the e2m1 nibble and an e4m3 byte
turned into a half by masking off its sign, plus the tiling's shape predicate.

The routed gate/up and down/combine kernels read the same bytes on both sides of the
step, so the header and the predicate live here rather than in either half.
"""

HEADER = """
    // The signed e2m1 lookup: the full nibble (sign bit included) indexes the value
    // directly, so the dot loop is a table load and an fma — no mask, no sign branch.
    // The table lives in threadgroup memory, not `constant`: each lane looks up a
    // different nibble and `constant` memory is broadcast-optimized, so divergent reads
    // serialize it. Seeded once per threadgroup.
    inline void nvfp4Lut(threadgroup float* L, uint tid) {
        if (tid < 16) {
            const float v[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                                 -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
            L[tid] = v[tid];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // The e4m3 group scale, decoded the way mlx decodes it (fp8.h, `fp8_e4m3::operator
    // float16_t`): the byte's seven magnitude bits land in a half's exponent and top
    // mantissa bits, and the 2^8 factor corrects the bias difference (15 against 7).
    // Every one of the 256 codes is exact in half — denormals become half denormals, the
    // largest is 480 — so the value is bit-for-bit mlx's, including 0x7F, which mlx reads
    // as 480 and not as a NaN (its quantizer saturates to 0x7E, so the code only appears
    // in a checkpoint that meant it).
    // Three register ops, and unlike the e2m1 nibble the scale is read once per sixteen
    // weights: a 256-entry threadgroup table would cost the same arithmetic to seed, add
    // a barrier, and then serialize on bank conflicts, since the scale bytes of 32 lanes
    // are unrelated to each other.
    inline float nvfp4Scale(uint8_t b) {
        half m = as_type<half>((ushort)((b & 0x7F) << 7));
        m *= 256.0f;
        return (b & 0x80) ? -(float)m : (float)m;
    }
    // Sixteen e2m1 nibbles from two uint32 words — one whole group — scaled once.
    inline float nvfp4Dot16(uint2 q, float s, const thread float* x,
                            threadgroup const float* L) {
        float d = 0.0f;
        for (uint n = 0; n < 8; n++) d += x[n] * L[(q.x >> (n * 4)) & 0xF];
        for (uint n = 0; n < 8; n++) d += x[8 + n] * L[(q.y >> (n * 4)) & 0xF];
        return d * s;
    }
"""


def applies(hidden: int, inner: int) -> bool:
    """A lane reads one whole 16-value group as a `uint2` and a threadgroup writes 16
    rows: both contractions have to cover whole groups, and the gate‖up half-axis must
    be a multiple of eight."""
    return hidden % 16 == 0 and inner % 16 == 0
