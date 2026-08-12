"""The affine dequant dot (`qmoeDot`): a templated unpack for every bit width, with the
word-at-a-time hardware unpack at 2 bits.

The routed gate/up and down/combine kernels read the same bytes on both sides of the
step, so the header lives here rather than in either half.
"""

HEADER = """
    template <int bits, int vpt>
    inline float qmoeDot(const device uint8_t* w, const thread float* x) {
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
        } else if (bits == 2 && vpt % 16 == 0) {
            // One hardware unpack per four bytes: the 0x03030303 mask leaves one 2-bit
            // field per byte and `unpack_unorm4x8_to_float` converts all four at once
            // (divided by 255, multiplied back at the end — folded, it is one multiply
            // per dot, not per weight). ~8% over the shift/mask/convert chain, which is
            // instruction-throughput-bound, not bandwidth-bound, at this bit width.
            const device uint* wq = (const device uint*)w;
            for (uint p = 0; p < vpt / 16; p++) {
                uint q = wq[p];
                float4 f0 = metal::unpack_unorm4x8_to_float(q & 0x03030303u);
                float4 f1 = metal::unpack_unorm4x8_to_float((q >> 2) & 0x03030303u);
                float4 f2 = metal::unpack_unorm4x8_to_float((q >> 4) & 0x03030303u);
                float4 f3 = metal::unpack_unorm4x8_to_float((q >> 6) & 0x03030303u);
                d += metal::dot(float4(x[p*16+ 0], x[p*16+ 4], x[p*16+ 8], x[p*16+12]), f0)
                   + metal::dot(float4(x[p*16+ 1], x[p*16+ 5], x[p*16+ 9], x[p*16+13]), f1)
                   + metal::dot(float4(x[p*16+ 2], x[p*16+ 6], x[p*16+10], x[p*16+14]), f2)
                   + metal::dot(float4(x[p*16+ 3], x[p*16+ 7], x[p*16+11], x[p*16+15]), f3);
            }
            d *= 255.0f;
        } else if (bits == 2) {
            const device ushort* wq = (const device ushort*)w;
            for (uint p = 0; p < vpt / 8; p++) {
                uint q = wq[p];
                for (uint j = 0; j < 8; j++) {
                    d += x[p*8+j] * (float)((q >> (2*j)) & 0x3);
                }
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
"""
