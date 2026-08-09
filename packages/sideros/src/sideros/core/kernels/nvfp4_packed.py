"""The NVFP4 decode the packed-scale kernels share, and the halved plane they read.

`gate_up/nvfp4.py`'s decode is the readable one: a threadgroup LUT for the e2m1 nibble, an
e4m3 byte turned into a half by masking off its sign. This one is the arithmetic-free
twin, and it is not the same value — it is the same value times `2^-22`, on purpose.

*The nibble.* The four bits of an e2m1 code already are a float's sign, exponent and
mantissa, so instead of a table lookup the code word is shifted until each nibble sits in
a half's bit 15 (sign), bits 11-10 (the low two exponent bits) and bit 9 (the mantissa's
top bit); `as_type<half2>` then reads two decoded values out of one 32-bit register. A
half whose exponent field is `0b000ee` is `2^-14` times the e2m1 value it encodes —
subnormals included, since e2m1's subnormal `0.5` lands on the half subnormal `2^-15`.
Eight values come out of one word in four masks and four shifts, with no divergent loads.

*The scale byte.* `bits << 7` puts e4m3's four exponent bits and three mantissa bits into
a half's low four exponent bits and top three mantissa bits, which is the e4m3 value times
`2^-8` — every code exact, denormals included. The byte's own sign bit lands *inside* the
half's exponent, so the decode is only the e4m3 value for **non-negative bytes**;
`halved_group32_scales` refuses a plane that contains one, which is why no kernel here
carries a sign branch.

*The rejoin.* `2^-14` from the nibbles times `2^-8` from the scale is `2^-22`, and the
`4194304.0f` every kernel multiplies by exactly once per output row, at the point where
the fp32 accumulator becomes bf16, is that factor coming back. It has to appear at one
site per `nvfp4_scale` call site or the row is off by a factor of four million; there is no
regime where it partially applies. Nothing rounds in between: the deficit rides in the
exponent of an fp32 accumulator with ~60 binades of headroom over any activation a
normalized residual stream can produce.

*The plane.* MLX's Metal `fp_quantize` takes its absmax over a simdgroup, so both halves of
every 32-weight span of a checkpoint it produced carry the same e4m3 byte — one byte per 16
weights stored, one byte per 32 weights of information. `halved_group32_scales` keeps the
even byte of each pair, which halves the scale traffic and lets a lane address its group
with `lane >> 1`. It is fail-closed: it returns `None` unless every discarded odd byte is
bitwise its even partner, except at the flat pair indices the caller declares — the
quantizer's first simdgroup writes the first span of a tensor twice, and those odd bytes
are copied into a `SCALE_PATCH_BYTES` header in front of the plane. Each kernel restores
them inline at the one lane that would have read them, which is also why the header exists
at all rather than a second buffer binding: the patch is two bytes and a cache line of
padding that keeps the plane itself aligned.
"""

from collections.abc import Sequence

import mlx.core as mx

SCALE_PATCH_BYTES = 128

QDOT_HEADER = """
static inline float nvfp4_scale(uint8_t bits) {
if (bits < 16u) {
    ushort fast_raw = ushort(bits) << 7;
    return float(as_type<half>(fast_raw));
}
    ushort raw = ushort(uint(bits) << 7);
    half converted = as_type<half>(raw);
    half signed_value = converted;
        return float(signed_value);
}

static inline float nvfp4_qdot_codes_16(
    uint2 codes,
    const thread float* input,
    float scale
) {
    float accum;
    {
        const uint c = codes.x;
        const uint xe = c & 0x0F0F0F0Fu;
        const uint ge = xe | (xe << 3);
        const uint yo = c & 0xF0F0F0F0u;
        const uint go = yo | (yo >> 3);
        const uint p0 = (ge << 9) & 0x8E008E00u;
        const uint p1 = (go << 8) & 0x8E008E00u;
        const uint p2 = (ge << 1) & 0x8E008E00u;
        const uint p3 = go & 0x8E008E00u;
        const float2 v04 = float2(as_type<half2>(p0));
        const float2 v15 = float2(as_type<half2>(p1));
        const float2 v26 = float2(as_type<half2>(p2));
        const float2 v37 = float2(as_type<half2>(p3));
        accum =
            (input[0] * v04.x +
             input[1] * v15.x +
             input[2] * v26.x +
             input[3] * v37.x);
        accum +=
            (input[4] * v04.y +
             input[5] * v15.y +
             input[6] * v26.y +
             input[7] * v37.y);
    }
    {
        const uint c = codes.y;
        const uint xe = c & 0x0F0F0F0Fu;
        const uint ge = xe | (xe << 3);
        const uint yo = c & 0xF0F0F0F0u;
        const uint go = yo | (yo >> 3);
        const uint p0 = (ge << 9) & 0x8E008E00u;
        const uint p1 = (go << 8) & 0x8E008E00u;
        const uint p2 = (ge << 1) & 0x8E008E00u;
        const uint p3 = go & 0x8E008E00u;
        const float2 v04 = float2(as_type<half2>(p0));
        const float2 v15 = float2(as_type<half2>(p1));
        const float2 v26 = float2(as_type<half2>(p2));
        const float2 v37 = float2(as_type<half2>(p3));
        accum +=
            (input[8] * v04.x +
             input[9] * v15.x +
             input[10] * v26.x +
             input[11] * v37.x);
        accum +=
            (input[12] * v04.y +
             input[13] * v15.y +
             input[14] * v26.y +
             input[15] * v37.y);
    }
    return scale * accum;
}

static inline float nvfp4_qdot_16(
    const device uint8_t* weight,
    const thread float* input,
    float scale
) {
    const device uint2* packed = (const device uint2*)weight;
    return nvfp4_qdot_codes_16(packed[0], input, scale);
}
"""


def _scalar(value: mx.array) -> int:
    item = value.item()
    assert isinstance(item, int)
    return item


def halved_group32_scales(
    scales: mx.array, allowed_flat_pairs: Sequence[int]
) -> mx.array | None:
    """A group-16 e4m3 plane as the one-byte-per-32-weights plane behind its patch header.

    `allowed_flat_pairs` are indices into the plane read as `[size // 2, 2]`; their odd
    byte survives in the header, slot by slot in the order given, and every other odd byte
    must equal its even partner. `None` means the checkpoint does not satisfy that — the
    caller keeps its full-resolution path rather than quantizing something twice."""
    assert scales.dtype == mx.uint8
    assert scales.ndim >= 1 and scales.shape[-1] % 2 == 0
    pair_count = scales.size // 2
    assert 0 < len(allowed_flat_pairs) <= SCALE_PATCH_BYTES
    assert all(0 <= pair < pair_count for pair in allowed_flat_pairs)

    pairs = scales.reshape(pair_count, 2)
    even, odd = pairs[:, 0], pairs[:, 1]
    allowed = mx.array(list(allowed_flat_pairs), dtype=mx.int32)
    mismatch = (even != odd).astype(mx.int32)
    violations = _scalar(mismatch.sum()) - _scalar(mismatch[allowed].sum())
    # A byte with its sign bit set decodes as the wrong exponent, not as a negative
    # number: the plane is only readable by `nvfp4_scale` while it stays non-negative.
    if violations != 0 or _scalar(mx.max(scales)) >= 128:
        return None

    patch = odd[allowed]
    padding = mx.zeros((SCALE_PATCH_BYTES - patch.size,), dtype=mx.uint8)
    return mx.concatenate([patch, padding, even])
