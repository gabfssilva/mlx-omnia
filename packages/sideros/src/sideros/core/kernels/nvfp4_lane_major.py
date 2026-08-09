"""The lane-major re-layout of an NVFP4 group-scale plane.

Both one-row NVFP4 projections below read a `[rows, kdim / 16]` uint8 e4m3 scale plane
strided by 32 groups per lane: lane `l` of a simdgroup owns group `l` of every 32-group
block, so the codes one lane needs sit `32` bytes apart and each block costs its own
byte request. This packing inverts that. A row keeps one base byte and stores every
group's code as a 4-bit offset from it, ordered lane-major, so one lane's whole row is a
contiguous nibble run and arrives in a single load.

Two conditions gate a row: its codes must span at most 15 values, and -- the *pairwise*
form, the only one the kernels here read -- lanes `2j` and `2j + 1` must carry the same
code, which they do when the producing quantizer emitted one scale per 32-element chunk
and the plane merely repeats it across the chunk's two group-16 halves. A row failing
either carries base `0xFF` and the kernel reads the stock plane for it instead, so the
bank is lossless by construction rather than by tolerance; `expand` is the check.
"""

from dataclasses import dataclass

import mlx.core as mx

_ESCAPE = 0xFF
_LANES = 32
_BLOCK_GROUPS = 32


@dataclass(frozen=True)
class LaneMajorScales:
    """`nibbles` [rows, groups / 4] uint8, `bases` [rows] uint8. A base of `0xFF` marks a
    row whose nibble run is meaningless and whose scales must be read from the plane."""

    nibbles: mx.array
    bases: mx.array


def lane_major_applies(groups: int) -> bool:
    """A row's per-lane nibble run has to be a whole number of bytes with no byte
    straddling two lanes: 32 groups per block, one nibble per kept lane-pair, two nibbles
    per byte."""
    return groups % (2 * _LANES) == 0


def lane_major_scales(scales: mx.array) -> LaneMajorScales:
    """A `[rows, groups]` uint8 e4m3 plane packed into its pairwise lane-major bank."""
    rows, groups = scales.shape
    assert lane_major_applies(groups)
    blocks = groups // _BLOCK_GROUPS

    row_min = scales.min(axis=1, keepdims=True)
    span = scales.max(axis=1, keepdims=True).astype(mx.int32) - row_min.astype(mx.int32)
    # [rows, block, group-in-block] -> [rows, lane, block]: lane `l` owns group `l` of
    # every block, so the codes it reads become adjacent.
    lanes = scales.reshape(rows, blocks, _LANES).transpose(0, 2, 1)
    halves = lanes.reshape(rows, _LANES // 2, 2, blocks)
    agree = mx.all(halves[:, :, 0] == halves[:, :, 1], axis=(1, 2)).reshape(rows, 1)
    fits = (span <= 15) & agree

    kept = halves[:, :, 0].reshape(rows, groups // 2).astype(mx.int32)
    offsets = mx.where(fits, kept - row_min.astype(mx.int32), mx.array(0, mx.int32))
    nibbles = (offsets[:, 0::2] | (offsets[:, 1::2] << 4)).astype(mx.uint8)
    bases = mx.where(
        fits.reshape(rows), row_min.reshape(rows), mx.array(_ESCAPE, mx.uint8)
    )
    return LaneMajorScales(nibbles=nibbles, bases=bases)


def expand(bank: LaneMajorScales, scales: mx.array) -> mx.array:
    """The plane the bank stands for, with escaped rows taken from `scales` itself.

    Undoing the permutation in mlx is what makes the bank auditable: equality with
    `scales` is a statement about the bytes the kernels will read, not about the packing.
    """
    rows, groups = scales.shape
    blocks = groups // _BLOCK_GROUPS
    nibbles = bank.nibbles.astype(mx.int32).reshape(rows, groups // 4, 1)
    pairs = mx.concatenate([nibbles & 0x0F, (nibbles >> 4) & 0x0F], axis=2)
    kept = pairs.reshape(rows, _LANES // 2, 1, blocks)
    lanes = mx.concatenate([kept, kept], axis=2).reshape(rows, _LANES, blocks)
    decoded = (bank.bases.astype(mx.int32).reshape(rows, 1, 1) + lanes).transpose(0, 2, 1)
    escaped = (bank.bases == _ESCAPE).reshape(rows, 1)
    return mx.where(escaped, scales, decoded.reshape(rows, groups).astype(mx.uint8))
