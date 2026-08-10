"""The one-row NVFP4 projection against `mx.quantized_matmul`'s own arithmetic.

Both paths read the same packed code plane, so what is compared is the decode and the
accumulation order, not the quantizer. The kernel rebuilds the e2m1 nibbles and the e4m3
group scale itself rather than calling mlx's, and it hoists the `2^22` those two decodes
compose to out of the K loop, so a slip in either exponent is a factor of two and lands
far above the floor.

The floor is the bf16 epilogue. Both sides accumulate in fp32 and round once to bf16;
`relative_diff` normalizes by the row maximum, so a one-ulp disagreement on the largest
element is `2^-8` and anything the fp32 orders can produce stays under two of them.
"""

import mlx.core as mx
import pytest
from conftest import relative_diff

from sideros.core.kernels.qmv.nvfp4 import nvfp4_qmv, nvfp4_qmv_applies
from sideros.core.kernels.shared.nvfp4 import (
    LaneMajorScales,
    expand,
    lane_major_scales,
)
from sideros.core.patch import original

KDIM = 2048
ROWS = 64
GROUP = 16
BITS = 4
BF16_FLOOR = 2.0**-7


def codes(rows: int, kdim: int, seed: int) -> mx.array:
    mx.random.seed(seed)
    dense = mx.random.normal((rows, kdim)).astype(mx.float32)
    packed, *_ = mx.quantize(dense, group_size=GROUP, bits=BITS, mode="nvfp4")
    mx.eval(packed)
    return packed


def fitting_plane(rows: int, groups: int, seed: int) -> mx.array:
    """A scale plane the pairwise lane-major bank carries whole: every 32-group block's
    two group-16 halves share a code, and a row spans at most 15 codes. `mx.quantize` on
    random weights produces neither, so a plane that exercises the nibble arm at all has
    to be written directly."""
    mx.random.seed(seed)
    base = mx.random.randint(20, 30, (rows, 1, 1))
    delta = mx.random.randint(0, 16, (rows, groups // 32, 16))
    return mx.repeat(base + delta, 2, axis=2).reshape(rows, groups).astype(mx.uint8)


def with_broken_pairs(plane: mx.array) -> mx.array:
    """Odd rows get a group whose pair partner disagrees, which no pairwise bank can
    carry: those rows must escape to the stock plane."""
    bump = mx.zeros(plane.shape, dtype=mx.uint8)
    bump[1::2, 1] = 1
    return plane + bump


def escaped(bank: LaneMajorScales) -> int:
    count = mx.sum(bank.bases == 0xFF).item()
    assert isinstance(count, int)
    return count


def reference(x: mx.array, weight: mx.array, scales: mx.array) -> mx.array:
    """`original`, not `mx.quantized_matmul`: importing sideros installs a replacement
    over that name, and comparing against it would compare a kernel with itself."""
    projected = original(mx, "quantized_matmul")(
        x, weight, scales=scales, transpose=True, group_size=GROUP, bits=BITS, mode="nvfp4"
    )
    assert isinstance(projected, mx.array)
    return projected


def test_applies_states_its_contract() -> None:
    assert nvfp4_qmv_applies(KDIM, ROWS, group_size=GROUP, bits=BITS)
    assert not nvfp4_qmv_applies(KDIM, ROWS, group_size=32, bits=BITS)
    assert not nvfp4_qmv_applies(KDIM, ROWS, group_size=GROUP, bits=8)
    # One `ushort` of scale nibbles is exactly four blocks: no other contraction fits.
    assert not nvfp4_qmv_applies(KDIM * 2, ROWS, group_size=GROUP, bits=BITS)
    assert not nvfp4_qmv_applies(KDIM // 2, ROWS, group_size=GROUP, bits=BITS)
    # Two simdgroups, one output row each.
    assert not nvfp4_qmv_applies(KDIM, ROWS + 1, group_size=GROUP, bits=BITS)


def test_bank_reproduces_the_plane_it_stands_for() -> None:
    plane = with_broken_pairs(fitting_plane(ROWS, KDIM // GROUP, seed=0))
    bank = lane_major_scales(plane)
    assert escaped(bank) == ROWS // 2
    assert bool(mx.all(expand(bank, plane) == plane).item())


def test_matches_quantized_matmul_through_the_nibble_arm() -> None:
    weight = codes(ROWS, KDIM, seed=1)
    scales = fitting_plane(ROWS, KDIM // GROUP, seed=2)
    bank = lane_major_scales(scales)
    assert escaped(bank) == 0
    x = mx.random.normal((1, 1, KDIM)).astype(mx.bfloat16)

    ours = nvfp4_qmv(x, weight, scales, bank)

    assert ours.shape == (1, 1, ROWS)
    assert ours.dtype == mx.bfloat16
    assert relative_diff(ours, reference(x, weight, scales)) < BF16_FLOOR


def test_matches_quantized_matmul_through_the_escape_arm() -> None:
    """Half the rows carry base `0xFF` and read the stock plane at the stock stride; the
    other half take the nibble arm, in the same dispatch."""
    weight = codes(ROWS, KDIM, seed=3)
    scales = with_broken_pairs(fitting_plane(ROWS, KDIM // GROUP, seed=4))
    bank = lane_major_scales(scales)
    assert escaped(bank) == ROWS // 2
    x = mx.random.normal((1, 1, KDIM)).astype(mx.bfloat16)

    ours = nvfp4_qmv(x, weight, scales, bank)

    assert relative_diff(ours, reference(x, weight, scales)) < BF16_FLOOR


@pytest.mark.parametrize("code", [0x01, 0x07, 0x38, 0x76, 0x7E])
def test_scale_codes_match_quantized_matmul(code: int) -> None:
    """`mx.quantize` never emits a denormal or a saturated scale, so the e4m3 decode's
    edges are only reachable by writing the plane directly. A uniform plane spans zero
    codes, so the bank carries every row through the nibble arm."""
    weight = codes(ROWS, KDIM, seed=5)
    scales = mx.full((ROWS, KDIM // GROUP), code, dtype=mx.uint8)
    bank = lane_major_scales(scales)
    assert escaped(bank) == 0
    x = mx.random.normal((1, 1, KDIM)).astype(mx.bfloat16)

    ours = nvfp4_qmv(x, weight, scales, bank)

    assert relative_diff(ours, reference(x, weight, scales)) < BF16_FLOOR


def test_output_shape_follows_the_input() -> None:
    weight = codes(ROWS, KDIM, seed=6)
    scales = fitting_plane(ROWS, KDIM // GROUP, seed=7)
    bank = lane_major_scales(scales)
    x = mx.random.normal((KDIM,)).astype(mx.bfloat16)

    assert nvfp4_qmv(x, weight, scales, bank).shape == (ROWS,)
