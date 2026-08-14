"""The fused gate product + NVFP4 projection against the two-dispatch chain it replaces.

The chain is a per-head broadcast multiply rounded to bf16 and then
`mx.quantized_matmul(..., mode="nvfp4")` over the same packed tensors, so the comparison
isolates the fusion and the decode: the gate has to reach the right head for every one of
the contraction's columns, and the e2m1/e4m3 decode has to reproduce mlx's. A gate routed
one head off is a wholesale change of the result, not a rounding difference.

The floor is the bf16 epilogue -- see `test_kernel_nvfp4_qmv` for the derivation of
`2^-7`. The gated activation is built in fp32 and rounded once, which is the kernel's
`bfloat(float(xp[i]) * g)` exactly.
"""

import mlx.core as mx
import pytest
from conftest import relative_diff

from mlx_omnia.engine.core.kernels.qmv.gated_nvfp4 import gated_nvfp4_qmv, gated_nvfp4_qmv_applies
from mlx_omnia.engine.core.kernels.qmv.nvfp4 import nvfp4_qmv
from mlx_omnia.engine.core.kernels.shared.nvfp4 import LaneMajorScales, lane_major_scales
from mlx_omnia.engine.core.patch import original

KDIM = 2048
ROWS = 512
HEADS = 16
HEAD_DIM = KDIM // HEADS
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
    two group-16 halves share a code, and a row spans at most 15 codes."""
    mx.random.seed(seed)
    base = mx.random.randint(20, 30, (rows, 1, 1))
    delta = mx.random.randint(0, 16, (rows, groups // 32, 16))
    return mx.repeat(base + delta, 2, axis=2).reshape(rows, groups).astype(mx.uint8)


def with_broken_pairs(plane: mx.array) -> mx.array:
    bump = mx.zeros(plane.shape, dtype=mx.uint8)
    bump[1::2, 1] = 1
    return plane + bump


def escaped(bank: LaneMajorScales) -> int:
    count = mx.sum(bank.bases == 0xFF).item()
    assert isinstance(count, int)
    return count


def pre_gated(x: mx.array, gate: mx.array) -> mx.array:
    """The gate product at the kernel's rounding point: fp32 multiply, one round to bf16."""
    scaled = x.reshape(HEADS, HEAD_DIM).astype(mx.float32) * gate.reshape(
        HEADS, 1
    ).astype(mx.float32)
    return scaled.astype(mx.bfloat16).reshape(1, 1, KDIM)


def reference(
    x: mx.array, gate: mx.array, weight: mx.array, scales: mx.array
) -> mx.array:
    """`original`, not `mx.quantized_matmul`: importing mlx_omnia installs a replacement
    over that name."""
    projected = original(mx, "quantized_matmul")(
        pre_gated(x, gate),
        weight,
        scales=scales,
        transpose=True,
        group_size=GROUP,
        bits=BITS,
        mode="nvfp4",
    )
    assert isinstance(projected, mx.array)
    return projected


def test_a_unit_gate_reduces_to_the_ungated_projection() -> None:
    """With `gate == 1` this kernel is `nvfp4_qmv` over the same tensors: same decode, same
    lane-major bank, same deferred `2^22`, only the output retile (four rows per simdgroup
    against one) and the nibble walk (a byte at a time against one `ushort`) differ.

    Comparing the two kernels directly takes both `mx.quantized_matmul` and the gate out of
    the comparison, which is what makes this the discriminating test: a failure here is in
    the retile or the walk, and a pass moves the fault into the gate or the reference."""
    weight = codes(ROWS, KDIM, seed=6)
    scales = fitting_plane(ROWS, KDIM // GROUP, seed=7)
    bank = lane_major_scales(scales)
    x = mx.random.normal((1, 1, KDIM)).astype(mx.bfloat16)
    unit = mx.ones((1, 1, HEADS), dtype=mx.bfloat16)

    ours = gated_nvfp4_qmv(x, unit, weight, scales, bank)

    assert relative_diff(ours, nvfp4_qmv(x, weight, scales, bank)) < BF16_FLOOR


def test_gate_product_matches_a_pre_gated_projection() -> None:
    """The other half of the split: the same two kernels, but the gate applied outside.
    Together with the unit-gate test this localizes any failure to exactly one of the
    retile, the gate index, or the reference."""
    weight = codes(ROWS, KDIM, seed=8)
    scales = fitting_plane(ROWS, KDIM // GROUP, seed=9)
    bank = lane_major_scales(scales)
    x = mx.random.normal((1, 1, KDIM)).astype(mx.bfloat16)
    gate = mx.random.uniform(shape=(1, 1, HEADS)).astype(mx.bfloat16)

    ours = gated_nvfp4_qmv(x, gate, weight, scales, bank)

    expected = nvfp4_qmv(pre_gated(x, gate), weight, scales, bank)
    assert relative_diff(ours, expected) < BF16_FLOOR


def test_applies_states_its_contract() -> None:
    assert gated_nvfp4_qmv_applies(KDIM, ROWS, HEADS, group_size=GROUP, bits=BITS)
    assert not gated_nvfp4_qmv_applies(KDIM, ROWS, HEADS, group_size=32, bits=BITS)
    assert not gated_nvfp4_qmv_applies(KDIM, ROWS, HEADS, group_size=GROUP, bits=8)
    # A lane pair's scale nibbles have to fill whole bytes: 1024 values per byte.
    assert not gated_nvfp4_qmv_applies(KDIM + 512, ROWS, HEADS, group_size=GROUP, bits=BITS)
    # Two simdgroups, four output rows each.
    assert not gated_nvfp4_qmv_applies(KDIM, ROWS + 4, HEADS, group_size=GROUP, bits=BITS)
    # `column >> head_shift` only names the right head when the contraction splits into
    # exactly `heads` heads of power-of-two width.
    assert not gated_nvfp4_qmv_applies(KDIM, ROWS, 5, group_size=GROUP, bits=BITS)
    assert gated_nvfp4_qmv_applies(KDIM, ROWS, KDIM // 256, group_size=GROUP, bits=BITS)


def test_matches_the_chain_through_the_nibble_arm() -> None:
    weight = codes(ROWS, KDIM, seed=0)
    scales = fitting_plane(ROWS, KDIM // GROUP, seed=1)
    bank = lane_major_scales(scales)
    assert escaped(bank) == 0
    x = mx.random.normal((1, 1, KDIM)).astype(mx.bfloat16)
    gate = mx.random.uniform(shape=(1, 1, HEADS)).astype(mx.bfloat16)

    ours = gated_nvfp4_qmv(x, gate, weight, scales, bank)

    assert ours.shape == (1, 1, ROWS)
    assert ours.dtype == mx.bfloat16
    assert relative_diff(ours, reference(x, gate, weight, scales)) < BF16_FLOOR


def test_matches_the_chain_through_the_escape_arm() -> None:
    weight = codes(ROWS, KDIM, seed=2)
    scales = with_broken_pairs(fitting_plane(ROWS, KDIM // GROUP, seed=3))
    bank = lane_major_scales(scales)
    assert escaped(bank) == ROWS // 2
    x = mx.random.normal((1, 1, KDIM)).astype(mx.bfloat16)
    gate = mx.random.uniform(shape=(1, 1, HEADS)).astype(mx.bfloat16)

    ours = gated_nvfp4_qmv(x, gate, weight, scales, bank)

    assert relative_diff(ours, reference(x, gate, weight, scales)) < BF16_FLOOR


def test_gate_reaches_one_head_only() -> None:
    """A one-hot gate leaves a single head's slice of the contraction alive. Every other
    head must contribute nothing, which is what pins `column >> head_shift`."""
    weight = codes(ROWS, KDIM, seed=4)
    scales = fitting_plane(ROWS, KDIM // GROUP, seed=5)
    bank = lane_major_scales(scales)
    x = mx.random.normal((1, 1, KDIM)).astype(mx.bfloat16)
    live = 5
    gate = (mx.arange(HEADS) == live).astype(mx.bfloat16).reshape(1, 1, HEADS)

    ours = gated_nvfp4_qmv(x, gate, weight, scales, bank)

    assert relative_diff(ours, reference(x, gate, weight, scales)) < BF16_FLOOR
    shifted = (mx.arange(HEADS) == live + 1).astype(mx.bfloat16).reshape(1, 1, HEADS)
    off_by_one = gated_nvfp4_qmv(x, shifted, weight, scales, bank)
    assert relative_diff(ours, off_by_one) > BF16_FLOOR


@pytest.mark.parametrize("batch", [2, 4])
def test_small_row_batch_matches_the_chain(batch: int) -> None:
    weight = codes(ROWS, KDIM, seed=6)
    scales = fitting_plane(ROWS, KDIM // GROUP, seed=7)
    bank = lane_major_scales(scales)
    x = mx.random.normal((batch, 1, KDIM)).astype(mx.bfloat16)
    gate = mx.random.uniform(shape=(batch, 1, HEADS)).astype(mx.bfloat16)

    ours = gated_nvfp4_qmv(x, gate, weight, scales, bank)

    assert ours.shape == (batch, 1, ROWS)
    expected = mx.concatenate(
        [
            reference(x[index : index + 1], gate[index : index + 1], weight, scales)
            for index in range(batch)
        ]
    )
    assert relative_diff(ours, expected) < BF16_FLOOR
