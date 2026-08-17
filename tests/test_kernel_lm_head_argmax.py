"""The pruned head chain against a stock projection -- ARGMAX only.

The chain returns a `[vocab]` row whose non-candidate slots hold an int5 approximation
certified below the winner, so there is exactly one property to assert: the row's argmax is
the stock projection's argmax. There is deliberately no test of logit closeness here; such
a test would encode a guarantee the kernels do not make, and it would pass today only by
accident of how far the int5 approximation happens to land.

What is also assertable, and tested, is the certificate the coarse pass emits: the fp32
logit lies inside `[coarse - delta, coarse + delta]` for every row, both for the one-pass
half-cell bound and for the base-plane full-cell one. That is the load-bearing claim -- the
argmax equality is a corollary of it plus the exact-winner threshold.

The argmax comparison filters samples whose top-two stock logits are closer than a bf16
ulp. That guard is about the *reference*: at these test shapes mlx may pick a gemv tiling
whose fp32 accumulation order differs from the `gemv_al` replica the kernels transcribe, so
a sample separated by less than the rounding of either path decides nothing about the
kernel. Samples above the guard decide everything.
"""

from collections.abc import Callable

import mlx.core as mx
import pytest

from mlx_omnia.engine.core.kernels.lm_head import (
    ArgmaxScreenedHead,
    DefaultScreenedHead,
    ScreenedHead,
)
from mlx_omnia.engine.core.kernels.lm_head.argmax import (
    Int5Planes,
    int5_planes,
    lm_head_argmax_applies,
    lm_head_argmax_row,
    lm_head_int5_base_coarse,
    lm_head_int5_coarse,
)

VOCAB = 4096
HIDDEN = 1024
GROUP = 32
SAMPLES = 32
BF16_ULP = 2.0**-8


def head_weight(seed: int = 0) -> mx.array:
    mx.random.seed(seed)
    weight = (mx.random.normal((VOCAB, HIDDEN)) * 0.05).astype(mx.bfloat16)
    mx.eval(weight)
    return weight


def planes_of(weight: mx.array) -> Int5Planes:
    planes = int5_planes(weight)
    assert planes is not None
    return planes


def hidden_states(count: int, seed: int) -> list[mx.array]:
    mx.random.seed(seed)
    states = [mx.random.normal((HIDDEN,)).astype(mx.bfloat16) for _ in range(count)]
    mx.eval(states)
    return states


def stock_logits(weight: mx.array, x: mx.array) -> mx.array:
    """The stock head projection: the whole `[vocab, hidden]` bf16 weight through mlx's own
    gemv, which is what the chain's exact pass replicates row by row."""
    return mx.matmul(weight, x)


def decided(reference: mx.array) -> bool:
    """Whether the reference's own rounding leaves the argmax unambiguous."""
    top = mx.sort(reference.astype(mx.float32))
    gap = float((top[-1] - top[-2]).item())
    return gap > 4 * BF16_ULP * float(mx.abs(top[-1]).item())


def decoded_int5(planes: Int5Planes) -> tuple[mx.array, mx.array]:
    """`(q, sd)` rebuilt from the planes the way the kernels rebuild them: the nibble plane
    carries `floor(q/2) + 8` (byte j: element 2j low, 2j+1 high), the bit plane carries
    `q - 2*floor(q/2)` (element j of a group at bit j of the group's word)."""
    lo = planes.codes_lo.astype(mx.uint32)
    nibbles = mx.stack([lo & 0xF, (lo >> 4) & 0xF], axis=-1).reshape(VOCAB, HIDDEN)
    hi = planes.codes_hi.astype(mx.uint32)
    shifts = mx.arange(8, dtype=mx.uint32)
    bits = ((hi[..., None] >> shifts) & 1).reshape(VOCAB, HIDDEN)
    q = ((nibbles << 1) | bits).astype(mx.int32) - 16
    byte = planes.scales.astype(mx.uint32)
    sd = mx.where(
        byte == 0,
        mx.array([0x00400000], dtype=mx.uint32).view(mx.float32),
        (byte << 23).view(mx.float32),
    )
    return q.astype(mx.float32), sd


def test_applies_accepts_the_geometry_the_launches_tile() -> None:
    assert lm_head_argmax_applies(VOCAB, HIDDEN, rows=1, dtype=mx.bfloat16)


def test_applies_refuses_batches_and_other_dtypes() -> None:
    assert not lm_head_argmax_applies(VOCAB, HIDDEN, rows=2, dtype=mx.bfloat16)
    assert not lm_head_argmax_applies(VOCAB, HIDDEN, rows=1, dtype=mx.float32)


def test_applies_refuses_shapes_the_launch_cannot_tile() -> None:
    assert not lm_head_argmax_applies(
        VOCAB + 32, HIDDEN, rows=1, dtype=mx.bfloat16
    )
    assert not lm_head_argmax_applies(
        VOCAB, HIDDEN + 32, rows=1, dtype=mx.bfloat16
    )


def test_int5_planes_hold_the_half_cell_certificate() -> None:
    """`|q| <= 15` (the ratio bound's premise) and `|w - sd*q| <= sd/2` (the d-term)."""
    weight = head_weight()
    q, sd = decoded_int5(planes_of(weight))

    assert float(mx.abs(q).max().item()) <= 15.0
    cell = mx.repeat(sd, GROUP, axis=1)
    residual = mx.abs(weight.astype(mx.float32) - cell * q)
    assert bool(mx.all(residual <= cell * 0.5).item())


def test_int5_planes_survive_a_tiny_row_and_a_wide_row() -> None:
    """The scale rule is exponent surgery on the group maximum, so the edges that would
    break the `|q| <= 15` guard are a group whose values are all near the same tiny
    magnitude and a group spanning several binades. Not an exactly-zero group: that decodes
    to a subnormal scale and `0 / 0` under fast-math division -- see `int5_planes`."""
    weight = head_weight()
    weight[0] = mx.full((HIDDEN,), 1e-8, dtype=mx.bfloat16)
    weight[1] = (mx.arange(HIDDEN).astype(mx.float32) * 1e-3 + 1e-6).astype(mx.bfloat16)
    mx.eval(weight)

    q, sd = decoded_int5(planes_of(weight))

    assert float(mx.abs(q).max().item()) <= 15.0
    cell = mx.repeat(sd, GROUP, axis=1)
    assert bool(mx.all(mx.abs(weight.astype(mx.float32) - cell * q) <= cell * 0.5).item())


@pytest.mark.parametrize("arm", ["one_pass", "base_plane"])
def test_coarse_bound_brackets_the_exact_logit(arm: str) -> None:
    """The certificate, directly: `|e_i - coarse_i| <= delta_i` for every row."""
    weight = head_weight()
    planes = planes_of(weight)
    (x,) = hidden_states(1, seed=7)

    coarse, delta = (
        lm_head_int5_coarse(x, planes)
        if arm == "one_pass"
        else lm_head_int5_base_coarse(x, planes)
    )
    exact = mx.matmul(weight.astype(mx.float32), x.astype(mx.float32))

    slack = mx.abs(exact - coarse) - delta.astype(mx.float32)
    assert float(slack.max().item()) <= 0.0


@pytest.mark.parametrize("refine", [False, True])
def test_argmax_matches_the_stock_projection(refine: bool) -> None:
    weight = head_weight()
    planes = planes_of(weight)

    judged = 0
    for x in hidden_states(SAMPLES, seed=11):
        reference = stock_logits(weight, x)
        if not decided(reference):
            continue
        judged += 1
        assembled = lm_head_argmax_row(x, weight, planes, refine=refine)
        assert mx.argmax(assembled).item() == mx.argmax(reference).item()
    assert judged >= SAMPLES // 2, f"only {judged}/{SAMPLES} samples were decidable"


@pytest.mark.parametrize("refine", [False, True])
def test_every_slot_is_written(refine: bool) -> None:
    """One lane writes each slot on exactly one path, so no slot may come back
    uninitialized -- which on a fresh buffer shows up as garbage, not as a NaN, unless the
    coverage argument holds."""
    weight = head_weight()
    planes = planes_of(weight)
    (x,) = hidden_states(1, seed=13)

    assembled = lm_head_argmax_row(x, weight, planes, refine=refine)

    assert assembled.shape == (VOCAB,)
    finite = mx.isfinite(assembled.astype(mx.float32))
    assert bool(mx.all(finite).item())


def test_both_arms_pick_the_same_token() -> None:
    """The three-level decode screen only changes which bytes are read on the way to the
    same decision."""
    weight = head_weight()
    planes = planes_of(weight)

    for x in hidden_states(8, seed=17):
        one_pass = lm_head_argmax_row(x, weight, planes, refine=False)
        refined = lm_head_argmax_row(x, weight, planes, refine=True)
        assert mx.argmax(one_pass).item() == mx.argmax(refined).item()


def projection_of(weight: mx.array) -> Callable[[mx.array], mx.array]:
    return lambda x: mx.matmul(x, weight.T)


def test_delegator_picks_the_pruned_chain_and_falls_back_on_geometry() -> None:
    weight = head_weight()
    assert isinstance(
        ScreenedHead(projection_of(weight), weight=weight).strategy, ArgmaxScreenedHead
    )
    assert isinstance(ScreenedHead(projection_of(weight)).strategy, DefaultScreenedHead)
    fp32 = weight.astype(mx.float32)
    assert isinstance(
        ScreenedHead(projection_of(fp32), weight=fp32).strategy, DefaultScreenedHead
    )


@pytest.mark.parametrize("refine", [False, True])
def test_both_strategies_agree_on_the_argmax(refine: bool) -> None:
    """The screened-row contract: the pruned row and the full projection share the one
    thing a caller may read, so the delegator's choice is invisible to a model."""
    weight = head_weight()
    pruned = ScreenedHead(projection_of(weight), weight=weight, refine=refine)
    stock = ScreenedHead(projection_of(weight))
    assert isinstance(pruned.strategy, ArgmaxScreenedHead)

    judged = 0
    for x in hidden_states(SAMPLES, seed=11):
        if not decided(stock_logits(weight, x)):
            continue
        judged += 1
        assert mx.argmax(pruned(x)).item() == mx.argmax(stock(x)).item()
    assert judged >= SAMPLES // 2, f"only {judged}/{SAMPLES} samples were decidable"


def test_the_screened_row_keeps_the_leading_shape() -> None:
    weight = head_weight()
    head = ScreenedHead(projection_of(weight), weight=weight)
    (x,) = hidden_states(1, seed=13)

    assert head(x).shape == (VOCAB,)
    assert head(x.reshape(1, 1, HIDDEN)).shape == (1, 1, VOCAB)
