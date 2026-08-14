"""The routed quantized MLP of a one-token step against its op chain (the affine
gate-up and down-combine kernels).

The replica is the whole step the two kernels replace — gather the routed gate‖up
stacks, silu(gate)·up, gather the down stacks, weight by the router, sum the experts,
add the residual — on random stacks at checkpoint-like magnitudes: rows scaled
1/sqrt(fan_in), a unit-variance token, and a residual the same order as the routed
sum, so no term of the chain hides inside another (a residual an order of magnitude
above the projections would let a dropped routing weight pass a relative bound).

Both paths read the *same* quantized tensors, so what is compared is arithmetic
order, not the quantizer: the fp32 bound holds identically at 2 bits and at 8. The
bit width is what the templated `qmoeDot` switches on — 4 has its own word-at-a-time
path, 2 a word (or short, on the down side's 8-value tiles) of sixteen (eight)
weights at a time, 3/5/6 stream a bit field across byte boundaries, 8 falls in the
generic branch byte-aligned.

The gate‖up epilogue optionally clamps before the activation (GPT-OSS order: gate
capped above, up clipped both ways); without a limit the guard keeps the path
bit-identical to the unclamped one, which the `limit=None` runs assert implicitly.

Shapes are the smallest the tiling accepts twice over (hidden 1024, inner 512: two
iterations of each kernel's k loop), and `moe_gemv_applies` is asserted so a shape
that stopped engaging the kernel fails here instead of passing vacuously. The shared
expert rides as the last slot of the index vector, with its own 2-D gate‖up and down
projections — the qwen3_5 layout — and its own bit width and group size, which is what
a per-leaf quantization plan produces and what the spare slot of both kernels decodes.
"""

import mlx.core as mx
import numpy as np
import pytest

from mlx_omnia.engine.core.kernels.down_combine.affine import _SOURCE as _DOWN_SOURCE
from mlx_omnia.engine.core.kernels.down_combine.affine import AffineDownCombine
from mlx_omnia.engine.core.kernels.down_combine.affine import applies as down_applies
from mlx_omnia.engine.core.kernels.gate_up.affine import _SOURCE as _GATE_UP_SOURCE
from mlx_omnia.engine.core.kernels.gate_up.affine import HEADER as _HEADER
from mlx_omnia.engine.core.kernels.gate_up.affine import AffineGateUp
from mlx_omnia.engine.core.kernels.gate_up.affine import applies as gate_up_applies
from mlx_omnia.engine.core.mxcompat import metal_kernel
from tests.conftest import relative_diff


def moe_gemv_applies(hidden: int, inner: int, gate_group: int, down_group: int) -> bool:
    return gate_up_applies(hidden, inner, gate_group) and down_applies(hidden, inner, down_group)

HIDDEN, INNER, EXPERTS, TOPK = 1024, 512, 8, 4
BITS = [2, 3, 4, 5, 6, 8]
ROUTED = [1, 3, 4, 6]


class Replica:
    """One token through the routed gate‖up, silu, down, router sum and residual."""

    def __init__(
        self,
        bits: int,
        gate_group: int = 64,
        down_group: int = 64,
        *,
        shared: bool,
        shared_bits: int | None = None,
        shared_group: int | None = None,
        seed: int = 3,
    ) -> None:
        rng = np.random.default_rng(seed)

        def normal(*shape: int, fan_in: int = 1) -> mx.array:
            return mx.array((rng.standard_normal(shape) / np.sqrt(fan_in)).astype(np.float32))

        self.bits, self.gate_group, self.down_group, self.shared = (
            bits, gate_group, down_group, shared
        )
        self.shared_bits = bits if shared_bits is None else shared_bits
        self.shared_group = gate_group if shared_group is None else shared_group
        self.x = normal(HIDDEN)
        self.residual = normal(HIDDEN)
        self.gw, self.gs, self.gb = mx.quantize(
            normal(EXPERTS, 2 * INNER, HIDDEN, fan_in=HIDDEN), group_size=gate_group, bits=bits
        )
        self.dw, self.ds, self.db = mx.quantize(
            normal(EXPERTS, HIDDEN, INNER, fan_in=INNER), group_size=down_group, bits=bits
        )
        self.shared_gate_up = (
            mx.quantize(
                normal(2 * INNER, HIDDEN, fan_in=HIDDEN),
                group_size=self.shared_group, bits=self.shared_bits,
            )
            if shared
            else None
        )
        self.shared_down = (
            mx.quantize(
                normal(HIDDEN, INNER, fan_in=INNER),
                group_size=self.shared_group, bits=self.shared_bits,
            )
            if shared
            else None
        )
        # The shared slot's index is never read on the correct path; it stays in bounds
        # so the "drop the shared slot" mutations gather a routed expert (a wrong answer)
        # instead of running off the stack (a crash).
        self.indices = mx.array(np.array(ROUTED + ([0] if shared else []), dtype=np.uint32))
        self.routing = mx.array(
            np.array([0.4, 0.3, 0.2, 0.1] + ([0.6] if shared else []), dtype=np.float32)
        )

    @property
    def slots(self) -> int:
        return self.indices.shape[0]

    @property
    def spare_gate_up(self) -> tuple[mx.array, mx.array, mx.array, int, int] | None:
        pair = self.shared_gate_up
        return None if pair is None else (*pair, self.shared_group, self.shared_bits)

    @property
    def spare_down(self) -> tuple[mx.array, mx.array, mx.array, int, int] | None:
        pair = self.shared_down
        return None if pair is None else (*pair, self.shared_group, self.shared_bits)

    def kernels(self) -> tuple[mx.array, mx.array]:
        act = AffineGateUp(
            self.gw, self.gs, self.gb, self.gate_group, self.bits, None, self.spare_gate_up
        )(self.x, self.indices)
        out = AffineDownCombine(
            self.dw, self.ds, self.db, self.down_group, self.bits, self.spare_down
        )(act, self.indices, self.routing, self.residual)
        return act, out

    def op_chain(self) -> tuple[mx.array, mx.array]:
        fused = mx.gather_qmm(
            self.x[None, None, None], self.gw, self.gs, self.gb,
            rhs_indices=self.indices[None, :TOPK], transpose=True,
            group_size=self.gate_group, bits=self.bits,
        ).reshape(TOPK, 2 * INNER)
        if self.shared_gate_up is not None:
            sw, ss, sb = self.shared_gate_up
            fused = mx.concatenate([
                fused,
                mx.quantized_matmul(
                    self.x[None], sw, ss, sb, transpose=True,
                    group_size=self.shared_group, bits=self.shared_bits,
                ),
            ], axis=0)
        pairs = fused.reshape(self.slots, INNER, 2)
        gated = pairs[..., 0]
        act = gated * mx.sigmoid(gated) * pairs[..., 1]
        routed = mx.gather_qmm(
            act[None, :TOPK, None], self.dw, self.ds, self.db,
            rhs_indices=self.indices[None, :TOPK], transpose=True,
            group_size=self.down_group, bits=self.bits,
        ).squeeze((0, 2))
        out = (routed * self.routing[:TOPK, None]).sum(axis=0) + self.residual
        if self.shared_down is not None:
            sw, ss, sb = self.shared_down
            out = out + self.routing[TOPK] * mx.quantized_matmul(
                act[TOPK][None], sw, ss, sb, transpose=True,
                group_size=self.shared_group, bits=self.shared_bits,
            ).reshape(-1)
        return act, out


def test_applies_covers_the_replica_and_rejects_untileable_shapes() -> None:
    assert moe_gemv_applies(HIDDEN, INNER, 64, 64)
    assert moe_gemv_applies(HIDDEN, INNER, 32, 32)
    assert moe_gemv_applies(HIDDEN, INNER, 128, 128)
    # hidden must tile 16 values x 32 lanes, inner 8 x 32, and a lane's slice sit
    # inside one quantization group.
    assert not moe_gemv_applies(HIDDEN + 256, INNER, 64, 64)
    assert not moe_gemv_applies(HIDDEN, INNER + 128, 64, 64)
    assert not moe_gemv_applies(HIDDEN, INNER, 8, 64)
    assert not moe_gemv_applies(HIDDEN, INNER, 64, 4)


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("bits", BITS)
def test_matches_op_chain_fp32(bits: int, shared: bool) -> None:
    """The fp32 template of the real kernels against gather_qmm: 1e-5, the house bound."""
    replica = Replica(bits, shared=shared)
    act, out = replica.kernels()
    ref_act, ref_out = replica.op_chain()
    assert act.shape == (replica.slots, INNER)
    assert relative_diff(act, ref_act) < 1e-5
    assert relative_diff(out, ref_out) < 1e-5


@pytest.mark.parametrize("widths", [(3, 8), (8, 3), (4, 2), (2, 5)])
def test_the_spare_slot_decodes_on_its_own_width(widths: tuple[int, int]) -> None:
    """A per-leaf plan gives the shared expert a width and a group of its own, on both
    halves of the step. Both kernels have to read it there while the routed stack keeps
    its own — the reference dequantizes each leaf with the parameters it was packed at."""
    bits, shared_bits = widths
    replica = Replica(bits, 64, 64, shared=True, shared_bits=shared_bits, shared_group=128)
    assert (replica.bits, replica.gate_group) != (replica.shared_bits, replica.shared_group)
    act, out = replica.kernels()
    ref_act, ref_out = replica.op_chain()
    assert relative_diff(act, ref_act) < 1e-5
    assert relative_diff(out, ref_out) < 1e-5


@pytest.mark.parametrize("bits", BITS)
def test_clamped_gate_up_matches_op_chain(bits: int) -> None:
    """With a limit low enough to bite, the kernel's clamp must match the stock order:
    gate capped above, up clipped both ways, before the activation."""
    replica = Replica(bits, shared=False)
    limit = 0.5
    act = AffineGateUp(
        replica.gw, replica.gs, replica.gb, replica.gate_group, replica.bits, limit, None
    )(replica.x, replica.indices)
    fused = mx.gather_qmm(
        replica.x[None, None, None], replica.gw, replica.gs, replica.gb,
        rhs_indices=replica.indices[None], transpose=True,
        group_size=replica.gate_group, bits=replica.bits,
    )
    pairs = fused.reshape(replica.slots, INNER, 2)
    gate = mx.minimum(pairs[..., 0], limit)
    up = mx.clip(pairs[..., 1], -limit, limit)
    ref = gate * mx.sigmoid(gate) * up
    assert relative_diff(mx.minimum(pairs[..., 0], limit), pairs[..., 0]) > 0  # the limit bites
    assert relative_diff(act, ref) < 1e-5


@pytest.mark.parametrize("groups", [(32, 32), (128, 128), (128, 32), (32, 128)])
def test_group_size_reaches_both_kernels(groups: tuple[int, int]) -> None:
    """Gate‖up and down carry independent group sizes; a frozen one shows here."""
    gate_group, down_group = groups
    replica = Replica(4, gate_group, down_group, shared=True)
    assert moe_gemv_applies(HIDDEN, INNER, gate_group, down_group)
    act, out = replica.kernels()
    ref_act, ref_out = replica.op_chain()
    assert relative_diff(act, ref_act) < 1e-5
    assert relative_diff(out, ref_out) < 1e-5


_GATE_UP_MUTATIONS = {
    "drop the zero point": ("result[row] += d * s + xsum * b;", "result[row] += d * s;"),
    "invert the silu sigmoid": ("metal::exp(-g)", "metal::exp(g)"),
    "drop the expert indirection": (
        "size_t wbase = last ? 0 : (size_t)IDX[expert] * rows;",
        "size_t wbase = last ? 0 : (size_t)expert * rows;",
    ),
    "drop the shared slot": (
        "bool last = SHARED && expert == (uint)TOPK - 1;",
        "bool last = false;",
    ),
}

_DOWN_MUTATIONS = {
    "drop the shared slot": ("bool last = SHARED && e == (uint)TOPK - 1;", "bool last = false;"),
    "drop the routing weight": ("float wt = (float)WTS[e];", "float wt = 1.0f;"),
    "drop the residual": (
        "Y[i] = (T)((float)(T)acc[row] + (float)RES[i]);",
        "Y[i] = (T)acc[row];",
    ),
    "drop the down zero point": ("res[row] += d * s + xs_ * b;", "res[row] += d * s;"),
}


def _gate_up(source: str, name: str, replica: Replica) -> mx.array:
    spare = replica.spare_gate_up
    assert spare is not None
    return metal_kernel(
        name=name,
        input_names=[
            "X", "W", "S", "Bs", "IDX", "LIM", "N", "KD", "GSIZE", "SW", "SS", "SB", "SGSIZE"
        ],
        output_names=["Y"],
        source=source,
        header=_HEADER,
    )(
        inputs=[
            replica.x, replica.gw, replica.gs, replica.gb, replica.indices,
            mx.array(float("inf"), dtype=mx.float32),
            mx.array(INNER, dtype=mx.int32),
            mx.array(HIDDEN, dtype=mx.int32),
            mx.array(replica.gate_group, dtype=mx.int32),
            spare[0], spare[1], spare[2],
            mx.array(replica.shared_group, dtype=mx.int32),
        ],
        template=[
            ("T", mx.float32), ("BITS", replica.bits), ("SBITS", replica.shared_bits),
            ("TOPK", replica.slots), ("SHARED", 1),
        ],
        grid=(64, 2 * INNER // 8, replica.slots),
        threadgroup=(64, 1, 1),
        output_shapes=[(replica.slots, INNER)],
        output_dtypes=[mx.float32],
    )[0]


def _down(source: str, name: str, replica: Replica, act: mx.array) -> mx.array:
    spare = replica.spare_down
    assert spare is not None
    return metal_kernel(
        name=name,
        input_names=[
            "ACT", "W", "S", "Bs", "IDX", "WTS", "RES", "SW", "SS", "SB", "N", "KD",
            "GSIZE", "SGSIZE",
        ],
        output_names=["Y"],
        source=source,
        header=_HEADER,
    )(
        inputs=[
            act.reshape(-1), replica.dw, replica.ds, replica.db, replica.indices,
            replica.routing, replica.residual, spare[0], spare[1], spare[2],
            mx.array(HIDDEN, dtype=mx.int32),
            mx.array(INNER, dtype=mx.int32),
            mx.array(replica.down_group, dtype=mx.int32),
            mx.array(replica.shared_group, dtype=mx.int32),
        ],
        template=[
            ("T", mx.float32), ("BITS", replica.bits), ("SBITS", replica.shared_bits),
            ("TOPK", replica.slots), ("SHARED", 1),
            ("VPT", 16 if replica.bits == 2 and INNER % 512 == 0 else 8),
        ],
        grid=(64, HIDDEN // 8, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(HIDDEN,)],
        output_dtypes=[mx.float32],
    )[0]


def test_dropping_the_clamp_breaks_the_clamped_parity() -> None:
    source = _GATE_UP_SOURCE.replace("g = metal::min(g, limit);", "")
    assert source != _GATE_UP_SOURCE
    replica = Replica(4, shared=False)
    limit = 0.5
    ref = AffineGateUp(
        replica.gw, replica.gs, replica.gb, replica.gate_group, replica.bits, limit, None
    )(replica.x, replica.indices)
    broken = metal_kernel(
        name="moe_gemv_gate_up_no_clamp",
        input_names=[
            "X", "W", "S", "Bs", "IDX", "LIM", "N", "KD", "GSIZE", "SW", "SS", "SB", "SGSIZE"
        ],
        output_names=["Y"],
        source=source,
        header=_HEADER,
    )(
        inputs=[
            replica.x, replica.gw, replica.gs, replica.gb, replica.indices,
            mx.array(limit, dtype=mx.float32),
            mx.array(INNER, dtype=mx.int32),
            mx.array(HIDDEN, dtype=mx.int32),
            mx.array(replica.gate_group, dtype=mx.int32),
            replica.gw, replica.gs, replica.gb,
            mx.array(replica.gate_group, dtype=mx.int32),
        ],
        template=[
            ("T", mx.float32), ("BITS", replica.bits), ("SBITS", replica.bits),
            ("TOPK", replica.slots), ("SHARED", 0),
        ],
        grid=(64, 2 * INNER // 8, replica.slots),
        threadgroup=(64, 1, 1),
        output_shapes=[(replica.slots, INNER)],
        output_dtypes=[mx.float32],
    )[0]
    assert relative_diff(broken, ref) > 1e-3


@pytest.mark.parametrize("mutation", sorted(_GATE_UP_MUTATIONS))
def test_gate_up_mutations_break_parity(mutation: str) -> None:
    """Each documented break is compiled as its own kernel and must fail the fp32 bound."""
    old, new = _GATE_UP_MUTATIONS[mutation]
    replica = Replica(4, shared=True)
    ref_act, _ = replica.op_chain()
    source = _GATE_UP_SOURCE.replace(old, new)
    assert source != _GATE_UP_SOURCE
    name = f"moe_gemv_gate_up_broken_{sorted(_GATE_UP_MUTATIONS).index(mutation)}"
    assert relative_diff(_gate_up(source, name, replica), ref_act) > 1e-2


@pytest.mark.parametrize("mutation", sorted(_DOWN_MUTATIONS))
def test_down_mutations_break_parity(mutation: str) -> None:
    old, new = _DOWN_MUTATIONS[mutation]
    replica = Replica(4, shared=True)
    ref_act, ref_out = replica.op_chain()
    source = _DOWN_SOURCE.replace(old, new)
    assert source != _DOWN_SOURCE
    name = f"moe_gemv_down_broken_{sorted(_DOWN_MUTATIONS).index(mutation)}"
    assert relative_diff(_down(source, name, replica, ref_act), ref_out) > 1e-2


def test_unmutated_replicas_of_the_dispatch_agree_with_the_module() -> None:
    """The two dispatch replicas above are only evidence if the untouched source
    reproduces what the module's own entry points return."""
    replica = Replica(4, shared=True)
    act, out = replica.kernels()
    assert relative_diff(_gate_up(_GATE_UP_SOURCE, "moe_gemv_gate_up_intact", replica), act) == 0.0
    assert relative_diff(_down(_DOWN_SOURCE, "moe_gemv_down_intact", replica, act), out) == 0.0
