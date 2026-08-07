"""P7 gate: parity vs the reference over the same 4-bit checkpoint, mutation, kernels.

No transformers ground truth exists at this size (fp32 of 30B is ~120GB): the
reference implementation over the same packed weights is the golden, bounded by
measured floors carried in the fixture (noise.logits, noise.batching).
"""

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from conftest import (
    assert_greedy_modulo_ties,
    checkpoint_dir,
    load_golden,
    relative_diff,
    requires_checkpoint,
)

from sideros import KVCache, stream_ids
from sideros.core.kernels.add_rms_norm import add_rms_norm
from sideros.core.kernels.moe_gemv import moe_down_combine, moe_gate_up_act
from sideros.core.kernels.moe_route import softmax_topk, softmax_topk_applies
from sideros.core.kernels.rope_epilogue import rope_epilogue
from sideros.core.mxcompat import softmax
from sideros.models.qwen3.model import Qwen3MoE
from sideros.models.qwen3.moe import CHECKPOINT

FIXTURE = Path(__file__).parent / "fixtures" / "qwen3_moe_mlxlm.safetensors"
REPO = "mlx-community/Qwen3-30B-A3B-4bit"


@pytest.fixture(scope="module")
def golden() -> dict[str, mx.array]:
    return load_golden(FIXTURE)


@pytest.fixture(scope="module")
def model() -> Qwen3MoE:
    return CHECKPOINT.load(checkpoint_dir(REPO), None)


@requires_checkpoint(REPO)
def test_logits_match_mlxlm(model: Qwen3MoE, golden: dict[str, mx.array]) -> None:
    logits = model(golden["input_ids"][None])
    assert relative_diff(logits, golden["logits"]) < golden["noise.logits"].item()


@requires_checkpoint(REPO)
def test_sorted_gather_matches_stepwise(model: Qwen3MoE, golden: dict[str, mx.array]) -> None:
    """Prefill takes the argsort/unsort reorder (168 routed rows); stepwise takes the
    fused T=1 kernels. The floor is 3x the reference's own measured batching noise."""
    ids = golden["greedy_ids"]
    assert ids.shape[0] * 8 >= 64
    prefill = model(ids[None])
    cache = model.make_cache()
    steps = [model(ids[None, i : i + 1], cache) for i in range(ids.shape[0])]
    gap = relative_diff(mx.concatenate(steps, axis=1), prefill)
    assert gap < 3 * golden["noise.batching"].item()


@requires_checkpoint(REPO)
def test_greedy_matches_mlxlm(model: Qwen3MoE, golden: dict[str, mx.array]) -> None:
    """The reference is 4-bit, so the ids compare modulo ties."""
    prompt = [int(i) for i in np.array(golden["input_ids"])]
    expected = [int(i) for i in np.array(golden["greedy_ids"])]
    generated = list(stream_ids(model, prompt, max_tokens=len(expected) - len(prompt)))
    assert_greedy_modulo_ties(
        prompt + generated,
        expected,
        lambda: model(golden["greedy_ids"][None])[0],
        golden["noise.logits"].item(),
    )


@requires_checkpoint(REPO)
def test_mutation_breaks_parity(model: Qwen3MoE, golden: dict[str, mx.array]) -> None:
    """Perturbing one expert stack's scales must blow past the fixture floor."""
    layer = model.model.layers[10]
    original = layer.mlp.switch_mlp.gate_up_proj.scales
    assert isinstance(original, mx.array)
    layer.mlp.switch_mlp.gate_up_proj.scales = original * 1.5
    try:
        logits = model(golden["input_ids"][None])
        assert relative_diff(logits, golden["logits"]) > golden["noise.logits"].item()
    finally:
        layer.mlp.switch_mlp.gate_up_proj.scales = original


@requires_checkpoint(REPO)
def test_cache_trim_rejected_only_when_untrimmable(model: Qwen3MoE) -> None:
    cache = model.make_cache()
    assert all(isinstance(c, KVCache) and c.is_trimmable for c in cache)


def test_softmax_topk_applies_predicate() -> None:
    """One simdgroup owns the row and writes the k winners from its own lanes."""
    assert softmax_topk_applies(128, 8)
    assert not softmax_topk_applies(100, 8)
    assert not softmax_topk_applies(16, 8)
    assert not softmax_topk_applies(128, 33)
    assert not softmax_topk_applies(128, 0)


def test_softmax_topk_matches_op_chain() -> None:
    rng = np.random.default_rng(7)
    for _ in range(8):
        logits = mx.array(rng.standard_normal(128).astype(np.float32)).astype(mx.bfloat16)
        indices, weights = softmax_topk(logits, 8)
        probs = softmax(logits, axis=-1, precise=True)
        chosen = mx.argpartition(probs, kth=120, axis=-1)[120:]
        reference = mx.take_along_axis(probs, chosen, axis=-1)
        reference = reference / reference.sum(keepdims=True)
        picked = np.array(indices), np.array(weights.astype(mx.float32))
        wanted = np.array(chosen), np.array(reference.astype(mx.float32))
        ours = {int(i): w for i, w in zip(*picked, strict=True)}
        theirs = {int(i): w for i, w in zip(*wanted, strict=True)}
        assert set(ours) == set(theirs)
        # Selection is bit-exact (ties included); the weights are not: the kernel's
        # exponential is Metal's, not mlx's, and the renormalizing sum accumulates in
        # the opposite order, rounding each partial in bf16. The bound is 8 bf16 ulps
        # plus the fp32 slack of the two exps (test_kernel_moe_route's _weight_bound).
        order = sorted(theirs)
        assert relative_diff(
            mx.array(np.array([ours[i] for i in order])),
            mx.array(np.array([theirs[i] for i in order])),
        ) < 8 * 2.0**-8 + 32 * 2.0**-23


def test_moe_gemv_matches_op_chain_fp32() -> None:
    """The fp32 template of the real kernels against the gather_qmm op chain."""
    experts, hidden, inner, k, group, bits = 8, 512, 256, 4, 64, 4
    rng = np.random.default_rng(3)
    gate_up = mx.array(rng.standard_normal((experts, 2 * inner, hidden)).astype(np.float32))
    down = mx.array(rng.standard_normal((experts, hidden, inner)).astype(np.float32))
    x = mx.array(rng.standard_normal(hidden).astype(np.float32))
    residual = mx.array(rng.standard_normal(hidden).astype(np.float32))
    indices = mx.array(np.array([1, 3, 4, 6], dtype=np.uint32))
    routing = mx.array(np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float32))

    gw, gs, gb = mx.quantize(gate_up, group_size=group, bits=bits)
    dw, ds, db = mx.quantize(down, group_size=group, bits=bits)

    act = moe_gate_up_act(x, gw, gs, gb, indices, group_size=group, bits=bits)
    out = moe_down_combine(
        act.reshape(-1), dw, ds, db, indices, routing, residual, group_size=group, bits=bits
    )

    fused = mx.gather_qmm(
        x[None, None, None], gw, gs, gb, rhs_indices=indices[None],
        transpose=True, group_size=group, bits=bits,
    )
    pairs = fused.reshape(1, k, inner, 2)
    gated = pairs[..., 0]
    ref_act = gated * mx.sigmoid(gated) * pairs[..., 1]
    projected = mx.gather_qmm(
        ref_act[:, :, None], dw, ds, db, rhs_indices=indices[None],
        transpose=True, group_size=group, bits=bits,
    ).squeeze(2)
    reference = (projected * routing[None, :, None]).sum(axis=1).reshape(-1) + residual

    act_gap = mx.abs(act - ref_act.reshape(k, inner)).max() / mx.abs(ref_act).max()
    assert act_gap.item() < 1e-5
    assert (mx.abs(out - reference).max() / mx.abs(reference).max()).item() < 1e-5


def never_head_dim(head_dim: int) -> bool:
    """A/B switch for rope_epilogue: falsify the predicate, the step takes the op chain."""
    return False


def never_hidden(hidden: int) -> bool:
    """A/B switch for add_rms_norm."""
    return False


def stepwise(model: Qwen3MoE, ids: mx.array) -> mx.array:
    cache = model.make_cache()
    return mx.concatenate([model(ids[None, i : i + 1], cache) for i in range(ids.shape[0])], axis=1)


@requires_checkpoint(REPO)
def test_step_kernels_match_op_path(
    model: Qwen3MoE, golden: dict[str, mx.array], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Kernels on (the default) against both predicates falsified. Same path, same
    inputs, only the rounding of the fused arithmetic differs — bounded by the
    fixture's own measured batching floor, which is the same kind of bf16 reordering."""
    monkeypatch.setattr("sideros.models.qwen3.layers.flags.ROPE_EPILOGUE_KERNEL", True)
    monkeypatch.setattr("sideros.models.qwen3.layers.flags.ADD_RMS_NORM_KERNEL", True)
    ids = golden["greedy_ids"]
    fused = stepwise(model, ids)
    monkeypatch.setattr(
        "sideros.models.qwen3.layers.attention.rope_epilogue_applies", never_head_dim
    )
    monkeypatch.setattr("sideros.models.qwen3.layers.block.add_rms_norm_applies", never_hidden)
    assert relative_diff(fused, stepwise(model, ids)) < 3 * golden["noise.batching"].item()


@requires_checkpoint(REPO)
def test_rope_epilogue_mutation_breaks_stepwise(
    model: Qwen3MoE, golden: dict[str, mx.array], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The seam hands the kernel one norm per side; swapping them must break the
    step-vs-prefill agreement. (Shifting `offset` is not a valid mutation: q and k are
    rotated by the same amount and rope is relative, so a uniform shift is invisible.)"""
    original = rope_epilogue

    def swapped(
        fused: mx.array,
        *,
        query_heads: int,
        kv_heads: int,
        head_dim: int,
        q_norm: mx.array,
        k_norm: mx.array,
        offset: int,
        base: float,
        eps: float,
    ) -> tuple[mx.array, mx.array]:
        return original(
            fused, query_heads=query_heads, kv_heads=kv_heads, head_dim=head_dim,
            q_norm=k_norm, k_norm=q_norm, offset=offset, base=base, eps=eps,
        )

    monkeypatch.setattr("sideros.models.qwen3.layers.flags.ROPE_EPILOGUE_KERNEL", True)
    monkeypatch.setattr("sideros.models.qwen3.layers.attention.rope_epilogue", swapped)
    ids = golden["greedy_ids"]
    gap = relative_diff(stepwise(model, ids), model(ids[None]))
    assert gap > 3 * golden["noise.batching"].item()


@requires_checkpoint(REPO)
def test_add_rms_norm_mutation_breaks_stepwise(
    model: Qwen3MoE, golden: dict[str, mx.array], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The residual join is the kernel's first output; dropping `x` from it must break
    the step-vs-prefill agreement."""

    def without_residual(
        x: mx.array, projected: mx.array, weight: mx.array, eps: float
    ) -> tuple[mx.array, mx.array]:
        return add_rms_norm(mx.zeros_like(x), projected, weight, eps)

    monkeypatch.setattr("sideros.models.qwen3.layers.flags.ADD_RMS_NORM_KERNEL", True)
    monkeypatch.setattr("sideros.models.qwen3.layers.block.add_rms_norm", without_residual)
    ids = golden["greedy_ids"]
    gap = relative_diff(stepwise(model, ids), model(ids[None]))
    assert gap > 3 * golden["noise.batching"].item()
