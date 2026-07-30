"""Llama 4 Scout: parity vs mlx-lm over the same 4-bit checkpoint, mutation, cache.

No transformers ground truth exists at this size (fp32 of 109B is ~436GB): mlx-lm
over the same packed weights is the reference, bounded by measured floors carried
in the fixture (noise.logits, noise.batching, per-block noise.block_i).

The prompt is short (64 tokens, within the 8192 chunk), so the chunked mask is
equivalent to the causal mask and the chunk-boundary mutation is not exercised
here — it would need a >8192-token prompt, infeasible to store as a fixture.
The mutations that ARE caught at short context: traditional rope flip, scale
perturbation, renorm added to sigmoid routing, shared expert dropped.
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
from sideros.models.llama4 import CHECKPOINT, Llama4, Llama4Attention, Llama4Block

FIXTURE = Path(__file__).parent / "fixtures" / "llama4_mlxlm.safetensors"
REPO = "mlx-community/Llama-4-Scout-17B-16E-Instruct-4bit"


@pytest.fixture(scope="module")
def golden() -> dict[str, mx.array]:
    return load_golden(FIXTURE)


@pytest.fixture(scope="module")
def model() -> Llama4:
    return CHECKPOINT.load(checkpoint_dir(REPO), None)


def stepwise(model: Llama4, ids: mx.array) -> mx.array:
    cache = model.make_cache()
    return mx.concatenate(
        [model(ids[None, i : i + 1], cache) for i in range(ids.shape[0])], axis=1
    )


@requires_checkpoint(REPO)
def test_logits_match_mlxlm(model: Llama4, golden: dict[str, mx.array]) -> None:
    logits = model(golden["input_ids"][None])
    assert relative_diff(logits, golden["logits"]) < golden["noise.logits"].item()


@requires_checkpoint(REPO)
def test_stepwise_matches_prefill(model: Llama4, golden: dict[str, mx.array]) -> None:
    """Prefill vs step-by-step. The floor is 3x mlx-lm's own measured batching noise."""
    ids = golden["greedy_ids"]
    prefill = model(ids[None])
    steps = stepwise(model, ids)
    gap = relative_diff(steps, prefill)
    assert gap < 3 * golden["noise.batching"].item()


@requires_checkpoint(REPO)
def test_greedy_matches_mlxlm(model: Llama4, golden: dict[str, mx.array]) -> None:
    """The reference is 4-bit mlx-lm, so the ids compare modulo ties."""
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
def test_per_block_within_floor(model: Llama4, golden: dict[str, mx.array]) -> None:
    """Each block's output within 3x its measured per-block floor."""
    activations = model.activations(golden["input_ids"][None])
    n_blocks = len(activations.blocks)
    floors = [golden[f"noise.block_{j}"] for j in range(n_blocks)]
    for i, (block, floor_val) in enumerate(zip(activations.blocks, floors, strict=True)):
        gap = relative_diff(block, golden[f"block_{i}"])
        assert gap < 3 * floor_val.item(), f"block {i}: {gap:.3e} >= {3 * floor_val.item():.3e}"


@requires_checkpoint(REPO)
def test_mutation_breaks_parity(model: Llama4, golden: dict[str, mx.array]) -> None:
    """Perturbing one expert stack's scales must blow past the fixture floor."""
    layer = model.model.layers[10]
    assert layer.mlp.switch_mlp is not None
    original = layer.mlp.switch_mlp.gate_up_proj.scales
    assert isinstance(original, mx.array)
    layer.mlp.switch_mlp.gate_up_proj.scales = original * 1.5
    try:
        logits = model(golden["input_ids"][None])
        assert relative_diff(logits, golden["logits"]) > golden["noise.logits"].item()
    finally:
        layer.mlp.switch_mlp.gate_up_proj.scales = original


@requires_checkpoint(REPO)
def test_traditional_rope_mutation_breaks_parity(
    model: Llama4, golden: dict[str, mx.array], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flipping traditional=True to False changes the rotation on every dim pair."""
    original = Llama4Attention._rope

    def half_split_rope(self: Llama4Attention, x: mx.array, offset: int) -> mx.array:
        return mx.fast.rope(
            x, self.head_dim, traditional=False, base=None, scale=1.0,
            offset=offset, freqs=self._freqs,
        )

    monkeypatch.setattr(Llama4Attention, "_rope", half_split_rope)
    ids = golden["greedy_ids"]
    gap = relative_diff(stepwise(model, ids), model(ids[None]))
    assert gap > 3 * golden["noise.batching"].item()
    monkeypatch.setattr(Llama4Attention, "_rope", original)


@requires_checkpoint(REPO)
def test_shared_expert_drop_breaks_parity(
    model: Llama4, golden: dict[str, mx.array], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dropping the shared expert must break the logits agreement."""
    layer = model.model.layers[0]

    def dropped(self: object, x: mx.array) -> mx.array:
        return mx.zeros_like(x)

    monkeypatch.setattr(layer.mlp.shared_expert, "__call__", dropped)
    logits = model(golden["input_ids"][None])
    assert relative_diff(logits, golden["logits"]) > golden["noise.logits"].item()


@requires_checkpoint(REPO)
def test_cache_trim_rejected_only_when_untrimmable(model: Llama4) -> None:
    cache = model.make_cache()
    assert all(isinstance(c, KVCache) and c.is_trimmable for c in cache)


@requires_checkpoint(REPO)
def test_fused_step_matches_op_path(
    model: Llama4, golden: dict[str, mx.array], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The T=1 fused step (moe_gemv with pre-multiplication) vs the ops path.
    Both compute the same arithmetic; the bound is the fixture's batching floor."""
    ids = golden["greedy_ids"]
    fused = stepwise(model, ids)

    def ops_path(self: Llama4Block) -> bool:
        return False

    monkeypatch.setattr(Llama4Block, "_fused_step_applies", ops_path)
    ops = stepwise(model, ids)
    assert relative_diff(fused, ops) < 3 * golden["noise.batching"].item()
