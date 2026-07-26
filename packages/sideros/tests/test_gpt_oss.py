"""GPT-OSS parity against mlx-lm over the same MXFP4 checkpoint.

No transformers ground truth exists here (no MXFP4 path on Apple Silicon, and fp32 of
a 20B is ~80GB): the golden is mlx-lm's float32 forward over the same packed weights,
bounded by floors measured in the fixture — `noise.logits`, `noise.block_i` (the
residual grows along the trunk, so the floor is per block) and `noise.batching`.
"""

from collections.abc import Iterator
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from conftest import (
    assert_greedy_modulo_ties,
    checkpoint_dir,
    load_golden,
    local_snapshot,
    relative_diff,
)

from sideros import KVCache, stream_ids
from sideros.checkpoint import load_gpt_oss
from sideros.core.kernels.mxfp4_moe_gemv import mxfp4_down_combine
from sideros.core.kernels.sink_attention import sink_attention
from sideros.models import gpt_oss as gpt_oss_module
from sideros.models.gpt_oss import GPTOSS

FIXTURE = Path(__file__).parent / "fixtures" / "gpt_oss_mlxlm.safetensors"
REPO = "openai/gpt-oss-20b"

requires_checkpoint = pytest.mark.skipif(
    local_snapshot(REPO) is None or not FIXTURE.exists(),
    reason="openai/gpt-oss-20b (or its fixture) not available locally",
)


@pytest.fixture(scope="module")
def golden() -> dict[str, mx.array]:
    return load_golden(FIXTURE)


@pytest.fixture(scope="module")
def model() -> GPTOSS:
    return load_gpt_oss(checkpoint_dir(REPO))


@requires_checkpoint
def test_logits_match_mlxlm(model: GPTOSS, golden: dict[str, mx.array]) -> None:
    """2x the floor, same triangle inequality as the blocks test: the golden is fp32,
    `noise.logits` is what mlx-lm's own bf16 rendering costs against it, and ours is a
    different bf16 rendering — the sorted prefill gather batches each expert's rows
    into one gemm (`sorted_indices=True`, mlx-lm does the same), which does not round
    like N single-row gemvs. Measured: 1.24x with the sorted gather, 0.85x without;
    the pure reorder with the hint off is bit-identical to the unsorted path."""
    logits = model(golden["input_ids"][None])
    assert relative_diff(logits, golden["logits"]) < 2 * golden["noise.logits"].item()


@requires_checkpoint
def test_blocks_match_mlxlm(model: GPTOSS, golden: dict[str, mx.array]) -> None:
    """Every block against its own measured floor: a failure names the layer.

    Bound is 2x the floor, and the reason is the triangle inequality, not two bf16
    sides: the golden is mlx-lm's *float32* forward, and `noise.block_i` is what
    mlx-lm's own bf16 rendering costs against it. Our bf16 rendering is a different one
    (fused projections do not round like separate ones), so what we can bound is
    |ours - fp32| <= |ours - mlx-lm bf16| + floor, and the first term is a quantity of
    the same class. Most blocks land exactly on 1.0x (bit-identical to mlx-lm); the
    widest measured is 1.56x.
    """
    blocks = model.activations(golden["input_ids"][None]).blocks
    for i, block in enumerate(blocks):
        floor = golden[f"noise.block_{i}"].item()
        assert relative_diff(block, golden[f"block.{i}"]) < 2 * floor, f"block {i}"


@requires_checkpoint
def test_stepwise_matches_prefill(model: GPTOSS, golden: dict[str, mx.array]) -> None:
    """The sliding mask over a full cache against the one-shot prefill mask, across the
    128-token window. Floor: 3x mlx-lm's own measured batching noise."""
    ids = golden["greedy_ids"]
    prefill = model(ids[None])
    cache = model.make_cache()
    steps = [model(ids[None, i : i + 1], cache) for i in range(ids.shape[0])]
    assert relative_diff(mx.concatenate(steps, axis=1), prefill) < 3 * golden[
        "noise.batching"
    ].item()


@requires_checkpoint
def test_greedy_matches_mlxlm(model: GPTOSS, golden: dict[str, mx.array]) -> None:
    """The reference ids were decoded by mlx-lm in bf16, so they compare modulo ties."""
    prompt = [int(i) for i in np.array(golden["input_ids"])]
    expected = [int(i) for i in np.array(golden["greedy_ids"])]
    generated = list(stream_ids(model, prompt, max_tokens=len(expected) - len(prompt)))
    assert_greedy_modulo_ties(
        prompt + generated,
        expected,
        lambda: model(golden["greedy_ids"][None])[0],
        golden["noise.logits"].item(),
    )


@requires_checkpoint
def test_zeroed_sink_breaks_parity(model: GPTOSS, golden: dict[str, mx.array]) -> None:
    """The sink only shows up in the softmax denominator — zeroing it is invisible to
    shapes and to the strict load, and must be caught numerically."""
    attention = model.model.layers[0].self_attn
    original = attention.sinks
    assert isinstance(original, mx.array)
    attention.sinks = mx.zeros_like(original)
    try:
        logits = model(golden["input_ids"][None])
        assert relative_diff(logits, golden["logits"]) > golden["noise.logits"].item()
    finally:
        attention.sinks = original


@requires_checkpoint
def test_swapped_gate_up_breaks_parity(model: GPTOSS, golden: dict[str, mx.array]) -> None:
    """gate‖up arrives interleaved row by row; reading the pair the other way round is
    a plausible port bug that no shape catches."""
    experts = model.model.layers[0].mlp.experts
    original = experts.gate_up_proj_weight
    assert isinstance(original, mx.array)
    experts.gate_up_proj_weight = original.reshape(
        original.shape[0], -1, 2, original.shape[2]
    )[:, :, ::-1].reshape(original.shape)
    try:
        logits = model(golden["input_ids"][None])
        assert relative_diff(logits, golden["logits"]) > golden["noise.logits"].item()
    finally:
        experts.gate_up_proj_weight = original


@requires_checkpoint
def test_sliding_layers_must_slide(model: GPTOSS, golden: dict[str, mx.array]) -> None:
    """Every other layer attends only 128 keys back. The prompt is 195 tokens, so
    letting those layers see everything shows up (0.111 against a 0.064 floor)."""
    original = model.config.layer_types
    object.__setattr__(model.config, "layer_types", ("full_attention",) * len(original))
    try:
        logits = model(golden["input_ids"][None])
        assert relative_diff(logits, golden["logits"]) > golden["noise.logits"].item()
    finally:
        object.__setattr__(model.config, "layer_types", original)


@requires_checkpoint
def test_affine_mode_is_rejected(model: GPTOSS, golden: dict[str, mx.array]) -> None:
    """MXFP4 has no biases; the affine path demands them, so a mode mix-up aborts
    instead of quietly computing something else."""
    experts = model.model.layers[0].mlp.experts
    with pytest.raises(ValueError, match=r"[Bb]iases"):
        mx.eval(
            mx.gather_qmm(
                mx.zeros((1, 1, 1, 1, model.config.hidden_size), dtype=mx.bfloat16),
                experts.gate_up_proj_weight,
                experts.gate_up_proj_scales,
                None,
                rhs_indices=mx.array([[[0]]]),
                transpose=True,
                group_size=32,
                bits=4,
                mode="affine",
            )
        )


@requires_checkpoint
def test_cache_is_trimmable(model: GPTOSS) -> None:
    cache = model.make_cache()
    assert all(isinstance(c, KVCache) and c.is_trimmable for c in cache)


def test_yarn_table_matches_reference() -> None:
    """The NTK-by-parts table and mscale against the values mlx-lm produces for this
    checkpoint's scaling block (theta 150000, factor 32, 4096 original context)."""
    from sideros.models.gpt_oss import GPTOSSRoPEScaling, yarn_rope

    scaling = GPTOSSRoPEScaling(
        factor=32.0, original_max_position_embeddings=4096, beta_fast=32.0, beta_slow=1.0
    )
    freqs, mscale = yarn_rope(64, 150000.0, scaling)
    assert abs(mscale - 1.3465735902799727) < 1e-12
    extra = 150000.0 ** (mx.arange(0, 64, 2, dtype=mx.float32) / 64)
    # Below the first correction dimension YaRN extrapolates (the raw frequency); above
    # the last it interpolates by the full factor.
    assert relative_diff(freqs[:1], extra[:1]) == 0.0
    assert relative_diff(freqs[-1:], 32.0 * extra[-1:]) < 1e-6


def _stepwise(model: GPTOSS, ids: mx.array) -> mx.array:
    cache = model.make_cache()
    return mx.concatenate([model(ids[None, i : i + 1], cache) for i in range(ids.shape[0])], axis=1)


@pytest.fixture
def kernels_off(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """The A/B switch the bench stage uses: two module attributes, no code edit."""
    monkeypatch.setattr(gpt_oss_module, "USE_MXFP4_MOE_GEMV", False)
    monkeypatch.setattr(gpt_oss_module, "USE_SINK_ATTENTION", False)
    yield


@requires_checkpoint
def test_kernels_are_engaged_at_step(model: GPTOSS, monkeypatch: pytest.MonkeyPatch) -> None:
    """Both predicates must hold on the real checkpoint at T=1 — otherwise every test
    below silently measures the ops path against itself."""
    block = model.model.layers[0]
    x = mx.zeros((1, 1, model.config.hidden_size), dtype=mx.bfloat16)
    assert block.mlp.fused_step_applies(x)

    engaged: list[bool] = []

    def spy(
        queries: mx.array, keys: mx.array, values: mx.array, sinks: mx.array,
        mask: mx.array | None, scale: float,
    ) -> mx.array:
        engaged.append(True)
        return sink_attention(queries, keys, values, sinks, mask, scale)

    monkeypatch.setattr(gpt_oss_module, "USE_SINK_ATTENTION", True)
    monkeypatch.setattr(gpt_oss_module, "sink_attention", spy)
    mx.eval(model(mx.array([[1]])))
    assert len(engaged) == len(model.model.layers)


@requires_checkpoint
def test_kernel_step_matches_ops_step(
    model: GPTOSS, golden: dict[str, mx.array], request: pytest.FixtureRequest
) -> None:
    """The kernels' decode path against the same path with both switches off — the ops
    chain is the parity reference. Floor: mlx-lm's own measured batching noise, since
    both sides are bf16 renderings of the same graph."""
    ids = golden["greedy_ids"]
    ours = _stepwise(model, ids)
    request.getfixturevalue("kernels_off")
    reference = _stepwise(model, ids)
    assert relative_diff(ours, reference) < 3 * golden["noise.batching"].item()


@requires_checkpoint
def test_fused_mlp_dropping_residual_breaks_parity(
    model: GPTOSS, golden: dict[str, mx.array], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Seam mutation: the down kernel folds the block residual itself, so a wiring that
    forgets to hand it over (or adds it twice outside) must be caught."""
    ids = golden["greedy_ids"]

    def without_residual(
        act: mx.array, weight: mx.array, scales: mx.array, bias: mx.array,
        indices: mx.array, routing: mx.array, residual: mx.array,
    ) -> mx.array:
        return mxfp4_down_combine(
            act, weight, scales, bias, indices, routing, mx.zeros_like(residual)
        )

    monkeypatch.setattr(gpt_oss_module, "mxfp4_down_combine", without_residual)
    broken = _stepwise(model, ids)
    monkeypatch.undo()
    assert relative_diff(broken, model(ids[None])) > 3 * golden["noise.batching"].item()


@requires_checkpoint
def test_sink_attention_ignoring_mask_breaks_parity(
    model: GPTOSS, golden: dict[str, mx.array], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Seam mutation: the sliding layers pass their band as the kernel's `mask`; passing
    `None` turns them into full attention, which the 128-token window must expose."""
    ids = golden["greedy_ids"]

    def without_mask(
        queries: mx.array, keys: mx.array, values: mx.array, sinks: mx.array,
        mask: mx.array | None, scale: float,
    ) -> mx.array:
        return sink_attention(queries, keys, values, sinks, None, scale)

    monkeypatch.setattr(gpt_oss_module, "USE_SINK_ATTENTION", True)
    monkeypatch.setattr(gpt_oss_module, "sink_attention", without_mask)
    broken = _stepwise(model, ids)
    monkeypatch.undo()
    assert relative_diff(broken, model(ids[None])) > 3 * golden["noise.batching"].item()
