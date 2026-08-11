"""GPT-OSS 120B parity against the reference implementation over the same MXFP4 checkpoint.

Config-only instance of the 20B port (128 experts vs 32, 36 layers vs 24); the model
tree, both kernels and the loader are unchanged. The golden is the reference's "exact"
forward — the same packed MXFP4 weights with everything else in float32 — which stays
feasible at ~70 GB because the MXFP4 expert weights (~61 GB) are not upcast. A full
fp32 forward (~480 GB) is infeasible on 128 GB. Bounded by floors measured in the
fixture: `noise.logits`, `noise.block_i` (the residual grows along a 36-layer trunk,
so the floor is per block) and `noise.batching`.

Architecture-level tests (YaRN table, synthetic MXFP4 leaf load, affine-mode rejection)
are in the 20B suite (`test_gpt_oss.py`); this suite pins the scale.
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

from mlx_omnia import KVCache, stream_ids
from mlx_omnia.checkpoint import stop_tokens
from mlx_omnia.core.config import load_config
from mlx_omnia.core.kernels.attention import SinkAttentionStep
from mlx_omnia.core.kernels.down_combine import DownCombine
from mlx_omnia.core.layers import QuantizedSwitchLinear
from mlx_omnia.models.gpt_oss import CHECKPOINT, GPTOSS, GPTOSSConfig
from mlx_omnia.models.gpt_oss.layers import flags

FIXTURE = Path(__file__).parent / "fixtures" / "gpt_oss_120b_mlxlm.safetensors"
REPO = "openai/gpt-oss-120b"

requires_checkpoint = pytest.mark.skipif(
    local_snapshot(REPO) is None or not FIXTURE.exists(),
    reason="openai/gpt-oss-120b (or its fixture) not available locally",
)


def _mxfp4_leaves(model: GPTOSS) -> tuple[QuantizedSwitchLinear, QuantizedSwitchLinear]:
    """The first layer's expert pair, as the spine built it."""
    leaves = model.model.layers[0].mlp.experts.mxfp4()
    assert leaves is not None
    return leaves


@pytest.fixture(scope="module")
def golden() -> dict[str, mx.array]:
    return load_golden(FIXTURE)


@pytest.fixture(scope="module")
def model() -> GPTOSS:
    return CHECKPOINT.load(checkpoint_dir(REPO), None)


@requires_checkpoint
def test_config_is_120b(model: GPTOSS) -> None:
    """The config-only delta: 128 experts, 36 layers, everything else unchanged."""
    assert model.config.num_hidden_layers == 36
    assert model.config.num_local_experts == 128
    assert model.config.num_experts_per_tok == 4
    assert len(model.config.layer_types) == 36
    assert len(model.model.layers) == 36
    assert model.config.hidden_size == 2880
    assert model.config.sliding_window == 128


@requires_checkpoint
def test_logits_match_mlxlm(model: GPTOSS, golden: dict[str, mx.array]) -> None:
    """2x the floor, same triangle inequality as the blocks test: the golden is the
    fp32-dense forward, `noise.logits` is what the reference's own bf16 rendering costs against
    it, and ours is a different bf16 rendering — the sorted prefill gather batches each
    expert's rows into one gemm (`sorted_indices=True`, the reference does the same), which does
    not round like N single-row gemvs."""
    logits = model(golden["input_ids"][None])
    assert relative_diff(logits, golden["logits"]) < 2 * golden["noise.logits"].item()


@requires_checkpoint
def test_blocks_match_mlxlm(model: GPTOSS, golden: dict[str, mx.array]) -> None:
    """Every block against its own measured floor: a failure names the layer.

    Bound is 2x the floor, and the reason is the triangle inequality: the golden is the
    fp32-dense forward, and `noise.block_i` is what the reference's own bf16 rendering costs
    against it. Our bf16 rendering is a different one (fused projections do not round
    like separate ones), so what we can bound is |ours - fp32| <= |ours - reference bf16| +
    floor, and the first term is a quantity of the same class.
    """
    blocks = model.activations(golden["input_ids"][None]).blocks
    for i, block in enumerate(blocks):
        floor = golden[f"noise.block_{i}"].item()
        assert relative_diff(block, golden[f"block.{i}"]) < 2 * floor, f"block {i}"


@requires_checkpoint
def test_stepwise_matches_prefill(model: GPTOSS, golden: dict[str, mx.array]) -> None:
    """The sliding mask over a full cache against the one-shot prefill mask, across the
    128-token window. Floor: 3x the reference's own measured batching noise."""
    ids = golden["greedy_ids"]
    prefill = model(ids[None])
    cache = model.make_cache()
    steps = [model(ids[None, i : i + 1], cache) for i in range(ids.shape[0])]
    assert relative_diff(mx.concatenate(steps, axis=1), prefill) < 3 * golden[
        "noise.batching"
    ].item()


@requires_checkpoint
def test_greedy_matches_mlxlm(model: GPTOSS, golden: dict[str, mx.array]) -> None:
    """The reference ids were decoded in bf16, so they compare modulo ties."""
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
    gate_up = _mxfp4_leaves(model)[0]
    original = gate_up.weight
    gate_up.weight = original.reshape(original.shape[0], -1, 2, original.shape[2])[
        :, :, ::-1
    ].reshape(original.shape)
    try:
        logits = model(golden["input_ids"][None])
        assert relative_diff(logits, golden["logits"]) > golden["noise.logits"].item()
    finally:
        gate_up.weight = original


@requires_checkpoint
def test_sliding_layers_must_slide(model: GPTOSS, golden: dict[str, mx.array]) -> None:
    """Every other layer attends only 128 keys back. The prompt crosses the 128-token
    sliding window, so letting those layers see everything breaks parity."""
    original = model.config.layer_types
    object.__setattr__(model.config, "layer_types", ("full_attention",) * len(original))
    try:
        logits = model(golden["input_ids"][None])
        assert relative_diff(logits, golden["logits"]) > golden["noise.logits"].item()
    finally:
        object.__setattr__(model.config, "layer_types", original)


@requires_checkpoint
def test_cache_is_trimmable(model: GPTOSS) -> None:
    cache = model.make_cache()
    assert all(isinstance(c, KVCache) and c.is_trimmable for c in cache)


def _stepwise(model: GPTOSS, ids: mx.array) -> mx.array:
    cache = model.make_cache()
    return mx.concatenate([model(ids[None, i : i + 1], cache) for i in range(ids.shape[0])], axis=1)


@pytest.fixture
def kernels_off(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """The A/B switch the bench stage uses: two module attributes, no code edit."""
    monkeypatch.setattr(flags, "USE_MXFP4_MOE_GEMV", False)
    monkeypatch.setattr(flags, "USE_SINK_ATTENTION", False)
    yield


@requires_checkpoint
def test_kernels_are_engaged_at_step(model: GPTOSS, monkeypatch: pytest.MonkeyPatch) -> None:
    """Both predicates must hold on the real checkpoint at T=1 — otherwise every test
    below silently measures the ops path against itself."""
    block = model.model.layers[0]
    x = mx.zeros((1, 1, model.config.hidden_size), dtype=mx.bfloat16)
    assert block.mlp.fused_step(x, x) is not None

    engaged: list[bool] = []
    original = SinkAttentionStep.__call__

    def spy(
        self: SinkAttentionStep, queries: mx.array, keys: mx.array, values: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        engaged.append(True)
        return original(self, queries, keys, values, mask)

    monkeypatch.setattr(flags, "USE_SINK_ATTENTION", True)
    monkeypatch.setattr(SinkAttentionStep, "__call__", spy)
    mx.eval(model(mx.array([[1]])))
    assert len(engaged) == len(model.model.layers)


@requires_checkpoint
def test_kernel_step_matches_ops_step(
    model: GPTOSS, golden: dict[str, mx.array], request: pytest.FixtureRequest
) -> None:
    """The kernels' decode path against the same path with both switches off — the ops
    chain is the parity reference. Floor: the reference's own measured batching noise, since
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

    original = DownCombine.__call__

    def without_residual(
        self: DownCombine, act: mx.array, chosen: mx.array, weights: mx.array,
        residual: mx.array,
    ) -> mx.array:
        return original(self, act, chosen, weights, mx.zeros_like(residual))

    monkeypatch.setattr(DownCombine, "__call__", without_residual)
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

    original = SinkAttentionStep.__call__

    def without_mask(
        self: SinkAttentionStep, queries: mx.array, keys: mx.array, values: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        return original(self, queries, keys, values, None)

    monkeypatch.setattr(flags, "USE_SINK_ATTENTION", True)
    monkeypatch.setattr(SinkAttentionStep, "__call__", without_mask)
    broken = _stepwise(model, ids)
    monkeypatch.undo()
    assert relative_diff(broken, model(ids[None])) > 3 * golden["noise.batching"].item()


@pytest.mark.skipif(
    local_snapshot(REPO) is None, reason="openai/gpt-oss-120b not available locally"
)
def test_the_token_that_ends_a_call_is_in_the_stop_set() -> None:
    """Same harmony protocol as the 20B: config declares `<|return|>` (200002),
    generation_config adds `<|end|>` (199999) and `<|call|>` (200012). The one that
    decides something is `<|call|>`: harmony ends a turn *that called a tool* with it,
    and a stop set without it means a model offered a function writes the call, does not
    stop, and spends the rest of the budget writing the result of its own call.

    The fixture is not needed: the two files are read, not the weights. Requires
    `generation_config.json` in the snapshot (not in the CHECKPOINT patterns — see the
    report's shared-file change request)."""
    directory = checkpoint_dir(REPO)
    declared = load_config(
        GPTOSSConfig, directory / "config.json", allowed_model_types=("gpt_oss",)
    ).eos

    stop = stop_tokens(directory, declared)

    assert declared == (200002,), "the config alone, which is what the trunk carries"
    assert stop[0] == 200002, "the checkpoint's own first eos stays first"
    assert set(stop) == {200002, 199999, 200012}
