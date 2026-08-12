"""LFM2.5-8B-A1B fp32 parity against transformers, plus cache and mutation gates.

Every tolerance is `3 x` the fixture's own measured fp32-vs-fp64 floor for that tensor:
the trunk is 24 layers deep and the residual grows along it, so a single number would be
vacuous at one end and impossible at the other.
"""

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from conftest import checkpoint_dir, floor, load_golden, relative_diff, requires_checkpoint
from mlx_omnia.engine.core.kernels.moe_gemv_dense import moe_dense_down

from mlx_omnia import KVCache, stream_ids
from mlx_omnia.engine.core.cache import ConvCache
from mlx_omnia.engine.core.kernels.conv_mix import conv_mix
from mlx_omnia.engine.models.lfm2.layers import conv as conv_layer
from mlx_omnia.engine.models.lfm2.layers import experts as experts_layer
from mlx_omnia.engine.models.lfm2.layers import flags
from mlx_omnia.engine.models.lfm2.layers.attention import LFM2Attention
from mlx_omnia.engine.models.lfm2.layers.conv import LFM2Conv
from mlx_omnia.engine.models.lfm2.layers.experts import LFM2SparseMLP
from mlx_omnia.engine.models.lfm2.moe import CHECKPOINT, LFM2MoE
from mlx_omnia.engine.models.lfm2.moe.model import LFM2MoEActivations

FIXTURE = Path(__file__).parent / "fixtures" / "lfm2_moe_forward.safetensors"
REPO = "LiquidAI/LFM2.5-8B-A1B"
N_LAYER = 24
ATTN_LAYER = 2


@pytest.fixture(scope="module")
def golden() -> dict[str, mx.array]:
    return load_golden(FIXTURE)


@pytest.fixture(scope="module")
def model() -> LFM2MoE:
    # The fixture is transformers in fp32; the checkpoint's bf16 upcasts losslessly.
    return CHECKPOINT.load(checkpoint_dir(REPO), mx.float32)


@pytest.fixture(scope="module")
def activations(model: LFM2MoE, golden: dict[str, mx.array]) -> LFM2MoEActivations:
    return model.activations(golden["input_ids"][None])


def conv_of(model: LFM2MoE, layer: int) -> LFM2Conv:
    """mlx.nn.Module's __getattr__ is untyped: every submodule reach is narrowed here."""
    conv = model.model.layers[layer].conv
    assert isinstance(conv, LFM2Conv)
    return conv


def attention_of(model: LFM2MoE, layer: int) -> LFM2Attention:
    attention = model.model.layers[layer].self_attn
    assert isinstance(attention, LFM2Attention)
    return attention


def sparse_mlp_of(model: LFM2MoE, layer: int) -> LFM2SparseMLP:
    mlp = model.model.layers[layer].feed_forward
    assert isinstance(mlp, LFM2SparseMLP)
    return mlp


@requires_checkpoint(REPO)
def test_embeddings_exact(
    activations: LFM2MoEActivations, golden: dict[str, mx.array]
) -> None:
    """bf16 upcast to fp32 is lossless and the lookup is a gather: no arithmetic yet."""
    assert relative_diff(activations.embeddings, golden["embeddings"]) == 0


@requires_checkpoint(REPO)
@pytest.mark.parametrize("layer", range(N_LAYER))
def test_block_within_floor(
    activations: LFM2MoEActivations, golden: dict[str, mx.array], layer: int
) -> None:
    assert relative_diff(activations.blocks[layer], golden[f"block_{layer}"]) < floor(
        golden, f"block_{layer}"
    )


@requires_checkpoint(REPO)
def test_norm_within_floor(
    activations: LFM2MoEActivations, golden: dict[str, mx.array]
) -> None:
    assert relative_diff(activations.norm, golden["norm"]) < floor(golden, "norm")


@requires_checkpoint(REPO)
def test_logits_within_floor(
    activations: LFM2MoEActivations, golden: dict[str, mx.array]
) -> None:
    assert relative_diff(activations.logits, golden["logits"]) < floor(golden, "logits")


@requires_checkpoint(REPO)
def test_greedy_predictions_match(
    activations: LFM2MoEActivations, golden: dict[str, mx.array]
) -> None:
    ours = mx.argmax(activations.logits, axis=-1)
    theirs = mx.argmax(golden["logits"], axis=-1)
    assert mx.array_equal(ours, theirs).item()


@requires_checkpoint(REPO)
def test_conv_block_internals_within_floor(model: LFM2MoE, golden: dict[str, mx.array]) -> None:
    """Block 0: short conv + dense MLP, each submodule against its own hook boundary."""
    block = model.model.layers[0]
    x = model.model.embed_tokens(golden["input_ids"][None])
    normed = block.operator_norm(x)
    assert relative_diff(normed, golden["b0_ln_1"]) < floor(golden, "b0_ln_1")

    mixed = conv_of(model, 0)(normed, ConvCache())
    assert relative_diff(mixed, golden["b0_conv"]) < floor(golden, "b0_conv")

    second = block.ffn_norm(x + mixed)
    assert relative_diff(second, golden["b0_ln_2"]) < floor(golden, "b0_ln_2")
    assert relative_diff(block.feed_forward(second), golden["b0_mlp"]) < floor(golden, "b0_mlp")


@requires_checkpoint(REPO)
def test_attention_moe_block_internals_within_floor(
    model: LFM2MoE, golden: dict[str, mx.array], activations: LFM2MoEActivations
) -> None:
    """Block 2: the first GQA layer, which is also the first routed MoE layer."""
    block = model.model.layers[ATTN_LAYER]
    x = activations.blocks[ATTN_LAYER - 1]
    normed = block.operator_norm(x)
    assert relative_diff(normed, golden["b2_ln_1"]) < floor(golden, "b2_ln_1")

    mixed = attention_of(model, ATTN_LAYER)(normed, KVCache())
    assert relative_diff(mixed, golden["b2_attn"]) < floor(golden, "b2_attn")

    second = block.ffn_norm(x + mixed)
    assert relative_diff(second, golden["b2_ln_2"]) < floor(golden, "b2_ln_2")

    mlp = sparse_mlp_of(model, ATTN_LAYER)
    router = mlp.gate(second)
    assert relative_diff(router, golden["b2_router"]) < floor(golden, "b2_router")
    assert relative_diff(mlp(second), golden["b2_moe"]) < floor(golden, "b2_moe")


@requires_checkpoint(REPO)
def test_qk_norm_and_rope_within_floor(model: LFM2MoE, golden: dict[str, mx.array]) -> None:
    """transformers hooks q_layernorm before its transpose, hence [1, L, heads, dim]."""
    attention = attention_of(model, ATTN_LAYER)
    q, k, _ = attention.split_heads(golden["b2_ln_1"])
    for normed, name in ((q, "b2_q_norm"), (k, "b2_k_norm")):
        reference = golden[name].transpose(0, 2, 1, 3)
        assert relative_diff(normed, reference) < floor(golden, name)
    for rotated, name in (
        (attention.rope(q, 0), "b2_q_rope"),
        (attention.rope(k, 0), "b2_k_rope"),
    ):
        assert relative_diff(rotated, golden[name]) < floor(golden, name)


@requires_checkpoint(REPO)
def test_stepwise_matches_prefill(model: LFM2MoE, golden: dict[str, mx.array]) -> None:
    """A wrong conv window or KV cache can survive a degenerate greedy; it does not
    survive a full-logits comparison. 25 ids x 4 experts pushes prefill through the
    expert-sorted gather (length*k >= 64)."""
    ids = golden["greedy_ids"]
    assert ids.shape[0] * 4 >= 64
    prefill = model(ids[None])
    cache = model.make_cache()
    steps = [model(ids[None, i : i + 1], cache) for i in range(ids.shape[0])]
    assert relative_diff(mx.concatenate(steps, axis=1), prefill) < 1e-5


@requires_checkpoint(REPO)
def test_cached_greedy_matches_fixture(model: LFM2MoE, golden: dict[str, mx.array]) -> None:
    prompt = [int(i) for i in np.array(golden["input_ids"])]
    expected = [int(i) for i in np.array(golden["greedy_ids"])]
    generated = list(stream_ids(model, prompt, max_tokens=len(expected) - len(prompt)))
    assert prompt + generated == expected


@requires_checkpoint(REPO)
def test_cache_trimmability_follows_layer_types(model: LFM2MoE) -> None:
    """The conv window keeps no history: nothing to rewind to, so no speculation."""
    cache = model.make_cache()
    for kind, entry in zip(model.config.layer_types, cache, strict=True):
        assert entry.is_trimmable == (kind == "full_attention")
    with pytest.raises(NotImplementedError):
        next(c for c in cache if isinstance(c, ConvCache)).trim(0)


@requires_checkpoint(REPO)
def test_cache_offsets_advance_together(model: LFM2MoE, golden: dict[str, mx.array]) -> None:
    """Conv and attention layers alike count tokens seen: prefill plus three steps leaves
    every entry at the same offset, which is what the attention layers rope against."""
    ids = golden["greedy_ids"]
    cache = model.make_cache()
    model(ids[None, :4], cache)
    for i in range(4, 7):
        model(ids[None, i : i + 1], cache)
    assert {entry.offset for entry in cache} == {7}


@requires_checkpoint(REPO)
def test_mutation_of_expert_stack_breaks_parity(
    model: LFM2MoE, golden: dict[str, mx.array]
) -> None:
    mlp = sparse_mlp_of(model, ATTN_LAYER)
    original = mlp.experts.w13.weight
    mlp.experts.w13.weight = original * (1 + 1e-3)
    try:
        logits = model(golden["input_ids"][None])
        assert relative_diff(logits, golden["logits"]) > floor(golden, "logits")
    finally:
        mlp.experts.w13.weight = original


@requires_checkpoint(REPO)
def test_mutation_of_expert_bias_breaks_selection(
    model: LFM2MoE, golden: dict[str, mx.array]
) -> None:
    """The float32 bias only shifts *which* experts win — a routing-only path that a
    weight mutation would not exercise."""
    mlp = sparse_mlp_of(model, ATTN_LAYER)
    original = mlp.expert_bias
    assert isinstance(original, mx.array)
    mlp.expert_bias = original + mx.arange(original.shape[0]).astype(mx.float32)
    try:
        logits = model(golden["input_ids"][None])
        assert relative_diff(logits, golden["logits"]) > floor(golden, "logits")
    finally:
        mlp.expert_bias = original


@requires_checkpoint(REPO)
def test_mutation_of_conv_taps_breaks_parity(
    model: LFM2MoE, golden: dict[str, mx.array]
) -> None:
    """The conv is unrolled by hand: a wrong tap order is the failure mode, and only a
    mutation of the taps themselves catches it."""
    conv = conv_of(model, 0).conv
    original = conv.weight
    conv.weight = original[:, :, ::-1]
    try:
        logits = model(golden["input_ids"][None])
        assert relative_diff(logits, golden["logits"]) > floor(golden, "logits")
    finally:
        conv.weight = original


def step_logits(model: LFM2MoE, ids: mx.array) -> mx.array:
    """Prefill all but the last id, then one T=1 step: the fused paths' only entry."""
    cache = model.make_cache()
    model(ids[None, :-1], cache)
    return model(ids[None, -1:], cache)


@requires_checkpoint(REPO)
def test_fused_step_paths_are_taken(model: LFM2MoE) -> None:
    """The two kernels are on by default at this checkpoint's shapes; every T=1 test
    above therefore exercises them."""
    assert conv_of(model, 0).fused_step_applies()
    assert sparse_mlp_of(model, ATTN_LAYER).fused_step_applies()


@requires_checkpoint(REPO)
@pytest.mark.parametrize("flag", ["CONV_MIX_FUSED", "MOE_GEMV_DENSE_FUSED", "MOE_ROUTE_FUSED"])
def test_fused_step_matches_ops_path(
    model: LFM2MoE, golden: dict[str, mx.array], monkeypatch: pytest.MonkeyPatch, flag: str
) -> None:
    """Each kernel against the op path it replaces, one at a time, on a complete step."""
    fused = step_logits(model, golden["input_ids"])
    with monkeypatch.context() as patch:
        patch.setattr(flags, flag, False)
        ops = step_logits(model, golden["input_ids"])
    assert relative_diff(fused, ops) < 1e-5


@requires_checkpoint(REPO)
def test_mutation_of_conv_window_writeback_breaks_stepwise(
    model: LFM2MoE, golden: dict[str, mx.array], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The seam is the returned window: swap its two rows and the next step convolves
    against the wrong history — invisible in a single step, fatal across three."""
    ids = golden["greedy_ids"]

    def swapped(
        x: mx.array, weights: mx.array, taps: mx.array, window: mx.array
    ) -> tuple[mx.array, mx.array]:
        gated, slid = conv_mix(x, weights, taps, window)
        return gated, slid[::-1]

    with monkeypatch.context() as patch:
        patch.setattr(conv_layer, "conv_mix", swapped)
        cache = model.make_cache()
        steps = [model(ids[None, i : i + 1], cache) for i in range(3)]
    assert relative_diff(mx.concatenate(steps, axis=1), model(ids[None, :3])) > 1e-5


@requires_checkpoint(REPO)
def test_mutation_of_routing_weight_breaks_step(
    model: LFM2MoE, golden: dict[str, mx.array], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The routing weight enters the fused down kernel and nowhere else on this path."""
    reference = step_logits(model, golden["input_ids"])

    def unweighted(
        gate_up: mx.array, weight: mx.array, indices: mx.array, routing: mx.array
    ) -> mx.array:
        return moe_dense_down(gate_up, weight, indices, mx.ones_like(routing))

    with monkeypatch.context() as patch:
        patch.setattr(experts_layer, "moe_dense_down", unweighted)
        assert relative_diff(step_logits(model, golden["input_ids"]), reference) > 1e-5
