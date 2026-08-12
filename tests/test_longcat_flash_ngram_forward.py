# pyright: basic
"""Longcat Flash Lite (ngram) against the reference implementation over the same checkpoint.

No transformers ground truth exists at this size: the reference implementation is the
golden, bounded by the floors the fixture measured. The floors are per block (``noise.block_i``)
because the residual grows down a 28-sublayer trunk.

The MLA latent cache, the softmax_bias_topk router, the identity experts and the
n-gram embedding are all exercised here:
- the stepwise-vs-prefill test catches a bad latent cache or an n-gram context
  that does not round-trip the boundary;
- the MoE internals test names the culprit (router, expert sum, identity sum);
- the mutation tests break each new numerical path and confirm the failure.
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

from mlx_omnia import stream_ids
from mlx_omnia.engine.models.longcat_flash_ngram import (
    CHECKPOINT,
    LongcatFlashNgram,
)

FIXTURE = Path(__file__).parent / "fixtures" / "longcat_flash_ngram_mlxlm.safetensors"
REPO = "meituan-longcat/LongCat-Flash-Lite"

QUANTUM = 2.0**-7


@pytest.fixture(scope="module")
def golden() -> dict[str, mx.array]:
    return load_golden(FIXTURE)


@pytest.fixture(scope="module")
def model() -> LongcatFlashNgram:
    return CHECKPOINT.load(checkpoint_dir(REPO), None)


@pytest.fixture(scope="module")
def blocks(model: LongcatFlashNgram, golden: dict[str, mx.array]) -> list[mx.array]:
    return model.activations(golden["input_ids"][None]).blocks


def bound(golden: dict[str, mx.array]) -> float:
    return golden["noise.batching"].item() + QUANTUM


@requires_checkpoint(REPO)
def test_shapes_after_fusion(model: LongcatFlashNgram) -> None:
    config = model.config
    assert config.num_layers == 14
    assert config.num_sublayers == 28
    assert (config.n_routed_experts, config.zero_expert_num) == (256, 128)
    assert config.total_experts == 384
    assert config.moe_topk == 12
    assert config.qk_head_dim == 192
    mlp = model.model.layers[0].mlp
    assert mlp.switch_mlp.gate_up_proj.weight.shape[0] == 256
    assert mlp.switch_mlp.down_proj.weight.shape[0] == 256
    assert mlp.router.classifier.weight.shape[0] == 384
    attn = model.model.layers[0].self_attn[0]
    assert attn.embed_q.weight.shape == (32, 512, 128)
    assert attn.unembed_out.weight.shape == (32, 128, 512)


@requires_checkpoint(REPO)
@pytest.mark.parametrize("layer", range(28))
def test_blocks_within_floor(
    blocks: list[mx.array], golden: dict[str, mx.array], layer: int
) -> None:
    floor = golden[f"noise.block_{layer}"].item() + QUANTUM
    assert relative_diff(blocks[layer], golden[f"block_{layer}"]) < floor


@requires_checkpoint(REPO)
def test_moe_internals_within_floor(
    model: LongcatFlashNgram, golden: dict[str, mx.array]
) -> None:
    """A failure here names the culprit: the router, the expert sum, or the
    identity pass-through."""
    from mlx_omnia.engine.core.mxcompat import softmax
    from mlx_omnia.engine.models.longcat_flash_ngram.layers.cache import MLACache, NgramCache

    ngram_cache = NgramCache(model.config.emb_neighbor_num)
    embeddings = model.model.ngram_embeddings(golden["input_ids"][None], ngram_cache)
    block = model.model.layers[0]
    mla_cache = MLACache()
    mixed = embeddings + block.self_attn[0](
        block.input_layernorm[0](embeddings), None, mla_cache
    )
    x = block.post_attention_layernorm[0](mixed)

    mlp = block.mlp
    router = mlp.router
    logits = router.classifier(x)
    scores = softmax(logits, axis=-1, precise=True)
    corrected = scores + router.e_score_correction_bias
    k = router.k
    indices = mx.argpartition(corrected, kth=-k, axis=-1)[..., -k:]
    weights = mx.take_along_axis(scores, indices, axis=-1) * router.scale
    identity_mask = indices >= mlp.n_routed
    clamped = mx.where(identity_mask, 0, indices)
    regular_weights = mx.where(identity_mask, 0.0, weights)
    routed = mlp.switch_mlp(x, clamped, sorted_indices=False)
    weighted = routed * mx.expand_dims(regular_weights, -1)
    expert_sum = mx.sum(weighted, axis=-2)
    identity_sum = mx.sum(
        mx.where(identity_mask, weights, 0.0), axis=-1, keepdims=True
    )

    def within(ours: mx.array, name: str) -> bool:
        return relative_diff(ours, golden[name]) < golden[f"noise.{name}"].item() + QUANTUM

    assert within(scores, "b0_moe_scores")
    assert mx.array_equal(
        mx.sort(indices.astype(mx.int32), axis=-1),
        mx.sort(golden["b0_moe_indices"], axis=-1),
    )
    assert within(weights, "b0_moe_weights")
    assert within(expert_sum, "b0_moe_expert_sum")
    assert within(identity_sum, "b0_moe_identity_sum")
    assert within(expert_sum + x * identity_sum, "b0_moe")


@requires_checkpoint(REPO)
def test_logits_match_mlxlm(model: LongcatFlashNgram, golden: dict[str, mx.array]) -> None:
    logits = model(golden["input_ids"][None])
    assert relative_diff(logits, golden["logits"]) < bound(golden)


@requires_checkpoint(REPO)
def test_teacher_forced_argmax_matches_mlxlm(
    model: LongcatFlashNgram, golden: dict[str, mx.array]
) -> None:
    """Argmax between two bf16 implementations compares modulo ties: a flip only
    counts when the golden separates the two candidates by more than the floor."""
    ids = golden["greedy_ids"]
    prompt = golden["input_ids"].shape[0]
    ours = model(ids[None]).astype(mx.float32)[0]
    picked = mx.argmax(ours, axis=-1)[prompt - 1 :]
    expected = ids[prompt:]
    tolerance = golden["noise.logits"].item() * float(mx.abs(ours).max().item())
    for position in range(expected.shape[0]):
        got, wanted = int(picked[position].item()), int(expected[position].item())
        if got == wanted:
            continue
        gap = float((ours[position, got] - ours[position, wanted]).item())
        assert gap < tolerance, f"position {position}: {got} over {wanted} by {gap}"


@requires_checkpoint(REPO)
def test_stepwise_matches_prefill(
    model: LongcatFlashNgram, golden: dict[str, mx.array]
) -> None:
    """The one test the MLA latent cache and the n-gram context have to pass: 28
    latent caches plus the n-gram window (3 ids) round-tripping the prefill→decode
    boundary, compared over full logits. A stale latent cache, a wrong pe_scores
    mask, or a n-gram context that does not reset at EOS all show up here."""
    ids = golden["greedy_ids"]
    prefill = model(ids[None])
    cache = model.make_cache()
    steps = [model(ids[None, i : i + 1], cache) for i in range(ids.shape[0])]
    floor = 3 * golden["noise.batching"].item()
    assert relative_diff(mx.concatenate(steps, axis=1), prefill) < floor


@requires_checkpoint(REPO)
def test_greedy_matches_mlxlm(
    model: LongcatFlashNgram, golden: dict[str, mx.array]
) -> None:
    """What the teacher-forced test above cannot see: the sampler and the cache
    driving the run themselves. Same rule on the ids — the first divergence has
    to be a tie."""
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
def test_mutation_of_expert_scales_breaks_parity(
    model: LongcatFlashNgram, golden: dict[str, mx.array]
) -> None:
    down = model.model.layers[0].mlp.switch_mlp.down_proj
    original = down.scales
    assert isinstance(original, mx.array)
    down.scales = original * 1.5
    try:
        logits = model(golden["input_ids"][None])
        assert relative_diff(logits, golden["logits"]) > bound(golden)
    finally:
        down.scales = original


@requires_checkpoint(REPO)
def test_mutation_of_identity_expert_breaks_parity(
    model: LongcatFlashNgram, golden: dict[str, mx.array]
) -> None:
    """Dropping the identity pass-through (``+ input x identity_weights_sum``)
    diverges on tokens routed to identity experts — about 1/3 of the top-12
    picks with 128 identity experts out of 384."""
    ids = golden["input_ids"]

    router = model.model.layers[0].mlp.router
    original_bias = router.e_score_correction_bias
    assert isinstance(original_bias, mx.array)
    # Zeroing the bias for identity experts keeps their softmax score but removes
    # the bias boost that pulls them in — the selection changes and so does the
    # identity contribution.
    n_routed = model.config.n_routed_experts
    router.e_score_correction_bias = mx.where(
        mx.arange(original_bias.shape[0]) >= n_routed, 0.0, original_bias
    )
    try:
        assert relative_diff(model(ids[None]), golden["logits"]) > bound(golden)
    finally:
        router.e_score_correction_bias = original_bias
