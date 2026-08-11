"""Gemma 3 270M fp32 parity against transformers, plus cache and mutation gates.

Every tolerance is `3 x` the fixture's own measured fp32-vs-fp64 floor for that tensor.
The 600-token sequence is what separates the sliding layers from the full ones: below
the 512-key window both masks agree, so a short prompt cannot see a wrong window or a
swapped rope base.
"""

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from conftest import floor, load_golden, relative_diff
from huggingface_hub import snapshot_download

from mlx_omnia import KVCache, stream_ids
from mlx_omnia.models.gemma3 import CHECKPOINT, Gemma3, Gemma3Activations

FIXTURE = Path(__file__).parent / "fixtures" / "gemma3_forward.safetensors"
N_LAYER = 18
SLIDING_LAYER, FULL_LAYER = 0, 5

PATTERNS = ["config.json", "model.safetensors"]


def gemma3_dir() -> Path:
    return Path(snapshot_download("google/gemma-3-270m", allow_patterns=PATTERNS))


@pytest.fixture(scope="module")
def golden() -> dict[str, mx.array]:
    return load_golden(FIXTURE)


@pytest.fixture(scope="module")
def model() -> Gemma3:
    # The fixture is transformers in fp32; the checkpoint's bf16 upcasts losslessly.
    return CHECKPOINT.load(gemma3_dir(), mx.float32)


@pytest.fixture(scope="module")
def activations(model: Gemma3, golden: dict[str, mx.array]) -> Gemma3Activations:
    return model.activations(golden["input_ids"][None])


@pytest.fixture(scope="module")
def long_activations(model: Gemma3, golden: dict[str, mx.array]) -> Gemma3Activations:
    return model.activations(golden["long_input_ids"][None])


def test_embeddings_within_floor(
    activations: Gemma3Activations, golden: dict[str, mx.array]
) -> None:
    """Not exact like a plain gather: the table is scaled by sqrt(hidden), a scalar
    transformers keeps in fp32 and casts to the weight dtype (25.298221 here)."""
    assert relative_diff(activations.embeddings, golden["embeddings"]) < floor(
        golden, "embeddings"
    )


@pytest.mark.parametrize("layer", range(N_LAYER))
def test_block_within_floor(
    activations: Gemma3Activations, golden: dict[str, mx.array], layer: int
) -> None:
    assert relative_diff(activations.blocks[layer], golden[f"block_{layer}"]) < floor(
        golden, f"block_{layer}"
    )


def test_norm_within_floor(activations: Gemma3Activations, golden: dict[str, mx.array]) -> None:
    assert relative_diff(activations.norm, golden["norm"]) < floor(golden, "norm")


def test_logits_within_floor(activations: Gemma3Activations, golden: dict[str, mx.array]) -> None:
    assert relative_diff(activations.logits, golden["logits"]) < floor(golden, "logits")


def test_greedy_predictions_match(
    activations: Gemma3Activations, golden: dict[str, mx.array]
) -> None:
    ours = mx.argmax(activations.logits, axis=-1)
    theirs = mx.argmax(golden["logits"], axis=-1)
    assert mx.array_equal(ours, theirs).item()


@pytest.mark.parametrize("layer", (SLIDING_LAYER, FULL_LAYER))
def test_long_sequence_block_within_floor(
    long_activations: Gemma3Activations, golden: dict[str, mx.array], layer: int
) -> None:
    """Past 512 keys the two layer types diverge: window mask and local rope base."""
    assert relative_diff(long_activations.blocks[layer], golden[f"long_block_{layer}"]) < floor(
        golden, f"long_block_{layer}"
    )


def test_long_sequence_last_logits_within_floor(
    long_activations: Gemma3Activations, golden: dict[str, mx.array]
) -> None:
    assert relative_diff(long_activations.logits[:, -1:, :], golden["long_logits_last"]) < floor(
        golden, "long_logits_last"
    )


@pytest.mark.parametrize("index", (SLIDING_LAYER, FULL_LAYER))
def test_block_internals_within_floor(
    model: Gemma3, golden: dict[str, mx.array], index: int
) -> None:
    """Naming the culprit: each of the four sandwich norms and both arms of one block of
    each layer type, against its own hook boundary. Every submodule is fed the *golden*
    input of its boundary, never our own chain: each floor is that tensor's fp32-vs-fp64
    noise, which cannot absorb the drift of the submodules before it."""
    block = model.model.layers[index]
    x = golden["embeddings"] if index == 0 else golden[f"block_{index - 1}"]

    def check(ours: mx.array, name: str) -> None:
        assert relative_diff(ours, golden[f"b{index}_{name}"]) < floor(golden, f"b{index}_{name}")

    check(block.input_layernorm(x), "input_layernorm")
    check(block.self_attn(golden[f"b{index}_input_layernorm"], KVCache()), "self_attn")
    check(
        block.post_attention_layernorm(golden[f"b{index}_self_attn"]),
        "post_attention_layernorm",
    )
    residual = x + golden[f"b{index}_post_attention_layernorm"]
    check(block.pre_feedforward_layernorm(residual), "pre_feedforward_layernorm")
    check(block.mlp(golden[f"b{index}_pre_feedforward_layernorm"]), "mlp")
    check(
        block.post_feedforward_layernorm(golden[f"b{index}_mlp"]),
        "post_feedforward_layernorm",
    )


@pytest.mark.parametrize("index", (SLIDING_LAYER, FULL_LAYER))
def test_qk_norm_within_floor(model: Gemma3, golden: dict[str, mx.array], index: int) -> None:
    """q/k rms-normed per head *between* projection and rotation. transformers hooks
    q_norm after its transpose, hence [1, heads, L, head_dim] already."""
    attention = model.model.layers[index].self_attn
    q, k, _ = attention.split_heads(golden[f"b{index}_input_layernorm"])
    for normed, name in ((q, f"b{index}_q_norm"), (k, f"b{index}_k_norm")):
        assert relative_diff(normed, golden[name]) < floor(golden, name)


def test_stepwise_matches_prefill(model: Gemma3, golden: dict[str, mx.array]) -> None:
    """A wrong cache can survive a degenerate greedy; it does not survive full logits."""
    ids = golden["greedy_ids"]
    prefill = model(ids[None])
    cache = model.make_cache()
    steps = [model(ids[None, i : i + 1], cache) for i in range(ids.shape[0])]
    assert relative_diff(mx.concatenate(steps, axis=1), prefill) < 1e-5


def test_stepwise_matches_prefill_past_the_window(
    model: Gemma3, golden: dict[str, mx.array]
) -> None:
    """T=1 drops the mask only while the cache fits the window; past 512 keys the
    sliding layers must still mask, and only the full-logits comparison sees it."""
    ids = golden["long_input_ids"][None]
    prefill = model(ids)
    cache = model.make_cache()
    model(ids[:, :-1], cache)
    step = model(ids[:, -1:], cache)
    assert relative_diff(step, prefill[:, -1:, :]) < 1e-5


def test_cached_greedy_matches_fixture(model: Gemma3, golden: dict[str, mx.array]) -> None:
    prompt = [int(i) for i in np.array(golden["input_ids"])]
    expected = [int(i) for i in np.array(golden["greedy_ids"])]
    generated = list(stream_ids(model, prompt, max_tokens=len(expected) - len(prompt)))
    assert prompt + generated == expected


def test_mutation_breaks_parity(model: Gemma3, golden: dict[str, mx.array]) -> None:
    """Perturbing one fused gate‖up must blow past the fixture floor."""
    mlp = model.model.layers[9].mlp
    original = mlp.gate_up_proj.weight
    mlp.gate_up_proj.weight = original * (1 + 1e-3)
    try:
        logits = model(golden["input_ids"][None])
        assert relative_diff(logits, golden["logits"]) > floor(golden, "logits")
    finally:
        mlp.gate_up_proj.weight = original


def test_mutation_of_folded_norm_scale_breaks_parity(
    model: Gemma3, golden: dict[str, mx.array]
) -> None:
    """The norm scale is `1 + w`, folded at load. Dropping the fold on one norm (weight
    back to the checkpoint's zero-centred value) must fail."""
    norm = model.model.layers[0].post_feedforward_layernorm
    original = norm.weight
    norm.weight = original - 1
    try:
        logits = model(golden["input_ids"][None])
        assert relative_diff(logits, golden["logits"]) > floor(golden, "logits")
    finally:
        norm.weight = original


def test_mutation_of_sliding_window_breaks_long_parity(
    model: Gemma3, golden: dict[str, mx.array]
) -> None:
    """Without the window a sliding layer is just a full one — invisible on a short
    prompt, which is why the 600-token sequence exists."""
    attention = model.model.layers[SLIDING_LAYER].self_attn
    original = attention.window
    attention.window = None
    try:
        blocks = model.activations(golden["long_input_ids"][None]).blocks
        assert relative_diff(blocks[SLIDING_LAYER], golden["long_block_0"]) > floor(
            golden, "long_block_0"
        )
    finally:
        attention.window = original


def test_mutation_of_local_rope_base_breaks_long_parity(
    model: Gemma3, golden: dict[str, mx.array]
) -> None:
    """A sliding layer rotates with rope_local_base_freq, not rope_theta."""
    attention = model.model.layers[SLIDING_LAYER].self_attn
    original = attention.rope_base
    attention.rope_base = model.config.rope_theta
    try:
        blocks = model.activations(golden["long_input_ids"][None]).blocks
        assert relative_diff(blocks[SLIDING_LAYER], golden["long_block_0"]) > floor(
            golden, "long_block_0"
        )
    finally:
        attention.rope_base = original


def test_mutation_of_attention_scale_breaks_parity(
    model: Gemma3, golden: dict[str, mx.array]
) -> None:
    """query_pre_attn_scalar^-0.5 coincides with head_dim^-0.5 on this checkpoint, so
    the scale is pinned by perturbing it rather than by swapping in head_dim."""
    attention = model.model.layers[FULL_LAYER].self_attn
    original = attention.scale
    attention.scale = original * 1.05
    try:
        logits = model(golden["input_ids"][None])
        assert relative_diff(logits, golden["logits"]) > floor(golden, "logits")
    finally:
        attention.scale = original
