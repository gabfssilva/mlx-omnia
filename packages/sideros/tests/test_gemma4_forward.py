"""Gemma 4 E2B fp32 parity against transformers, plus cache and mutation gates.

Every tolerance is `3x` the fixture's own measured fp32-vs-fp64 floor for that tensor.
The 600-token sequence separates the 4:1 sliding/full split (full at idx 4,9,...).

Gemma 4 parity surface:
- scale=1.0 (not query_pre_attn_scalar**-0.5 as in Gemma 3)
- proportional partial-rotary RoPE on full layers (manual, not mx.fast.rope)
- PLE per-layer embeddings + third residual arm
- KV sharing (20 trailing layers read shared full-length KV)
- standard (non-zero-centered) RMSNorm — NO _fold_norm_scales (+1)
- logit softcap tanh(logits/30)*30 in fp32
- dual head_dim (256 sliding / 512 full)
- double-wide MLP on KV-shared layers
- v_norm scale-less (RMSNormNoScale)
- layer_scalar per block
"""

import dataclasses
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from conftest import floor, load_golden, relative_diff
from huggingface_hub import snapshot_download

from sideros import stream_ids
from sideros.models.gemma4 import CHECKPOINT, Gemma4, Gemma4Activations

FIXTURE = Path(__file__).parent / "fixtures" / "gemma4_forward.safetensors"
N_LAYER = 35
SLIDING_LAYER, FULL_LAYER = 0, 4

PATTERNS = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"]


def gemma4_dir() -> Path:
    return Path(snapshot_download("google/gemma-4-e2b-it", allow_patterns=PATTERNS))


@pytest.fixture(scope="module")
def golden() -> dict[str, mx.array]:
    return load_golden(FIXTURE)


@pytest.fixture(scope="module")
def model() -> Gemma4:
    return CHECKPOINT.load(gemma4_dir(), mx.float32)


@pytest.fixture(scope="module")
def activations(model: Gemma4, golden: dict[str, mx.array]) -> Gemma4Activations:
    return model.activations(golden["input_ids"][None])


@pytest.fixture(scope="module")
def long_activations(model: Gemma4, golden: dict[str, mx.array]) -> Gemma4Activations:
    return model.activations(golden["long_input_ids"][None])


def test_embeddings_within_floor(
    activations: Gemma4Activations, golden: dict[str, mx.array]
) -> None:
    """sqrt(hidden_size) — 1536 is not a perfect square, so the bf16 cast matters."""
    assert relative_diff(activations.embeddings, golden["embeddings"]) < floor(
        golden, "embeddings"
    )


@pytest.mark.parametrize("layer", range(N_LAYER))
def test_block_within_floor(
    activations: Gemma4Activations, golden: dict[str, mx.array], layer: int
) -> None:
    assert relative_diff(activations.blocks[layer], golden[f"block_{layer}"]) < floor(
        golden, f"block_{layer}"
    )


def test_norm_within_floor(activations: Gemma4Activations, golden: dict[str, mx.array]) -> None:
    assert relative_diff(activations.norm, golden["norm"]) < floor(golden, "norm")


def test_logits_within_floor(
    activations: Gemma4Activations, golden: dict[str, mx.array]
) -> None:
    assert relative_diff(activations.logits, golden["logits"]) < floor(golden, "logits")


def test_greedy_predictions_match(
    activations: Gemma4Activations, golden: dict[str, mx.array]
) -> None:
    ours = mx.argmax(activations.logits, axis=-1)
    theirs = mx.argmax(golden["logits"], axis=-1)
    assert mx.array_equal(ours, theirs).item()


@pytest.mark.parametrize("layer", (SLIDING_LAYER, FULL_LAYER))
def test_long_sequence_block_within_floor(
    long_activations: Gemma4Activations, golden: dict[str, mx.array], layer: int
) -> None:
    """Past 512 keys the two layer types diverge: window mask and proportional rope."""
    assert relative_diff(
        long_activations.blocks[layer], golden[f"long_block_{layer}"]
    ) < floor(golden, f"long_block_{layer}")


def test_long_sequence_last_logits_within_floor(
    long_activations: Gemma4Activations, golden: dict[str, mx.array]
) -> None:
    assert relative_diff(
        long_activations.logits[:, -1:, :], golden["long_logits_last"]
    ) < floor(golden, "long_logits_last")


@pytest.mark.parametrize("index", (SLIDING_LAYER, FULL_LAYER))
def test_block_internals_within_floor(
    model: Gemma4, golden: dict[str, mx.array], index: int
) -> None:
    block = model.model.layers[index]
    x = golden["embeddings"] if index == 0 else golden[f"block_{index - 1}"]

    def check(ours: mx.array, name: str) -> None:
        assert relative_diff(ours, golden[f"b{index}_{name}"]) < floor(
            golden, f"b{index}_{name}"
        )

    check(block.input_layernorm(x), "input_layernorm")
    if not block.self_attn.kv_shared:
        from sideros.core.cache import KVCache

        check(
            block.self_attn(golden[f"b{index}_input_layernorm"], KVCache()),
            "self_attn",
        )
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


def test_stepwise_matches_prefill(model: Gemma4, golden: dict[str, mx.array]) -> None:
    ids = golden["greedy_ids"]
    prefill = model(ids[None])
    cache = model.make_cache()
    steps = [model(ids[None, i : i + 1], cache) for i in range(ids.shape[0])]
    assert relative_diff(mx.concatenate(steps, axis=1), prefill) < 1e-5


def test_stepwise_matches_prefill_past_the_window(
    model: Gemma4, golden: dict[str, mx.array]
) -> None:
    ids = golden["long_input_ids"][None]
    prefill = model(ids)
    cache = model.make_cache()
    model(ids[:, :-1], cache)
    step = model(ids[:, -1:], cache)
    assert relative_diff(step, prefill[:, -1:, :]) < 1e-5


def test_cached_greedy_matches_fixture(
    model: Gemma4, golden: dict[str, mx.array]
) -> None:
    prompt = [int(i) for i in np.array(golden["input_ids"])]
    expected = [int(i) for i in np.array(golden["greedy_ids"])]
    generated = list(stream_ids(model, prompt, max_tokens=len(expected) - len(prompt)))
    assert prompt + generated == expected


def test_scale_is_one(model: Gemma4) -> None:
    """Gemma 4 uses scale=1.0, NOT query_pre_attn_scalar**-0.5 (Gemma 3)."""
    for layer in model.model.layers:
        assert layer.self_attn.scale == 1.0


# --- Mutation tests ---


def test_mutation_breaks_parity(model: Gemma4, golden: dict[str, mx.array]) -> None:
    mlp = model.model.layers[0].mlp
    original = mlp.gate_proj.weight
    mlp.gate_proj.weight = original * (1 + 1e-3)
    try:
        logits = model(golden["input_ids"][None])
        assert relative_diff(logits, golden["logits"]) > floor(golden, "logits")
    finally:
        mlp.gate_proj.weight = original


def test_mutation_of_attention_scale_breaks_parity(
    model: Gemma4, golden: dict[str, mx.array]
) -> None:
    """scale=1.0 is intentional (Q and K are RMSNormed). Perturbing it must fail."""
    attention = model.model.layers[FULL_LAYER].self_attn
    original = attention.scale
    attention.scale = 1.0 / mx.sqrt(mx.array(attention.head_dim, mx.float32)).item()
    try:
        logits = model(golden["input_ids"][None])
        assert relative_diff(logits, golden["logits"]) > floor(golden, "logits")
    finally:
        attention.scale = original


def test_mutation_of_sliding_window_breaks_long_parity(
    model: Gemma4, golden: dict[str, mx.array]
) -> None:
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


def test_mutation_of_norm_scale_breaks_parity(
    model: Gemma4, golden: dict[str, mx.array]
) -> None:
    """Gemma 4 norms are standard (no +1 fold). Adding +1 (simulating the Gemma 3
    fold accidentally applied) must break parity."""
    norm = model.model.layers[0].input_layernorm
    original = norm.weight
    norm.weight = original + 1
    try:
        logits = model(golden["input_ids"][None])
        assert relative_diff(logits, golden["logits"]) > floor(golden, "logits")
    finally:
        norm.weight = original


def test_mutation_of_softcap_breaks_parity(
    model: Gemma4, golden: dict[str, mx.array]
) -> None:
    """The logit softcap tanh(logits/30)*30 must be applied (in fp32)."""
    original = model.config.final_logit_softcapping
    model.config = dataclasses.replace(model.config, final_logit_softcapping=None)
    try:
        logits = model(golden["input_ids"][None])
        assert relative_diff(logits, golden["logits"]) > floor(golden, "logits")
    finally:
        model.config = dataclasses.replace(
            model.config, final_logit_softcapping=original
        )


def test_mutation_of_layer_scalar_breaks_parity(
    model: Gemma4, golden: dict[str, mx.array]
) -> None:
    """layer_scalar is a no-op at init (1.0) but the multiply must run."""
    block = model.model.layers[0]
    original = block.layer_scalar
    block.layer_scalar = original * 2.0
    try:
        logits = model(golden["input_ids"][None])
        assert relative_diff(logits, golden["logits"]) > floor(golden, "logits")
    finally:
        block.layer_scalar = original
