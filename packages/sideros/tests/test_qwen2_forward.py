"""Qwen2.5-0.5B fp32 parity against transformers, plus cache and mutation gates.

Every tolerance is `3 x` the fixture's own measured fp32-vs-fp64 floor for that
tensor. The Qwen2 delta against qwen3 is the bias on q/k/v, so the projections get
their own pinned tensors and their own mutation.
"""

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from conftest import assert_greedy_modulo_ties, floor, load_golden, relative_diff
from huggingface_hub import snapshot_download

from sideros import KVCache, stream_ids
from sideros.models.qwen2 import CHECKPOINT, Qwen2, Qwen2Activations

FIXTURE = Path(__file__).parent / "fixtures" / "qwen2_forward.safetensors"
Q4_FIXTURE = Path(__file__).parent / "fixtures" / "qwen2_q4_mlxlm.safetensors"
N_LAYER = 24

PATTERNS = ["config.json", "model.safetensors"]


def qwen2_dir() -> Path:
    return Path(snapshot_download("Qwen/Qwen2.5-0.5B", allow_patterns=PATTERNS))


def quantized_dir() -> Path:
    return Path(
        snapshot_download("mlx-community/Qwen2.5-0.5B-Instruct-4bit", allow_patterns=PATTERNS)
    )


@pytest.fixture(scope="module")
def golden() -> dict[str, mx.array]:
    return load_golden(FIXTURE)


@pytest.fixture(scope="module")
def model() -> Qwen2:
    # The fixture is transformers in fp32; the checkpoint's bf16 upcasts losslessly.
    return CHECKPOINT.load(qwen2_dir(), mx.float32)


@pytest.fixture(scope="module")
def activations(model: Qwen2, golden: dict[str, mx.array]) -> Qwen2Activations:
    return model.activations(golden["input_ids"][None])


def test_embeddings_exact(activations: Qwen2Activations, golden: dict[str, mx.array]) -> None:
    """bf16 upcast to fp32 is lossless and the lookup is a gather: no arithmetic yet."""
    assert relative_diff(activations.embeddings, golden["embeddings"]) == 0


@pytest.mark.parametrize("layer", range(N_LAYER))
def test_block_within_floor(
    activations: Qwen2Activations, golden: dict[str, mx.array], layer: int
) -> None:
    assert relative_diff(activations.blocks[layer], golden[f"block_{layer}"]) < floor(
        golden, f"block_{layer}"
    )


def test_norm_within_floor(activations: Qwen2Activations, golden: dict[str, mx.array]) -> None:
    assert relative_diff(activations.norm, golden["norm"]) < floor(golden, "norm")


def test_logits_within_floor(activations: Qwen2Activations, golden: dict[str, mx.array]) -> None:
    assert relative_diff(activations.logits, golden["logits"]) < floor(golden, "logits")


def test_greedy_predictions_match(
    activations: Qwen2Activations, golden: dict[str, mx.array]
) -> None:
    ours = mx.argmax(activations.logits, axis=-1)
    theirs = mx.argmax(golden["logits"], axis=-1)
    assert mx.array_equal(ours, theirs).item()


def test_block0_internals_within_floor(model: Qwen2, golden: dict[str, mx.array]) -> None:
    """Naming the culprit: each submodule of block 0 against its own hook boundary."""
    block = model.model.layers[0]
    x = model.model.embed_tokens(golden["input_ids"][None])
    normed = block.input_layernorm(x)
    assert relative_diff(normed, golden["b0_ln_1"]) < floor(golden, "b0_ln_1")

    attended = block.self_attn(normed, KVCache())
    assert relative_diff(attended, golden["b0_attn"]) < floor(golden, "b0_attn")

    second = block.post_attention_layernorm(x + attended)
    assert relative_diff(second, golden["b0_ln_2"]) < floor(golden, "b0_ln_2")
    assert relative_diff(block.mlp(second), golden["b0_mlp"]) < floor(golden, "b0_mlp")


def test_block0_biased_projections_and_rope_within_floor(
    model: Qwen2, golden: dict[str, mx.array]
) -> None:
    """The fused qkv must reproduce q/k/v *including* the bias slice each one owns —
    transformers hooks the projections flat, so the golden is [1, L, count*head_dim]."""
    attention = model.model.layers[0].self_attn
    q, k, v = attention.split_heads(golden["b0_ln_1"])
    length = q.shape[2]
    for part, name in ((q, "b0_q_proj"), (k, "b0_k_proj"), (v, "b0_v_proj")):
        flat = part.transpose(0, 2, 1, 3).reshape(1, length, -1)
        assert relative_diff(flat, golden[name]) < floor(golden, name)
    for rotated, name in ((attention.rope(q, 0), "b0_q_rope"), (attention.rope(k, 0), "b0_k_rope")):
        assert relative_diff(rotated, golden[name]) < floor(golden, name)


def test_stepwise_matches_prefill(model: Qwen2, golden: dict[str, mx.array]) -> None:
    """A wrong cache can survive a degenerate greedy; it does not survive full logits."""
    ids = golden["greedy_ids"]
    prefill = model(ids[None])
    cache = model.make_cache()
    steps = [model(ids[None, i : i + 1], cache) for i in range(ids.shape[0])]
    assert relative_diff(mx.concatenate(steps, axis=1), prefill) < 1e-5


def test_cached_greedy_matches_fixture(model: Qwen2, golden: dict[str, mx.array]) -> None:
    prompt = [int(i) for i in np.array(golden["input_ids"])]
    expected = [int(i) for i in np.array(golden["greedy_ids"])]
    generated = list(stream_ids(model, prompt, max_tokens=len(expected) - len(prompt)))
    assert prompt + generated == expected


def test_mutation_breaks_parity(model: Qwen2, golden: dict[str, mx.array]) -> None:
    """Perturbing one fused gate‖up must blow past the fixture floor."""
    mlp = model.model.layers[11].mlp
    original = mlp.gate_up_proj.weight
    mlp.gate_up_proj.weight = original * (1 + 1e-3)
    try:
        logits = model(golden["input_ids"][None])
        assert relative_diff(logits, golden["logits"]) > floor(golden, "logits")
    finally:
        mlp.gate_up_proj.weight = original


def test_mutation_of_qkv_bias_breaks_parity(model: Qwen2, golden: dict[str, mx.array]) -> None:
    """The Qwen2 delta: zeroing the fused qkv bias must fail. Without this the loader
    could drop the bias (or misalign it against the concatenated rows) unnoticed."""
    attention = model.model.layers[0].self_attn
    original = attention.qkv_proj.bias
    attention.qkv_proj.bias = mx.zeros_like(original)
    try:
        logits = model(golden["input_ids"][None])
        assert relative_diff(logits, golden["logits"]) > floor(golden, "logits")
    finally:
        attention.qkv_proj.bias = original


@pytest.fixture(scope="module")
def q4_golden() -> dict[str, mx.array]:
    return load_golden(Q4_FIXTURE)


@pytest.fixture(scope="module")
def q4_model() -> Qwen2:
    return CHECKPOINT.load(quantized_dir(), None)


def test_q4_rejects_dtype_cast() -> None:
    """`dtype=` over packed uint32 weights survives the shape check and silently ruins
    the numbers; the loader has to refuse instead."""
    with pytest.raises(ValueError):
        CHECKPOINT.load(quantized_dir(), mx.float32)


def test_q4_logits_match_mlxlm(q4_model: Qwen2, q4_golden: dict[str, mx.array]) -> None:
    """Pre-quantized path (`quantization` in config.json, tied head): only the leaves
    the dict packed get quantized. Floor is mlx-lm's own measured rounding cost."""
    logits = q4_model(q4_golden["input_ids"][None])
    assert relative_diff(logits, q4_golden["logits"]) < q4_golden["noise.logits"].item()


def test_q4_greedy_matches_mlxlm(q4_model: Qwen2, q4_golden: dict[str, mx.array]) -> None:
    """The reference is quantized mlx-lm, so the ids compare modulo ties."""
    prompt = [int(i) for i in np.array(q4_golden["input_ids"])]
    expected = [int(i) for i in np.array(q4_golden["greedy_ids"])]
    generated = list(stream_ids(q4_model, prompt, max_tokens=len(expected) - len(prompt)))
    assert_greedy_modulo_ties(
        prompt + generated,
        expected,
        lambda: q4_model(q4_golden["greedy_ids"][None])[0],
        q4_golden["noise.logits"].item(),
    )


def test_q4_stepwise_matches_prefill(q4_model: Qwen2, q4_golden: dict[str, mx.array]) -> None:
    """3x mlx-lm's own prefill-vs-stepwise noise over this checkpoint; 1e-5 is an fp32
    tolerance and would fail here for the wrong reason."""
    ids = q4_golden["greedy_ids"]
    prefill = q4_model(ids[None])
    cache = q4_model.make_cache()
    steps = [q4_model(ids[None, i : i + 1], cache) for i in range(ids.shape[0])]
    gap = relative_diff(mx.concatenate(steps, axis=1), prefill)
    assert gap < 3 * q4_golden["noise.batching"].item()


def test_q4_mutation_breaks_parity(q4_model: Qwen2, q4_golden: dict[str, mx.array]) -> None:
    """The quantized qkv fusion must line its bias up with the packed rows: perturbing
    the fused scales must blow past the floor."""
    attention = q4_model.model.layers[9].self_attn
    original = attention.qkv_proj.scales
    assert isinstance(original, mx.array)
    attention.qkv_proj.scales = original * 1.5
    try:
        logits = q4_model(q4_golden["input_ids"][None])
        assert relative_diff(logits, q4_golden["logits"]) > q4_golden["noise.logits"].item()
    finally:
        attention.qkv_proj.scales = original
