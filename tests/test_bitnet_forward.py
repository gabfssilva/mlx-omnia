"""BitNet b1.58-2B fp32 parity against transformers, plus cache and mutation gates.

The fixture is transformers in fp32 (the 2B ternary model fits fp32 in ~10 GB), so
mlx_omnia loads in fp32 and the tolerance is ``3x`` the fixture's measured fp32-vs-fp64
floor per tensor — the trunk is 30 layers, so a single number would be vacuous at one
end and impossible at the other.

Hypothesis flagged, not measured here: transformers' ``AutoBitLinear`` quantizes the
activation with ``torch.round`` (round-half-to-even); mlx_omnia uses ``mx.round``. If
the two round modes disagree on a half-value, an int8 level flips by 1 and the output
drifts beyond the fp32-vs-fp64 floor. ``mx.round`` is expected to match (IEEE
round-to-nearest-even); if the suite fails past the floor, the act-quant round-mode
floor needs to be measured into the fixture as ``noise.actquant`` and the tolerance
widened to it — never to make a pass, only to carry a measured offset.
"""

import math
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from huggingface_hub import snapshot_download

from mlx_omnia import KVCache, stream_ids
from mlx_omnia.engine.models.bitnet import CHECKPOINT, BitNet, BitNetActivations
from mlx_omnia.engine.models.bitnet.layers.mlp import _relu2
from tests.conftest import floor, load_golden, relative_diff

FIXTURE = Path(__file__).parent / "fixtures" / "bitnet_forward.safetensors"
N_LAYER = 30
REPO = "microsoft/bitnet-b1.58-2B-4T"

PATTERNS = ["config.json", "model*.safetensors", "tokenizer.json", "tokenizer_config.json"]


def bitnet_dir() -> Path:
    return Path(snapshot_download(REPO, allow_patterns=PATTERNS))


@pytest.fixture(scope="module")
def golden() -> dict[str, mx.array]:
    return load_golden(FIXTURE)


@pytest.fixture(scope="module")
def model() -> BitNet:
    # The fixture is transformers in fp32; the ternary weights stay uint8, the rest
    # casts to fp32 (the loader skips the packed .weight).
    return CHECKPOINT.load(bitnet_dir(), mx.float32)


@pytest.fixture(scope="module")
def activations(model: BitNet, golden: dict[str, mx.array]) -> BitNetActivations:
    return model.activations(golden["input_ids"][None])


def test_embeddings_exact(activations: BitNetActivations, golden: dict[str, mx.array]) -> None:
    assert relative_diff(activations.embeddings, golden["embeddings"]) == 0


@pytest.mark.parametrize("layer", range(N_LAYER))
def test_block_within_floor(
    activations: BitNetActivations, golden: dict[str, mx.array], layer: int
) -> None:
    assert relative_diff(activations.blocks[layer], golden[f"block_{layer}"]) < floor(
        golden, f"block_{layer}"
    )


def test_norm_within_floor(activations: BitNetActivations, golden: dict[str, mx.array]) -> None:
    assert relative_diff(activations.norm, golden["norm"]) < floor(golden, "norm")


def test_logits_within_floor(activations: BitNetActivations, golden: dict[str, mx.array]) -> None:
    assert relative_diff(activations.logits, golden["logits"]) < floor(golden, "logits")


def test_greedy_predictions_match(
    activations: BitNetActivations, golden: dict[str, mx.array]
) -> None:
    ours = mx.argmax(activations.logits, axis=-1)
    theirs = mx.argmax(golden["logits"], axis=-1)
    assert mx.array_equal(ours, theirs).item()


def test_block0_internals_within_floor(
    model: BitNet, golden: dict[str, mx.array]
) -> None:
    """Naming the culprit: each submodule of block 0 against its own hook boundary."""
    block = model.model.layers[0]
    ids = golden["input_ids"][None]
    x = model.model.embed_tokens(ids)
    normed = block.input_layernorm(x)
    assert relative_diff(normed, golden["b0_ln_1"]) < floor(golden, "b0_ln_1")

    attended = block.self_attn(normed, KVCache())
    assert relative_diff(attended, golden["b0_attn"]) < floor(golden, "b0_attn")

    second = block.post_attention_layernorm(x + attended)
    assert relative_diff(second, golden["b0_ln_2"]) < floor(golden, "b0_ln_2")
    assert relative_diff(block.mlp(second), golden["b0_mlp"]) < floor(golden, "b0_mlp")


def test_block0_sub_norms_within_floor(model: BitNet, golden: dict[str, mx.array]) -> None:
    """The BitNet deltas: an attn_sub_norm before o_proj and an ffn_sub_norm before
    down_proj. Each is pinned at its own boundary, reconstructed from the projections
    so the test does not depend on a module that hides the boundary."""
    block = model.model.layers[0]
    attn = block.self_attn
    ids = golden["input_ids"][None]
    x = model.model.embed_tokens(ids)
    normed = block.input_layernorm(x)
    length = normed.shape[1]

    def heads(proj: mx.array, count: int) -> mx.array:
        return proj.reshape(1, length, count, attn.head_dim).transpose(0, 2, 1, 3)

    q = heads(attn.q_proj(normed), attn.heads)
    k = heads(attn.k_proj(normed), attn.kv_heads)
    v = heads(attn.v_proj(normed), attn.kv_heads)
    qr, kr = attn.rope(q, 0), attn.rope(k, 0)
    keys, values = KVCache().update_and_fetch(kr, v)
    attended = mx.fast.scaled_dot_product_attention(
        qr, keys, values, scale=1 / math.sqrt(attn.head_dim), mask="causal"
    )
    pre_o = attended.transpose(0, 2, 1, 3).reshape(1, length, attn.heads * attn.head_dim)
    assert relative_diff(attn.attn_sub_norm(pre_o), golden["b0_attn_sub_norm"]) < floor(
        golden, "b0_attn_sub_norm"
    )

    second = block.post_attention_layernorm(x + block.self_attn(normed, KVCache()))
    gated = _relu2(block.mlp.gate_proj(second)) * block.mlp.up_proj(second)
    assert relative_diff(block.mlp.ffn_sub_norm(gated), golden["b0_ffn_sub_norm"]) < floor(
        golden, "b0_ffn_sub_norm"
    )


def test_stepwise_matches_prefill(model: BitNet, golden: dict[str, mx.array]) -> None:
    """A wrong cache can survive a degenerate greedy; it does not survive full logits.

    The bar is the fixture's measured ``noise.batching`` (transformers fp32, prefill vs
    step-by-step), not the usual fp32 1e-5: the per-token act-quant round amplifies a
    mere change in matmul row count chaotically along the 30-layer trunk."""
    ids = golden["greedy_ids"]
    prefill = model(ids[None])
    cache = model.make_cache()
    steps = [model(ids[None, i : i + 1], cache) for i in range(ids.shape[0])]
    assert relative_diff(mx.concatenate(steps, axis=1), prefill) < floor(golden, "batching")


def test_cached_greedy_matches_fixture(model: BitNet, golden: dict[str, mx.array]) -> None:
    prompt = [int(i) for i in np.array(golden["input_ids"])]
    expected = [int(i) for i in np.array(golden["greedy_ids"])]
    generated = list(stream_ids(model, prompt, max_tokens=len(expected) - len(prompt)))
    assert prompt + generated == expected


def test_mutation_breaks_parity(model: BitNet, golden: dict[str, mx.array]) -> None:
    """Perturbing one weight_scale must blow past the fixture floor — it is the only
    BitLinear parameter that is cheap to mutate (the ternary weight is packed uint8).
    The target is ``down_proj``: gate/up scales are cancelled by the ``ffn_sub_norm``
    RMSNorm right after the gated product, so mutating them is a near no-op. The factor
    is 3x — the downstream layernorms absorb most of a 1.5x on a single layer (measured
    0.052 against a 0.084 floor; 3x lands at 0.143)."""
    mlp = model.model.layers[13].mlp
    original = mlp.down_proj.weight_scale
    mlp.down_proj.weight_scale = original * mx.array([3.0], dtype=original.dtype)
    try:
        logits = model(golden["input_ids"][None])
        assert relative_diff(logits, golden["logits"]) > floor(golden, "logits")
    finally:
        mlp.down_proj.weight_scale = original
