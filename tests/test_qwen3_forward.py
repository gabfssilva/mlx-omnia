"""Qwen3-0.6B fp32 parity against transformers, plus cache and mutation gates.

Every tolerance is `3 x` the fixture's own measured fp32-vs-fp64 floor for that
tensor — the trunk is 28 layers deep, so a single number would be vacuous at one
end and impossible at the other.
"""

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from conftest import floor, load_golden, relative_diff
from huggingface_hub import snapshot_download

from mlx_omnia import KVCache, stream_ids
from mlx_omnia.engine.core.kernels.qkv_rope.epilogue import rope_epilogue
from mlx_omnia.engine.models.qwen3.dense import CHECKPOINT
from mlx_omnia.engine.models.qwen3.model import Qwen3, Qwen3Activations

FIXTURE = Path(__file__).parent / "fixtures" / "qwen3_forward.safetensors"
N_LAYER = 28

PATTERNS = ["config.json", "model.safetensors"]


def qwen3_dir() -> Path:
    return Path(snapshot_download("Qwen/Qwen3-0.6B", allow_patterns=PATTERNS))


@pytest.fixture(scope="module")
def golden() -> dict[str, mx.array]:
    return load_golden(FIXTURE)


@pytest.fixture(scope="module")
def model() -> Qwen3:
    # The fixture is transformers in fp32; the checkpoint's bf16 upcasts losslessly.
    return CHECKPOINT.load(qwen3_dir(), mx.float32)


@pytest.fixture(scope="module")
def activations(model: Qwen3, golden: dict[str, mx.array]) -> Qwen3Activations:
    return model.activations(golden["input_ids"][None])


def test_embeddings_exact(activations: Qwen3Activations, golden: dict[str, mx.array]) -> None:
    """bf16 upcast to fp32 is lossless and the lookup is a gather: no arithmetic yet."""
    assert relative_diff(activations.embeddings, golden["embeddings"]) == 0


@pytest.mark.parametrize("layer", range(N_LAYER))
def test_block_within_floor(
    activations: Qwen3Activations, golden: dict[str, mx.array], layer: int
) -> None:
    assert relative_diff(activations.blocks[layer], golden[f"block_{layer}"]) < floor(
        golden, f"block_{layer}"
    )


def test_norm_within_floor(activations: Qwen3Activations, golden: dict[str, mx.array]) -> None:
    assert relative_diff(activations.norm, golden["norm"]) < floor(golden, "norm")


def test_logits_within_floor(activations: Qwen3Activations, golden: dict[str, mx.array]) -> None:
    assert relative_diff(activations.logits, golden["logits"]) < floor(golden, "logits")


def test_greedy_predictions_match(
    activations: Qwen3Activations, golden: dict[str, mx.array]
) -> None:
    ours = mx.argmax(activations.logits, axis=-1)
    theirs = mx.argmax(golden["logits"], axis=-1)
    assert mx.array_equal(ours, theirs).item()


def test_block0_internals_within_floor(model: Qwen3, golden: dict[str, mx.array]) -> None:
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


def test_block0_qk_norm_and_rope_within_floor(
    model: Qwen3, golden: dict[str, mx.array]
) -> None:
    """The Qwen3 delta: q/k rms-normed per head *between* projection and rotation.
    transformers hooks q_norm before its transpose, hence [1, L, heads, head_dim]."""
    attention = model.model.layers[0].self_attn
    q, k, _ = attention.split_heads(golden["b0_ln_1"])
    for normed, name in ((q, "b0_q_norm"), (k, "b0_k_norm")):
        reference = golden[name].transpose(0, 2, 1, 3)
        assert relative_diff(normed, reference) < floor(golden, name)
    for rotated, name in ((attention.rope(q, 0), "b0_q_rope"), (attention.rope(k, 0), "b0_k_rope")):
        assert relative_diff(rotated, golden[name]) < floor(golden, name)


def test_stepwise_matches_prefill(model: Qwen3, golden: dict[str, mx.array]) -> None:
    """A wrong cache can survive a degenerate greedy; it does not survive full logits."""
    ids = golden["greedy_ids"]
    prefill = model(ids[None])
    cache = model.make_cache()
    steps = [model(ids[None, i : i + 1], cache) for i in range(ids.shape[0])]
    assert relative_diff(mx.concatenate(steps, axis=1), prefill) < 1e-5


def test_cached_greedy_matches_fixture(model: Qwen3, golden: dict[str, mx.array]) -> None:
    prompt = [int(i) for i in np.array(golden["input_ids"])]
    expected = [int(i) for i in np.array(golden["greedy_ids"])]
    generated = list(stream_ids(model, prompt, max_tokens=len(expected) - len(prompt)))
    assert prompt + generated == expected


def test_mutation_breaks_parity(model: Qwen3, golden: dict[str, mx.array]) -> None:
    """Perturbing one fused gate‖up must blow past the fixture floor."""
    mlp = model.model.layers[13].mlp
    original = mlp.gate_up_proj.weight
    mlp.gate_up_proj.weight = original * (1 + 1e-3)
    try:
        logits = model(golden["input_ids"][None])
        assert relative_diff(logits, golden["logits"]) > floor(golden, "logits")
    finally:
        mlp.gate_up_proj.weight = original


def test_mutation_of_q_norm_breaks_parity(model: Qwen3, golden: dict[str, mx.array]) -> None:
    """q_norm is the Qwen3 delta: without it the trunk still runs, so it needs its own
    mutation or a missing per-head norm would go unnoticed."""
    attention = model.model.layers[0].self_attn
    original = attention.q_norm.weight
    attention.q_norm.weight = mx.ones_like(original)
    try:
        logits = model(golden["input_ids"][None])
        assert relative_diff(logits, golden["logits"]) > floor(golden, "logits")
    finally:
        attention.q_norm.weight = original


def never(head_dim: int) -> bool:
    """The A/B switch: monkeypatching the predicate to False puts the step back on the
    op chain without touching the model file."""
    return False


def stepwise(model: Qwen3, ids: mx.array) -> mx.array:
    cache = model.make_cache()
    return mx.concatenate([model(ids[None, i : i + 1], cache) for i in range(ids.shape[0])], axis=1)


def test_rope_epilogue_matches_op_path(
    model: Qwen3, golden: dict[str, mx.array], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The T=1 step runs the fused kernel by default; the op chain stays reachable by
    falsifying the predicate (that is also the A/B switch for the bench)."""
    ids = golden["greedy_ids"]
    fused = stepwise(model, ids)
    monkeypatch.setattr(
        "mlx_omnia.engine.models.qwen3.layers.attention.rope_epilogue_applies", never
    )
    assert relative_diff(fused, stepwise(model, ids)) < 1e-5


def test_rope_epilogue_mutation_breaks_stepwise(
    model: Qwen3, golden: dict[str, mx.array], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The seam has to hand the kernel each norm on the right side: swapping q_norm and
    k_norm breaks the step-vs-prefill agreement it otherwise holds. (Shifting `offset`
    is not a valid mutation here — the step rotates q and k by the same amount, and rope
    is relative, so a uniform shift is invisible by construction.)"""
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

    monkeypatch.setattr("mlx_omnia.engine.models.qwen3.layers.attention.rope_epilogue", swapped)
    ids = golden["greedy_ids"]
    assert relative_diff(stepwise(model, ids), model(ids[None])) > 1e-5
