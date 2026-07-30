# pyright: basic
"""Falcon-H1 7B fp32 parity against transformers, plus cache and mutation gates.

The 7B (``n_groups=1``) is the only valid mlx-lm cross-check reference: mlx-lm
calls bare ``mx.fast.rms_norm`` with no grouping, which matches transformers
only when ``n_groups=1``. The 34B (``n_groups=2``) diverges and must be checked
against transformers fp32 directly.

Every tolerance is ``3x`` the fixture's own measured fp32-vs-fp64 floor for that
tensor. The μP folding floor (``noise.fold``) measures the bf16-ulp difference
between folded-at-load and unfolded-runtime multipliers.

Cannot run in this environment (parallel MLX runs reboot the M5). The
orchestrator runs all validation serially.
"""

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from conftest import floor, load_golden, relative_diff
from huggingface_hub import snapshot_download

from sideros.models.falcon_h1 import CHECKPOINT, FalconH1, FalconH1Activations

FIXTURE = Path(__file__).parent / "fixtures" / "falcon_h1_forward.safetensors"

# The 7B is the fixture model: small enough for fp32, n_groups=1 (valid mlx-lm ref).
MODEL_REPO = "tiiuae/Falcon-H1-7B-Base"
PATTERNS = ["config.json", "model*.safetensors", "tokenizer.json"]


def falcon_h1_dir() -> Path:
    return Path(snapshot_download(MODEL_REPO, allow_patterns=PATTERNS))


@pytest.fixture(scope="module")
def golden() -> dict[str, mx.array]:
    return load_golden(FIXTURE)


@pytest.fixture(scope="module")
def model() -> FalconH1:
    return CHECKPOINT.load(falcon_h1_dir(), mx.float32)


@pytest.fixture(scope="module")
def activations(model: FalconH1, golden: dict[str, mx.array]) -> FalconH1Activations:
    return model.activations(golden["input_ids"][None])


def test_embeddings_within_floor(
    activations: FalconH1Activations, golden: dict[str, mx.array]
) -> None:
    """The embedding is scaled by ``embedding_multiplier`` (folded at load)."""
    assert relative_diff(activations.embeddings, golden["embeddings"]) < floor(
        golden, "embeddings"
    )


@pytest.mark.parametrize("layer", [0, 1, 2, 3])
def test_block_within_floor(
    activations: FalconH1Activations, golden: dict[str, mx.array], layer: int
) -> None:
    assert relative_diff(activations.blocks[layer], golden[f"block_{layer}"]) < floor(
        golden, f"block_{layer}"
    )


def test_norm_within_floor(activations: FalconH1Activations, golden: dict[str, mx.array]) -> None:
    assert relative_diff(activations.norm, golden["norm"]) < floor(golden, "norm")


def test_logits_within_floor(activations: FalconH1Activations, golden: dict[str, mx.array]) -> None:
    assert relative_diff(activations.logits, golden["logits"]) < floor(golden, "logits")


def test_greedy_predictions_match(
    activations: FalconH1Activations, golden: dict[str, mx.array]
) -> None:
    ours = mx.argmax(activations.logits, axis=-1)
    theirs = mx.argmax(golden["logits"], axis=-1)
    assert mx.array_equal(ours, theirs).item()


def test_stepwise_matches_prefill(model: FalconH1, golden: dict[str, mx.array]) -> None:
    """A wrong composite cache (conv + SSM state + KV) cannot survive a
    full-logits comparison."""
    ids = golden["greedy_ids"]
    prefill = model(ids[None])
    cache = model.make_cache()
    steps = [model(ids[None, i : i + 1], cache) for i in range(ids.shape[0])]
    assert relative_diff(mx.concatenate(steps, axis=1), prefill) < 1e-5


def test_cached_greedy_matches_fixture(model: FalconH1, golden: dict[str, mx.array]) -> None:
    from sideros import stream_ids

    prompt = [int(i) for i in np.array(golden["input_ids"])]
    expected = [int(i) for i in np.array(golden["greedy_ids"])]
    generated = list(stream_ids(model, prompt, max_tokens=len(expected) - len(prompt)))
    assert prompt + generated == expected


def test_mutation_breaks_parity(model: FalconH1, golden: dict[str, mx.array]) -> None:
    """Perturbing one fused gate‖up must blow past the fixture floor."""
    mlp = model.model.layers[0].mlp
    original = mlp.gate_up_proj.weight
    mlp.gate_up_proj.weight = original * (1 + 1e-3)
    try:
        logits = model(golden["input_ids"][None])
        assert relative_diff(logits, golden["logits"]) > floor(golden, "logits")
    finally:
        mlp.gate_up_proj.weight = original


def test_mutation_of_folded_mup_breaks_parity(
    model: FalconH1, golden: dict[str, mx.array]
) -> None:
    """The μP multipliers are folded into the weights at load. Reversing the
    fold on the lm_head (un-multiplying) must fail."""
    if not hasattr(model, "lm_head"):
        pytest.skip("tied embeddings: no lm_head to mutate")
    original = model.lm_head.weight
    # Un-fold: divide by lm_head_multiplier (the fold was a multiply)
    # The exact value comes from the config; for the 7B it's ~0.039
    config = model.config
    model.lm_head.weight = original / config.lm_head_multiplier
    try:
        logits = model(golden["input_ids"][None])
        assert relative_diff(logits, golden["logits"]) > floor(golden, "logits")
    finally:
        model.lm_head.weight = original


def test_mutation_of_grouped_norm_breaks_parity(
    model: FalconH1, golden: dict[str, mx.array]
) -> None:
    """The gated norm must group the variance (n_groups=1 for the 7B, so this
    is a no-op test — but it guards the 34B path). Perturbing the norm weight
    must break parity."""
    if not model.config.mamba_rms_norm:
        pytest.skip("mamba_rms_norm is false")
    norm = model.model.layers[0].mamba.norm
    original = norm.weight
    norm.weight = original * 1.5
    try:
        logits = model(golden["input_ids"][None])
        assert relative_diff(logits, golden["logits"]) > floor(golden, "logits")
    finally:
        norm.weight = original
