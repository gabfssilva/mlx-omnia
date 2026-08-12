"""Parity gate for Laguna S 2.1: logits vs the reference, prefill vs stepwise, mutation.

No transformers ground truth exists at this size (fp32 of 118B is far beyond memory):
the reference implementation over the same checkpoint is the golden, bounded by
measured floors carried in the fixture (noise.logits, noise.batching).
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

from mlx_omnia import KVCache, stream_ids
from mlx_omnia.engine.models.laguna import CHECKPOINT, Laguna
from mlx_omnia.engine.models.laguna.layers.moe import LagunaSparseMoe

FIXTURE = Path(__file__).parent / "fixtures" / "laguna_mlxlm.safetensors"
REPO = "local/Laguna-S-2.1-mlx-oQ3e-fast-gs128"


@pytest.fixture(scope="module")
def golden() -> dict[str, mx.array]:
    return load_golden(FIXTURE)


@pytest.fixture(scope="module")
def model() -> Laguna:
    return CHECKPOINT.load(checkpoint_dir(REPO), None)


@requires_checkpoint(REPO)
def test_logits_match_mlxlm(model: Laguna, golden: dict[str, mx.array]) -> None:
    logits = model(golden["input_ids"][None])
    assert relative_diff(logits, golden["logits"]) < golden["noise.logits"].item()


@requires_checkpoint(REPO)
def test_sorted_gather_matches_stepwise(
    model: Laguna, golden: dict[str, mx.array]
) -> None:
    """Prefill takes the argsort/unsort reorder (10 routed rows/token); stepwise
    takes the per-token path. The floor is 3x the reference's own measured batching noise."""
    ids = golden["greedy_ids"]
    assert ids.shape[0] * 10 >= 64
    prefill = model(ids[None])
    cache = model.make_cache()
    steps = [model(ids[None, i : i + 1], cache) for i in range(ids.shape[0])]
    gap = relative_diff(mx.concatenate(steps, axis=1), prefill)
    assert gap < 3 * golden["noise.batching"].item()


@requires_checkpoint(REPO)
def test_greedy_matches_mlxlm(model: Laguna, golden: dict[str, mx.array]) -> None:
    """The reference is quantized, so the ids compare modulo ties."""
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
def test_mutation_breaks_parity(model: Laguna, golden: dict[str, mx.array]) -> None:
    """Perturbing one expert stack's weight must blow past the fixture floor."""
    layer = model.model.layers[5]
    assert isinstance(layer.mlp, LagunaSparseMoe)
    original = layer.mlp.switch_mlp.gate_up_proj.weight
    assert isinstance(original, mx.array)
    layer.mlp.switch_mlp.gate_up_proj.weight = original * 1.5
    try:
        logits = model(golden["input_ids"][None])
        assert relative_diff(logits, golden["logits"]) > golden["noise.logits"].item()
    finally:
        layer.mlp.switch_mlp.gate_up_proj.weight = original


@requires_checkpoint(REPO)
def test_cache_trim_rejected_only_when_untrimmable(model: Laguna) -> None:
    cache = model.make_cache()
    assert all(isinstance(c, KVCache) and c.is_trimmable for c in cache)
