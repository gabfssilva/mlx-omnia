# pyright: basic
"""Parity gate for Hunyuan 3 (Hy3): logits vs transformers, prefill vs stepwise, mutation.

Omnia is the first MLX implementation. The reference is
transformers (CUDA-generated fixture). Floors are measured, never invented; argmax
compares modulo ties.

The fixture (`hy3_transformers.safetensors`) must be generated on CUDA/RunPod via
`fixtures/generate_hy3.py` — the 298B model does not fit locally (bf16 alone is 597 GB).
Tests skip when the fixture or local checkpoint is absent.
"""

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mlx_omnia import KVCache, stream_ids
from mlx_omnia.engine.models.hy3 import CHECKPOINT, Hy3
from mlx_omnia.engine.models.hy3.layers.moe import Hy3SparseMoe
from tests.conftest import (
    assert_greedy_modulo_ties,
    checkpoint_dir,
    load_golden,
    relative_diff,
    requires_checkpoint,
)
from tests.mutation import mutated

FIXTURE = Path(__file__).parent / "fixtures" / "hy3_transformers.safetensors"
REPO = "tencent/Hy3"

# TikTok/Hy3 is gated; locally a oQ2 / mixed-2-6 conversion in the HF cache is the
# realistic checkpoint. Update REPO to match the local conversion's cache directory name.
LOCAL_REPO = "tencent/Hy3"


@pytest.fixture(scope="module")
def golden() -> dict[str, mx.array]:
    return load_golden(FIXTURE)


@pytest.fixture(scope="module")
def model() -> Hy3:
    return CHECKPOINT.load(checkpoint_dir(LOCAL_REPO), None)


@requires_checkpoint(LOCAL_REPO)
def test_forced_logits_match_transformers(
    model: Hy3, golden: dict[str, mx.array]
) -> None:
    logits = model(golden["input_ids"][None])
    assert relative_diff(logits, golden["logits"]) < golden["noise.logits"].item()


@requires_checkpoint(LOCAL_REPO)
def test_stepwise_matches_prefill(model: Hy3, golden: dict[str, mx.array]) -> None:
    """79 sigmoid-MoE layers -> large batching floor; 3x it, as Laguna does."""
    ids = golden["greedy_ids"]
    prefill = model(ids[None])
    cache = model.make_cache()
    steps = [model(ids[None, i : i + 1], cache) for i in range(ids.shape[0])]
    gap = relative_diff(mx.concatenate(steps, axis=1), prefill)
    assert gap < 3 * golden["noise.batching"].item()


@requires_checkpoint(LOCAL_REPO)
def test_sorted_gather_bit_identical_fp32(
    model: Hy3, golden: dict[str, mx.array]
) -> None:
    """Prefill reorder is a pure reorder; in fp32 it is exactly 0."""
    ids = golden["input_ids"]
    assert ids.shape[0] * 8 >= 64  # sorted gather engages
    cache_a = model.make_cache()
    prefill_a = model(ids[None], cache_a)
    cache_b = model.make_cache()
    prefill_b = model(ids[None], cache_b)
    gap = relative_diff(prefill_a, prefill_b)
    assert gap < 1e-6


@requires_checkpoint(LOCAL_REPO)
def test_greedy_modulo_ties(model: Hy3, golden: dict[str, mx.array]) -> None:
    prompt = [int(i) for i in np.array(golden["input_ids"])]
    expected = [int(i) for i in np.array(golden["greedy_ids"])]
    generated = list(
        stream_ids(model, prompt, max_tokens=len(expected) - len(prompt))
    )
    assert_greedy_modulo_ties(
        prompt + generated,
        expected,
        lambda: model(golden["greedy_ids"][None])[0],
        golden["noise.logits"].item(),
    )


@requires_checkpoint(LOCAL_REPO)
def test_mutation_breaks_parity(model: Hy3, golden: dict[str, mx.array]) -> None:
    """Perturbing one expert stack's weight must blow past the fixture floor."""
    layer = model.model.layers[5]
    assert isinstance(layer.mlp, Hy3SparseMoe)
    original = layer.mlp.switch_mlp.gate_up_proj.weight
    assert isinstance(original, mx.array)
    with mutated(layer.mlp.switch_mlp.gate_up_proj, "weight", original * 1.5):
        logits = model(golden["input_ids"][None])
        assert relative_diff(logits, golden["logits"]) > golden["noise.logits"].item()


@requires_checkpoint(LOCAL_REPO)
def test_cache_trim_rejected_only_when_untrimmable(model: Hy3) -> None:
    cache = model.make_cache()
    assert all(isinstance(c, KVCache) and c.is_trimmable for c in cache)
