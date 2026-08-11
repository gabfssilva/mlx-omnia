"""Parity gate for Step 3.7 Flash: logits vs mlx-vlm, prefill vs stepwise, mutation.

No transformers ground truth exists at 198B (fp32 of 198B is ~800 GB): mlx-vlm over
the same checkpoint is the reference, bounded by measured floors carried in the
fixture (``noise.logits``, ``noise.batching``).
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
from mlx_omnia.models.step3p7 import CHECKPOINT, Step3p7, Step3p7MoE

FIXTURE = Path(__file__).parent / "fixtures" / "step3p7_mlxlm.safetensors"
REPO = "stepfun-ai/Step-3.7-Flash"


@pytest.fixture(scope="module")
def golden() -> dict[str, mx.array]:
    return load_golden(FIXTURE)


@pytest.fixture(scope="module")
def model() -> Step3p7:
    return CHECKPOINT.load(checkpoint_dir(REPO), None)


@requires_checkpoint(REPO)
def test_logits_match_mlxlm(model: Step3p7, golden: dict[str, mx.array]) -> None:
    logits = model(mx.array(golden["input_ids"])[None])
    assert relative_diff(logits, golden["logits"]) < golden["noise.logits"].item()


@requires_checkpoint(REPO)
def test_sorted_gather_matches_stepwise(
    model: Step3p7, golden: dict[str, mx.array]
) -> None:
    """Prefill takes the argsort/unsort reorder (8 routed rows/token); stepwise takes
    the per-token path. The floor is 3x mlx-vlm's own measured batching noise."""
    ids = golden["input_ids"]
    assert ids.shape[0] * 8 >= 64
    prefill = model(ids[None])
    cache = model.make_cache()
    steps = [model(ids[None, i : i + 1], cache) for i in range(ids.shape[0])]
    gap = relative_diff(mx.concatenate(steps, axis=1), prefill)
    assert gap < 3 * golden["noise.batching"].item()


@requires_checkpoint(REPO)
def test_greedy_matches_mlxlm(model: Step3p7, golden: dict[str, mx.array]) -> None:
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
def test_mutation_breaks_parity(model: Step3p7, golden: dict[str, mx.array]) -> None:
    """Perturbing one expert stack's weight must blow past the fixture floor."""
    layer = model.model.layers[10]
    assert isinstance(layer.moe, Step3p7MoE)
    original = layer.moe.switch_mlp.gate_up_proj.weight
    assert isinstance(original, mx.array)
    layer.moe.switch_mlp.gate_up_proj.weight = original * 1.5
    try:
        logits = model(golden["input_ids"][None])
        assert relative_diff(logits, golden["logits"]) > golden["noise.logits"].item()
    finally:
        layer.moe.switch_mlp.gate_up_proj.weight = original


@requires_checkpoint(REPO)
def test_cache_trim_rejected_only_when_untrimmable(model: Step3p7) -> None:
    cache = model.make_cache()
    assert all(isinstance(c, KVCache) and c.is_trimmable for c in cache)
