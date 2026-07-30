"""Qwen3.6-27B 6-bit against mlx-lm (git main) over the same packed weights.

No transformers ground truth exists at this size: the reference is mlx-lm, bounded by
the floors the fixture measured (`noise.logits`, `noise.batching`). This is also the
only checkpoint of the pair with kv-ratio 3 (16 key heads into 48 value heads) and with
the mlx dialect of the checkpoint (`language_model.model.*`, conv `[dim, kernel, 1]`,
the zero-centering already folded in).
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

from sideros import stream_ids
from sideros.models.qwen3_5 import CHECKPOINT, Qwen35

FIXTURE = Path(__file__).parent / "fixtures" / "qwen3_5_27b_mlxlm.safetensors"
REPO = "mlx-community/Qwen3.6-27B-6bit"
# The fixture was measured on the local conversion, which sits next to a hub revision
# of the same repo in the cache; the revision is part of the checkpoint's identity here.
REVISION = "local"


@pytest.fixture(scope="module")
def golden() -> dict[str, mx.array]:
    return load_golden(FIXTURE)


@pytest.fixture(scope="module")
def model() -> Qwen35:
    return CHECKPOINT.load(checkpoint_dir(REPO, REVISION), None)


@requires_checkpoint(REPO, REVISION)
def test_dialect_and_shapes(model: Qwen35) -> None:
    """The mlx dialect: untied head, ratio 3, conv squeezed off the trailing axis."""
    config = model.config
    assert not config.tie_word_embeddings
    assert config.eos_token_id == (248046, 248044)
    assert config.linear_num_value_heads // config.linear_num_key_heads == 3
    conv = model.model.layers[0].linear_attn.conv1d.weight
    assert conv.shape == (config.conv_dim, config.linear_conv_kernel_dim)


@requires_checkpoint(REPO, REVISION)
def test_logits_match_mlxlm(model: Qwen35, golden: dict[str, mx.array]) -> None:
    """The bound is `noise.batching`, not `noise.logits`, and the difference is
    measured, not slack: `noise.logits` is one graph against itself in fp32, while two
    bf16 implementations of the *same* graph also differ by evaluation order — three
    projections fused into one matmul do not round like three. `noise.batching` is
    exactly that class of difference, measured on mlx-lm against itself (3.7e-2); we sit
    at 1.2e-2. Swapping our l2norm for mlx-lm's moves it to 1.1e-2, so the deliberate
    transformers eps-in-the-sum is not what the gap is made of; mutating one packed
    scale by 1.5 takes it to 8.7e-2."""
    logits = model(golden["input_ids"][None])
    assert relative_diff(logits, golden["logits"]) < golden["noise.batching"].item()


@requires_checkpoint(REPO, REVISION)
def test_teacher_forced_argmax_matches_mlxlm(
    model: Qwen35, golden: dict[str, mx.array]
) -> None:
    """Argmax between two bf16 implementations compares modulo ties: a flip only counts
    when the golden separates the two candidates by more than the measured floor."""
    ids = golden["greedy_ids"]
    prompt = golden["input_ids"].shape[0]
    ours = model(ids[None]).astype(mx.float32)[0]
    # Only the generated positions are forced: what the model would say after the
    # prompt's first token is nobody's prediction.
    reference = mx.argmax(ours, axis=-1)[prompt - 1 :]
    expected = ids[prompt:]
    tolerance = golden["noise.logits"].item() * float(mx.abs(ours).max().item())
    for position in range(expected.shape[0]):
        picked = int(reference[position].item())
        wanted = int(expected[position].item())
        if picked == wanted:
            continue
        gap = float((ours[position, picked] - ours[position, wanted]).item())
        assert gap < tolerance, f"position {position}: {picked} over {wanted} by {gap}"


@requires_checkpoint(REPO, REVISION)
def test_stepwise_matches_prefill(model: Qwen35, golden: dict[str, mx.array]) -> None:
    """A stale conv window or recurrent state does not survive a full-logits
    comparison. The floor is 3x mlx-lm's own measured batching noise."""
    ids = golden["greedy_ids"]
    prefill = model(ids[None])
    cache = model.make_cache()
    steps = [model(ids[None, i : i + 1], cache) for i in range(ids.shape[0])]
    assert relative_diff(mx.concatenate(steps, axis=1), prefill) < 3 * golden[
        "noise.batching"
    ].item()


@requires_checkpoint(REPO, REVISION)
def test_greedy_matches_mlxlm(model: Qwen35, golden: dict[str, mx.array]) -> None:
    """What the teacher-forced test above cannot see: the sampler and the cache driving
    the run themselves. Same rule on the ids — the first divergence has to be a tie."""
    prompt = [int(i) for i in np.array(golden["input_ids"])]
    expected = [int(i) for i in np.array(golden["greedy_ids"])]
    generated = list(stream_ids(model, prompt, max_tokens=len(expected) - len(prompt)))
    assert_greedy_modulo_ties(
        prompt + generated,
        expected,
        lambda: model(golden["greedy_ids"][None])[0],
        golden["noise.logits"].item(),
    )


@requires_checkpoint(REPO, REVISION)
def test_mutation_breaks_parity(model: Qwen35, golden: dict[str, mx.array]) -> None:
    """Perturbing one DeltaNet's packed scales must blow past the fixture floor."""
    delta = model.model.layers[1].linear_attn
    original = delta.fused_proj.scales
    assert isinstance(original, mx.array)
    delta.fused_proj.scales = original * 1.5
    try:
        logits = model(golden["input_ids"][None])
        assert relative_diff(logits, golden["logits"]) > golden["noise.batching"].item()
    finally:
        delta.fused_proj.scales = original
