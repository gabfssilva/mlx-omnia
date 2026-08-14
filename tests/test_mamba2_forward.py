"""Mamba2 fp32 parity against transformers, plus cache, kernel A/B and mutation
gates.

Every tolerance is `3 x` the fixture's own measured fp32-vs-fp64 floor for that
tensor: the trunk is 64 layers of recurrence, so a single number would be vacuous
at one end and impossible at the other.
"""

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from huggingface_hub import snapshot_download

from mlx_omnia import stream_ids
from mlx_omnia.engine.core.cache import DeltaCache
from mlx_omnia.engine.core.kernels.ssm.step import ssm_step
from mlx_omnia.engine.models.mamba2 import CHECKPOINT, Mamba2, Mamba2Activations
from mlx_omnia.engine.models.mamba2.layers import flags, ssd
from tests.conftest import floor, load_golden, relative_diff

FIXTURE = Path(__file__).parent / "fixtures" / "mamba2_forward.safetensors"
N_LAYER = 64
LAYER = 0

PATTERNS = ["config.json", "*.safetensors", "*.json"]


def mamba2_dir() -> Path:
    return Path(snapshot_download("state-spaces/mamba2-2.8b-hf", allow_patterns=PATTERNS))


@pytest.fixture(scope="module")
def golden() -> dict[str, mx.array]:
    return load_golden(FIXTURE)


@pytest.fixture(scope="module")
def model() -> Mamba2:
    return CHECKPOINT.load(mamba2_dir(), mx.float32)


@pytest.fixture(scope="module")
def activations(model: Mamba2, golden: dict[str, mx.array]) -> Mamba2Activations:
    return model.activations(golden["input_ids"][None])


def test_embeddings_exact(activations: Mamba2Activations, golden: dict[str, mx.array]) -> None:
    assert relative_diff(activations.embeddings, golden["embeddings"]) == 0


@pytest.mark.parametrize("layer", range(N_LAYER))
def test_block_within_floor(
    activations: Mamba2Activations, golden: dict[str, mx.array], layer: int
) -> None:
    assert relative_diff(activations.blocks[layer], golden[f"block_{layer}"]) < floor(
        golden, f"block_{layer}"
    )


def test_norm_within_floor(activations: Mamba2Activations, golden: dict[str, mx.array]) -> None:
    assert relative_diff(activations.norm, golden["norm"]) < floor(golden, "norm")


def test_logits_within_floor(activations: Mamba2Activations, golden: dict[str, mx.array]) -> None:
    assert relative_diff(activations.logits, golden["logits"]) < floor(golden, "logits")


def test_greedy_predictions_match(
    activations: Mamba2Activations, golden: dict[str, mx.array]
) -> None:
    ours = mx.argmax(activations.logits, axis=-1)
    theirs = mx.argmax(golden["logits"], axis=-1)
    assert mx.array_equal(ours, theirs).item()


def test_mixer_internals_within_floor(model: Mamba2, golden: dict[str, mx.array]) -> None:
    """Naming the culprit inside the mixer: the pre-norm and the mixer output."""
    block = model.model.layers[LAYER]
    mixer = block.mixer
    x = model.model.embed_tokens(golden["input_ids"][None])
    normed = block.norm(x)
    assert relative_diff(normed, golden["b0_ln"]) < floor(golden, "b0_ln")

    mixer_out = mixer(normed, DeltaCache())
    assert relative_diff(mixer_out, golden["b0_mixer"]) < floor(golden, "b0_mixer")


def test_stepwise_matches_prefill(model: Mamba2, golden: dict[str, mx.array]) -> None:
    """A wrong conv window or a stale recurrent state can survive a degenerate
    greedy; it does not survive full logits."""
    ids = golden["greedy_ids"]
    prefill = model(ids[None])
    cache = model.make_cache()
    steps = [model(ids[None, i : i + 1], cache) for i in range(ids.shape[0])]
    assert relative_diff(mx.concatenate(steps, axis=1), prefill) < 1e-5


def test_cached_greedy_matches_fixture(model: Mamba2, golden: dict[str, mx.array]) -> None:
    prompt = [int(i) for i in np.array(golden["input_ids"])]
    expected = [int(i) for i in np.array(golden["greedy_ids"])]
    generated = list(stream_ids(model, prompt, max_tokens=len(expected) - len(prompt)))
    assert prompt + generated == expected


def test_delta_cache_is_not_trimmable(model: Mamba2) -> None:
    cache = model.make_cache()
    for entry in cache:
        assert not entry.is_trimmable
    with pytest.raises(NotImplementedError):
        next(iter(cache)).trim(0)


def test_mutation_of_conv_breaks_parity(model: Mamba2, golden: dict[str, mx.array]) -> None:
    conv = model.model.layers[LAYER].mixer.conv1d
    original = conv.weight
    conv.weight = original * 1.01
    try:
        logits = model(golden["input_ids"][None])
        assert relative_diff(logits, golden["logits"]) > floor(golden, "logits")
    finally:
        conv.weight = original


def test_mutation_of_decay_breaks_parity(model: Mamba2, golden: dict[str, mx.array]) -> None:
    mixer = model.model.layers[LAYER].mixer
    original = mixer.A_log
    mixer.A_log = original + 0.5
    try:
        logits = model(golden["input_ids"][None])
        assert relative_diff(logits, golden["logits"]) > floor(golden, "logits")
    finally:
        mixer.A_log = original


# --- Kernel seams --------------------------------------------------------


def stepwise(model: Mamba2, ids: mx.array) -> mx.array:
    cache = model.make_cache()
    return mx.concatenate(
        [model(ids[None, i : i + 1], cache) for i in range(ids.shape[0])], axis=1
    )


def test_kernel_is_the_default_path(
    model: Mamba2, golden: dict[str, mx.array], monkeypatch: pytest.MonkeyPatch
) -> None:
    assert flags.SSM_KERNEL and flags.COMPILED_STEP

    calls = {"ssm": 0}

    def counted_ssm(
        x: mx.array,
        a_log: mx.array,
        b: mx.array,
        c: mx.array,
        d: mx.array,
        dt: mx.array,
        dt_bias: mx.array,
        state: mx.array,
        time_step_limit: tuple[float, float],
    ) -> tuple[mx.array, mx.array]:
        calls["ssm"] += 1
        return ssm_step(x, a_log, b, c, d, dt, dt_bias, state, time_step_limit)

    monkeypatch.setattr(ssd, "ssm_step", counted_ssm)
    mx.eval(model(golden["input_ids"][None, :1]))
    assert calls == {"ssm": N_LAYER}


def test_kernel_matches_ops(
    model: Mamba2, golden: dict[str, mx.array], monkeypatch: pytest.MonkeyPatch
) -> None:
    ids = golden["input_ids"][None]
    with_kernel = model(ids)
    monkeypatch.setattr(flags, "SSM_KERNEL", False)
    assert relative_diff(with_kernel, model(ids)) < 1e-5


def test_compiled_step_matches_eager(
    model: Mamba2, golden: dict[str, mx.array], monkeypatch: pytest.MonkeyPatch
) -> None:
    ids = golden["greedy_ids"]
    with_compiled = stepwise(model, ids)
    monkeypatch.setattr(flags, "COMPILED_STEP", False)
    assert relative_diff(with_compiled, stepwise(model, ids)) < 1e-5
