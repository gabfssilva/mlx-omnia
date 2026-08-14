from pathlib import Path

import mlx.core as mx
import pytest
from huggingface_hub import snapshot_download

from mlx_omnia.engine.models.gpt2 import CHECKPOINT, GPT2, GPT2Activations
from tests.conftest import floor, load_golden, relative_diff

FIXTURE = Path(__file__).parent / "fixtures" / "gpt2_forward.safetensors"

N_LAYER = 12


def gpt2_dir() -> Path:
    return Path(
        snapshot_download(
            "gpt2",
            allow_patterns=["config.json", "model.safetensors", "vocab.json", "merges.txt"],
        )
    )


@pytest.fixture(scope="module")
def golden() -> dict[str, mx.array]:
    return load_golden(FIXTURE)


@pytest.fixture(scope="module")
def model() -> GPT2:
    return CHECKPOINT.load(gpt2_dir(), None)


@pytest.fixture(scope="module")
def activations(model: GPT2, golden: dict[str, mx.array]) -> GPT2Activations:
    return model.activations(golden["input_ids"][None])


def test_embeddings_exact(activations: GPT2Activations, golden: dict[str, mx.array]) -> None:
    assert relative_diff(activations.embeddings, golden["embeddings"]) == 0


@pytest.mark.parametrize("layer", range(N_LAYER))
def test_block_within_floor(
    activations: GPT2Activations, golden: dict[str, mx.array], layer: int
) -> None:
    assert relative_diff(activations.blocks[layer], golden[f"block_{layer}"]) < floor(
        golden, f"block_{layer}"
    )


def test_ln_f_within_floor(activations: GPT2Activations, golden: dict[str, mx.array]) -> None:
    assert relative_diff(activations.ln_f, golden["ln_f"]) < floor(golden, "ln_f")


def test_logits_within_floor(activations: GPT2Activations, golden: dict[str, mx.array]) -> None:
    assert relative_diff(activations.logits, golden["logits"]) < floor(golden, "logits")


def test_block0_internals_within_floor(model: GPT2, golden: dict[str, mx.array]) -> None:
    """Naming the culprit: each submodule of block 0 against its own hook boundary."""
    block = model.h[0]
    ids = golden["input_ids"][None]
    x = model.wte(ids) + model.wpe(mx.arange(ids.shape[-1]))
    normed = block.ln_1(x)
    assert relative_diff(normed, golden["b0_ln_1"]) < floor(golden, "b0_ln_1")

    attended = block.attn(normed)
    assert relative_diff(attended, golden["b0_attn"]) < floor(golden, "b0_attn")

    second = block.ln_2(x + attended)
    assert relative_diff(second, golden["b0_ln_2"]) < floor(golden, "b0_ln_2")
    assert relative_diff(block.mlp(second), golden["b0_mlp"]) < floor(golden, "b0_mlp")


def test_greedy_predictions_match(
    activations: GPT2Activations, golden: dict[str, mx.array]
) -> None:
    ours = mx.argmax(activations.logits, axis=-1)
    theirs = mx.argmax(golden["logits"], axis=-1)
    assert mx.array_equal(ours, theirs).item()


def test_mutation_breaks_parity(golden: dict[str, mx.array]) -> None:
    model = CHECKPOINT.load(gpt2_dir(), None)
    model.h[5].mlp.c_fc.weight = model.h[5].mlp.c_fc.weight * (1 + 1e-3)
    mutated = model.activations(golden["input_ids"][None])
    assert relative_diff(mutated.logits, golden["logits"]) > floor(golden, "logits")
