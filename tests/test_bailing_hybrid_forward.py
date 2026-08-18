"""Ling 3.0 (`bailing_hybrid`) in fp32 against the reference MLX port, over a synthetic
12-layer model that exercises both mixers, both MLP kinds and every load-time fusion.

There is no transformers ground truth to have: the released modeling file delegates the
KDA recurrence to fla's triton kernels, which do not run here, and the only published
checkpoint is 124B. The fixture's own docstring says what the reference is and how the
floors were measured; every tolerance below is `3 x` a floor from the fixture, never a
number chosen to make a test pass.
"""

import json
from collections.abc import Sequence
from pathlib import Path

import mlx.core as mx
import pytest

from mlx_omnia import stream_ids
from mlx_omnia.engine.core.cache import LayerCache
from mlx_omnia.engine.core.cache_file import dump, load
from mlx_omnia.engine.core.layers import MultiLinear
from mlx_omnia.engine.core.prefix import Payload, Prefixes, PrefixStore, Slot, Vault
from mlx_omnia.engine.models.bailing_hybrid import (
    CHECKPOINT,
    BailingHybrid,
    BailingHybridActivations,
)
from mlx_omnia.engine.models.bailing_hybrid.layers.attention import BailingHybridLatentAttention
from mlx_omnia.engine.models.bailing_hybrid.layers.kda import KimiDeltaAttention
from tests.conftest import floor, load_golden, relative_diff

FIXTURES = Path(__file__).parent / "fixtures"
FIXTURE = FIXTURES / "bailing_hybrid_forward.safetensors"
CONFIG = FIXTURES / "bailing_hybrid_tiny" / "config.json"
N_LAYER = 12
KDA_LAYER = 0
MLA_LAYER = 5


@pytest.fixture(scope="module")
def golden() -> dict[str, mx.array]:
    return load_golden(FIXTURE)


@pytest.fixture(scope="module")
def model(golden: dict[str, mx.array], tmp_path_factory: pytest.TempPathFactory) -> BailingHybrid:
    """The fixture's weights are the checkpoint: written back out in the HF names they
    were generated in, so the loader's fusions run exactly as they would on Ling 3.0."""
    directory = tmp_path_factory.mktemp("bailing_hybrid")
    weights = {
        name.removeprefix("weight."): value
        for name, value in golden.items()
        if name.startswith("weight.")
    }
    mx.save_safetensors(str(directory / "model.safetensors"), weights)
    (directory / "config.json").write_text(CONFIG.read_text())
    return CHECKPOINT.load(directory, None)


@pytest.fixture(scope="module")
def activations(model: BailingHybrid, golden: dict[str, mx.array]) -> BailingHybridActivations:
    return model.activations(golden["input_ids"][None])


def kda_of(model: BailingHybrid, layer: int) -> KimiDeltaAttention:
    """mlx.nn.Module's __getattr__ is untyped: every submodule reach is narrowed here."""
    attention = model.model.layers[layer].attention
    assert isinstance(attention, KimiDeltaAttention)
    return attention


def mla_of(model: BailingHybrid, layer: int) -> BailingHybridLatentAttention:
    attention = model.model.layers[layer].attention
    assert isinstance(attention, BailingHybridLatentAttention)
    return attention


def test_config_matches_the_fixture(model: BailingHybrid) -> None:
    raw = json.loads(CONFIG.read_text())
    config = model.config
    assert config.num_hidden_layers == N_LAYER
    assert config.attends == tuple(layer in (5, 11) for layer in range(N_LAYER))
    assert config.routes == tuple(layer >= raw["first_k_dense_replace"] for layer in range(N_LAYER))
    assert config.kda_safe_gate and config.kda_lower_bound == -5.0


def test_kv_b_proj_is_split_per_head(model: BailingHybrid) -> None:
    """The load-time split is the one fusion with no sibling in another port: the tree
    declares the two halves and the checkpoint carries neither."""
    attention = mla_of(model, MLA_LAYER)
    embed_q, unembed = attention.embed_q, attention.unembed_out
    assert isinstance(embed_q, MultiLinear) and isinstance(unembed, MultiLinear)
    assert embed_q.weight.shape == (4, 64, 32)
    assert unembed.weight.shape == (4, 32, 64)


@pytest.mark.parametrize("layer", range(N_LAYER))
def test_block_activations(
    activations: BailingHybridActivations, golden: dict[str, mx.array], layer: int
) -> None:
    diff = relative_diff(activations.blocks[layer], golden[f"block.{layer}"])
    assert diff < floor(golden, f"block_{layer}")


def test_final_norm(activations: BailingHybridActivations, golden: dict[str, mx.array]) -> None:
    assert relative_diff(activations.norm, golden["norm"]) < floor(golden, "block_11")


def test_logits(activations: BailingHybridActivations, golden: dict[str, mx.array]) -> None:
    assert relative_diff(activations.logits, golden["logits"]) < floor(golden, "batching")


def test_greedy_matches_the_reference(
    model: BailingHybrid, golden: dict[str, mx.array]
) -> None:
    ids = [int(token) for token in golden["input_ids"]]
    generated = list(stream_ids(model, ids, max_tokens=len(golden["greedy"])))
    assert generated == [int(token) for token in golden["greedy"]]


def test_stepwise_matches_prefill(model: BailingHybrid, golden: dict[str, mx.array]) -> None:
    """A wrong cache survives a degenerate greedy; it does not survive full logits. The
    ids are 16 long and the router keeps 4 of 16 experts, so prefill goes through the
    sorted gather (16 · 4 ≥ 64) and the step does not."""
    ids = [int(token) for token in golden["input_ids"]]
    prefill = model(mx.array([ids]))
    cache = model.make_cache()
    stepped = mx.concatenate(
        [model(mx.array([[token]]), cache) for token in ids], axis=1
    )
    assert relative_diff(stepped, prefill) < floor(golden, "batching")


class Recording:
    """The trunk plus what reached it: rows per forward, and the logits each one answered
    with. Ids are not evidence about a cache — a run off by a row still decodes fluently."""

    def __init__(self, model: BailingHybrid) -> None:
        self.model = model
        self.fed: list[int] = []
        self.logits: list[mx.array] = []

    def make_cache(self) -> list[LayerCache]:
        return self.model.make_cache()

    def __call__(self, ids: mx.array, cache: Sequence[LayerCache] | None = None) -> mx.array:
        out = self.model(ids, cache)
        self.fed.append(ids.shape[1])
        self.logits.append(out[:, -1, :])
        return out


SPAN = 4


def conversation(
    model: BailingHybrid, golden: dict[str, mx.array]
) -> tuple[Prefixes, list[int]]:
    """One turn generated with the store in hand, and the prompt of the turn after it: the
    first prompt, the ids the model wrote, and a new tail."""
    prefix = Prefixes(PrefixStore(1 << 30, span=SPAN), "bailing", "a-stamp")
    first = [int(token) for token in golden["input_ids"]]
    grown = list(stream_ids(model, first, max_tokens=4, prefix=prefix))
    return prefix, [*first, *grown, *first[:3]]


def test_a_second_turn_reuses_the_hybrid_cache(
    model: BailingHybrid, golden: dict[str, mx.array]
) -> None:
    """10 of these 12 layers keep a recurrent state that cannot be rewound, and a
    conversation never asks them to: the second turn extends the first, so the stored entry
    is a prefix of it and is handed over whole. The logits of every step the sampler read
    have to be the cold ones."""
    prefix, second = conversation(model, golden)
    warm, cold = Recording(model), Recording(model)

    reused = list(stream_ids(warm, second, max_tokens=4, prefix=prefix))

    assert reused == list(stream_ids(cold, second, max_tokens=4))
    covered = (len(second) - 1) // SPAN * SPAN
    assert sum(warm.fed[:-4]) == len(second) - covered, "the spans covered all but the tail"
    assert cold.fed[0] == len(second)
    for hot, fresh in zip(warm.logits, cold.logits, strict=True):
        assert relative_diff(hot, fresh) < floor(golden, "batching")


def test_a_diverging_turn_is_prefilled_instead_of_rewound(
    model: BailingHybrid, golden: dict[str, mx.array]
) -> None:
    """The conversation edited in the middle. A KV-only trunk rewinds the stored entry to
    the branch point and resumes from it; here 10 of the 12 layers hold a state with no way
    back, so the candidate is dropped and the turn is prefilled whole. The assertion is that
    the answer is the cold one — a state rewound to a step it never took reads as a fluent
    wrong answer."""
    prefix, second = conversation(model, golden)
    apart = 1 if second[6] != 1 else 2
    edited = [*second[:6], apart, *second[7:10]]
    warm, cold = Recording(model), Recording(model)

    resumed = list(stream_ids(warm, edited, max_tokens=4, prefix=prefix))

    assert resumed == list(stream_ids(cold, edited, max_tokens=4))
    # Two forwards and not one: the prefill stops on the last span boundary so the turn
    # leaves an anchor. What it must not do is skip a row.
    assert sum(warm.fed[:-4]) == len(edited), "a state was rewound to a branch point"
    # The decode's own steps, which is where the two runs stand on the same positions: the
    # prefill forwards are cut differently by construction, so pairing them would be pairing
    # position 7 against position 9.
    for hot, fresh in zip(warm.logits[-4:], cold.logits[-4:], strict=True):
        assert relative_diff(hot, fresh) < floor(golden, "batching")


def test_stepwise_matches_prefill_over_a_reused_hybrid_cache(
    model: BailingHybrid, golden: dict[str, mx.array]
) -> None:
    """The mandatory invariant, on the branch the reuse opened: the tail stepped one row at
    a time over a cache that came out of the trie against the same rows of a cold prefill.
    A recurrent state restored to the wrong step is exactly what this catches and what the
    ids above do not."""
    prefix, second = conversation(model, golden)
    cache = model.make_cache()
    walk = prefix.begin(cache)
    assert walk is not None
    covered = walk.resume(second, cache)
    assert covered == (len(second) - 1) // SPAN * SPAN

    stepped = mx.concatenate(
        [model(mx.array([[token]]), cache) for token in second[covered:]], axis=1
    )

    prefill = model(mx.array([second]))
    assert relative_diff(stepped, prefill[:, covered:]) < floor(golden, "batching")


def test_stepwise_matches_prefill_over_spans_read_back_from_disk(
    model: BailingHybrid, golden: dict[str, mx.array], tmp_path: Path
) -> None:
    """The mandatory invariant on the other branch this track opened: spans that went to
    files and came back have to step exactly like a cache that never left memory.

    It is the test that catches a restore that is right about the attention layers and wrong
    about the recurrent ones — 10 of these 12 hold a state, and a state restored to the wrong
    step reads as a fluent wrong answer that a greedy decode never exposes.
    """
    ids = [int(token) for token in golden["input_ids"]]
    cut = (len(ids) - 1) // SPAN * SPAN
    files = _Files(tmp_path)
    prefix = Prefixes(PrefixStore(1, files, span=SPAN), "bailing", "a-stamp")
    warm = model.make_cache()
    walk = prefix.begin(warm)
    assert walk is not None
    model(mx.array([ids[:cut]]), warm)
    mx.eval([tensor for layer in warm for tensor in layer.tensors])
    walk.commit(ids, warm, cut)

    read = model.make_cache()
    second = prefix.begin(read)
    assert second is not None
    assert second.resume(ids, read) == cut
    assert files.reads > 0, "nothing came off disk, so the parity below proves nothing"
    stepped = mx.concatenate(
        [model(mx.array([[token]]), read) for token in ids[cut:]], axis=1
    )

    prefill = model(mx.array([ids]))
    assert relative_diff(stepped, prefill[:, cut:]) < floor(golden, "batching")


def test_a_generation_off_a_restored_cache_writes_the_same_ids(
    model: BailingHybrid, golden: dict[str, mx.array], tmp_path: Path
) -> None:
    """The same question the store's own test asks, with a disk in the middle: a turn resumed
    off files is the turn that was never interrupted. The ceiling is one byte, so every span
    is pushed into the vault the moment it is stored and the second turn can only be answered
    from there."""
    ids = [int(token) for token in golden["input_ids"]]
    files = _Files(tmp_path)
    prefix = Prefixes(PrefixStore(1, files, span=SPAN), "bailing", "a-stamp")
    grown = list(stream_ids(model, ids, max_tokens=4, prefix=prefix))
    second = [*ids, *grown, *ids[:3]]

    resumed = list(stream_ids(model, second, max_tokens=4, prefix=prefix))

    assert resumed == list(stream_ids(model, second, max_tokens=4))
    assert files.reads > 0, "nothing came off disk, so the parity above proves nothing"


class _Files(Vault):
    """A vault that is a directory, which is what the daemon's own is minus a ceiling, an
    index and a thread — none of which is what this test is about."""

    def __init__(self, root: Path) -> None:
        self._root = root / "spans"
        self._root.mkdir(parents=True, exist_ok=True)
        self.reads = 0

    def _path(self, key: Slot) -> Path:
        return self._root / f"{key[0]}-{key[1]}.safetensors"

    def holds(self, key: Slot) -> bool:
        return self._path(key).exists()

    def read(self, key: Slot) -> Payload | None:
        path = self._path(key)
        if not path.exists():
            return None
        self.reads += 1
        return load(path)

    def write(self, key: Slot, payload: Payload, nbytes: int) -> bool:
        dump(payload, self._path(key))
        return True

    def forget(self, key: Slot) -> None:
        self._path(key).unlink(missing_ok=True)
