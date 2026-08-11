"""Ling 3.0 (`bailing_hybrid`) in fp32 against the reference MLX port, over a synthetic
12-layer model that exercises both mixers, both MLP kinds and every load-time fusion.

There is no transformers ground truth to have: the released modeling file delegates the
KDA recurrence to fla's triton kernels, which do not run here, and the only published
checkpoint is 124B. The fixture's own docstring says what the reference is and how the
floors were measured; every tolerance below is `3 ×` a floor from the fixture, never a
number chosen to make a test pass.
"""

import json
from collections.abc import Sequence
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from conftest import floor, load_golden, relative_diff

from mlx_omnia import stream_ids
from mlx_omnia.core import cache_file
from mlx_omnia.core.cache import DeltaCache, KVCache, LayerCache
from mlx_omnia.core.kernels.gated_delta import GatedDelta, gated_delta
from mlx_omnia.core.layers import MultiLinear, SharedMLP, SwitchGLU
from mlx_omnia.core.prompt_cache import PromptCache, Reuse
from mlx_omnia.models.bailing_hybrid import (
    CHECKPOINT,
    BailingHybrid,
    BailingHybridActivations,
)
from mlx_omnia.models.bailing_hybrid.layers.attention import BailingHybridLatentAttention
from mlx_omnia.models.bailing_hybrid.layers.kda import KimiDeltaAttention
from mlx_omnia.models.bailing_hybrid.layers.moe import LimitedSharedMLP, LimitedSwitchGLU

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

    def __call__(self, ids: mx.array, cache: list[LayerCache] | None = None) -> mx.array:
        out = self.model(ids, cache)
        self.fed.append(ids.shape[1])
        self.logits.append(out[:, -1, :])
        return out


def conversation(
    model: BailingHybrid, golden: dict[str, mx.array]
) -> tuple[PromptCache[LayerCache], list[int]]:
    """One turn generated with the trie in hand, and the prompt of the turn after it: the
    first prompt, the ids the model wrote, and a new tail."""
    trie = PromptCache[LayerCache](budget=1 << 30)
    first = [int(token) for token in golden["input_ids"]]
    grown = list(stream_ids(model, first, max_tokens=4, prefix=trie))
    return trie, [*first, *grown, *first[:3]]


def test_a_second_turn_reuses_the_hybrid_cache(
    model: BailingHybrid, golden: dict[str, mx.array]
) -> None:
    """10 of these 12 layers keep a recurrent state that cannot be rewound, and a
    conversation never asks them to: the second turn extends the first, so the stored entry
    is a prefix of it and is handed over whole. The logits of every step the sampler read
    have to be the cold ones."""
    trie, second = conversation(model, golden)
    warm, cold = Recording(model), Recording(model)

    reused = list(stream_ids(warm, second, max_tokens=4, prefix=trie))

    assert reused == list(stream_ids(cold, second, max_tokens=4))
    assert warm.fed[0] == 3, "the stored entry covered everything but the new tail"
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
    trie, second = conversation(model, golden)
    apart = 1 if second[6] != 1 else 2
    edited = [*second[:6], apart, *second[7:10]]
    warm, cold = Recording(model), Recording(model)

    resumed = list(stream_ids(warm, edited, max_tokens=4, prefix=trie))

    assert resumed == list(stream_ids(cold, edited, max_tokens=4))
    assert warm.fed[0] == len(edited), "a state was rewound to a branch point"
    for hot, fresh in zip(warm.logits, cold.logits, strict=True):
        assert relative_diff(hot, fresh) < floor(golden, "batching")


def test_stepwise_matches_prefill_over_a_reused_hybrid_cache(
    model: BailingHybrid, golden: dict[str, mx.array]
) -> None:
    """The mandatory invariant, on the branch the reuse opened: the tail stepped one row at
    a time over a cache that came out of the trie against the same rows of a cold prefill.
    A recurrent state restored to the wrong step is exactly what this catches and what the
    ids above do not."""
    trie, second = conversation(model, golden)
    reuse = trie.take(second)
    assert reuse is not None and reuse.length == len(second) - 3

    stepped = mx.concatenate(
        [model(mx.array([[token]]), reuse.caches) for token in second[reuse.length :]], axis=1
    )

    prefill = model(mx.array([second]))
    assert relative_diff(stepped, prefill[:, reuse.length :]) < floor(golden, "batching")


def test_stepwise_matches_prefill_over_a_cache_read_back_from_disk(
    model: BailingHybrid, golden: dict[str, mx.array], tmp_path: Path
) -> None:
    """The mandatory invariant on the other branch this track opened: a cache that went to a
    file and came back has to step exactly like one that never left memory.

    It is the test that catches a restore that is right about the attention layers and wrong
    about the recurrent ones — 10 of these 12 hold a state, and a state restored to the wrong
    step reads as a fluent wrong answer that a greedy decode never exposes.
    """
    ids = [int(token) for token in golden["input_ids"]]
    cut = len(ids) - 4
    warm = model.make_cache()
    model(mx.array([ids[:cut]]), warm)
    mx.eval([tensor for layer in warm for tensor in layer.tensors])
    path = tmp_path / "entry.safetensors"
    cache_file.dump(warm, path)

    read = model.make_cache()
    cache_file.restore(read, path)
    stepped = mx.concatenate(
        [model(mx.array([[token]]), read) for token in ids[cut:]], axis=1
    )

    prefill = model(mx.array([ids]))
    assert relative_diff(stepped, prefill[:, cut:]) < floor(golden, "batching")


def test_a_generation_off_a_restored_cache_writes_the_same_ids(
    model: BailingHybrid, golden: dict[str, mx.array], tmp_path: Path
) -> None:
    """The same question the trie's own test asks, with a file in the middle: a turn resumed
    from disk is the turn that was never interrupted."""
    ids = [int(token) for token in golden["input_ids"]]
    trie = PromptCache[LayerCache](budget=1 << 30)
    grown = list(stream_ids(model, ids, max_tokens=4, prefix=trie))
    second = [*ids, *grown, *ids[:3]]
    stored = trie.take(second)
    assert stored is not None
    path = tmp_path / "entry.safetensors"
    cache_file.dump(stored.caches, path)

    from_disk = model.make_cache()
    cache_file.restore(from_disk, path)
    resumed = list(
        stream_ids(model, second, max_tokens=4, prefix=_Spilled(from_disk, stored.length))
    )

    assert resumed == list(stream_ids(model, second, max_tokens=4))


class _Spilled(PromptCache[LayerCache]):
    """A trie that holds nothing and answers one recall: the file already read. It stands in
    for the daemon's disk tier, which knows about directories, keys and a ceiling — none of
    which is what this test is about."""

    def __init__(self, caches: list[LayerCache], length: int) -> None:
        super().__init__(budget=0)
        self._read = (caches, length)

    def take(
        self, tokens: Sequence[int], *, into: list[LayerCache] | None = None
    ) -> Reuse[LayerCache] | None:
        caches, length = self._read
        return Reuse(caches, length)


def test_latent_cache_holds_the_latent(model: BailingHybrid, golden: dict[str, mx.array]) -> None:
    """The MLA cache holds the latent under one head — `kv_lora_rank` columns of it plus
    the shared rotary key — not the four expanded heads a decompressed port would keep."""
    ids = [int(token) for token in golden["input_ids"]]
    cache = model.make_cache()
    model(mx.array([ids]), cache)
    layer = cache[MLA_LAYER]
    assert isinstance(layer, KVCache)
    latent, rotary = layer.fetch()
    assert latent.shape == (1, 1, len(ids), 64)
    assert rotary.shape == (1, 1, len(ids), 16)


def test_kda_projections_fuse_into_one_matrix(model: BailingHybrid) -> None:
    """The six projections read the same input, so the loader concatenates them on the
    output axis and the tree declares only the fused leaf. Row-aligned, so the numerical
    tests above are what say it is bit-exact; the shape is what the one-token read wants."""
    config = model.config
    mixer = kda_of(model, KDA_LAYER)
    width, heads = config.linear_width, config.num_attention_heads
    assert mixer.fused_proj.weight.shape == (5 * width + heads, config.hidden_size)
    assert mixer.conv1d.weight.shape == (3 * width, config.short_conv_kernel_size)
    fused = ("q", "k", "v", "f", "g", "b")
    assert not any(f"{name}_proj" in mixer for name in fused)
    assert not any(f"{name}_conv1d" in mixer for name in ("q", "k", "v"))


def test_kda_cache_holds_one_window_over_the_three_channels(
    model: BailingHybrid, golden: dict[str, mx.array]
) -> None:
    """q, k and v share one conv window, `kernel - 1` rows of the concatenated channel
    axis — the shape the fused projection leaves them in."""
    ids = [int(token) for token in golden["input_ids"]]
    cache = model.make_cache()
    model(mx.array([ids]), cache)
    layer = cache[KDA_LAYER]
    assert isinstance(layer, DeltaCache)
    assert layer.window is not None and layer.window.shape == (1, 3, 3 * 128)
    assert layer.state is not None and layer.state.shape == (1, 4, 32, 32)


def test_per_channel_kernel_matches_the_ops_rule() -> None:
    """The `gated_delta` kernel's per-channel variant against the ops recurrence the
    facade falls back to — the decay indexes the key channel, not the head."""
    mx.random.seed(7)
    shape = (1, 6, 4, 32)
    q, k, v = (mx.random.normal(shape) for _ in range(3))
    decay = mx.sigmoid(mx.random.normal(shape))
    beta = mx.sigmoid(mx.random.normal((1, 6, 4)))
    state = mx.random.normal((1, 4, 32, 32))
    kernel_out, kernel_state = gated_delta(q, k, v, decay, beta, state)
    ops = GatedDelta(key_dim=32, key_heads=4, value_heads=4, value_dim=32, enabled=False)
    ops_out, ops_state = ops(q, k, v, decay, beta, state)
    assert relative_diff(kernel_out, ops_out) < 1e-5
    assert relative_diff(kernel_state, ops_state) < 1e-5


def test_per_head_decay_is_not_silently_accepted() -> None:
    """A `[B, T, H]` decay still takes the per-head kernel: the new variant must not have
    changed the shape the DeltaNet ports pass."""
    mx.random.seed(11)
    shape = (1, 4, 4, 32)
    q, k, v = (mx.random.normal(shape) for _ in range(3))
    per_head = mx.sigmoid(mx.random.normal((1, 4, 4)))
    beta = mx.sigmoid(mx.random.normal((1, 4, 4)))
    state = mx.zeros((1, 4, 32, 32))
    out, _ = gated_delta(q, k, v, per_head, beta, state)
    broadcast, _ = gated_delta(q, k, v, mx.repeat(per_head[..., None], 32, axis=-1), beta, state)
    assert relative_diff(out, broadcast) < 1e-6


MUTATIONS = (
    ("router bias", "model.layers.2.mlp.gate.expert_bias"),
    ("kda decay bias", "model.layers.0.attention.dt_bias"),
    ("conv taps", "model.layers.0.attention.k_conv1d.weight"),
    ("latent key half", "model.layers.5.attention.kv_b_proj.weight"),
    # Both land inside the fused projection, at different offsets: a fusion that
    # concatenated them in another order would move these two and nothing else.
    ("kda output gate", "model.layers.0.attention.g_proj.weight"),
    ("kda write strength", "model.layers.0.attention.b_proj.weight"),
)


@pytest.mark.parametrize(("name", "leaf"), MUTATIONS)
def test_mutation_is_caught(
    golden: dict[str, mx.array],
    tmp_path: Path,
    name: str,
    leaf: str,
) -> None:
    """Break one leaf, confirm the logits move past the bound, and let the next
    parametrization rebuild from the untouched fixture."""
    weights = {
        key.removeprefix("weight."): value
        for key, value in golden.items()
        if key.startswith("weight.")
    }
    weights[leaf] = weights[leaf] * 1.5 + 0.1
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    (tmp_path / "config.json").write_text(CONFIG.read_text())
    mutated = CHECKPOINT.load(tmp_path, None)
    logits = mutated(golden["input_ids"][None])
    assert relative_diff(logits, golden["logits"]) > floor(golden, "batching"), name


def rebuild(
    golden: dict[str, mx.array], directory: Path, **overrides: object
) -> BailingHybrid:
    """The fixture's weights under a patched config — the only way to reach a code path
    the tiny config does not turn on."""
    weights = {
        key.removeprefix("weight."): value
        for key, value in golden.items()
        if key.startswith("weight.")
    }
    mx.save_safetensors(str(directory / "model.safetensors"), weights)
    raw = json.loads(CONFIG.read_text()) | overrides
    (directory / "config.json").write_text(json.dumps(raw))
    return CHECKPOINT.load(directory, None)


# Small enough to bite on the fixture's activations; the limits Ling 3.0 ships (4, 5, 7)
# are calibrated to a 2560-wide model and clamp nothing at this scale.
BITING = 0.05
WIDE = 1e4


def test_the_limit_reaches_only_the_layers_that_declare_one(
    golden: dict[str, mx.array], tmp_path: Path
) -> None:
    model = rebuild(
        golden,
        tmp_path,
        expert_swiglu_limit_list=[0.0] * 11 + [BITING],
        share_expert_swiglu_limit_list=[0.0] * 10 + [BITING] + [0.0],
    )
    routed = [type(layer.mlp.switch_mlp) for layer in model.model.layers[2:]]
    shared = [type(layer.mlp.shared_experts) for layer in model.model.layers[2:]]
    assert routed == [SwitchGLU] * 9 + [LimitedSwitchGLU]
    assert shared == [SharedMLP] * 8 + [LimitedSharedMLP, SharedMLP]


def test_limited_activation_matches_the_explicit_form() -> None:
    """`silu(gate).clamp(max=L) * up.clamp(-L, L)` against the interleaved pair the
    load-time fusion actually hands the leaf."""
    mx.random.seed(3)
    inner, limit = 8, 0.7
    limited = LimitedSwitchGLU(2, 4, inner, limit)
    fused = mx.random.normal((5, 2 * inner))
    pairs = fused.reshape(5, inner, 2)
    gate, up = pairs[..., 0], pairs[..., 1]
    expected = mx.minimum(gate * mx.sigmoid(gate), limit) * mx.clip(up, -limit, limit)
    assert relative_diff(limited.activate(fused), expected) < 1e-6

    plain = SwitchGLU(2, 4, inner)
    wide = LimitedSwitchGLU(2, 4, inner, WIDE)
    assert relative_diff(wide.activate(fused), plain.activate(fused)) < 1e-6


def test_a_wide_limit_is_the_unclamped_path(
    golden: dict[str, mx.array], tmp_path: Path
) -> None:
    """The clamp is opt-in per layer and reduces to the plain SwiGLU above the data —
    the golden was generated without it."""
    model = rebuild(
        golden,
        tmp_path,
        expert_swiglu_limit_list=[WIDE] * N_LAYER,
        share_expert_swiglu_limit_list=[WIDE] * N_LAYER,
    )
    logits = model(golden["input_ids"][None])
    assert relative_diff(logits, golden["logits"]) < floor(golden, "batching")


def test_a_biting_limit_moves_the_logits(
    golden: dict[str, mx.array], tmp_path: Path
) -> None:
    """The mutation for this path: if the clamp were dropped on the way to the leaf, the
    logits would still match the unclamped golden."""
    model = rebuild(
        golden,
        tmp_path,
        expert_swiglu_limit_list=[0.0] * 2 + [BITING] * (N_LAYER - 2),
        share_expert_swiglu_limit_list=[0.0] * 2 + [BITING] * (N_LAYER - 2),
    )
    logits = model(golden["input_ids"][None])
    assert relative_diff(logits, golden["logits"]) > floor(golden, "batching")


def test_fixture_floors_are_measured(golden: dict[str, mx.array]) -> None:
    """Every floor came off the reference; none is zero, and none is a round number."""
    floors = [float(golden[f"noise.block_{layer}"]) for layer in range(N_LAYER)]
    floors.append(float(golden["noise.batching"]))
    assert all(0 < value < 1e-3 for value in floors)
    assert not np.allclose(floors, floors[0])
