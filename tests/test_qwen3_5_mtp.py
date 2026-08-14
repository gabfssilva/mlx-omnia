"""The MTP step of Qwen3.8, and the speculative round its hybrid trunk needs to run it.

Two things are being protected and they fail differently. The step's arithmetic — two norms
*before* the concat, the embedding half first, the end norm at the end, the tap on the
trunk's **normed** stream — is checked by mutation, because there is no golden to compare
against: transformers ignores `mtp.*` (`_keys_to_ignore_on_load_unexpected`) and nothing in
MLX implements this head, so the only authority is vllm's `qwen3_5_mtp.py` read as source.

The round is checked by equality, in **fp32**, on a trunk with DeltaNet layers in it: three
of every four caches cannot be trimmed, the round has to restore and replay, and a replay
that keeps one row too many or too few is a decode running off a sequence that never
happened. Random fp32 weights put no near-ties in the way, so `==` means what it says.
"""

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten, tree_unflatten

import mlx_omnia.engine.task as task
from mlx_omnia import CompositeModel, load
from mlx_omnia.engine.core.cache import DeltaCache, KVCache
from mlx_omnia.engine.generate import stream_ids
from mlx_omnia.engine.models.qwen3_5.config import (
    Qwen35Config,
    Qwen35RoPEParameters,
    Qwen35TextConfig,
)
from mlx_omnia.engine.models.qwen3_5.model import Qwen35
from mlx_omnia.engine.models.qwen3_5.mtp import Qwen35MTP, Qwen35MTPBlock
from mlx_omnia.engine.quant.quantization import Affine
from mlx_omnia.engine.speculative import (
    Acceptance,
    Chained,
    Persistent,
    stream_speculative_ids,
)

PROMPT = [3, 1, 2, 1, 0, 3, 2, 1]

# Deliberately tiny. An untrained head proposes at chance, and the test needs the accepted
# branch exercised as much as the rejected one: on four ids, chance alone puts the odds of
# a run with zero acceptances at 0.75 ** 39, which is not a coin this test flips.
VOCAB = 4


def _config(*, experts: int = 0) -> Qwen35Config:
    inner = 32
    return Qwen35Config(
        text_config=Qwen35TextConfig(
            hidden_size=64,
            num_hidden_layers=4,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=32,
            vocab_size=VOCAB,
            rms_norm_eps=1e-6,
            layer_types=(
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ),
            linear_num_key_heads=2,
            linear_num_value_heads=2,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            linear_conv_kernel_dim=4,
            rope_parameters=Qwen35RoPEParameters(
                rope_theta=10000.0, partial_rotary_factor=0.25, mrope_section=(4, 2, 2)
            ),
            eos_token_id=None,
            tie_word_embeddings=False,
            intermediate_size=inner if not experts else 0,
            num_experts=experts,
            num_experts_per_tok=2 if experts else 0,
            moe_intermediate_size=inner if experts else 0,
            shared_expert_intermediate_size=inner if experts else 0,
        ),
    )


_NORMS = ("norm.weight", "input_layernorm.weight", "post_attention_layernorm.weight")
_DELTA = ("conv1d.weight", "dt_bias")


def _spread(model: object, seed: int) -> None:
    """Two families of leaf are born constant and mean nothing until they are moved.

    Every **RMSNorm** weight is one, which makes the norm of an already-normed vector the
    identity — and that would quietly pass a test claiming the step's rows are not normed
    a second time. The **DeltaNet** conv kernel and `dt_bias` are zero, and the conv is
    the one that matters: a zero kernel makes the mixer's input zero, so the recurrent
    state stays zero forever and every test of a rewind compares nothing to nothing.
    `A_log` keeps its zeros: the decay is `-exp(A_log)` and a random one there saturates
    the recurrence instead of exercising it. The projections keep their own random init,
    which is what keeps the greedy stream from collapsing to a constant token — an
    untrained head only exercises the accepted branch when the target's argmax moves.
    """
    assert isinstance(model, nn.Module)
    mx.random.seed(seed)
    moved: list[tuple[str, mx.array]] = []
    for path, leaf in tree_flatten(model.parameters()):
        if not isinstance(leaf, mx.array):
            continue
        if path.endswith(_NORMS) or "norm" in path.rsplit(".", 2)[-2]:
            moved.append((path, mx.ones(leaf.shape) + mx.random.normal(leaf.shape) * 0.3))
        elif path.endswith(_DELTA):
            moved.append((path, mx.random.normal(leaf.shape) * 0.3))
    model.update(tree_unflatten(moved))


@pytest.fixture
def trunk() -> Qwen35:
    model = Qwen35(_config())
    _spread(model, 11)
    mx.eval(model.parameters())
    return model


@pytest.fixture
def step(trunk: Qwen35) -> Qwen35MTP:
    head = Qwen35MTP(trunk.config.text_config)
    _spread(head, 23)
    mx.eval(head.parameters())
    return head


def _relative(a: mx.array, b: mx.array) -> float:
    a32, b32 = a.astype(mx.float32), b.astype(mx.float32)
    return float((mx.max(mx.abs(a32 - b32)) / mx.max(mx.abs(b32))).item())


def test_the_trunk_needs_both_rewinds(trunk: Qwen35) -> None:
    """What makes this family the hard case, stated as an assertion so it cannot quietly
    stop being true: three caches in four are DeltaNet, and none of those can trim."""
    cache = trunk.make_cache()
    assert [type(layer) for layer in cache] == [DeltaCache, DeltaCache, DeltaCache, KVCache]
    assert not all(layer.is_trimmable for layer in cache)
    assert all(layer.is_replayable for layer in cache)


@pytest.mark.parametrize("kept", [0, 1, 2])
def test_verify_rewinds_the_recurrent_state_to_the_rows_that_were_kept(
    trunk: Qwen35, kept: int
) -> None:
    """What the ids cannot see. A rewind that keeps a rejected row, or forgets to replay,
    or moves the wrong offset, leaves a cache that still decodes fluently — so the state is
    compared directly, against a run that only ever saw the rows that were kept."""
    prefix, rows = mx.array([PROMPT]), mx.array([[1, 2, 3]])

    cache = trunk.make_cache()
    trunk(prefix, cache)
    _, _, rewind = trunk.verify(rows, cache, at=(-1,))
    rewind(kept)

    stepwise = trunk.make_cache()
    trunk(prefix, stepwise)
    if kept:
        trunk(rows[:, :kept], stepwise)
    whole = trunk.make_cache()
    trunk(mx.concatenate([prefix, rows[:, :kept]], axis=1), whole)

    assert [layer.offset for layer in cache] == [layer.offset for layer in stepwise]
    kinds = trunk.config.text_config.layer_types
    recurrent = [index for index, kind in enumerate(kinds) if kind == "linear_attention"]
    assert recurrent, "the point of this trunk is the layers that cannot trim"
    for index in recurrent:
        got, want, other = cache[index], stepwise[index], whole[index]
        assert isinstance(got, DeltaCache)
        assert isinstance(want, DeltaCache) and isinstance(other, DeltaCache)
        assert got.state is not None and want.state is not None and other.state is not None
        floor = _relative(want.state, other.state)
        assert _relative(got.state, want.state) <= max(3 * floor, 1e-6)
        assert got.window is not None and want.window is not None
        assert _relative(got.window, want.window) <= 1e-6


def test_the_step_fuses_before_it_concatenates(trunk: Qwen35, step: Qwen35MTP) -> None:
    """The arithmetic vllm spells (`qwen3_5_mtp.py:152-155`), reproduced by hand: the two
    halves normed separately, the embedding first, the projection over the pair. Written
    out rather than asserted against a fixture because there is no fixture to have — this
    *is* the reference, and what defends it is the mutation list below it."""
    rows = mx.random.normal((1, 5, 64))
    embedded = trunk.raw_embed(mx.array(PROMPT[:5])[None])

    fused = step.fuse(embedded, rows)
    assert mx.allclose(
        fused,
        step.fc(
            mx.concatenate(
                [step.pre_fc_norm_embedding(embedded), step.pre_fc_norm_hidden(rows)], axis=-1
            )
        ),
    )

    # Swapped halves: same two vectors, each into the other's columns of `fc`.
    swapped = step.fc(
        mx.concatenate(
            [step.pre_fc_norm_hidden(rows), step.pre_fc_norm_embedding(embedded)], axis=-1
        )
    )
    assert not mx.allclose(fused, swapped)

    # Normed *after* the concat instead of before, which mixes the two statistics into one.
    together = mx.fast.rms_norm(mx.concatenate([embedded, rows], axis=-1), None, 1e-6)
    assert not mx.allclose(fused, step.fc(together))


def test_the_step_is_fuse_then_block_then_its_own_norm(trunk: Qwen35, step: Qwen35MTP) -> None:
    """The whole step written out: the fusion into the block, the end norm on its output.
    Reproducing it by hand is what pins the order — a step that dropped `norm`, or applied
    it in the middle, still returns rows of the right shape.

    And `model.norm` is *not* applied on top: what comes back goes straight to
    `raw_logits`. Normalizing twice is a different model."""
    ids = mx.array(PROMPT[:3])[None]
    rows = mx.random.normal((1, 3, 64))
    embedded = trunk.raw_embed(ids)
    (block,) = step.layers
    assert isinstance(block, Qwen35MTPBlock)

    by_hand = block(step.fuse(embedded, rows), step.make_cache()[0])
    out = step(embedded, rows, step.make_cache())

    assert mx.allclose(out, step.norm(by_hand))
    assert not mx.allclose(out, by_hand), "the end norm did nothing"
    assert not mx.allclose(out, trunk.model.norm(out)), "normed a second time"


def test_the_tap_is_the_normed_stream(trunk: Qwen35) -> None:
    """The `hidden` half of the pair is what the trunk's forward *returns* — vllm's runner
    hands the step `norm(hidden)` (`gpu_model_runner.py:5328`), not the last block's raw
    output. A tap on the raw block is fluent, plausible, and a different model."""
    ids = mx.array([PROMPT])
    activations = trunk.activations(ids, trunk.make_cache())
    _, features = trunk.block_outputs(ids, trunk.make_cache(), at=(-1,))
    assert mx.allclose(features, activations.norm)
    assert not mx.allclose(features, activations.blocks[-1])


def test_a_sparse_head_is_refused_by_name(trunk: Qwen35) -> None:
    with pytest.raises(ValueError, match="not ported"):
        Qwen35MTP(_config(experts=4).text_config)


def test_speculation_reproduces_the_greedy_stream(trunk: Qwen35, step: Qwen35MTP) -> None:
    """The whole claim, on a trunk whose cache cannot be trimmed.

    fp32 and random weights, so `==` is the test and not a tolerance. `Acceptance.rate` is
    asserted above zero in the same run, because equality passes for free when every
    proposal is rejected — and rejection is exactly the path that restores and replays."""
    plain = list(stream_ids(trunk, PROMPT, max_tokens=40))

    for block in (2, 3, 4):
        counts = Acceptance()
        drafted = list(
            stream_speculative_ids(
                trunk,
                Chained(trunk, step, block=block, tap=-1),
                PROMPT,
                max_tokens=40,
                acceptance=counts,
            )
        )
        assert drafted == plain, f"block {block}"
        assert counts.rounds > 0
        assert counts.rate is not None and counts.rate > 0.0, f"block {block} never accepted"


def _wide_config() -> Qwen35Config:
    """The small config again, with the delta head widened to 32 so the fused kernel
    tiles it (`gated_delta_applies` wants `Dk % 32 == 0`) — the compiled verify refuses
    a trunk whose recurrence runs on the ops path, so these are the shapes the compiled
    tests must wear."""
    base = _config().text_config
    from dataclasses import replace

    return Qwen35Config(text_config=replace(base, linear_key_head_dim=32))


@pytest.fixture
def wide_trunk() -> Qwen35:
    model = Qwen35(_wide_config())
    _spread(model, 11)
    mx.eval(model.parameters())
    return model


@pytest.fixture
def wide_step(wide_trunk: Qwen35) -> Qwen35MTP:
    head = Qwen35MTP(wide_trunk.config.text_config)
    _spread(head, 23)
    mx.eval(head.parameters())
    return head


class _Eager:
    """A `speculative.Step` shim that hides `compile_chain`, so a `Chained` over it walks
    the eager path — the reference the compiled chain is compared against."""

    def __init__(self, inner: Qwen35MTP) -> None:
        self.inner = inner
        self.block = inner.block

    def make_cache(self) -> list[KVCache]:
        return self.inner.make_cache()

    def __call__(
        self, embeddings: mx.array, hidden: mx.array, cache: list[KVCache] | None = None
    ) -> mx.array:
        return self.inner(embeddings, hidden, cache)


def test_the_compiled_chain_matches_the_eager_proposal(
    trunk: Qwen35, step: Qwen35MTP
) -> None:
    """The chain's compiled graph against the same chain walked eagerly: same features
    absorbed, same committed ids, same proposals. fp32 and random weights, so an argmax
    that moved is arithmetic that moved."""
    features = mx.random.normal((1, 4, 64))
    committed = [*PROMPT, 2]

    compiled = Chained(trunk, step, block=4, tap=-1)
    compiled.absorb(features)
    eager = Chained(trunk, _Eager(step), block=4, tap=-1)
    eager.absorb(features)

    first = compiled.propose(committed)
    assert mx.array_equal(first, eager.propose(committed))
    # A second round must start from a reset buffer, not attend the first round's rows.
    assert mx.array_equal(compiled.propose(committed), eager.propose(committed))


@pytest.mark.parametrize("kept", [1, 2])
@pytest.mark.parametrize("trace", [True, False], ids=["trace-kernel", "unrolled"])
def test_compiled_verify_rewinds_to_the_rows_that_were_kept(
    wide_trunk: Qwen35, kept: int, trace: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The slot-pick rewind against a run that only ever saw the kept rows: window,
    state, and the graph-visible attention position. Both delta routes take the same
    walk through the same f32 buffers, so both are held to the same floor."""
    from mlx_omnia.engine.core.cache import FixedDeltaCache, FixedKVCache
    from mlx_omnia.engine.models.qwen3_5.layers import flags

    monkeypatch.setattr(flags, "VERIFY_TRACE_KERNEL", trace)
    prefix, rows = mx.array([PROMPT]), mx.array([1, 2, 3])

    cache = wide_trunk.make_cache()
    wide_trunk(prefix, cache)
    compiled = wide_trunk.compile_verify(cache, width=2, at=(-1,))
    assert compiled is not None, "the wide shapes are exactly what the kernel tiles"
    verify, rewind = compiled
    verify(rows)
    rewind(kept)

    stepwise = wide_trunk.make_cache()
    wide_trunk(prefix, stepwise)
    wide_trunk(rows[None, :kept], stepwise)
    whole = wide_trunk.make_cache()
    wide_trunk(mx.concatenate([prefix, rows[None, :kept]], axis=1), whole)

    kinds = wide_trunk.config.text_config.layer_types
    for index, kind in enumerate(kinds):
        got, want, other = cache[index], stepwise[index], whole[index]
        assert got.offset == want.offset
        if kind == "linear_attention":
            assert isinstance(got, FixedDeltaCache)
            assert isinstance(want, DeltaCache) and isinstance(other, DeltaCache)
            assert want.state is not None and other.state is not None
            state, window = got.graph[1], got.graph[0]
            floor = _relative(want.state, other.state)
            assert _relative(state, want.state) <= max(3 * floor, 1e-6)
            assert want.window is not None
            assert _relative(window, want.window) <= 1e-6
        else:
            assert isinstance(got, FixedKVCache)
            assert got.rows == want.offset


@pytest.mark.parametrize("trace", [True, False], ids=["trace-kernel", "unrolled"])
def test_the_compiled_verify_forward_is_the_eager_forward(
    wide_trunk: Qwen35, trace: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fixed-shape graph against the growing eager forward over the same rows, at the
    house floor. The stream test alone cannot hold this: on four ids an argmax survives a
    wrong mask (measured: the off-by-one moves the logits 0.11 and flips nothing), so the
    logits are compared directly — which is what a verify *is*."""
    from mlx_omnia.engine.models.qwen3_5.layers import flags

    monkeypatch.setattr(flags, "VERIFY_TRACE_KERNEL", trace)
    prefix, rows = mx.array([PROMPT]), mx.array([1, 2, 3])

    reference = wide_trunk.make_cache()
    wide_trunk(prefix, reference)
    expected = wide_trunk(rows[None], reference)[0]

    cache = wide_trunk.make_cache()
    wide_trunk(prefix, cache)
    compiled = wide_trunk.compile_verify(cache, width=2, at=(-1,))
    assert compiled is not None
    verify, _ = compiled
    logits, features = verify(rows)

    assert _relative(logits, expected) < 1e-5
    assert features.shape == (1, 3, 64)


@pytest.mark.parametrize("trace", [True, False], ids=["trace-kernel", "unrolled"])
def test_compiled_speculation_reproduces_the_greedy_stream(
    wide_trunk: Qwen35, wide_step: Qwen35MTP, trace: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole compiled round — chain, fixed-shape verify, slot-pick rewind — emits the
    plain greedy stream, in fp32 where `==` means what it says, on both delta routes."""
    from mlx_omnia.engine.models.qwen3_5.layers import flags

    monkeypatch.setattr(flags, "VERIFY_TRACE_KERNEL", trace)
    plain = list(stream_ids(wide_trunk, PROMPT, max_tokens=40))

    for block in (2, 3):
        counts = Acceptance()
        drafted = list(
            stream_speculative_ids(
                wide_trunk,
                Chained(wide_trunk, wide_step, block=block, tap=-1),
                PROMPT,
                max_tokens=40,
                acceptance=counts,
            )
        )
        assert drafted == plain, f"block {block}"
        assert counts.rounds > 0
        assert counts.rate is not None and counts.rate > 0.0, f"block {block} never accepted"


def test_persistent_speculation_reproduces_the_greedy_stream(
    trunk: Qwen35, step: Qwen35MTP
) -> None:
    """The persistent proposer under the replay round: same claim as `Chained`'s, on the
    trunk whose cache cannot trim — the drafter's own cache *can*, which is the whole of
    what `Persistent` asks for."""
    plain = list(stream_ids(trunk, PROMPT, max_tokens=40))

    for block in (2, 3, 4):
        counts = Acceptance()
        drafted = list(
            stream_speculative_ids(
                trunk,
                Persistent(trunk, step, block=block, tap=-1),
                PROMPT,
                max_tokens=40,
                acceptance=counts,
            )
        )
        assert drafted == plain, f"block {block}"
        assert counts.rounds > 0
        assert counts.rate is not None and counts.rate > 0.0, f"block {block} never accepted"


@pytest.mark.parametrize("trace", [True, False], ids=["trace-kernel", "unrolled"])
def test_persistent_speculation_survives_the_compiled_round(
    wide_trunk: Qwen35, wide_step: Qwen35MTP, trace: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same claim under `_compiled_round`, whose absorb/settle cadence is the one the
    real model runs: features arrive already cut to the kept rows, and the rejected tail
    must never become a pair."""
    from mlx_omnia.engine.models.qwen3_5.layers import flags

    monkeypatch.setattr(flags, "VERIFY_TRACE_KERNEL", trace)
    plain = list(stream_ids(wide_trunk, PROMPT, max_tokens=40))

    counts = Acceptance()
    drafted = list(
        stream_speculative_ids(
            wide_trunk,
            Persistent(wide_trunk, wide_step, block=3, tap=-1),
            PROMPT,
            max_tokens=40,
            acceptance=counts,
        )
    )
    assert drafted == plain
    assert counts.rounds > 0
    assert counts.rate is not None and counts.rate > 0.0


def test_the_persistent_catchup_is_the_scratch_walk(trunk: Qwen35, step: Qwen35MTP) -> None:
    """The incremental cache against a proposer handed the whole history from scratch.

    The lossless stream cannot hold this — a shifted pair or a trim that keeps a guessed
    row only moves *acceptance*, never a token — so the proposals themselves are pinned:
    at every round, catching up over the cached rows must propose exactly what a fresh
    cache walking all pairs proposes. Rounds advance without accepting anything, which is
    the cadence that exercises trim-past-the-guesses hardest."""
    cache = trunk.make_cache()
    logits, features = trunk.block_outputs(mx.array([PROMPT]), cache, at=(-1,))
    committed = [*PROMPT, int(mx.argmax(logits[:, -1, :], axis=-1).item())]

    incremental = Persistent(trunk, step, block=3, tap=-1)
    incremental.absorb(features)
    history = features

    for _ in range(4):
        proposed = incremental.propose(committed)

        scratch = Persistent(trunk, step, block=3, tap=-1)
        scratch.absorb(history)
        assert mx.array_equal(proposed, scratch.propose(committed))
        expected = len(committed) - 1 + scratch.width - 1
        assert incremental._cache[0].offset == expected, "the chain never appended"

        # The target advances one committed token — a round that accepted nothing.
        logits, row = trunk.block_outputs(mx.array([committed[-1:]]), cache, at=(-1,))
        committed.append(int(mx.argmax(logits[:, -1, :], axis=-1).item()))
        incremental.absorb(row)
        incremental.settle(len(committed))
        assert incremental._cache[0].offset == len(committed) - 2, "a guessed row survived"
        history = mx.concatenate([history, row], axis=1)


def test_the_persistent_pairs_are_the_chained_pairs(trunk: Qwen35, step: Qwen35MTP) -> None:
    """The alignment `Persistent` cannot pin by itself: which token pairs with which
    hidden. With exactly one position of history the two proposers see the same pairs in
    the same empty cache — absolute rotation *is* relative rotation there — so their
    proposals must be identical, and `Chained`'s alignment is the one the real-weights
    acceptance was measured on. A pair shifted by one diverges here and nowhere else."""
    ids = mx.array([[PROMPT[0]]])
    logits, features = trunk.block_outputs(ids, trunk.make_cache(), at=(-1,))
    committed = [PROMPT[0], int(mx.argmax(logits[:, -1, :], axis=-1).item())]

    chained = Chained(trunk, step, block=3, tap=-1)
    chained.absorb(features)
    persistent = Persistent(trunk, step, block=3, tap=-1)
    persistent.absorb(features)

    assert mx.array_equal(chained.propose(committed), persistent.propose(committed))


def test_the_ops_trunk_refuses_the_compiled_verify(trunk: Qwen35) -> None:
    """Dk=16 is a shape the fused kernel does not tile: the recurrence runs on the ops
    strategy and the compiled verify's checkpoints are the kernel's, so the answer is the
    replay round — `None`, not a wrong number."""
    cache = trunk.make_cache()
    trunk(mx.array([PROMPT]), cache)
    assert trunk.compile_verify(cache, width=2, at=(-1,)) is None


# The four bytes `VOCAB` allows, in the byte-level alphabet `ByteLevelBPE` decodes through.
_TOKENIZER = {
    "added_tokens": [],
    "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False},
    "model": {"vocab": {"!": 0, '"': 1, "#": 2, "$": 3}, "merges": []},
}


def _text_config_json(config: Qwen35TextConfig) -> dict[str, object]:
    written = dict(vars(config))
    written["layer_types"] = list(config.layer_types)
    written["rope_parameters"] = {
        "rope_theta": config.rope_parameters.rope_theta,
        "partial_rotary_factor": config.rope_parameters.partial_rotary_factor,
        "mrope_section": list(config.rope_parameters.mrope_section),
    }
    return written


def _checkpoint(directory: Path, trunk: Qwen35, step: Qwen35MTP | None) -> None:
    """A source checkpoint on disk in the mlx dialect: house names, the conv widened back
    to `[dim, K, 1]` (which is also what marks the norms as already shifted), the head
    under `mtp.` when there is one, and the least tokenizer that loads."""
    tensors = dict(tree_flatten(trunk.parameters()))
    if step is not None:
        tensors |= {f"mtp.{path}": leaf for path, leaf in tree_flatten(step.parameters())}
    for name, value in tensors.items():
        if name.endswith("conv1d.weight"):
            tensors[name] = value[:, :, None]
    mx.save_safetensors(str(directory / "model.safetensors"), tensors)
    config = {
        "model_type": "qwen3_5",
        "text_config": _text_config_json(trunk.config.text_config),
        "tie_word_embeddings": False,
        "eos_token_id": 0,
    }
    (directory / "config.json").write_text(json.dumps(config))
    (directory / "tokenizer.json").write_text(json.dumps(_TOKENIZER))


def test_the_raw_dialect_head_gets_its_norms_shifted(
    trunk: Qwen35, step: Qwen35MTP, tmp_path: Path
) -> None:
    """The one dialect that actually ships the head is raw HF — the mlx conversions drop
    `mtp.*` — and there every norm is still zero-centered (`Qwen3_5RMSNorm` is
    `GemmaRMSNorm`, scale `1 + w`). The probe is the trunk's conv layout, so the head's
    fold rides the same detection the trunk's does."""
    from mlx_omnia.engine.models.qwen3_5.checkpoint import mtp_weights

    tensors = {f"mtp.{path}": leaf for path, leaf in tree_flatten(step.parameters())}
    conv_dim = trunk.config.text_config.conv_dim
    tensors["model.language_model.layers.0.linear_attn.conv1d.weight"] = mx.zeros(
        (conv_dim, 1, 4)
    )
    mx.save_safetensors(str(tmp_path / "model.safetensors"), tensors)

    loaded = mtp_weights(tmp_path, trunk.config)

    assert mx.allclose(loaded["norm.weight"], step.norm.weight + 1)
    assert mx.allclose(
        loaded["pre_fc_norm_embedding.weight"], step.pre_fc_norm_embedding.weight + 1
    )
    assert mx.allclose(loaded["pre_fc_norm_hidden.weight"], step.pre_fc_norm_hidden.weight + 1)
    block = step.layers[0]
    assert isinstance(block, Qwen35MTPBlock)
    assert mx.allclose(
        loaded["layers.0.self_attn.q_norm.weight"], block.self_attn.q_norm.weight + 1
    )
    assert mx.allclose(loaded["layers.0.input_layernorm.weight"], block.input_layernorm.weight + 1)
    # The projections are weights, not norms: the shift never touches them.
    assert mx.allclose(loaded["fc.weight"], step.fc.weight)


def test_quantizing_keeps_the_head_instead_of_deleting_it(
    trunk: Qwen35, step: Qwen35MTP, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Four things have to hold together, and any one alone is an entry that cannot
    speculate: the tensors are in the file, the plan declares them, the trunk still loads
    *without* them, and the head loads back out."""
    monkeypatch.setattr(task, "_CACHE", tmp_path / "quantized")
    source = tmp_path / "source"
    source.mkdir()
    _checkpoint(source, trunk, step)

    load(source, quantize=Affine(group_size=32, bits=4))

    (entry,) = [path for path in (tmp_path / "quantized").glob("*/*") if path.is_dir()]
    packed = mx.load(str(entry / "model.safetensors"))
    assert [name for name in packed if name.startswith("mtp.")], "the entry lost the head"

    declared = json.loads((entry / "config.json").read_text())["quantization"]["leaves"]
    assert [leaf for leaf in declared if leaf.startswith("mtp.")], "the plan forgot the head"

    assert isinstance(load(entry), CompositeModel)
    assert task.has_mtp(entry)

    rebuilt = task.mtp_head(entry)
    assert isinstance(rebuilt, Qwen35MTP)
    rows = mx.random.normal((1, 2, 64))
    assert mx.all(mx.isfinite(rebuilt(rows, rows))).item()


def test_an_entry_written_without_a_head_says_so(
    trunk: Qwen35, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A checkpoint of the same architecture that ships no head — every mlx-community
    conversion — quantizes as it always did, and `has_mtp` answers for the file rather
    than for the family."""
    monkeypatch.setattr(task, "_CACHE", tmp_path / "quantized")
    source = tmp_path / "headless"
    source.mkdir()
    _checkpoint(source, trunk, None)

    load(source, quantize=Affine(group_size=32, bits=4))

    (entry,) = [path for path in (tmp_path / "quantized").glob("*/*") if path.is_dir()]
    assert not task.has_mtp(entry)
    with pytest.raises(ValueError, match="no MTP head"):
        task.mtp_head(entry)
