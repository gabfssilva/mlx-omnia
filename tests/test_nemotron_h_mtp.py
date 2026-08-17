"""The MTP step of Nemotron-H, and the speculative round a hybrid trunk needs to run it.

Two things are being protected here and they fail differently. The step's arithmetic — two
norms *before* the concat, the embedding half first, the end norm at the end — is checked by
mutation, because there is no golden to compare against: transformers ignores `mtp.*` and
nothing in MLX implements this head, so the only authority is vllm's
`nemotron_h_mtp.py` read as source.

The round is checked by equality, in **fp32**, on a trunk with mamba layers in it. That is
the point of building a small one instead of reaching for the 65 GB checkpoint: 23 of the
real trunk's 52 layers are recurrent, and their cache cannot be trimmed — the round has to
restore and replay, and a replay that keeps one row too many or one too few is a decode
running off a sequence that never happened. In bf16 that claim is untestable by equality,
because the checkpoint's own stepwise decode disagrees with its own batched forward at about
one position in eighty-five, always on two logits a ULP apart. Here the weights are random
and the width is fp32, so a tie has measure zero and `==` means what it says.
"""

import dataclasses
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten, tree_unflatten

import mlx_omnia.engine.task as task
from mlx_omnia import CompositeModel, load
from mlx_omnia.engine.core.cache import DeltaCache, FixedDeltaCache, KVCache
from mlx_omnia.engine.core.decode import compiled_verify
from mlx_omnia.engine.generate import stream_ids
from mlx_omnia.engine.models.nemotron_h.config import NemotronHConfig
from mlx_omnia.engine.models.nemotron_h.model import NemotronH
from mlx_omnia.engine.models.nemotron_h.mtp import NemotronHMTP
from mlx_omnia.engine.quant.quantization import Affine
from mlx_omnia.engine.speculative import (
    Acceptance,
    Chained,
    SpeculationRefused,
    stream_speculative_ids,
)
from tests.conftest import relative_diff

PROMPT = [3, 1, 2, 1, 0, 3, 2, 1]

# Deliberately tiny. An untrained head proposes at chance, and the test needs the accepted
# branch exercised as much as the rejected one: on four ids, chance alone puts the odds of a
# run with zero acceptances at 0.75 ** 39, which is not a coin this test flips.
VOCAB = 4


def config_of(pattern: str) -> NemotronHConfig:
    return NemotronHConfig(
        hidden_size=64,
        num_hidden_layers=len(pattern),
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=VOCAB,
        layer_norm_epsilon=1e-5,
        intermediate_size=128,
        mamba_num_heads=4,
        mamba_head_dim=16,
        ssm_state_size=16,
        conv_kernel=4,
        n_groups=2,
        # Below the sequences here, so the chunked scan runs rather than only the step.
        chunk_size=4,
        moe_intermediate_size=32,
        moe_shared_expert_intermediate_size=32,
        n_routed_experts=4,
        num_experts_per_tok=2,
        hybrid_override_pattern=pattern,
    )


_NORMS = ("norm.weight", "norm_f.weight", "enorm.weight", "hnorm.weight", "final_layernorm.weight")
_ROUTER = ("gate.weight", "e_score_correction_bias")
_MAMBA = ("conv1d.weight", "A_log", "D", "dt_bias")


def spread(model: nn.Module) -> None:
    """Three families of leaf are born constant and mean nothing until they are moved.

    The **router**'s weight and correction bias are zero, which makes every expert tie and the
    routing an artefact of `argpartition`'s order. Random ones give the gate something to
    separate, and a routing that varies by token is what makes the sparse block real.

    Every **RMSNorm** weight is one, which makes the norm of an already-normed vector the
    identity — and that would quietly pass a test claiming the step's rows are not normed a
    second time.

    The **mamba** conv kernel, `A_log`, `D` and `dt_bias` are zero, and the conv is the one
    that matters: a zero kernel makes the mixer's input zero, so the recurrent state stays
    zero forever and every test of a rewind compares nothing to nothing. Found that way.
    """
    moved: list[tuple[str, mx.array]] = []
    for path, leaf in tree_flatten(model.parameters()):
        if not isinstance(leaf, mx.array):
            continue
        if path.endswith(_ROUTER):
            moved.append((path, mx.random.normal(leaf.shape) * 0.5))
        elif path.endswith(_NORMS):
            moved.append((path, mx.ones(leaf.shape) + mx.random.normal(leaf.shape) * 0.3))
        elif path.endswith(_MAMBA):
            moved.append((path, mx.random.normal(leaf.shape) * 0.3))
    model.update(tree_unflatten(moved))


@pytest.fixture
def trunk() -> NemotronH:
    mx.random.seed(11)
    model = NemotronH(config_of("M*EM*E"))
    spread(model)
    mx.eval(model.parameters())
    return model


@pytest.fixture
def step(trunk: NemotronH) -> NemotronHMTP:
    mx.random.seed(23)
    head = NemotronHMTP(trunk.config, ("*", "E"))
    spread(head)
    mx.eval(head.parameters())
    return head


def test_the_trunk_needs_both_rewinds(trunk: NemotronH) -> None:
    """What makes this family the hard case, stated as an assertion so it cannot quietly
    stop being true: the cache is three kinds, and only one of them can trim."""
    cache = trunk.make_cache()
    assert [type(layer) for layer in cache[:3]] == [DeltaCache, KVCache, type(cache[2])]
    assert not all(layer.is_trimmable for layer in cache)
    assert all(layer.is_replayable for layer in cache)


@pytest.mark.parametrize("kept", [1, 2])
def test_the_round_rewinds_the_recurrent_state_to_the_rows_that_were_kept(kept: int) -> None:
    """What the ids cannot see. A rewind that keeps a rejected row, or picks the wrong slot,
    or moves the wrong offset, leaves a cache that still decodes fluently — the argmax
    survives a wrong state for a long time, and on four ids it survives it almost always. So
    the state is compared directly, against a run that only ever saw the rows that were kept.

    The floor is measured in the same test rather than chosen: the reference is built twice,
    once as two calls and once as one, and whatever the chunked scan does differently between
    those groupings is the noise a real difference has to clear.

    `ssm_state_size=32` because the per-row checkpoints are the fused step's, and the fused
    step is what `NemotronH.checkpoints` requires — at 16 the round replays instead, which
    the greedy-stream tests below cover.
    """
    mx.random.seed(11)
    trunk = NemotronH(dataclasses.replace(config_of("M*EM*E"), ssm_state_size=32))
    spread(trunk)
    mx.eval(trunk.parameters())

    prefix, rows = mx.array([PROMPT]), mx.array([1, 2, 3])
    tap = len(trunk.config.pattern) - 1

    cache = trunk.make_cache()
    trunk(prefix, cache)
    compiled = compiled_verify(trunk, cache, rows=3, taps=(tap,))
    assert compiled is not None, "the fused step is what this trunk was built to reach"
    verify, rewind = compiled
    verify(rows)
    rewind(kept, len(PROMPT) + kept)

    stepwise = trunk.make_cache()
    trunk(prefix, stepwise)
    trunk(rows[None, :kept], stepwise)
    whole = trunk.make_cache()
    trunk(mx.concatenate([prefix, rows[None, :kept]], axis=1), whole)

    assert [layer.offset for layer in cache] == [layer.offset for layer in stepwise]
    recurrent = [i for i, kind in enumerate(trunk.config.pattern) if kind == "M"]
    assert recurrent, "the point of this trunk is the layers that cannot trim"
    for index in recurrent:
        got, want, other = cache[index], stepwise[index], whole[index]
        assert isinstance(got, FixedDeltaCache)
        assert isinstance(want, DeltaCache) and isinstance(other, DeltaCache)
        assert want.state is not None and other.state is not None
        floor = relative_diff(want.state, other.state)
        assert relative_diff(got.graph[1], want.state) <= max(3 * floor, 1e-6)
        assert want.window is not None
        assert relative_diff(got.graph[0], want.window) <= 1e-6


def test_the_step_fuses_before_it_concatenates(trunk: NemotronH, step: NemotronHMTP) -> None:
    """The arithmetic vllm spells (`nemotron_h_mtp.py:99-106`), reproduced by hand: the two
    halves normed separately, the embedding first, the projection over the pair. Written out
    rather than asserted against a fixture because there is no fixture to have — this *is*
    the reference, and what defends it is the mutation list below it."""
    rows = mx.random.normal((1, 5, trunk.config.hidden_size))
    embedded = trunk.raw_embed(mx.array(PROMPT[:5])[None])
    first = step.layers[0]

    fused = first.fuse(embedded, rows)
    assert mx.allclose(
        fused, first.eh_proj(mx.concatenate([first.enorm(embedded), first.hnorm(rows)], axis=-1))
    )

    # Swapped halves: same two vectors, each into the other's columns of `eh_proj`.
    swapped = first.eh_proj(mx.concatenate([first.hnorm(rows), first.enorm(embedded)], axis=-1))
    assert not mx.allclose(fused, swapped)

    # Normed *after* the concat instead of before, which mixes the two statistics into one.
    together = mx.fast.rms_norm(
        mx.concatenate([embedded, rows], axis=-1), None, trunk.config.layer_norm_epsilon
    )
    assert not mx.allclose(fused, first.eh_proj(together))


def test_the_step_is_fuse_then_blocks_then_its_own_norm(
    trunk: NemotronH, step: NemotronHMTP
) -> None:
    """The whole step written out: the fusion into the first block, the blocks composed, the
    end norm on the last. Reproducing it by hand is what pins the order — a step that dropped
    `final_layernorm`, or applied it in the middle, still returns rows of the right shape.

    And `norm_f` is *not* applied on top: what comes back goes straight to `raw_logits`.
    Normalizing twice is a different model, and it is the mistake a reader who knows the
    trunk would make.
    """
    ids = mx.array(PROMPT[:3])[None]
    rows = mx.random.normal((1, 3, trunk.config.hidden_size))
    embedded = trunk.raw_embed(ids)
    first, second = step.layers

    by_hand = second(first(first.fuse(embedded, rows), step.make_cache()[0]), step.make_cache()[1])
    out = step(embedded, rows, step.make_cache())

    assert mx.allclose(out, second.finish(by_hand))
    assert not mx.allclose(out, by_hand), "the end norm did nothing"
    assert not mx.allclose(out, trunk.backbone.norm_f(out)), "normed a second time"


def test_a_step_with_a_recurrent_block_is_refused(trunk: NemotronH) -> None:
    """The chain re-runs the step per drafted token from a cache it just built, so a mamba
    state inside it would carry across runs that describe different sequences."""
    with pytest.raises(ValueError, match="recurrent block"):
        NemotronHMTP(trunk.config, ("M", "E"))


def test_speculation_reproduces_the_greedy_stream(trunk: NemotronH, step: NemotronHMTP) -> None:
    """The whole claim, on a trunk whose cache cannot be trimmed.

    fp32 and random weights, so `==` is the test and not a tolerance. `Acceptance.rate` is
    asserted above zero in the same run, because equality passes for free when every
    proposal is rejected — and rejection is exactly the path that restores and replays, so a
    run with no acceptances would also be a run that never exercised the other branch.
    """
    plain = list(stream_ids(trunk, PROMPT, max_tokens=40))

    for block in (2, 3, 4):
        counts = Acceptance()
        drafted = list(
            stream_speculative_ids(
                trunk,
                Chained(trunk, step, block=block, tap=len(trunk.config.pattern) - 1),
                PROMPT,
                max_tokens=40,
                acceptance=counts,
            )
        )
        assert drafted == plain, f"block {block}"
        assert counts.rounds > 0
        assert counts.rate is not None and counts.rate > 0.0, f"block {block} never accepted"


def test_the_chain_reads_the_last_row_and_starts_its_cache_fresh(
    trunk: NemotronH, step: NemotronHMTP
) -> None:
    """The proposer's mechanics, asserted directly — and they have to be, because nothing
    downstream can see them. A proposer that reads the wrong row, or feeds the target's
    hidden back instead of its own, or lets its cache outlive the round, still emits the
    target's own ids: it is only ever slower. The equality test above would pass on all
    three, and `rate > 0` survives them at chance on four ids.
    """
    proposer = Chained(trunk, step, block=3, tap=len(trunk.config.pattern) - 1)
    features = mx.random.normal((1, 4, trunk.config.hidden_size))
    proposer.absorb(features)
    committed = [*PROMPT, 2]

    drafted = proposer.propose(committed)

    hidden, token = features[:, -1:, :], mx.array([committed[-1]])[None]
    cache = step.make_cache()
    expected: list[mx.array] = []
    for _ in range(2):
        hidden = step(trunk.raw_embed(token), hidden, cache)
        token = mx.argmax(trunk.raw_logits(hidden)[:, -1, :], axis=-1)[None]
        expected.append(token[0])

    assert drafted.size == 2
    assert mx.array_equal(drafted, mx.concatenate(expected))

    # And the cache belonged to that round: a second proposal starts it over rather than
    # attending to the rows of the first. Asserted on the rows and not on the ids, because a
    # head that kept them would still answer the same ids most of the time — the extra rows
    # move the logits, and four of them is not a vocabulary that notices.
    # The attention layer and not every layer: the stateless kinds count their offset in
    # Python, which a traced link advances once for the whole graph, and nothing in a chain
    # ever reads it. What decides which rows a link attends is this one.
    watched = Chained(trunk, step, block=3, tap=0)
    watched.absorb(features)
    watched.propose(committed)
    assert watched._cache[0].rows == 2
    watched.propose(committed)
    assert watched._cache[0].rows == 2


def test_the_chain_refuses_a_target_that_lends_nothing(step: NemotronHMTP) -> None:
    with pytest.raises(SpeculationRefused, match="Draftable"):
        Chained(object(), step, block=2, tap=0)  # pyright: ignore[reportArgumentType]


def _unstack(leaves: dict[str, mx.array], prefix: str) -> dict[str, mx.array]:
    """The loader's `_stack_experts` run backwards, so a fixture can write a checkpoint in the
    spelling a *source* uses: one tensor per expert, which is what the entry no longer has."""
    written: dict[str, mx.array] = {}
    for name, value in leaves.items():
        if ".switch_mlp." not in name:
            written[f"{prefix}{name}"] = value
            continue
        head, tail = name.split(".switch_mlp.")
        leaf = {"fc1.weight": "up_proj", "fc2.weight": "down_proj"}[tail]
        for expert in range(value.shape[0]):
            written[f"{prefix}{head}.experts.{expert}.{leaf}.weight"] = value[expert]
    return written


# The four bytes `VOCAB` allows, in the byte-level alphabet `ByteLevelBPE` decodes through.
_TOKENIZER = {
    "added_tokens": [],
    "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False},
    "model": {"vocab": {"!": 0, '"': 1, "#": 2, "$": 3}, "merges": []},
}


def _checkpoint(directory: Path, trunk: NemotronH, step: NemotronHMTP | None) -> None:
    """A source checkpoint on disk: the trunk under its own names, the head under `mtp.` when
    there is one, and the least tokenizer that loads — `load` ends in the task, and a task
    needs one whatever this test is about."""
    tensors = _unstack(dict(tree_flatten(trunk.parameters())), "")
    if step is not None:
        tensors |= _unstack(dict(tree_flatten(step.parameters())), "mtp.")
    mx.save_safetensors(str(directory / "model.safetensors"), tensors)
    config = {
        key: value
        for key, value in vars(trunk.config).items()
        if not isinstance(value, tuple) or key == "hybrid_override_pattern"
    }
    config["model_type"] = "nemotron_h"
    config["hybrid_override_pattern"] = "".join(trunk.config.pattern)
    (directory / "config.json").write_text(json.dumps(config))
    (directory / "tokenizer.json").write_text(json.dumps(_TOKENIZER))


def test_quantizing_keeps_the_head_instead_of_deleting_it(
    trunk: NemotronH, step: NemotronHMTP, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bug this test exists for: the entry was written from the trunk's weight dict, which
    `_drop_mtp` had already emptied of the head — so quantizing did not compress the head, it
    deleted it, and the only way back was the source checkpoint.

    Four things have to hold together, and any one of them alone is an entry that cannot
    speculate: the tensors are in the file, the plan declares them, the trunk still loads
    *without* them, and the head loads back out.
    """
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

    # The trunk loads from an entry that now holds a second tree in the same file.
    assert isinstance(load(entry), CompositeModel)
    assert task.has_mtp(entry)

    rebuilt = task.mtp_head(entry)
    assert isinstance(rebuilt, NemotronHMTP)
    assert rebuilt.pattern == step.pattern
    rows = mx.random.normal((1, 2, trunk.config.hidden_size))
    assert mx.all(mx.isfinite(rebuilt(rows, rows))).item()


def test_an_entry_written_without_a_head_says_so(
    trunk: NemotronH, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A checkpoint of the same architecture that ships no head quantizes as it always did,
    and `has_mtp` answers for the file rather than for the family."""
    monkeypatch.setattr(task, "_CACHE", tmp_path / "quantized")
    source = tmp_path / "headless"
    source.mkdir()
    _checkpoint(source, trunk, None)

    load(source, quantize=Affine(group_size=32, bits=4))

    (entry,) = [path for path in (tmp_path / "quantized").glob("*/*") if path.is_dir()]
    assert not task.has_mtp(entry)
    with pytest.raises(ValueError, match="no MTP head"):
        task.mtp_head(entry)


def test_write_entry_is_where_the_head_is_merged_not_the_caller(
    trunk: NemotronH, step: NemotronHMTP, tmp_path: Path
) -> None:
    """There are two callers that write an entry — `load(quantize=…)` and the daemon's own
    quantization job — and the first version of this merged the head in one of them. That
    produced entries that quantized fine and could not speculate, twice: once for the library
    path and once for the daemon's. So the merge is asserted *here*, at the floor both stand
    on, over the plain weight dict a caller hands over.
    """
    source = tmp_path / "source"
    source.mkdir()
    _checkpoint(source, trunk, step)
    plan = {"backbone.layers.1.mixer.q_proj": Affine(group_size=32, bits=4)}

    task.write_entry(
        tmp_path / "entry",
        source,
        ("config.json", "tokenizer.json"),
        {"model_type": "nemotron_h"},
        {},
        plan,
        selection=Affine(group_size=32, bits=4),
    )

    packed = mx.load(str(tmp_path / "entry" / "model.safetensors"))
    assert [name for name in packed if name.startswith("mtp.")]
    declared = json.loads((tmp_path / "entry" / "config.json").read_text())["quantization"]
    assert [leaf for leaf in declared["leaves"] if leaf.startswith("mtp.")]
    # The caller's own plan survives beside the head's.
    assert "backbone.layers.1.mixer.q_proj" in declared["leaves"]


def test_an_entry_written_with_no_selection_carries_no_head(
    trunk: NemotronH, step: NemotronHMTP, tmp_path: Path
) -> None:
    """`selection=None` is a caller writing an entry for something that is not a model —
    a drafter's own, which has no head of its own to pack."""
    source = tmp_path / "source"
    source.mkdir()
    _checkpoint(source, trunk, step)

    task.write_entry(
        tmp_path / "entry", source, ("config.json",), {"model_type": "nemotron_h"}, {}, {}
    )

    packed = mx.load(str(tmp_path / "entry" / "model.safetensors"))
    assert not [name for name in packed if name.startswith("mtp.")]
