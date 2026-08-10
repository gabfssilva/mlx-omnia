"""DFlash on the pair it was trained for: the drafter proposes, the target's argmax decides.

The claim is the one every speculative path in this package makes — the ids are the greedy
ids, whatever proposed them — and here it is made against two real checkpoints, because
what the scripted doubles in `test_speculative` cannot exercise is the arithmetic: rope at
two different offsets in one forward, a cache that grows by the accepted run, and a
proposal read through the *target's* embedding table and head.

The second claim is that the drafter is being consulted at all. A misfed drafter still
emits the target's ids — it only never has one accepted — so a rate is asserted and not
just the stream.
"""

import math
from collections.abc import Sequence
from dataclasses import replace

import mlx.core as mx
import mlx.nn as nn
import pytest
from conftest import checkpoint_dir, requires_checkpoint

from sideros import (
    GenerationOptions,
    load,
    repetition_penalty,
    sampler,
    stream_ids,
    top_k,
)
from sideros.bpe import ByteLevelBPE
from sideros.models.muse_glimmer import CHECKPOINT, MuseGlimmer, load_assistant
from sideros.models.muse_glimmer.dflash import (
    DEFAULT_BLOCK,
    MuseGlimmerAssistant,
    MuseGlimmerDFlash,
)
from sideros.models.muse_glimmer.model import MuseGlimmerLanguageModel
from sideros.speculative import Acceptance, Proposer

REPO = "local/Muse-Glimmer-30B-4bit"
"""The quantized target, not the bf16 one: this is the shape the pair actually decodes in
— the drafter is 5.1 GB of bf16 against 17 GB of 4-bit — and it is 57 GB less to read."""
DRAFTER = "meta-models/Muse-Glimmer-30B-assistant"

PROMPT = "The three laws of thermodynamics are"
TOKENS = 24


@pytest.fixture(scope="module")
def target() -> MuseGlimmer:
    return CHECKPOINT.load(checkpoint_dir(REPO), None)


@pytest.fixture(scope="module")
def drafter() -> MuseGlimmerAssistant:
    return load_assistant(checkpoint_dir(DRAFTER))


@pytest.fixture(scope="module")
def tokenizer() -> ByteLevelBPE:
    return ByteLevelBPE.from_file(checkpoint_dir(REPO) / "tokenizer.json")


def _ulp(magnitude: float, dtype: mx.Dtype) -> float:
    """The spacing of the dtype the head writes in, at this magnitude. Not a tolerance —
    two logits a single ulp apart are the same number as far as the checkpoint can say."""
    bits = {mx.bfloat16: 8, mx.float16: 11, mx.float32: 24}[dtype]
    return math.ldexp(1.0, math.frexp(abs(magnitude))[1] - bits)


@requires_checkpoint(REPO)
@requires_checkpoint(DRAFTER)
def test_dflash_reproduces_the_greedy_stream(
    target: MuseGlimmer, drafter: MuseGlimmerAssistant, tokenizer: ByteLevelBPE
) -> None:
    """Ids for ids, except where the two candidates are the same number.

    A round judges its proposals from a 16-row forward and the free loop draws from a
    one-row one; on a 4-bit checkpoint the two regimes round differently, and where the top
    two logits sit within an ulp of each other the argmax flips. That is a tie and not an
    acceptance the target would have refused — which is why the divergence is weighed in
    the *free* loop's own regime, replaying the prefix and feeding the last id alone: there
    the two candidates have to be indistinguishable, or the round accepted something.
    """
    prompt = tokenizer.encode(PROMPT)
    plain = list(stream_ids(target, prompt, max_tokens=TOKENS))

    counts = Acceptance()
    proposer = MuseGlimmerDFlash(target, drafter)
    drafted = list(stream_ids(target, prompt, max_tokens=TOKENS, draft=proposer, acceptance=counts))

    assert len(drafted) == len(plain)
    steps = enumerate(zip(drafted, plain, strict=True))
    divergence = next((step for step, (ours, free) in steps if ours != free), None)
    if divergence is not None:
        cache = target.make_cache()
        context = prompt + plain[:divergence]
        target(mx.array(context[:-1])[None], cache)
        logits = target(mx.array(context[-1:])[None], cache)[0, -1]
        pair = mx.array([logits[plain[divergence]], logits[drafted[divergence]]])
        gap = float(mx.abs(pair[0] - pair[1]).item())
        floor = _ulp(float(mx.abs(pair).max().item()), logits.dtype)
        assert gap <= floor, (
            f"id {drafted[divergence]} over {plain[divergence]} at step {divergence} by "
            f"{gap}, and the head's own spacing there is {floor}"
        )

    assert counts.rate is not None and counts.rate > 0.1, (
        f"{counts.accepted}/{counts.proposed} accepted: the drafter is being fed something "
        "it cannot read"
    )
    # The whole point of a block: fewer reads of the target's weights than tokens out.
    assert counts.rounds < len(drafted)


@requires_checkpoint(DRAFTER)
def test_the_drafter_is_a_proposer_and_says_what_it_reads(drafter: MuseGlimmerAssistant) -> None:
    """The contract the loop is written against, and the two numbers a round is shaped by.

    The taps are the checkpoint's; the block is *not*. What the checkpoint declares is what
    the drafter was trained to write in one forward (16), and what a round is worth
    proposing is where the target's rows are still free — four here, measured. Reading the
    trained length as the default is what made the feature a loss out of the box."""
    proposer = MuseGlimmerDFlash(None, drafter)  # pyright: ignore[reportArgumentType]

    assert isinstance(proposer, Proposer)
    assert proposer.width == DEFAULT_BLOCK - 1
    assert drafter.config.block_size > DEFAULT_BLOCK
    assert list(proposer.taps) == list(drafter.config.target_layer_ids)


@requires_checkpoint(DRAFTER)
def test_a_shorter_block_is_a_shorter_proposal(drafter: MuseGlimmerAssistant) -> None:
    """The knob the settings expose. Below the checkpoint's own length it is a choice; above
    it there is nothing to write with, and asking is a `ValueError` rather than a block of
    mask rows the drafter never saw in training."""
    shorter = MuseGlimmerDFlash(None, drafter, block_size=8)  # pyright: ignore[reportArgumentType]
    assert shorter.width == 7
    with pytest.raises(ValueError, match="writes"):
        MuseGlimmerDFlash(None, drafter, block_size=32)  # pyright: ignore[reportArgumentType]


@requires_checkpoint(REPO)
def test_block_outputs_is_the_same_forward(target: MuseGlimmer, tokenizer: ByteLevelBPE) -> None:
    """`BlockOutputs` is a selection and not a second path: the logits it returns are the
    logits `__call__` returns, and the features are the trunk's own rows at those depths."""
    ids = mx.array(tokenizer.encode(PROMPT))[None]
    at = (1, 13)

    logits, features = target.block_outputs(ids, target.make_cache(), at=at)
    reference = target.activations(ids, target.make_cache())

    assert mx.array_equal(logits, reference.logits)
    hidden = target.config.hidden_size
    assert features.shape == (1, ids.shape[1], len(at) * hidden)
    for slot, block in enumerate(at):
        taken = features[..., slot * hidden : (slot + 1) * hidden]
        assert mx.array_equal(taken, reference.blocks[block])


class _WrongAnchor(MuseGlimmerDFlash):
    """The block written against a token that is not the one the target is about to
    continue from."""

    def propose(self, committed: Sequence[int]) -> mx.array:
        return super().propose([*committed[:-1], committed[0]])


class _RotatedTaps(MuseGlimmerDFlash):
    """The five blocks handed over in the wrong order — the encoder folds them by position,
    so block 13's rows arrive where block 1's belong.

    Reading one block *deeper* is not a mutation this catches, and it is not one: the
    residual stream carries block i into block i + 1, and the drafter goes on reading
    nearly the same signal. What the order breaks is the fold itself.
    """

    def absorb(self, features: mx.array) -> None:
        super().absorb(mx.roll(features, features.shape[-1] // len(self.taps), axis=-1))


@requires_checkpoint(REPO)
@requires_checkpoint(DRAFTER)
@pytest.mark.parametrize("mutant", [_WrongAnchor, _RotatedTaps])
def test_a_broken_pairing_shows_up_as_acceptance_and_never_as_text(
    target: MuseGlimmer,
    drafter: MuseGlimmerAssistant,
    tokenizer: ByteLevelBPE,
    mutant: type[MuseGlimmerDFlash],
) -> None:
    """Both halves of the contract at once. Feed the drafter the wrong anchor, or the wrong
    depth of the trunk, and it goes on writing plausible ids that the target then refuses:
    the acceptance collapses and the stream does not move. It is the mutation test for
    everything the round wires up — a loop that verified nothing would keep the rate."""
    prompt = tokenizer.encode(PROMPT)
    honest, broken = Acceptance(), Acceptance()
    good = list(
        stream_ids(
            target,
            prompt,
            max_tokens=TOKENS,
            draft=MuseGlimmerDFlash(target, drafter),
            acceptance=honest,
        )
    )
    bad = list(
        stream_ids(
            target, prompt, max_tokens=TOKENS, draft=mutant(target, drafter), acceptance=broken
        )
    )

    assert honest.rate is not None and broken.rate is not None
    assert broken.rate < honest.rate / 2, f"{broken.rate} against {honest.rate}"
    assert bad == good


@requires_checkpoint(REPO)
@requires_checkpoint(DRAFTER)
def test_the_facade_only_speculates_where_the_target_can_be_verified(
    drafter: MuseGlimmerAssistant,
) -> None:
    """The pairing as the daemon does it, and the four requests it does not speculate on.

    Not refusals: a sampled request, a penalized one and a constrained one are answered the
    way a model with no drafter answers them, because the acceptance rule that keeps a
    sampled distribution is not reachable through a `Sampler` (`speculative`'s docstring).
    """
    model = load(REPO, local_files_only=True)
    facade = model.model
    assert isinstance(facade, MuseGlimmerLanguageModel)
    assert facade._proposer(GenerationOptions(max_tokens=1)) is None

    facade.speculate_with(drafter, block_size=4)

    free = GenerationOptions(max_tokens=1)
    proposer = facade._proposer(free)
    assert proposer is not None and proposer.width == 3
    assert facade._proposer(replace(free, speculate=False)) is None
    assert facade._proposer(replace(free, sampler=sampler(top_k(1)))) is None
    assert facade._proposer(replace(free, penalty=repetition_penalty(1.1))) is None


@requires_checkpoint(DRAFTER)
def test_a_tree_that_is_not_a_drafter_is_refused_by_name(drafter: MuseGlimmerAssistant) -> None:
    """`speculate_with` is where a checkpoint id becomes a tree, so it is where a wrong one
    stops — before a generation reads five blocks that are not there."""
    model = load(REPO, local_files_only=True)
    facade = model.model
    assert isinstance(facade, MuseGlimmerLanguageModel)

    with pytest.raises(TypeError, match="not a Muse-Glimmer DFlash drafter"):
        facade.speculate_with(nn.Linear(4, 4))
    assert facade.drafter is None
