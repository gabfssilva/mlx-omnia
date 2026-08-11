"""Speculation may change the speed and nothing else.

The invariant is equality, not a tolerance: the ids a greedy run emits with a draft are the
ids it emits without one. Everything here is built so that a loop accepting one token the
target would have rejected shows up as a different id in the stream — the scripted double
below makes the target's script exact, and the counts pinned in
`test_the_acceptance_is_read_out_of_the_loop` are hand-derived from that script, so an
over-accepting loop breaks both the ids and the numbers.
"""

from collections.abc import Sequence
from pathlib import Path

import mlx.core as mx
import pytest
from huggingface_hub import hf_hub_download, snapshot_download

from sideros import GPT2, GPT2Tokenizer, KVCache, repetition_penalty, sampler, stream_ids, top_k
from sideros.core.cache import DeltaCache, RingKVCache
from sideros.generate import Meter
from sideros.models.gpt2 import CHECKPOINT
from sideros.speculative import Acceptance, SpeculationRefused, stream_speculative_ids

VOCAB = 64
PROMPT = [0]
TARGET = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
# Wrong at every position: `TARGET[i] + 1` is never `TARGET[i]`.
HOSTILE = [token + 1 for token in TARGET]
# Right on the even positions only.
ALTERNATING = [token if i % 2 == 0 else token + 1 for i, token in enumerate(TARGET)]
LOOKAHEAD = 2


class ScriptedLM:
    """Logits that depend only on the row's absolute position, read from the cache's offset.

    A rewound cache replays the same script, which is what makes the double usable under the
    trims a speculation round performs — a step counter would not survive the first rewind.
    The peak is unique (`-|arange - token|`), so the argmax has no tie to break, and the row
    at position p predicts `script[p]`: with a one-token prompt the i-th id generated is
    `script[i]`.
    """

    def __init__(self, script: list[int], vocab: int) -> None:
        self.script = script
        self.vocab = vocab

    def make_cache(self) -> list[KVCache]:
        return [KVCache()]

    def __call__(self, ids: mx.array, cache: list[KVCache] | None = None) -> mx.array:
        assert cache is not None, "the speculative loop always carries a cache"
        offset = cache[0].offset
        length = ids.shape[-1]
        rows = mx.array([self.script[min(offset + i, len(self.script) - 1)] for i in range(length)])
        consumed = mx.zeros((1, 1, length, 1))
        cache[0].update_and_fetch(consumed, consumed)
        distance = mx.abs(mx.arange(self.vocab)[None] - rows[:, None]).astype(mx.float32)
        return (-distance)[None]


class RecurrentLM:
    """A DeltaNet-shaped cache: the state a trim would cut cannot be rebuilt. As a *draft* it
    is still refused — the round holds no forward of the draft to replay through."""

    def make_cache(self) -> list[DeltaCache]:
        return [DeltaCache()]

    def __call__(self, ids: mx.array, cache: list[DeltaCache] | None = None) -> mx.array:
        raise AssertionError("the refusal must land before any forward")


class RotatingLM:
    """A ring, which can neither trim nor restore: slot `j` holds absolute position
    `j % window`, so a rejected row is written over one the ring still needs."""

    def make_cache(self) -> list[RingKVCache]:
        return [RingKVCache(8)]

    def __call__(self, ids: mx.array, cache: list[RingKVCache] | None = None) -> mx.array:
        raise AssertionError("the refusal must land before any forward")


class ReplayingLM(ScriptedLM):
    """The same script over a cache that cannot trim, only start over.

    `state` is the running sum of every id the cache has taken, which is the cheapest thing
    that is *wrong* if a rewind keeps a row the target rejected: a trim would leave the sum
    with the rejected id still in it, and a restore that forgot to replay would leave it
    short. The script keeps the logits a pure function of position, so the ids alone would
    not catch either — the sum is what does.
    """

    def __init__(self, script: list[int], vocab: int) -> None:
        super().__init__(script, vocab)
        self.handed: list[list[DeltaCache]] = []
        """Every cache this model made, so a test can read the one the loop kept — the
        speculative stream builds its own and hands it to nobody."""

    def make_cache(self) -> list[DeltaCache]:  # type: ignore[override]
        self.handed.append([DeltaCache()])
        return self.handed[-1]

    def __call__(self, ids: mx.array, cache: list[DeltaCache] | None = None) -> mx.array:  # type: ignore[override]
        assert cache is not None, "the speculative loop always carries a cache"
        layer = cache[0]
        offset = layer.offset
        length = ids.shape[-1]
        running = layer.state if layer.state is not None else mx.zeros((1,))
        # Reassigned, never mutated: that is what makes `checkpoint()` a restore point.
        layer.state = running + mx.sum(ids.astype(mx.float32))
        layer.offset = offset + length
        rows = mx.array([self.script[min(offset + i, len(self.script) - 1)] for i in range(length)])
        distance = mx.abs(mx.arange(self.vocab)[None] - rows[:, None]).astype(mx.float32)
        return (-distance)[None]


@pytest.fixture
def target() -> ScriptedLM:
    return ScriptedLM(TARGET, VOCAB)


@pytest.mark.parametrize("script", [TARGET, HOSTILE, ALTERNATING])
def test_speculation_reproduces_the_greedy_stream(target: ScriptedLM, script: list[int]) -> None:
    """Three drafts — one that is always right, one that is never right, one that is right
    every other position — and one stream. The hostile draft is the discriminating one: its
    proposals differ from the target's script at every slot, so a loop that accepted any of
    them would spell a different id here."""
    drafted = list(
        stream_ids(
            target,
            PROMPT,
            max_tokens=6,
            draft=ScriptedLM(script, VOCAB),
            lookahead=LOOKAHEAD,
        )
    )
    assert drafted == list(stream_ids(target, PROMPT, max_tokens=6))
    assert drafted == TARGET[:6]


@pytest.mark.parametrize(
    ("script", "rounds", "proposed", "accepted"),
    [
        # Six ids: the prompt's forward yields one, and each round settles the tokens it
        # accepted plus the target's own. Perfect: two rounds of three. Hostile: nothing is
        # accepted, so five rounds of one. Alternating: one round accepts nothing and two
        # accept a single proposal each.
        (TARGET, 2, 4, 4),
        (HOSTILE, 5, 10, 0),
        (ALTERNATING, 3, 6, 2),
    ],
)
def test_the_acceptance_is_read_out_of_the_loop(
    target: ScriptedLM, script: list[int], rounds: int, proposed: int, accepted: int
) -> None:
    """The count the bench reads. It is the loop's own tally, not a rate inferred from how
    many tokens came out — and it moves with the draft, which is also what proves the draft
    is being consulted at all."""
    counts = Acceptance()
    generated = list(
        stream_ids(
            target,
            PROMPT,
            max_tokens=6,
            draft=ScriptedLM(script, VOCAB),
            lookahead=LOOKAHEAD,
            acceptance=counts,
        )
    )
    assert generated == TARGET[:6]
    assert (counts.rounds, counts.proposed, counts.accepted) == (rounds, proposed, accepted)
    assert counts.rate == accepted / proposed


def test_a_stop_token_ends_the_round_it_lands_in(target: ScriptedLM) -> None:
    """A round settles three ids at once; the stop token is the third. What the stream owes
    is the two before it — a loop that flushed the round it already computed would emit the
    stop token itself."""
    stop = {TARGET[3]}
    drafted = list(
        stream_ids(
            target,
            PROMPT,
            max_tokens=6,
            stop=stop,
            draft=ScriptedLM(TARGET, VOCAB),
            lookahead=LOOKAHEAD,
        )
    )
    assert drafted == TARGET[:3]
    assert drafted == list(stream_ids(target, PROMPT, max_tokens=6, stop=stop))


def test_the_meter_counts_the_ids_the_speculative_loop_emitted(target: ScriptedLM) -> None:
    """`usage` is the same contract on this path: the prompt before the first forward, one
    mark per id handed to the consumer — not per id a round settled."""
    meter = Meter()
    generated = list(
        stream_speculative_ids(
            target,
            ScriptedLM(TARGET, VOCAB),
            PROMPT,
            max_tokens=5,
            lookahead=LOOKAHEAD,
            meter=meter,
        )
    )
    assert len(generated) == 5
    assert meter.prompt_tokens == len(PROMPT)
    assert meter.completion_tokens == 5
    assert meter.ttft is not None


def test_a_meter_carries_the_acceptance_without_being_asked(target: ScriptedLM) -> None:
    """What a screen needs to say a turn speculated at all. It is not in the rate — the same
    model answers a sampled request with no drafter at all — and a caller that measures a
    generation should not have to know that speculation is a second object to pass.

    A run with no draft leaves it at zero, which is what "did not speculate" is: the same
    reading for a model with no drafter and for a request one could not verify."""
    meter = Meter()
    list(
        stream_ids(
            target,
            PROMPT,
            max_tokens=6,
            meter=meter,
            draft=ScriptedLM(TARGET, VOCAB),
            lookahead=LOOKAHEAD,
        )
    )
    assert meter.speculation.rounds > 0
    assert meter.speculation.accepted == meter.speculation.proposed

    plain = Meter()
    list(stream_ids(target, PROMPT, max_tokens=6, meter=plain))

    assert (plain.speculation.rounds, plain.speculation.proposed) == (0, 0)


def test_an_explicit_acceptance_still_wins(target: ScriptedLM) -> None:
    """The meter's is a default and not a redirection: a caller with somewhere else to put
    the counts — the bench keeps them per arm — gets them there, and the meter's own stays
    at zero rather than holding half a tally."""
    counts = Acceptance()
    meter = Meter()
    list(
        stream_ids(
            target,
            PROMPT,
            max_tokens=6,
            meter=meter,
            acceptance=counts,
            draft=ScriptedLM(TARGET, VOCAB),
            lookahead=LOOKAHEAD,
        )
    )

    assert counts.rounds > 0
    assert meter.speculation.rounds == 0


def test_a_cache_that_can_neither_trim_nor_replay_refuses_the_draft(target: ScriptedLM) -> None:
    """Refused by name at either end, before a single forward runs — the doubles assert as
    much. A ring is the target's case: it cannot drop the rejected rows and it cannot be
    restored either, because it wrote over rows it still needs. A recurrent *draft* is the
    other: replay is a forward, and the round holds the target's and not the draft's."""
    with pytest.raises(SpeculationRefused, match="target's cache keeps recurrent state"):
        list(stream_ids(RotatingLM(), PROMPT, max_tokens=4, draft=ScriptedLM(TARGET, VOCAB)))
    with pytest.raises(SpeculationRefused, match="draft's cache keeps recurrent state"):
        list(stream_ids(target, PROMPT, max_tokens=4, draft=RecurrentLM()))


@pytest.mark.parametrize("script", [TARGET, HOSTILE, ALTERNATING])
def test_a_recurrent_target_speculates_by_replay(script: list[int]) -> None:
    """The rewind a state cannot do by trimming, done by starting over.

    Two claims, and the second is the one a trim would pass: the ids are the greedy ids, and
    the cache lands on the state a run that never speculated would hold. `HOSTILE` is the
    discriminating script — every round rejects, so every round restores and replays.
    """
    draft = ScriptedLM(script, VOCAB)
    speculated = list(
        stream_ids(ReplayingLM(TARGET, VOCAB), PROMPT, max_tokens=8, draft=draft)
    )
    plain = list(stream_ids(ReplayingLM(TARGET, VOCAB), PROMPT, max_tokens=8))
    assert speculated == plain

    model = ReplayingLM(TARGET, VOCAB)
    list(stream_ids(model, PROMPT, max_tokens=8, draft=ScriptedLM(script, VOCAB)))
    (kept,) = model.handed
    # A round settles what it accepted plus one, so the loop stops holding ids it never
    # emitted: the cache is longer than the stream and the reference has to be read from the
    # true continuation, cut to exactly what the cache took.
    reference = ReplayingLM(TARGET, VOCAB)
    truth = [*PROMPT, *stream_ids(ReplayingLM(TARGET, VOCAB), PROMPT, max_tokens=16)]
    # Both ends of the offset, and the floor is the one that bites: a rewind that restores
    # and forgets to replay leaves a cache that is *consistent* and short, which the state
    # comparison below would happily confirm against its own smaller prefix.
    assert len(PROMPT) + 8 - 1 <= kept[0].offset <= len(truth)
    straight = reference.make_cache()
    reference(mx.array(truth[: kept[0].offset])[None], straight)
    assert kept[0].state is not None and straight[0].state is not None
    assert mx.array_equal(kept[0].state, straight[0].state)


def test_what_cannot_be_verified_is_refused_instead_of_approximated(target: ScriptedLM) -> None:
    """A `Sampler` returns an id and no distribution, so the ratio the sampled acceptance
    rule needs is not reachable — including through a filter chain that happens to be
    greedy. A penalty is refused for its own reason: a verification row would have to be
    penalized against a history the round has not committed yet."""
    draft = ScriptedLM(TARGET, VOCAB)
    with pytest.raises(SpeculationRefused, match="greedy-only"):
        list(
            stream_ids(
                target,
                PROMPT,
                max_tokens=4,
                sampler=sampler(top_k(1), seed=0),
                draft=draft,
            )
        )
    with pytest.raises(SpeculationRefused, match="penalty"):
        list(
            stream_ids(
                target, PROMPT, max_tokens=4, penalty=repetition_penalty(1.1), draft=draft
            )
        )
    with pytest.raises(ValueError, match="lookahead"):
        list(stream_ids(target, PROMPT, max_tokens=4, draft=draft, lookahead=0))


@pytest.fixture(scope="module")
def gpt2() -> GPT2:
    directory = Path(snapshot_download("gpt2", allow_patterns=["config.json", "model.safetensors"]))
    return CHECKPOINT.load(directory, None)


@pytest.fixture(scope="module")
def tokenizer() -> GPT2Tokenizer:
    return GPT2Tokenizer.from_files(
        Path(hf_hub_download("gpt2", "vocab.json")),
        Path(hf_hub_download("gpt2", "merges.txt")),
    )


def test_speculation_reproduces_greedy_on_a_real_model(
    gpt2: GPT2, tokenizer: GPT2Tokenizer
) -> None:
    """The scripted double keeps its own positions; a real trunk keeps keys, and a rejected
    proposal leaves rows in twelve caches that describe a sequence that never happened. The
    draft proposes one word forever, so most rounds reject and every round rewinds."""
    prompt = list(tokenizer.encode("Hello, my name is"))
    plain = list(stream_ids(gpt2, prompt, max_tokens=16))
    counts = Acceptance()
    drafted = list(
        stream_ids(
            gpt2,
            prompt,
            max_tokens=16,
            draft=ScriptedLM(list(tokenizer.encode(" the")), len(tokenizer.encoder)),
            acceptance=counts,
        )
    )
    assert drafted == plain
    assert counts.accepted < counts.proposed, "no rejection: the rewind was never exercised"


def test_a_draft_that_is_the_target_is_almost_always_accepted(
    gpt2: GPT2, tokenizer: GPT2Tokenizer
) -> None:
    """Identity does not price the speculation: a loop that fed the draft the wrong context
    would still emit the target's ids, only never accept anything. With the target as its own
    draft the two only disagree where a batched row rounds away from a single one (fp32 here,
    ~1e-6 by `test_stepwise_matches_prefill`), so a rate at half is a misfeed and not
    arithmetic."""
    prompt = list(tokenizer.encode("Hello, my name is"))
    counts = Acceptance()
    drafted = list(stream_ids(gpt2, prompt, max_tokens=16, draft=gpt2, acceptance=counts))
    assert drafted == list(stream_ids(gpt2, prompt, max_tokens=16))
    assert counts.rate is not None
    assert counts.rate > 0.5


class ScriptedBlockLM(ScriptedLM):
    """The same script, through `generate.BlockOutputs`. The features are the rows' own
    absolute positions repeated once per tap, which is what lets a test say exactly which
    positions the loop handed over — a loop that passed the rejected tail, or passed a
    block twice, spells it here as a gap or a repeat."""

    def block_outputs(
        self, ids: mx.array, cache: list[KVCache], *, at: Sequence[int]
    ) -> tuple[mx.array, mx.array]:
        offset = cache[0].offset
        length = ids.shape[-1]
        logits = self(ids, cache)
        positions = mx.arange(offset, offset + length).astype(mx.float32)
        return logits, mx.stack([positions] * len(at), axis=-1)[None]


class BlockProposer:
    """One forward per round, like DFlash: it proposes `width` ids at once and reads the
    target's blocks instead of its ids."""

    def __init__(self, script: list[int], width: int, taps: tuple[int, ...] = (1, 3)) -> None:
        self._script = script
        self._width = width
        self._taps = taps
        self.absorbed: list[int] = []
        self.anchors: list[int] = []
        self.settled: list[int] = []

    @property
    def taps(self) -> Sequence[int]:
        return self._taps

    @property
    def width(self) -> int:
        return self._width

    def absorb(self, features: mx.array) -> None:
        assert features.shape[-1] == len(self._taps)
        rows = features[0, :, 0].tolist()
        assert isinstance(rows, list)
        self.absorbed.extend(int(row) for row in rows)

    def propose(self, committed: Sequence[int]) -> mx.array:
        self.anchors.append(committed[-1])
        base = len(committed) - 1
        last = len(self._script) - 1
        return mx.array([self._script[min(base + i, last)] for i in range(self._width)])

    def settle(self, length: int) -> None:
        self.settled.append(length)


@pytest.mark.parametrize("script", [TARGET, HOSTILE, ALTERNATING])
def test_a_block_proposer_reproduces_the_greedy_stream(script: list[int]) -> None:
    """The same invariant as the autoregressive draft, over a proposer that writes a whole
    block per round: what changes is who proposes, never what is emitted."""
    target = ScriptedBlockLM(TARGET, VOCAB)
    drafted = list(stream_ids(target, PROMPT, max_tokens=6, draft=BlockProposer(script, 3)))
    assert drafted == list(stream_ids(ScriptedLM(TARGET, VOCAB), PROMPT, max_tokens=6))
    assert drafted == TARGET[:6]


def test_the_proposer_reads_every_kept_position_exactly_once() -> None:
    """What `absorb` is: the positions the target's cache took and kept, in order, with the
    rejected tail of every round left out. Contiguous from zero and as long as the cache is
    — a loop that handed over the whole verification block would repeat positions here, and
    one that forgot the prefill would start past zero.

    The hostile script is the discriminating one: every round rejects, so every round has a
    tail to leave out."""
    target = ScriptedBlockLM(TARGET, VOCAB)
    proposer = BlockProposer(HOSTILE, 3)

    generated = list(stream_ids(target, PROMPT, max_tokens=6, draft=proposer))

    assert generated == TARGET[:6]
    assert proposer.absorbed == list(range(len(proposer.absorbed)))
    # One row per position the target committed: the prompt, plus every id but the last —
    # which is the anchor of the round that never ran.
    assert len(proposer.absorbed) == len(PROMPT) + len(generated) - 1
    assert proposer.anchors == TARGET[:5]
    assert proposer.settled == [2, 3, 4, 5, 6]


def test_a_target_with_no_blocks_to_read_refuses_the_proposer() -> None:
    """A proposer that conditions on the trunk against a model that does not hand it out.
    Refused by name before the first forward: what it would read instead is nothing."""
    with pytest.raises(SpeculationRefused, match="block_outputs"):
        list(
            stream_ids(
                ScriptedLM(TARGET, VOCAB), PROMPT, max_tokens=4, draft=BlockProposer(TARGET, 3)
            )
        )
