"""Speculation may change the speed and nothing else.

The invariant is equality, not a tolerance: the ids a greedy run emits with a draft are the
ids it emits without one. Everything here is built so that a loop accepting one token the
target would have rejected shows up as a different id in the stream — the scripted double
below makes the target's script exact, and the counts pinned in
`test_the_acceptance_is_read_out_of_the_loop` are hand-derived from that script, so an
over-accepting loop breaks both the ids and the numbers.
"""

from pathlib import Path

import mlx.core as mx
import pytest
from huggingface_hub import hf_hub_download, snapshot_download

from sideros import GPT2, GPT2Tokenizer, KVCache, repetition_penalty, sampler, stream_ids, top_k
from sideros.core.cache import DeltaCache
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
    """A DeltaNet-shaped cache: the recurrent state a trim would cut cannot be rebuilt."""

    def make_cache(self) -> list[DeltaCache]:
        return [DeltaCache()]

    def __call__(self, ids: mx.array, cache: list[DeltaCache] | None = None) -> mx.array:
        raise AssertionError("the refusal must land before any forward")


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


def test_a_recurrent_cache_refuses_the_draft(target: ScriptedLM) -> None:
    """Qwen3.6's DeltaNet state cannot be rewound to the accepted prefix. Refused by name at
    either end, before a single forward runs — the double asserts as much."""
    with pytest.raises(SpeculationRefused, match="target's cache keeps recurrent state"):
        list(stream_ids(RecurrentLM(), PROMPT, max_tokens=4, draft=ScriptedLM(TARGET, VOCAB)))
    with pytest.raises(SpeculationRefused, match="draft's cache keeps recurrent state"):
        list(stream_ids(target, PROMPT, max_tokens=4, draft=RecurrentLM()))


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
    prompt = tokenizer.encode("Hello, my name is")
    plain = list(stream_ids(gpt2, prompt, max_tokens=16))
    counts = Acceptance()
    drafted = list(
        stream_ids(
            gpt2,
            prompt,
            max_tokens=16,
            draft=ScriptedLM(tokenizer.encode(" the"), len(tokenizer.encoder)),
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
    prompt = tokenizer.encode("Hello, my name is")
    counts = Acceptance()
    drafted = list(stream_ids(gpt2, prompt, max_tokens=16, draft=gpt2, acceptance=counts))
    assert drafted == list(stream_ids(gpt2, prompt, max_tokens=16))
    assert counts.rate is not None
    assert counts.rate > 0.5
